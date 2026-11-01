r"""Mixed-precision and certified-factorization coverage, audit of October 31, 2026 (package tests-infra).

* A229 (gap-mixed-precision-routes-10): fp16 training end to end through ``train_step`` with an enabled CPU GradScaler,
  for the six optimizer classes the builder returns, each on the non-fused path (``GradScaler`` unscales and skips) and
  on the fused path (``_step_supports_amp_scaling``: the optimizer reads ``grad_scale`` and ``found_inf``), as on the
  RTX 5090. Overflows are injected into a Fisher table group, a clipped group and the frame table, on a silent and on a
  log step, and under gradient accumulation. A skipped step must leave every parameter and every optimizer-state
  tensor bitwise unchanged, back the scale off, hold the scheduler and the successful-update clock, and run no frame
  condition bound; an accepted step advances both clocks by one. The decode-bank Fisher step is checked the same way.
  CPU fused AdamW exists in torch 2.4 and later, so both paths run here; the builder selects the fused path from the
  device of the first parameter, which these tests answer as CUDA while the optimizers are built.
* A227 (gap-certified-factorization-routes-6, part): an aligned ill-conditioned key map at kappa(X) = 300, r = 8,
  where the factor certificate accepts every entry and the Gram certificate rejects it, under ``gram_certified``: the
  float64 autograd gradient agrees with float64 central differences, every key entry is rebuilt, and the float32
  gradients stay within 1e-3 of float64 on every input.

Tiny CPU shapes only.
"""
import copy
import math
import types
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest
import torch

import vfe3.relational_scoring as rs
import vfe3.train as T
from tests.profiles import config_values
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel

INF = float("inf")
F32, F64 = torch.float32, torch.float64


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# --------------------------------------------------------------------------------------------- A229


def _fp16_config(**overrides: Any) -> VFE3Config:
    values = dict(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=3, batch_size=2, n_layers=1, decode_mode="linear",
                  decode_logits_amp=True, amp_dtype="fp16", grad_clip_scope="group", fp32_matmul_precision="highest",
                  mu_init_std=0.3, phi_scale=0.05)
    values.update(overrides)
    return VFE3Config(**config_values(**values))


_VARIANTS: Dict[str, Tuple[str, Dict[str, Any]]] = {
    # name: (optimizer class, overrides)
    "adamw":                 ("AdamW", dict(phi_frame_condition_max=1.05)),
    "table_fisher":          ("TableFisherAdamW", dict(m_table_update_mode="fisher", phi_frame_condition_max=1.05)),
    "row_lazy":              ("RowLazyAdamW", dict(m_phi_sparse_rows=True, phi_frame_condition_max=1.05)),
    "row_lazy_table_fisher": ("RowLazyTableFisherAdamW", dict(m_phi_sparse_rows=True, m_table_update_mode="fisher")),
    "manifold":              ("GaugeManifoldAdamW", dict(m_phi_update_mode="pullback_group", pos_phi="none")),
    "manifold_table_fisher": ("TableFisherGaugeManifoldAdamW", dict(m_phi_update_mode="pullback_group",
                                                                     m_table_update_mode="fisher", pos_phi="none")),
}

# (label, injected parameter or None, log step, accumulation steps, expected did_step)
_SCENARIOS: Tuple[Tuple[str, Optional[str], bool, int, bool], ...] = (
    ("normal", None, False, 1, True),
    ("overflow in a Fisher table group (silent)", "prior_bank.mu_embed", False, 1, False),
    ("overflow in a clipped group (silent)", "prior_bank.output_proj_weight", False, 1, False),
    ("overflow in a Fisher table group (log step)", "prior_bank.mu_embed", True, 1, False),
    ("overflow in the frame table (silent)", "prior_bank.phi_embed", False, 1, False),
    ("normal again", None, False, 1, True),
    ("normal, two accumulation steps", None, False, 2, True),
    ("overflow, two accumulation steps", "prior_bank.mu_embed", False, 2, False),
)


@contextmanager
def _device_reads_as_cuda(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    r"""While the optimizers are built, the builder's ``next(model.parameters()).is_cuda`` reads True, so it builds the
    fused AdamW path that CUDA parameters take."""
    monkeypatch.setattr(T, "next", lambda iterator, *default: types.SimpleNamespace(is_cuda=True), raising=False)
    try:
        yield
    finally:
        monkeypatch.delattr(T, "next")


def _build(cfg: VFE3Config, fused: bool, monkeypatch: pytest.MonkeyPatch) -> Tuple[Any, ...]:
    torch.manual_seed(0)
    model = VFEModel(cfg)
    if fused:
        with _device_reads_as_cuda(monkeypatch):
            optimizer = T.build_optimizer(model, cfg)
    else:
        optimizer = T.build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    T._set_successful_update_count(optimizer, 0)
    return model, optimizer, scheduler, torch.amp.GradScaler(device="cpu", enabled=True)


def _snapshot(model: VFEModel, optimizer: torch.optim.Optimizer) -> Tuple[Dict[str, torch.Tensor], Dict[int, dict]]:
    params = {name: p.detach().clone() for name, p in model.named_parameters()}
    state = {id(p): {key: (value.detach().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value))
                     for key, value in entry.items()} for p, entry in optimizer.state.items()}
    return params, state


def _changes(before: Tuple[dict, dict], after: Tuple[dict, dict]) -> Tuple[List[str], List[str]]:
    params = [name for name in before[0] if not torch.equal(before[0][name], after[0][name])]
    moved_state = []
    for key in set(before[1]) | set(after[1]):
        a, b = before[1].get(key, {}), after[1].get(key, {})
        for name in set(a) | set(b):
            x, y = a.get(name), b.get(name)
            same = (torch.equal(x, y) if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) else x == y)
            if not same:
                moved_state.append(name)
    return params, sorted(set(moved_state))


def _step(model: VFEModel, optimizer: Any, scheduler: Any, scaler: Any, cfg: VFE3Config, seed: int,
          inject: Optional[str], log: bool, accumulation: int, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    handle = None
    if inject is not None:
        handle = dict(model.named_parameters())[inject].register_hook(lambda grad: torch.full_like(grad, INF))
    bound_calls: List[int] = []
    real_bound = T.bound_phi_frame_condition_

    def counted(*args: Any, **kwargs: Any) -> Any:
        bound_calls.append(1)
        return real_bound(*args, **kwargs)

    monkeypatch.setattr(T, "bound_phi_frame_condition_", counted)
    gen = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len), generator=gen)
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len), generator=gen)
    before = _snapshot(model, optimizer)
    status: Dict[str, Any] = {}
    scale, epoch, updates = float(scaler.get_scale()), scheduler.last_epoch, T._successful_update_count(optimizer)
    try:
        T.train_step(model, optimizer, scheduler, tokens, targets, grad_clip=1.0, grad_clip_scope="group",
                     grad_accum_steps=accumulation, scaler=scaler, metrics_out={} if log else None,
                     status_out=status)
    finally:
        monkeypatch.setattr(T, "bound_phi_frame_condition_", real_bound)
        if handle is not None:
            handle.remove()
    params, moved_state = _changes(before, _snapshot(model, optimizer))
    return dict(did_step=status.get("did_step"), scale=(scale, float(scaler.get_scale())),
                epoch=(epoch, scheduler.last_epoch), updates=(updates, T._successful_update_count(optimizer)),
                params=params, state=moved_state, bound_calls=len(bound_calls),
                finite=all(bool(torch.isfinite(p).all()) for p in model.parameters()))


def _check(label: str, outcome: Dict[str, Any], accepted: bool) -> None:
    assert outcome["did_step"] is accepted, (label, outcome)
    assert outcome["finite"], label
    if accepted:
        assert outcome["params"], label
        assert outcome["epoch"][1] == outcome["epoch"][0] + 1, label
        assert outcome["updates"][1] == outcome["updates"][0] + 1, label
    else:
        assert outcome["params"] == [] and outcome["state"] == [], (label, outcome)
        assert outcome["epoch"][1] == outcome["epoch"][0] and outcome["updates"][1] == outcome["updates"][0], label
        assert outcome["scale"][1] < outcome["scale"][0], label
        assert outcome["bound_calls"] == 0, label


@pytest.mark.parametrize("fused", [False, True], ids=["non_fused", "fused"])
@pytest.mark.parametrize("variant", sorted(_VARIANTS))
def test_fp16_train_step_skips_overflows_and_takes_finite_steps(variant: str, fused: bool,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    expected_class, overrides = _VARIANTS[variant]
    cfg = _fp16_config(**overrides)
    model, optimizer, scheduler, scaler = _build(cfg, fused, monkeypatch)
    assert type(optimizer).__name__ == expected_class
    assert bool(getattr(optimizer, "_step_supports_amp_scaling", False)) is fused
    for seed, (label, inject, log, accumulation, accepted) in enumerate(_SCENARIOS, start=1):
        _check(label, _step(model, optimizer, scheduler, scaler, cfg, seed, inject, log, accumulation, monkeypatch),
               accepted)


@pytest.mark.parametrize("fused", [False, True], ids=["non_fused", "fused"])
def test_fp16_decode_bank_fisher_step_skips_overflows(fused: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _fp16_config(decode_mode="diagonal_chunked", decode_logits_amp=False, untie_decode_bank=True,
                       m_decode_update_mode="fisher", m_table_update_mode="fisher", decode_av_precision="fp32",
                       decode_ranking_fp64_escalation=False)
    model, optimizer, scheduler, scaler = _build(cfg, fused, monkeypatch)
    marks = sorted(group["table_fisher"] for group in optimizer.param_groups if group.get("table_fisher"))
    assert marks == ["decode_mu", "decode_sigma", "mu", "sigma"]
    assert bool(getattr(optimizer, "_step_supports_amp_scaling", False)) is fused
    for seed, (label, inject, accepted) in enumerate((
            ("normal", None, True),
            ("overflow in the decode mean table", "prior_bank.decode_mu_embed", False),
            ("overflow in the decode log-variance table", "prior_bank.decode_sigma_log_embed", False),
            ("normal again", None, True)), start=1):
        _check(label, _step(model, optimizer, scheduler, scaler, cfg, seed, inject, False, 1, monkeypatch), accepted)


# --------------------------------------------------------------------------------------------- A227


def _conditioned(kappa: float, rows: int, cols: int, gen: torch.Generator) -> torch.Tensor:
    left = torch.linalg.qr(torch.randn(rows, rows, generator=gen, dtype=F64))[0]
    right = torch.linalg.qr(torch.randn(cols, rows, generator=gen, dtype=F64))[0]
    return (left * torch.logspace(0, -math.log10(kappa), rows, dtype=F64)) @ right.T


def _aligned_key_case(kappa: float = 300.0, seed: int = 1) -> Tuple[List[torch.Tensor], torch.Tensor]:
    r"""Comparison maps with S_k = S_q R, R a small rotation, so that the key map is as ill-conditioned as the query
    map and every pair is attention-relevant (r = d = 8, N = 6, one head)."""
    gen = torch.Generator().manual_seed(seed)
    N, H, K = 6, 1, 8
    S_q = _conditioned(kappa, 8, K, gen)[None]                                   # (H, r, K)
    A = 0.02 * torch.randn(K, K, generator=gen, dtype=F64)
    S_k = S_q @ torch.linalg.matrix_exp(A - A.T)                                  # (H, r, K)
    mu = 0.3 * torch.randn(1, N, K, generator=gen, dtype=F64)
    sigma = torch.exp(0.05 * torch.randn(1, N, K, generator=gen, dtype=F64))
    inverse = torch.linalg.matrix_exp(0.02 * torch.randn(1, N, H, K, K, generator=gen, dtype=F64))
    return [mu, sigma, inverse, S_q, S_k], torch.randn(1, H, N, N, generator=gen, dtype=F64)


def _aligned_cost(weights: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, inverse: torch.Tensor,
                  S_q: torch.Tensor, S_k: torch.Tensor) -> torch.Tensor:
    N, K = mu.shape[-2:]
    mu_h, sigma_h = mu.unsqueeze(-2).expand(1, N, 1, K), sigma.unsqueeze(-2).expand(1, N, 1, K)
    a_q, G_q = rs.projected_moments(mu_h, sigma_h, inverse, S_q)
    a_k, G_k = rs.projected_moments(mu_h, sigma_h, inverse, S_k)
    D = rs.projected_gaussian_kl(a_q, G_q, a_k, G_k, projection_q=(sigma_h, inverse, S_q, mu_h),
                                 projection_k=(sigma_h, inverse, S_k, mu_h), symmetric_inputs=True)  # (1, H, N, N)
    return (D * weights.to(D.dtype)).sum()


def _central_differences(weights: torch.Tensor, inputs: List[torch.Tensor], index: int, positions: List[int],
                         step: float = 1e-6) -> List[float]:
    r"""Float64 central differences of the weighted cost along coordinates ``positions`` of input ``index``."""
    out = []
    for position in positions:
        values = []
        for sign in (1.0, -1.0):
            moved = [x.detach().clone() for x in inputs]
            flat = moved[index].reshape(-1)
            flat[position] += sign * step
            with torch.no_grad():
                values.append(float(_aligned_cost(weights, *moved)))
        out.append((values[0] - values[1]) / (2.0 * step))
    return out


def test_gram_certified_rebuilds_an_ill_conditioned_key_map_to_float64_gradient_accuracy() -> None:
    r"""The arrangement no certified-route test had: the key map is as ill-conditioned as the query map (kappa(X) =
    300, r = 8), the factor certificate accepts every entry and the Gram certificate rejects it. Under
    ``gram_certified`` every entry is rebuilt and the float32 gradients agree with float64 central differences."""
    inputs, weights = _aligned_key_case()
    rs.set_projected_condition_recovery(False)
    rs.set_projected_factor_route("gram")
    leaves64 = [x.clone().requires_grad_(True) for x in inputs]
    _aligned_cost(weights, *leaves64).backward()
    rs.set_projected_condition_recovery(True)                                # gram_certified
    rs.reset_source_recovery_elements()
    leaves32 = [x.detach().to(F32).clone().requires_grad_(True) for x in inputs]
    _aligned_cost(weights, *leaves32).backward()
    assert rs.source_recovery_elements() > 0                                 # the flagged entries were rebuilt
    rs.set_projected_condition_recovery(False)
    gen = torch.Generator().manual_seed(2)
    for index, name in enumerate(("mu", "sigma", "inverse", "S_q", "S_k")):
        assert float((leaves32[index].grad.double() - leaves64[index].grad).norm()
                     / leaves64[index].grad.norm()) < 1e-3, name
        positions = torch.randint(0, inputs[index].numel(), (2,), generator=gen).tolist()
        for position, central in zip(positions, _central_differences(weights, inputs, index, positions)):
            tolerance = 1e-3 * max(1.0, abs(central))
            assert abs(float(leaves64[index].grad.reshape(-1)[position]) - central) <= tolerance, (name, position)
            assert abs(float(leaves32[index].grad.reshape(-1)[position]) - central) <= tolerance, (name, position)
