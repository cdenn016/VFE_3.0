r"""Route-level regression tests for test gaps of the October 31, 2026 audit (remediation package tests-infra).

* A219 (evaluation-metrics-6): strided evaluation on the live route against an independent dense per-target
  cross-entropy, with an exact and a padded final window, exact target accounting, prefix invariance, and the
  evaluation-protocol training CE.
* A220 (prior-bank-decode-7, torch part): float64 finite-difference checks of the chunked KL decode cross-entropy
  (both checkpoint branches, ragged chunks at V = 13) and its agreement with an independent
  :math:`-\mathrm{KL}(q\|p_v)/\tau + \log\pi_v` cross-entropy; the interpreted fused rows are in
  ``test_audit_20261031_tests_infra_triton.py``.
* A222 (config-launcher-9): the principled exact route, resolved from the launcher, under a whole-model local
  change of frame, at initialization scales where the beliefs-only control discriminates.
* A230 (gap-mstep-self-coupling-handoff-8): with the self divergence comparable to ``b0``, the gradient of the M-step
  self term is the detached-alpha one and differs from the unweighted and the undetached ones.
* A234 (posthoc-routing-10): the post-hoc Table M5 rows on the live profile.
* A238 (train-loop-lifecycle-9): a resume on the live profile with the log, evaluation and checkpoint cadence on.

Tiny CPU models only.
"""
import logging
import math
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tests.profiles import _CONFIG_VALUES, _LIVE_OMITTED, launcher_values, live_config, live_tiny, make_prior_bank
from tests.profiles import test_config
from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel

CPU = torch.device("cpu")
F64 = torch.float64


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# --------------------------------------------------------------------------------------------- A219


def _live_eval_model(N: int, stride: int, V: int) -> VFEModel:
    r"""The tiny live route under a sliding evaluation window, with a nonuniform unigram prior and every parameter
    moved off its initialization."""
    cfg = live_config(max_seq_len=N, eval_stride=stride, vocab_size=V)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    gen = torch.Generator().manual_seed(4)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and p.is_floating_point():
                p.add_(0.05 * torch.randn(p.shape, generator=gen, dtype=p.dtype))
    if cfg.decode_unigram_prior:
        model.prior_bank.set_unigram_log_prior(torch.randint(0, 40, (V,), generator=gen).double())
    return model.eval()


@pytest.mark.parametrize("T", [45, 46], ids=["exact_final_window", "padded_final_window"])
def test_strided_evaluation_matches_a_dense_per_target_ce_on_the_live_route(T: int) -> None:
    from vfe3.train import _eval_protocol_train_ce, evaluate

    N, s, V = 8, 3, 7
    model = _live_eval_model(N, s, V)
    stream = torch.randint(0, V, (T,), generator=torch.Generator().manual_seed(1))
    windows = TokenWindows(stream, N, stride=s, pad_final=True)
    # Every transition is scored exactly once, and the counts are the window arithmetic.
    covered = [k * s + p for k in range(len(windows)) for p in range(N) if int(windows[k][1][p]) != -100]
    assert sorted(covered) == list(range(T - 1))
    metrics = evaluate(model, DataLoader(windows, batch_size=3, shuffle=False), device=CPU)
    n_real_last = (T - 1) - (len(windows) - 1) * s
    assert metrics["scored_targets"] == T - 1
    assert metrics["expected_targets"] == len(windows) * N
    assert metrics["overlap_context_targets"] == (len(windows) - 1) * (N - s)
    assert metrics["padding_targets"] == N - n_real_last
    assert metrics["padding_targets"] == (0 if T == 45 else 2)
    assert metrics["excluded_targets"] == metrics["expected_targets"] - metrics["scored_targets"]
    # The independent oracle: dense logits of a full-context forward, a float64 cross-entropy per scored target.
    total, count = 0.0, 0
    with torch.no_grad():
        for k in range(len(windows)):
            x, y = windows[k]
            _, logits = model.forward_beliefs(x.unsqueeze(0), return_logits=True)          # (1, N, V)
            total += float(F.cross_entropy(logits.reshape(-1, V).double(), y, ignore_index=-100, reduction="sum"))
            count += int((y != -100).sum())
    assert count == T - 1
    assert abs(metrics["ce"] - total / count) < 1e-5
    assert metrics["ppl"] == pytest.approx(math.exp(metrics["ce"]), rel=1e-12)
    assert metrics["bits_per_token"] == pytest.approx(metrics["ce"] / math.log(2.0), rel=1e-12)
    # Prefix invariance: a replaced suffix, random or the zero padding of the final window, moves no earlier logit.
    gen = torch.Generator().manual_seed(2)
    x, _ = windows[1]
    with torch.no_grad():
        _, full = model.forward_beliefs(x.unsqueeze(0), return_logits=True)
        for cut in range(1, N):
            for suffix in (torch.randint(0, V, (N - cut,), generator=gen), torch.zeros(N - cut, dtype=x.dtype)):
                changed = torch.cat([x[:cut], suffix])
                _, logits = model.forward_beliefs(changed.unsqueeze(0), return_logits=True)
                assert torch.equal(logits[0, :cut], full[0, :cut]), cut
    # The evaluation-protocol training CE scores the last s targets of each disjoint window.
    tokens = torch.stack([stream[i * N: i * N + N] for i in range(4)])
    targets = torch.stack([stream[i * N + 1: i * N + N + 1] for i in range(4)])
    model.train()
    protocol = _eval_protocol_train_ce(model, tokens, targets, s)
    assert model.training
    model.eval()
    with torch.no_grad():
        _, logits = model.forward_beliefs(tokens, return_logits=True)
    tail = F.cross_entropy(logits[:, N - s:].reshape(-1, V).double(), targets[:, N - s:].reshape(-1))
    assert abs(protocol - float(tail)) < 1e-5


# --------------------------------------------------------------------------------------------- A220


_V, _K, _B, _N, _TAU = 13, 4, 2, 3, 0.1


def _decode_bank(decode_mode: str, *, full: bool = False) -> Any:
    r"""A float64 untied decode bank at V = 13 with a unigram prior, temperature 0.1, chunks of 5 (ragged: 5, 5, 3)."""
    bank = make_prior_bank(_V, _K, _K * _K, mu_init_std=0.5, decode_tau=_TAU, decode_mode=decode_mode,
                           decode_chunk_size=5, decode_av_precision="fp32", decode_unigram_prior=True,
                           untie_decode_bank=True, family="gaussian_full" if full else "gaussian_diagonal",
                           diagonal_covariance=not full, irrep_dims=[_K])
    bank.decode_log_scale.requires_grad_(False)
    bank = bank.double()
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        bank.decode_mu_embed.copy_(0.7 * torch.randn(_V, _K, generator=gen, dtype=F64))
        bank.decode_sigma_log_embed.copy_(0.5 * torch.randn(_V, _K, generator=gen, dtype=F64))
    counts = torch.randint(0, 50, (_V,), generator=gen).double()
    bank.set_unigram_log_prior(counts)
    return bank, torch.log((counts + 1.0) / (counts.sum() + _V))


def _query() -> tuple:
    gen = torch.Generator().manual_seed(2)
    mu_q = 0.8 * torch.randn(_B, _N, _K, generator=gen, dtype=F64)
    sigma_q = torch.exp(0.4 * torch.randn(_B, _N, _K, generator=gen, dtype=F64))
    targets = torch.randint(0, _V, (_B, _N), generator=gen)
    targets[0, 1] = -100                                       # an ignored target
    targets[1, 2] = _V - 1                                     # a target in the ragged last chunk
    return mu_q, sigma_q, targets


def _independent_ce(mu_q: torch.Tensor, sigma_q: torch.Tensor, mu_v: torch.Tensor, log_s_v: torch.Tensor,
                    log_pi: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    r"""Mean CE of the logits :math:`-\mathrm{KL}(\mathcal N(\mu_q, \mathrm{diag}\,\sigma_q) \| \mathcal N(\mu_v,
    \mathrm{diag}\,s_v))/\tau + \log\pi_v` over the scored targets."""
    s_v = torch.exp(log_s_v).clamp(min=1e-6)
    d = mu_q.unsqueeze(-2) - mu_v                                                     # (B, N, V, K)
    sq = sigma_q.unsqueeze(-2)                                                        # (B, N, 1, K)
    kl = 0.5 * ((sq + d ** 2) / s_v - 1.0 + torch.log(s_v) - torch.log(sq)).sum(-1)   # (B, N, V)
    logits = -kl / _TAU + log_pi
    return F.cross_entropy(logits.reshape(-1, _V), targets.reshape(-1), ignore_index=-100)


def _with_tables(bank: Any, mu_table: torch.Tensor, log_table: torch.Tensor, call: Any) -> torch.Tensor:
    bank._decode_mu_table = lambda: mu_table
    bank._decode_sigma_log_table = lambda: log_table
    try:
        return call()
    finally:
        del bank._decode_mu_table
        del bank._decode_sigma_log_table


@pytest.mark.parametrize("checkpoint", ["never", "always"])
def test_chunked_kl_decode_ce_has_the_finite_difference_gradient_of_an_independent_formula(checkpoint: str) -> None:
    bank, log_pi = _decode_bank("diagonal_chunked")
    bank.decode_ce_checkpoint = checkpoint                     # both branches of _decode_ce_should_checkpoint
    mu_q, sigma_q, targets = _query()

    def chunked(mu: torch.Tensor, sig: torch.Tensor, mu_v: torch.Tensor, log_v: torch.Tensor) -> torch.Tensor:
        return _with_tables(bank, mu_v, log_v, lambda: bank.decode_ce_diagonal_chunked(mu, sig, targets))

    leaves = [x.detach().clone().requires_grad_(True)
              for x in (mu_q, sigma_q, bank.decode_mu_embed, bank.decode_sigma_log_embed)]
    assert torch.autograd.gradcheck(chunked, tuple(leaves), eps=1e-6, atol=1e-6, rtol=1e-5)
    value = chunked(*leaves)
    value.backward()
    reference_leaves = [x.detach().clone().requires_grad_(True) for x in leaves]
    reference = _independent_ce(*reference_leaves, log_pi, targets)
    reference.backward()
    assert abs(float(value) - float(reference)) < 1e-12
    for got, want in zip(leaves, reference_leaves):
        assert torch.allclose(got.grad, want.grad, rtol=0, atol=1e-12)


def test_full_and_expected_likelihood_chunked_ce_pass_float64_gradcheck() -> None:
    mu_q, sigma_q, targets = _query()
    bank, _ = _decode_bank("full_chunked", full=True)
    bank.decode_ce_checkpoint = "always"
    tril = torch.tril_indices(_K, _K)

    def full(mu: torch.Tensor, packed: torch.Tensor, mu_v: torch.Tensor, log_v: torch.Tensor) -> torch.Tensor:
        L = torch.zeros(_B, _N, _K, _K, dtype=F64)
        L[..., tril[0], tril[1]] = packed
        Sigma = L @ L.transpose(-1, -2) + 0.3 * torch.eye(_K, dtype=F64)              # (B, N, K, K) SPD
        return _with_tables(bank, mu_v, log_v, lambda: bank.decode_ce_full_chunked(mu, Sigma, targets))

    packed = 0.5 * torch.randn(_B, _N, tril.shape[1], generator=torch.Generator().manual_seed(3), dtype=F64)
    assert torch.autograd.gradcheck(full, tuple(x.detach().clone().requires_grad_(True) for x in (
        mu_q, packed, bank.decode_mu_embed, bank.decode_sigma_log_embed)), eps=1e-6, atol=1e-6, rtol=1e-5)
    el_bank, _ = _decode_bank("expected_likelihood_chunked")
    el_bank.decode_ce_checkpoint = "never"

    def expected(mu: torch.Tensor, sig: torch.Tensor, mu_v: torch.Tensor, log_v: torch.Tensor) -> torch.Tensor:
        return _with_tables(el_bank, mu_v, log_v,
                            lambda: el_bank.decode_ce_expected_likelihood_chunked(mu, sig, targets))

    assert torch.autograd.gradcheck(expected, tuple(x.detach().clone().requires_grad_(True) for x in (
        mu_q, sigma_q, el_bank.decode_mu_embed, el_bank.decode_sigma_log_embed)), eps=1e-6, atol=1e-6, rtol=1e-5)


# --------------------------------------------------------------------------------------------- A222


@pytest.mark.parametrize("regime", ["sticky", "learned"])
def test_the_launcher_exact_route_is_invariant_under_a_local_change_of_frame(regime: str) -> None:
    r"""The principled exact route resolved from ``train_vfe3.py``, at tiny scale, with unit initialization scales.

    At the launcher's ``mu_init_std`` and ``sigma_init`` the logits barely depend on the encoded means (the audit's
    beliefs-only control read 1.3e-4), so the control could not show that the frames carry the invariance; the
    initialization scales are therefore raised, and no route switch is touched."""
    from tests import test_audit_20261016_a1_passive_gauge_model as gauge
    from vfe3.exact_route import principled_exact_route_failures, principled_exact_route_overrides

    launcher = launcher_values()
    try:
        overrides = principled_exact_route_overrides(launcher)
    except ValueError as exc:                                      # a refusal is the switch's contract
        pytest.skip(f"the launcher route has no exact-route reading: {exc}")
    cfg = VFE3Config(**{**_CONFIG_VALUES, **_LIVE_OMITTED, **launcher, **live_tiny(launcher), **overrides,
                        "max_seq_len": gauge.N, "batch_size": gauge.B, "mu_init_std": 1.0, "sigma_init": 1.0,
                        "phi_scale": 0.3, "block_memory_regime_transition": regime})
    assert principled_exact_route_failures(cfg) == []
    logits_change, loss_change, beliefs_only_change = gauge._invariance(gauge._model(cfg))
    assert logits_change < gauge._REL and loss_change < gauge._REL
    assert beliefs_only_change > 1e-3                               # the frames carry the invariance


# --------------------------------------------------------------------------------------------- A230


@pytest.mark.parametrize("alpha_mode", ["state_dependent", "state_dependent_per_coord"])
def test_the_self_term_gradient_carries_the_detached_alpha_weighting(alpha_mode: str) -> None:
    r"""At :math:`b_0 = 0.1` the self divergence :math:`D` exceeds :math:`b_0`, so :math:`\alpha^* = c_0/(b_0 + D)`
    is far from :math:`c_0/b_0`: the gradient of the term must be :math:`w\,\overline{\alpha^*\,\partial D}` with
    :math:`\alpha^*` held fixed, and the unweighted gradient :math:`w\,\overline{\partial D}` and the undetached one
    :math:`w\,\overline{\partial(\alpha^* D)}` must both differ from it. Only gradients are compared, so the check holds
    whichever value (the detached product or the profiled energy) the term reports."""
    from vfe3.alpha_i import alpha_is_per_coord, self_coupling_alpha
    from vfe3.families import get_family
    from vfe3.free_energy import self_divergence_for_alpha

    w, b0, c0 = 0.5, 0.1, 1.0
    values = dict(vocab_size=11, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                  attention_mode="projected_predictive", predictive_covariance="moment_matched",
                  lambda_alpha_mode=alpha_mode, b0=b0, c0=c0, mu_init_std=1.0, kl_max=1e4, seed=0)
    gen = torch.Generator().manual_seed(5)
    tokens = torch.randint(0, 11, (3, 5), generator=gen)
    targets = torch.randint(0, 11, (3, 5), generator=gen)

    def gradients(weight: float, reference: Any = None) -> Dict[str, torch.Tensor]:
        torch.manual_seed(0)
        model = VFEModel(test_config(**values, mstep_self_coupling_weight=weight)).double()
        if reference is not None:                                    # the self term alone, recomputed
            cfg = model.cfg
            capture: Dict[str, Any] = {}
            model.forward_beliefs(tokens, capture=capture)
            q, (mu_p, sigma_p) = capture["converged"], capture["final_block_prior"]
            fam = get_family(cfg.family)
            D = self_divergence_for_alpha(fam(q.mu, q.sigma), fam(mu_p, sigma_p), kl_max=cfg.kl_max, eps=cfg.eps,
                                          lambda_alpha_mode=alpha_mode)
            assert float(D.mean()) > b0                              # not the D << b0 regime where alpha* = c0/b0
            alpha, _ = self_coupling_alpha(D, mode=alpha_mode, b0=torch.tensor(b0, dtype=F64),
                                           c0=torch.tensor(c0, dtype=F64))
            term = reference(alpha, D)
            loss = w * (term.sum(dim=-1) if alpha_is_per_coord(alpha_mode) else term).mean()
        else:
            _, loss, _ = model(tokens, targets)
        loss.backward()
        return {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}

    with_term, ce_only = gradients(w), gradients(0.0)
    term = {name: with_term[name] - ce_only.get(name, torch.zeros_like(with_term[name])) for name in with_term}
    detached = gradients(w, lambda alpha, D: alpha.detach() * D)
    unweighted = gradients(w, lambda alpha, D: D)
    undetached = gradients(w, lambda alpha, D: alpha * D)
    scale = max(float(g.abs().max()) for g in detached.values())
    assert scale > 0.0
    for name, grad in detached.items():
        assert torch.allclose(term[name], grad, rtol=1e-7, atol=1e-10 * scale), name
    assert max(float((unweighted[name] - detached[name]).abs().max()) for name in detached) > 1e-2 * scale
    assert max(float((undetached[name] - detached[name]).abs().max()) for name in detached) > 1e-2 * scale


# --------------------------------------------------------------------------------------------- A234


def test_posthoc_table_m5_rows_on_the_live_profile() -> None:
    from vfe3.posthoc import evaluate as ev
    from vfe3.posthoc import interventions as iv
    from vfe3.posthoc import report as rp
    from vfe3.posthoc.attention import AttentionStatsObserver
    from vfe3.posthoc.klcap import KLCapObserver

    V, N = 9, 8
    cfg = live_config(vocab_size=V, max_seq_len=N, eval_stride=4)
    torch.manual_seed(0)
    model = VFEModel(cfg).eval()
    gen = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for name, p in model.named_parameters():
            scale = 0.3 if "phi_embed" in name else 0.5 if "sigma" in name else 0.2
            p.add_(scale * torch.randn(p.shape, generator=gen, dtype=p.dtype))
    stream = torch.randint(0, V, (37,), generator=torch.Generator().manual_seed(1))
    dataset = TokenWindows(stream, N, stride=4, pad_final=True)
    indices = np.arange(len(dataset), dtype=np.int64)
    # The baseline with the report's observers is bitwise the plain pass the arms use.
    observers = [AttentionStatsObserver(None), KLCapObserver(float(cfg.kl_max), cfg.routing_kl_max_reference)]
    base = ev.nll_pass(model, dataset, indices, batch_size=2, device=CPU, observers=observers)
    plain = ev.nll_pass(model, dataset, indices, batch_size=2, device=CPU)
    assert np.array_equal(base.nll, plain.nll) and base.n == plain.n > 0
    manuscript_rows = [i.key for i in iv.INTERVENTIONS if i.key != "baseline"]        # Table M5, rows 2 to 12
    rows = rp._run_interventions(iv, model, dataset, indices, base, base.window.copy(), batch_size=2, device=CPU,
                                 keys=manuscript_rows, n_resamples=50, log=lambda message: None, save_to=None)
    by_key = {row["key"]: row for row in rows}
    not_applicable = {"positional_frames_identity", "frames_removed_decoder"}   # no positional frame; KL decoder
    for key in manuscript_rows:
        expected = "not_applicable" if key in not_applicable else "ok"
        assert by_key[key]["status"] == expected, (key, by_key[key])
        if expected == "ok":
            assert by_key[key]["delta_ce"] != 0.0, key                # the arm intercepts and moves the output
    assert by_key["baseline"]["status"] == "ok" and by_key["baseline"]["delta_ce"] == 0.0
    restoration = by_key["restoration_check"]
    assert restoration["status"] == "ok" and restoration["max_abs_nll_change"] == 0.0
    # With every realized frame the identity, removing the frames from one role is the identity map.
    pb = model.prior_bank
    with iv.swap_parameter_data(pb.phi_embed, iv.zeros_view(pb.phi_embed)):
        identity = ev.nll_pass(model, dataset, indices, batch_size=2, device=CPU)
        for key in ("frames_removed_input_moments", "frames_removed_routing_covariance", "frames_removed_norm_mlp"):
            with iv.by_key([key])[-1].apply(model):
                out = ev.nll_pass(model, dataset, indices, batch_size=2, device=CPU)
            assert np.array_equal(out.nll, identity.nll), key


# --------------------------------------------------------------------------------------------- A238


_RESUME_LEGS = ("step", "model_state", "optimizer_state", "rng_state", "data_state", "scaler_state",
                "best_val_ppl", "best_step", "numerics_counters", "divergence_monitor", "weight_ema")


def test_live_resume_with_the_log_eval_and_checkpoint_cadence_on_is_bitwise(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""The live cadence scaled down (log every 2, evaluate every 3, checkpoint at 6 of 12 keeping the last one,
    checkpoint at the end, sliding evaluation stride 2 of N = 4, block variance at log), resumed from step 6, which is
    a log, evaluation and checkpoint step at once; the shuffled loader crosses an epoch boundary at step 7."""
    import vfe3.run_artifacts as ra
    import vfe3.train as T
    from tests.test_audit_20261016_c1_resume_artifacts import _assert_same

    monkeypatch.setattr(ra.RunArtifacts, "save_attention_maps", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    saved: List[Path] = []
    real_save = ra.RunArtifacts.save_checkpoint

    def keep_a_copy(self: Any, step: int, *args: Any, **kwargs: Any) -> Path:
        path = real_save(self, step, *args, **kwargs)
        copies = Path(self.run_dir).parent / f"copies_{Path(self.run_dir).name}"
        copies.mkdir(exist_ok=True)
        shutil.copy2(path, copies / Path(path).name)
        saved.append(Path(path))
        return path

    monkeypatch.setattr(ra.RunArtifacts, "save_checkpoint", keep_a_copy)
    cfg = live_config(max_steps=12, log_interval=2, eval_interval=3, checkpoint_interval=6, checkpoint_keep_last=1,
                      checkpoint_at_end=True, eval_stride=2, block_variance_at_log=True)
    V, N, B = cfg.vocab_size, cfg.max_seq_len, cfg.batch_size
    train_tokens = torch.randint(0, V, (61,), generator=torch.Generator().manual_seed(123))
    val_tokens = (torch.arange(29) * 3 + 1) % V

    def run(run_dir: Path, resume_from: Any = None) -> List[float]:
        torch.manual_seed(5)
        model = VFEModel(cfg)
        artifacts = ra.RunArtifacts(run_dir, cfg, model)
        torch.manual_seed(11)
        loader = DataLoader(TokenWindows(train_tokens, N), batch_size=B, shuffle=True, drop_last=True,
                            generator=torch.Generator().manual_seed(3))
        val = DataLoader(TokenWindows(val_tokens, N, stride=cfg.eval_stride, pad_final=True), batch_size=2,
                         shuffle=False)
        losses = T.train(model, loader, cfg, n_steps=12, grad_clip=cfg.grad_clip, log_interval=2, eval_interval=3,
                         val_loader=val, device=CPU, logger=logging.getLogger("tests-infra-cadence-resume"),
                         artifacts=artifacts, generate_samples=False, resume_from=resume_from)
        artifacts.wait_background()
        return losses

    straight = run(tmp_path / "straight")
    assert sorted(p.name for p in (tmp_path / "straight" / "checkpoints").glob("step_*.pt")) == ["step_12.pt"]
    resumed = run(tmp_path / "resumed", resume_from=tmp_path / "copies_straight" / "step_6.pt")
    assert resumed == straight[6:]
    reference = torch.load(tmp_path / "copies_straight" / "step_12.pt", weights_only=True)
    final = torch.load(tmp_path / "copies_resumed" / "step_12.pt", weights_only=True)
    for leg in _RESUME_LEGS:
        _assert_same(final.get(leg), reference.get(leg), leg)
    with (tmp_path / "straight" / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        import csv
        straight_rows = [row for row in csv.DictReader(handle) if int(row["step"]) > 6]
    with (tmp_path / "resumed" / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        resumed_rows = list(csv.DictReader(handle))
    assert [row["step"] for row in resumed_rows] == [row["step"] for row in straight_rows] == ["8", "9", "10", "12"]
    timing = {"wall_clock_s", "train_step_ms_mean", "train_step_tokens_per_s", "pipeline_tokens_per_s",
              "tokens_per_s", "checkpoint_ms", *T.LOG_TIMING_COLUMNS, *T.TRAIN_STEP_PHASE_COLUMNS.values()}
    for a, b in zip(straight_rows, resumed_rows):
        assert {key for key in a if key not in timing and a.get(key) != b.get(key)} == set(), a["step"]
