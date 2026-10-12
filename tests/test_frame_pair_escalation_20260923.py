r"""Audit 2026-09-23 #12: measured-condition float64 escalation of the stored frame pair.

The group-frame pair :math:`(U, U^{-1})` stays float32 by default. A producer call escalates the
WHOLE call's pair to float64 iff :math:`\max_b \lVert U_b\rVert_F \lVert U_b^{-1}\rVert_F
\varepsilon_{32} > 10^{-4}`, decided inside the inverse's existing single host read. The escalated
pair is ``exp(M)`` recomputed in float64 and inverted from that float64 element. Every consumer then
computes at float64 and returns the belief's working dtype.

Escalation is FORCED here by monkeypatching ``transport._FRAME_PAIR_CONDITION_TOL`` to zero, and
reached from data by planting a worst-direction generator
:math:`M = Q\,\mathrm{diag}(r/\sqrt{2}, -r/\sqrt{2}, 0, \ldots)\,Q^{T}`, for which
:math:`\lVert M\rVert_F = r` and :math:`\kappa_2(e^{M}) = e^{\sqrt{2}\,r}`. Every model is tiny and
CPU-bound.
"""

import json
import math
import warnings
from typing import Callable, Dict, Optional, Tuple

import pytest
import torch
from torch.utils.data import DataLoader

import train_vfe3
import vfe3.geometry.transport as transport
import vfe3.model.model as model_module
from tests.profiles import config_values, test_config
from tests.profiles import tiny_config as _profile_tiny_config
from tests.test_remediation_20260923_C import _HostReads
from vfe3.config import VFE3Config
from vfe3.contracts import CanonicalFrameContext
from vfe3.data.datasets import TokenWindows
from vfe3.geometry.rope import build_rope_rotation
from vfe3.geometry.transport import (
    RopeTransport,
    _checked_group_inverse_certified,
    _frame_pair_kappa,
    _stable_compact_glk_exp_pair,
    build_factored_transport,
    frame_pair_fp64_escalations,
    reset_frame_pair_fp64_escalations,
    stable_matrix_exp_pair,
)
from vfe3 import metrics
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts
from vfe3.train import (
    _terminal_validation_signature,
    build_optimizer,
    lr_lambda,
    train,
    train_step,
)


# Every test here exercises the opt-in escalation (unconditional from 2026-10-02 to 2026-10-04,
# opt-in again since), so configs enable it and direct producer calls run with the process switch
# on; the switch is restored after each test.
def tiny_config(**overrides: object) -> VFE3Config:
    return _profile_tiny_config(**{
        "frame_pair_escalation_tol": 1e-4, "attention_mode": "projected_predictive",
        "attention_residual": False,
        "predictive_covariance": "passthrough", "lambda_alpha_mode": "constant",
        "mu_init_std": 0.02, "sigma_init": 1.0, "phi_scale": 0.01, **overrides,
    })


@pytest.fixture(autouse=True)
def _fresh_counter():
    previous = transport.set_frame_pair_fp64_escalation(True)
    reset_frame_pair_fp64_escalations()
    yield
    reset_frame_pair_fp64_escalations()
    transport.set_frame_pair_fp64_escalation(previous)


def _force(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Escalate every producer call. A model built afterwards would install its config tolerance
    (frame_pair_escalation_tol), so its installer keeps the forced zero."""
    monkeypatch.setattr(transport, "_FRAME_PAIR_CONDITION_TOL", 0.0)
    monkeypatch.setattr(model_module, "set_frame_pair_condition_tol", lambda tol: 0.0)


def _worst_direction(r: float, d: int, *, seed: int = 0) -> torch.Tensor:
    r"""(d, d) float32 generator with ||M||_F = r and kappa_2(exp M) = exp(sqrt(2) r)."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    lam = torch.zeros(d, dtype=torch.float64)
    lam[0], lam[1] = r / math.sqrt(2.0), -r / math.sqrt(2.0)
    return (Q @ torch.diag(lam) @ Q.T).float()


def _pair_residual(U: torch.Tensor, V: torch.Tensor) -> float:
    r""":math:`\lVert U V - I\rVert_{\max}` evaluated in float64."""
    eye = torch.eye(U.shape[-1], dtype=torch.float64)
    return float((U.double() @ V.double() - eye).abs().amax())


# -- producers: certificate boundary, exact float64 recompute, fail-closed order -----------------

def _compact_input(r: float) -> torch.Tensor:           # (2, H=2, 2, 2)
    small = 0.2 * torch.randn(2, 2, 2, 2, generator=torch.Generator().manual_seed(1))
    small[0, 0] = _worst_direction(r, 2)
    return small


def _dense_input(r: float, block_dims: Optional[list]) -> torch.Tensor:
    K = 3 if block_dims is None else sum(block_dims)
    M = torch.zeros(2, K, K)
    last = K - (3 if block_dims is None else block_dims[-1])
    M[0, last:last + 2, last:last + 2] = _worst_direction(r, 2)
    if block_dims is not None:
        start = 0
        for b in block_dims[:-1]:
            M[1, start:start + b, start:start + b] = 0.2 * torch.eye(b)
            start += b
    return M


def _produce(kind: str, r: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if kind == "compact":
        M = _compact_input(r)
        return (*_stable_compact_glk_exp_pair(M),
                torch.linalg.matrix_exp(M.double()))
    dims = {"dense_full": None, "dense_equal": [2, 2], "dense_unequal": [2, 3]}[kind]
    M = _dense_input(r, dims)
    reference = (torch.linalg.matrix_exp(M.double()) if dims is None
                 else transport._blockwise_matrix_exp(M.double(), dims))
    return (*stable_matrix_exp_pair(M, block_dims=dims),
            reference)


@pytest.mark.parametrize("kind", ["compact", "dense_full", "dense_equal", "dense_unequal"])
def test_certificate_boundary_and_exact_float64_recompute(kind: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")                     # escalation is silent
        U, V, _ = _produce(kind, 4.5)                      # kappa_F eps_32 ~ 7e-5: stays float32
        assert U.dtype == V.dtype == torch.float32
        assert frame_pair_fp64_escalations() == 0
        for count, r in enumerate((5.0, 12.0), start=1):    # kappa_F eps_32 > 1e-4: float64
            U, V, reference = _produce(kind, r)
            assert U.dtype == V.dtype == torch.float64
            assert torch.equal(U, reference)                # exp of the SAME generators, in float64
            assert _pair_residual(U, V) <= 1e-8
            assert frame_pair_fp64_escalations() == count


@pytest.mark.parametrize("kind", ["compact", "dense_full", "dense_equal"])
def test_escalated_pair_is_consistent_where_float32_is_not(kind: str, monkeypatch) -> None:
    monkeypatch.setattr(transport, "_FRAME_PAIR_CONDITION_TOL", math.inf)
    U32, V32, _ = _produce(kind, 10.0)                       # the float32 pair, never escalated
    monkeypatch.setattr(transport, "_FRAME_PAIR_CONDITION_TOL", 1e-4)
    U64, V64, _ = _produce(kind, 10.0)
    assert U32.dtype == torch.float32 and U64.dtype == torch.float64
    assert _pair_residual(U32, V32) > 5e-3                  # ~2e-2 measured
    assert _pair_residual(U64, V64) <= 1e-10


def test_failures_still_raise_before_any_escalation(monkeypatch) -> None:
    _force(monkeypatch)
    nan = _compact_input(12.0)
    nan[1, 1, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite matrix norm"):    # the chart check runs first
        _stable_compact_glk_exp_pair(nan)
    with pytest.raises(ValueError, match="nonfinite matrix norm"):
        _stable_compact_glk_exp_pair(nan, validity_max_norm=100.0)
    with pytest.raises(ValueError, match="validity bound exceeded"):
        stable_matrix_exp_pair(_dense_input(12.0, None), validity_max_norm=5.0)
    with pytest.raises(ValueError, match="singular"):
        _checked_group_inverse_certified(torch.zeros(2, 3, 3), kappa=_frame_pair_kappa())
    assert frame_pair_fp64_escalations() == 0


def test_certificate_covers_the_composed_pair_not_each_factor() -> None:
    r"""exp(A) exp(A) = exp(2A): each factor passes, their stored composition does not."""
    A = _worst_direction(4.0, 2)                             # each factor: kappa_F eps_32 ~ 3e-5
    phi = torch.zeros(3, 8)
    right = torch.zeros(3, 8)
    phi[1, :4] = A.reshape(-1)
    right[1, :4] = A.reshape(-1)
    group = VFEModel(tiny_config()).group
    for compact in (True, False):
        reset_frame_pair_fp64_escalations()
        alone = build_factored_transport(phi, group, compact_blocks=compact)
        assert (alone.exp_blocks if compact else alone.exp_phi).dtype == torch.float32
        built = build_factored_transport(phi, group, right_phi=right, compact_blocks=compact)
        forward = built.exp_blocks if compact else built.exp_phi
        inverse = built.inv_blocks if compact else built.exp_neg_phi
        assert forward.dtype == inverse.dtype == torch.float64
        assert _pair_residual(forward, inverse) <= 1e-10
        assert frame_pair_fp64_escalations() == 2          # left escalated, right rebuilt


@pytest.mark.parametrize("compact", [True, False])
@pytest.mark.parametrize("side", ["left", "right"])
def test_one_escalated_factor_promotes_the_composed_pair_jointly(side: str, compact: bool) -> None:
    group = VFEModel(tiny_config()).group
    g = torch.Generator().manual_seed(3)
    phi = 0.1 * torch.randn(2, 3, 8, generator=g)
    right = 0.1 * torch.randn(3, 8, generator=g)
    target = phi[1, 2] if side == "left" else right[2]
    target[4:] = _worst_direction(10.0, 2).reshape(-1)      # head 1 of one vertex
    built = build_factored_transport(phi, group, right_phi=right, compact_blocks=compact)
    forward = built.exp_blocks if compact else built.exp_phi
    inverse = built.inv_blocks if compact else built.exp_neg_phi
    assert forward.dtype == inverse.dtype == torch.float64
    assert _pair_residual(forward, inverse) <= 1e-9
    CanonicalFrameContext(forward=forward, inverse=inverse, block_diagonal=compact)
    assert frame_pair_fp64_escalations() == 2


@pytest.mark.parametrize("kind", ["compact", "dense_equal", "dense_full"])
def test_escalated_gradient_matches_the_float64_reference(kind: str) -> None:
    r"""d/dM of <W, U> + <W, U^-1> through the escalated producer equals the float64 chain."""
    if kind == "compact":
        M = _compact_input(8.0)
    else:
        M = _dense_input(8.0, [2, 2] if kind == "dense_equal" else None)
    W = torch.randn(M.shape, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    leaf = M.clone().requires_grad_(True)
    if kind == "compact":
        U, V = _stable_compact_glk_exp_pair(leaf)
    else:
        U, V = stable_matrix_exp_pair(
            leaf, block_dims=[2, 2] if kind == "dense_equal" else None)
    assert U.dtype == torch.float64
    ((U * W).sum() + (V * W).sum()).backward()
    assert leaf.grad.dtype == torch.float32

    leaf64 = M.double().requires_grad_(True)
    if kind == "dense_equal":
        U64 = transport._blockwise_matrix_exp(leaf64, [2, 2])
        V64 = transport._blockwise_group_inverse(U64, [2, 2])
    else:
        U64 = torch.linalg.matrix_exp(leaf64)
        V64 = torch.linalg.inv(U64)
    ((U64 * W).sum() + (V64 * W).sum()).backward()
    rel = float((leaf.grad.double() - leaf64.grad).norm() / leaf64.grad.norm())
    assert rel <= 1e-6


# -- no extra host synchronization on the healthy path ------------------------------------------

def _producer_reads(monkeypatch: pytest.MonkeyPatch) -> Tuple[int, ...]:
    g = torch.Generator().manual_seed(5)
    blocks = 0.4 * torch.randn(2, 3, 2, 2, 2, generator=g)
    dense = torch.block_diag(*(0.4 * torch.randn(2, 2, generator=g) for _ in range(2)))
    dense = dense.expand(3, 4, 4).contiguous()
    full = 0.4 * torch.randn(3, 4, 4, generator=g)
    group = VFEModel(tiny_config()).group
    phi = 0.3 * torch.randn(2, 3, 8, generator=g)
    right = 0.2 * torch.randn(3, 8, generator=g)
    calls: Tuple[Callable[[], object], ...] = (
        lambda: _stable_compact_glk_exp_pair(blocks),
        lambda: stable_matrix_exp_pair(dense, block_dims=[2, 2]),
        lambda: stable_matrix_exp_pair(full),
        lambda: build_factored_transport(phi, group, right_phi=right, compact_blocks=True),
        lambda: build_factored_transport(phi, group, right_phi=right),
    )
    reads = _HostReads(monkeypatch)
    counts = []
    for call in calls:
        reads.count = 0
        call()
        counts.append(reads.count)
    return tuple(counts)


def test_healthy_producers_pay_no_extra_host_read(monkeypatch) -> None:
    # Pre-edit counts: one pre-exp read plus one strict-inverse read per producer call (pinned by
    # test_remediation_20260923_C), two producer calls per composed build.
    assert _producer_reads(monkeypatch) == (2, 2, 2, 4, 6)
    assert frame_pair_fp64_escalations() == 0


def test_escalated_producers_pay_one_read_for_the_float64_inverse(monkeypatch) -> None:
    _force(monkeypatch)
    # +1 for the float64 inverse of the float64 U; the composed builds skip the second fp32 attempt.
    assert _producer_reads(monkeypatch) == (3, 3, 3, 5, 7)


@pytest.mark.parametrize("route", ['launcher'])
def test_healthy_model_step_reads_do_not_depend_on_the_certificate(route: str, monkeypatch) -> None:
    # Measured on HEAD ba066d5 before this change: launcher 16, transported kernel 9, glk 8.
    cfg = _ROUTES[route]()
    tokens = _tokens(cfg)

    def reads_at(tol: float) -> int:
        monkeypatch.setattr(transport, "_FRAME_PAIR_CONDITION_TOL", tol)
        torch.manual_seed(0)
        model = VFEModel(cfg)
        model(tokens, tokens)
        counter = _HostReads(monkeypatch)
        _, loss, _ = model(tokens, tokens)
        loss.backward()
        return counter.count

    healthy = reads_at(1e-4)
    monkeypatch.undo()
    assert healthy == reads_at(math.inf)
    assert frame_pair_fp64_escalations() == 0


# -- forced escalation end to end ----------------------------------------------------------------

_LAUNCHER_TINY = dict(
    vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=8, eval_stride=8, attention_window=8,
    n_layers=2, batch_size=2, amp_dtype=None, kl_max=32.0, generate_embedding_figures=False,
    block_mlp_expansion=2, warmup_steps=1, max_steps=4, deterministic=False)


def _launcher(**overrides: object) -> VFE3Config:
    values = config_values(
        attention_mode="projected_predictive", pos_phi="learned",
        skip_belief_sigma_update=True,

        use_block_mlp=True, norm_type_block="layernorm", norm_type_final="layernorm",
        layernorm_affine=True, decode_mode="diagonal_chunked",
        decode_av_precision="fp32", decode_ranking_fp64_escalation=False,
        lambda_alpha_mode="constant", mu_init_std=0.02, sigma_init=1.0, phi_scale=0.01,
    )
    values.update(_LAUNCHER_TINY)
    values["frame_pair_escalation_tol"] = 1e-4
    values.update(overrides)
    return test_config(**values)


def _tiny(**overrides: object) -> Callable[[], VFE3Config]:
    return lambda: tiny_config(warmup_steps=1, max_steps=4, **overrides)


_ROUTES: Dict[str, Callable[[], VFE3Config]] = {
    "launcher": _launcher,                                  # projected_predictive, pos_phi learned
    "launcher_bf16": lambda: _launcher(amp_dtype="bf16"),
    "launcher_noisy_query": lambda: _launcher(relational_routing="noisy_query"),
    "projected_predictive_diagonal": _tiny(attention_mode="projected_predictive",


                                           attention_residual=True),
    # RoPE folded into a CanonicalFrameContext: the float32 rotation must meet the float64 pair
    # re-orthogonalized, or the float64 contract rejects the folded frame (review of #12).
    "launcher_rope": lambda: _launcher(rope_mode="frame"),
    "launcher_rope_on_value": lambda: _launcher(rope_mode="frame_value"),
    "projected_predictive_rope": _tiny(attention_mode="projected_predictive",

                                       rope_mode="frame_value"),
    # Intrinsic frame of a RoPE'd frame-diagonal state: built from R, never through U (review r2).
}


def _tokens(cfg: VFE3Config) -> torch.Tensor:
    n = min(cfg.max_seq_len, 8)
    g = torch.Generator().manual_seed(3)
    return torch.randint(0, cfg.vocab_size, (2, n), generator=g)


def _run(cfg: VFE3Config, mutate: Optional[Callable[[VFEModel], None]] = None) -> Dict[str, object]:
    reset_frame_pair_fp64_escalations()
    torch.manual_seed(0)
    model = VFEModel(cfg)
    if mutate is not None:
        mutate(model)
    tokens = _tokens(cfg)
    with torch.no_grad():
        belief, logits = model.forward_beliefs(tokens, return_logits=True)
    _, loss, _ = model(tokens, tokens)
    loss.backward()
    grads = {name: p.grad.detach().clone()
             for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_lambda(s, cfg))
    train_step(model, optimizer, scheduler, tokens, tokens, grad_clip=1.0)
    return dict(model=model, loss=loss.detach(), logits=logits, mu=belief.mu, sigma=belief.sigma,
                grads=grads, escalations=frame_pair_fp64_escalations(),
                params_finite=all(bool(torch.isfinite(p).all()) for p in model.parameters()))


@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_forced_escalation_runs_end_to_end_at_unchanged_dtypes(route: str, monkeypatch) -> None:
    cfg = _ROUTES[route]()
    healthy = _run(cfg)
    assert healthy["escalations"] == 0
    _force(monkeypatch)
    forced = _run(cfg)
    assert forced["escalations"] > 0
    for key in ("logits", "mu", "sigma", "loss"):
        assert forced[key].dtype == healthy[key].dtype, key
    assert forced["loss"].dtype == torch.float32 and math.isfinite(float(forced["loss"]))
    assert abs(float(forced["loss"]) - float(healthy["loss"])) <= 1e-5 * abs(float(healthy["loss"]))
    # A frame-intrinsic route cancels the frame tables exactly (U^-1 (U R^T) = R^T): the escalated
    # build yields their exact zero gradient (no graph), float32 only a roundoff residue.
    missing = set(healthy["grads"]) - set(forced["grads"])
    assert set(forced["grads"]) <= set(healthy["grads"])
    assert missing <= {"pos_phi_free", "prior_bank.phi_embed"}, missing
    assert all(float(healthy["grads"][name].abs().max()) < 1e-8 for name in missing)
    parameters = dict(forced["model"].named_parameters())
    for name, grad in forced["grads"].items():
        assert grad.dtype == parameters[name].dtype, name
        assert torch.isfinite(grad).all(), name
        reference = healthy["grads"][name]
        assert float((grad - reference).norm()) <= 5e-3 * float(reference.norm()) + 1e-12, name
    assert forced["params_finite"]


@pytest.mark.parametrize("route", ['launcher'])
def test_forced_escalation_diagnostics(route: str, monkeypatch) -> None:
    _force(monkeypatch)
    cfg = _ROUTES[route]()
    torch.manual_seed(0)
    model = VFEModel(cfg).eval()
    model.diagnostics(_tokens(cfg))
    assert frame_pair_fp64_escalations() > 0


# -- escalation reached from data, no monkeypatching (the regression pin) -----------------------

def _plant(r: float, where: str) -> Callable[[VFEModel], None]:
    def mutate(model: VFEModel) -> None:
        with torch.no_grad():
            if where == "token":                        # left factor: phi_embed of a used token
                token = int(_tokens(model.cfg)[0, 0])
                model.prior_bank.phi_embed[token].view(2, 2, 2)[0] = _worst_direction(r, 2)
            else:                                       # right factor only: one position's pos_phi
                model.pos_phi_free[1].view(2, 2, 2)[1] = _worst_direction(r, 2)
    return mutate


@pytest.mark.parametrize("route", ['launcher', 'launcher_bf16', 'launcher_rope', 'launcher_rope_on_value', 'projected_predictive_rope'])
def test_ill_conditioned_frames_escalate_from_data(route: str) -> None:
    r"""At ||M||_F = 12, inside the M-step radius, HEAD ba066d5 raised ValueError on the launcher
    (CanonicalFrameContext residual 1.13 > 1e-2); the escalated pair now trains."""
    cfg = _ROUTES[route]()
    assert _run(cfg, _plant(4.0, "token"))["escalations"] == 0
    for r, where in ((6.0, "token"), (12.0, "token"), (6.0, "position")):
        result = _run(cfg, _plant(r, where))
        assert result["escalations"] > 0, (r, where)
        assert math.isfinite(float(result["loss"])) and result["params_finite"]
        assert all(bool(torch.isfinite(g).all()) for g in result["grads"].values())
        assert result["mu"].dtype == torch.float32


# -- consumers on synthetic float64 containers --------------------------------------------------


# -- RoPE meets escalated frames at float64 orthogonality ----------------------------------------

def test_rope_is_reorthogonalized_only_against_float64_frames() -> None:
    rope = build_rope_rotation(torch.arange(6), [2, 2])             # float32, as the model builds it
    rope64 = build_rope_rotation(torch.arange(6), [2, 2], dtype=torch.float64)
    assert transport._rope_at_frame_precision(rope, torch.float32) is rope        # healthy: untouched
    assert transport._rope_at_frame_precision(rope64, torch.float64) is rope64
    lifted = transport._rope_at_frame_precision(rope, torch.float64)
    eye = torch.eye(4, dtype=torch.float64)
    assert float((rope.double().mT @ rope.double() - eye).abs().amax()) > 1e-8
    assert float((lifted.mT @ lifted - eye).abs().amax()) <= 1e-15
    assert torch.equal(lifted == 0, rope == 0)                     # block-diagonal zeros stay exact
    assert float((lifted - rope64).abs().amax()) <= 1e-7            # the same rotation


def _escalated_base(compact: bool) -> Tuple[object, torch.Tensor]:
    group = VFEModel(tiny_config()).group
    phi = 0.1 * torch.randn(3, 8, generator=torch.Generator().manual_seed(6))
    phi[1, 4:] = _worst_direction(10.0, 2).reshape(-1)              # head 1 of token 1
    base = build_factored_transport(phi, group, compact_blocks=compact)
    assert transport._factor_dtype(base) == torch.float64
    return base, build_rope_rotation(torch.arange(3), group.irrep_dims)


@pytest.mark.parametrize("compact", [True, False])
def test_right_fold_of_escalated_frames_passes_the_float64_frame_contract(compact: bool) -> None:
    r"""Review of #12: casting the float32 rotation left the folded float64 pair with its 4e-8
    orthogonality defect, which ``CanonicalFrameContext`` rejects against ~1e-13 on every route
    that builds a canonical frame from a RoPE fold."""
    base, rope = _escalated_base(compact)

    def pair(folded: object) -> Tuple[torch.Tensor, torch.Tensor]:
        return ((folded.exp_blocks, folded.inv_blocks) if compact
                else (folded.exp_phi, folded.exp_neg_phi))

    with pytest.raises(ValueError, match="mutual inverses"):         # the cast rotation (pre-fix)
        CanonicalFrameContext(*pair(transport.fold_rope_into_frame(base, rope.double())),
                              block_diagonal=compact)
    forward, inverse = pair(RopeTransport(base, rope, insertion="right").score_operator())
    assert forward.dtype == inverse.dtype == torch.float64
    CanonicalFrameContext(forward=forward, inverse=inverse, block_diagonal=compact)
    assert _pair_residual(forward, inverse) <= 1e-9


# -- counter wiring ------------------------------------------------------------------------------

def test_counter_reads_and_resets() -> None:
    assert frame_pair_fp64_escalations() == 0
    _stable_compact_glk_exp_pair(_compact_input(12.0))
    assert frame_pair_fp64_escalations() == 1
    reset_frame_pair_fp64_escalations()
    assert frame_pair_fp64_escalations() == 0


def _loader(cfg: VFE3Config) -> DataLoader:
    tokens = torch.randint(0, cfg.vocab_size, (60,), generator=torch.Generator().manual_seed(9))
    windows = TokenWindows(tokens.to(torch.long), cfg.max_seq_len, stride=None, pad_final=False)
    return DataLoader(windows, batch_size=2, shuffle=False, drop_last=True)


@pytest.mark.parametrize("forced", [False, True])
def test_train_records_the_escalation_count(forced: bool, tmp_path, monkeypatch) -> None:
    cfg = tiny_config(max_seq_len=4, warmup_steps=1, max_steps=2)
    _stable_compact_glk_exp_pair(_compact_input(12.0))      # stale count from before the run
    assert frame_pair_fp64_escalations() == 1
    if forced:
        _force(monkeypatch)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    artifacts = RunArtifacts(tmp_path, cfg, model)
    train(model, _loader(cfg), cfg, n_steps=1, artifacts=artifacts, generate_samples=False)
    recorded = artifacts.realized_updates["frame_pair_fp64_escalations"]
    assert json.loads(json.dumps(artifacts.realized_updates))
    if forced:
        assert recorded == frame_pair_fp64_escalations() > 0
    else:
        assert recorded == 0                              # reset at run start


def test_terminal_validation_signature_tracks_the_escalation_count() -> None:
    cfg = tiny_config(max_seq_len=4)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    loader = _loader(cfg)
    before = _terminal_validation_signature(model, loader)
    assert before is not None and before == _terminal_validation_signature(model, loader)
    _stable_compact_glk_exp_pair(_compact_input(12.0))
    assert _terminal_validation_signature(model, loader) != before


@pytest.mark.parametrize("forced", [False, True])
def test_an_escalating_terminal_validation_is_not_handed_off(forced: bool, monkeypatch) -> None:
    r"""Like the other fallback counts, an escalation inside the final evaluation withdraws the
    private launcher handoff, so the launcher recomputes the terminal metrics itself."""
    cfg = tiny_config(max_seq_len=4, warmup_steps=1, max_steps=2)
    if forced:
        _force(monkeypatch)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    handoff: Dict[str, Optional[float]] = {}
    train(model, _loader(cfg), cfg, n_steps=1, eval_interval=1, val_loader=_loader(cfg),
          generate_samples=False, _final_validation_out=handoff)
    assert bool(handoff) is not forced


# -- recorded diagnostic bank --------------------------------------------------------------------


# -- the opt-in switch and its threshold ------------------------------------------------------------

def test_escalation_is_off_by_default_and_installed_from_config() -> None:
    assert VFE3Config.frame_pair_escalation_tol is None             # dataclass default: off
    assert test_config().frame_pair_escalation_tol is None
    VFEModel(_profile_tiny_config())
    assert transport.frame_pair_fp64_escalation() is False
    VFEModel(tiny_config(frame_pair_escalation_tol=3e-4))
    assert transport.frame_pair_fp64_escalation() is True
    assert transport._FRAME_PAIR_CONDITION_TOL == 3e-4
    VFEModel(tiny_config())
    assert transport._FRAME_PAIR_CONDITION_TOL == 1e-4
    for bad in (0.0, 1.0, -1e-4, math.inf, math.nan, True, "1e-4"):
        with pytest.raises(ValueError, match="frame_pair_escalation_tol"):
            test_config(frame_pair_escalation_tol=bad)


@pytest.mark.parametrize("kind", ["compact", "dense_full", "dense_equal", "dense_unequal"])
def test_disabled_escalation_forms_no_certificate_and_keeps_float32(kind: str, monkeypatch) -> None:
    transport.set_frame_pair_fp64_escalation(False)
    _force(monkeypatch)                                          # would escalate every call if enabled

    def no_certificate(*args: object, **kwargs: object) -> None:
        raise AssertionError("the condition certificate was formed with the escalation disabled")

    monkeypatch.setattr(transport, "_frame_pair_kappa", no_certificate)
    U, V, _ = _produce(kind, 12.0)
    assert U.dtype == V.dtype == torch.float32
    assert frame_pair_fp64_escalations() == 0


@pytest.mark.parametrize("kind", ["compact", "dense_full", "dense_equal", "dense_unequal"])
def test_condition_tolerance_moves_the_escalation_boundary(kind: str) -> None:
    U, V, _ = _produce(kind, 5.0)                                 # kappa_F eps_32 > 1e-4: escalates
    assert U.dtype == torch.float64 and frame_pair_fp64_escalations() == 1
    previous = transport.set_frame_pair_condition_tol(0.5)
    try:
        U, V, _ = _produce(kind, 5.0)                             # same pair, looser tolerance
        assert U.dtype == V.dtype == torch.float32
        assert frame_pair_fp64_escalations() == 1
    finally:
        transport.set_frame_pair_condition_tol(previous)
    with pytest.raises(ValueError, match="condition tolerance"):
        transport.set_frame_pair_condition_tol(0.0)


def test_disabled_escalation_trains_the_launcher_route_in_float32(monkeypatch) -> None:
    _force(monkeypatch)
    result = _run(_launcher(frame_pair_escalation_tol=None))
    assert result["escalations"] == 0
    assert math.isfinite(float(result["loss"])) and result["mu"].dtype == torch.float32
