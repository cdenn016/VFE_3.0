r"""Audit 2026-10-12 item 3 / T4: the opt-in frame condition bound ``phi_frame_condition_max``.

The bound keeps every learned frame factor's skew part and trace and limits the spread of its traceless
symmetric part :math:`S_0` to :math:`2c`, :math:`c = \log\kappa_{\max} / (2 n_{\mathrm{factors}})`, so that the
composed frame :math:`U_i = e^{X_i} e^{Y_i}` has :math:`\kappa_2(U_i) \le \kappa_{\max}`.
"""
import math

import pytest
import torch

import vfe3.train as train_module
from tests.profiles import test_config
from vfe3.gauge_optim import bound_phi_frame_condition_
from vfe3.geometry.transport import build_factored_transport
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer


def _head_blocks(phi: torch.Tensor, model: VFEModel) -> torch.Tensor:   # (rows, H, d, d) float64
    group = model.group
    M = torch.einsum("ra,aij->rij", phi.detach().to(torch.float64), group.generators.to(torch.float64))
    d = group.irrep_dims[0]
    return torch.stack([M[:, h * d:(h + 1) * d, h * d:(h + 1) * d] for h in range(len(group.irrep_dims))], dim=1)


def _spread_skew_trace(blocks: torch.Tensor) -> tuple:
    S = 0.5 * (blocks + blocks.transpose(-1, -2))
    trace = S.diagonal(dim1=-2, dim2=-1).sum(-1)                                   # (rows, H)
    d = blocks.shape[-1]
    lam = torch.linalg.eigvalsh(S - (trace / d)[..., None, None] * torch.eye(d, dtype=S.dtype))
    return lam[..., -1] - lam[..., 0], 0.5 * (blocks - blocks.transpose(-1, -2)), trace


@pytest.mark.parametrize("overrides", [
    dict(gauge_group="block_glk", embed_dim=6, n_heads=2),
    dict(gauge_group="block_glk", embed_dim=6, n_heads=2, gauge_tying="shared"),
    dict(gauge_group="block_glmc", embed_dim=8, n_heads=2),
])
def test_projection_bounds_spread_and_keeps_skew_and_trace(overrides: dict) -> None:
    cfg = test_config(vocab_size=6, max_seq_len=3, pos_phi="learned", phi_frame_condition_max=20.0, **overrides)
    model = VFEModel(cfg)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        model.prior_bank.phi_embed.copy_(torch.randn(model.prior_bank.phi_embed.shape, generator=generator))
        model.prior_bank.phi_embed[0].mul_(1e-3)                  # a row already inside the bound
    tables = (model.prior_bank.phi_embed, model.pos_phi_free)
    before = [table.detach().clone() for table in tables]
    c = math.log(20.0) / (2 * len(tables))
    spread0, _, _ = _spread_skew_trace(_head_blocks(before[0], model))
    assert bool((spread0[1:] > 2 * c).any())                       # the stretch is actually active

    changed = bound_phi_frame_condition_(model, 20.0)

    assert int(changed) > 0
    assert torch.equal(model.prior_bank.phi_embed[0], before[0][0])   # feasible row bitwise unchanged
    for table, old in zip(tables, before):
        spread_new, skew_new, trace_new = _spread_skew_trace(_head_blocks(table, model))
        _, skew_old, trace_old = _spread_skew_trace(_head_blocks(old, model))
        assert float(spread_new.max()) <= 2 * c * (1 + 1e-5)
        torch.testing.assert_close(skew_new, skew_old, rtol=0.0, atol=1e-6)
        torch.testing.assert_close(trace_new, trace_old, rtol=0.0, atol=1e-5)
    # Idempotent up to fp32 roundoff of the boundary rows.
    snapshot = model.prior_bank.phi_embed.detach().clone()
    bound_phi_frame_condition_(model, 20.0)
    torch.testing.assert_close(model.prior_bank.phi_embed.detach(), snapshot, rtol=0.0, atol=1e-5)


def _stretched_model(overrides: dict) -> VFEModel:
    cfg = test_config(vocab_size=12, max_seq_len=3, pos_phi="none", phi_frame_condition_max=20.0, **overrides)
    model = VFEModel(cfg)                                                      # one frame factor: c = log(kappa)/2
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        phi = model.prior_bank.phi_embed
        phi.copy_(1e-3 * torch.randn(phi.shape, generator=generator))       # inside the bound
        phi[::3].copy_(2.0 * torch.randn(phi[::3].shape, generator=generator))   # every third row stretched
    return model


def _float64_reference(model: VFEModel, kappa_max: float) -> torch.Tensor:
    r"""The unscreened float64 projection of every row (the pre-screen algorithm)."""
    from vfe3.gauge_optim import _frame_bound_rows
    group = model.group
    H, d = len(group.irrep_dims), group.irrep_dims[0]
    basis = gram = None
    if group.elementary_block_shape is None:
        basis = group.head_block_generators().detach().to(torch.float64)
        gram = basis.square().sum(dim=(1, 2, 3))
    phi = model.prior_bank.phi_embed.detach()
    new, over, _ = _frame_bound_rows(phi, basis, gram, H, d, math.log(kappa_max) / 2, torch.float64)
    return torch.where(over[:, None], new.to(phi.dtype), phi)


@pytest.mark.parametrize("overrides", [
    dict(gauge_group="block_glk", embed_dim=6, n_heads=2),
    dict(gauge_group="block_glk", embed_dim=6, n_heads=2, gauge_tying="shared"),
    dict(gauge_group="block_glmc", embed_dim=8, n_heads=2),
])
def test_screened_float32_projection_matches_the_float64_projection(overrides: dict) -> None:
    r"""The screen skips only rows already within the bound (bitwise unchanged), and the certified float32
    projection of the flagged rows equals the float64 projection to float32 rounding."""
    model = _stretched_model(overrides)
    before = model.prior_bank.phi_embed.detach().clone()
    reference = _float64_reference(model, 20.0)
    bound_phi_frame_condition_(model, 20.0)
    after = model.prior_bank.phi_embed.detach()
    unchanged = torch.equal(reference, before)
    assert not unchanged                                                       # the projection acts
    torch.testing.assert_close(after, reference, rtol=0.0, atol=1e-5)
    untouched = (reference == before).all(-1)
    assert bool(untouched.any()) and torch.equal(after[untouched], before[untouched])


def test_uncertified_float32_blocks_are_recomputed_in_float64(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""A float32 eigendecomposition that fails the orthogonality certificate is redone in float64."""
    model = _stretched_model(dict(gauge_group="block_glk", embed_dim=6, n_heads=2))
    reference = _float64_reference(model, 20.0)
    eigh, calls = torch.linalg.eigh, []

    def corrupted(A: torch.Tensor) -> tuple:
        lam, Q = eigh(A)
        calls.append(A.dtype)
        return (lam, 1.01 * Q) if A.dtype == torch.float32 else (lam, Q)

    monkeypatch.setattr(torch.linalg, "eigh", corrupted)
    bound_phi_frame_condition_(model, 20.0)
    monkeypatch.setattr(torch.linalg, "eigh", eigh)
    assert torch.float32 in calls and torch.float64 in calls
    torch.testing.assert_close(model.prior_bank.phi_embed.detach(), reference, rtol=0.0, atol=1e-6)


def test_screen_is_a_sound_bound_on_the_spread() -> None:
    r""":math:`\lambda_{\max} - \lambda_{\min} \le \sqrt2\lVert S_0\rVert_F` for a traceless symmetric :math:`S_0`."""
    generator = torch.Generator().manual_seed(2)
    X = torch.randn(4000, 7, 7, generator=generator, dtype=torch.float64) * torch.logspace(-3, 2, 4000,
                                                                                          dtype=torch.float64)[:, None, None]
    S = 0.5 * (X + X.transpose(-1, -2))
    S0 = S - S.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * torch.eye(7, dtype=torch.float64)
    lam = torch.linalg.eigvalsh(S0)
    assert bool(((lam[:, -1] - lam[:, 0]) <= math.sqrt(2.0) * S0.norm(dim=(-2, -1)) * (1 + 1e-12)).all())


def _train_case(**overrides: object) -> tuple:
    cfg = test_config(vocab_size=8, embed_dim=6, n_heads=2, max_seq_len=4, n_layers=1, n_e_steps=1,
                      pos_phi="learned", **overrides)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    tokens = torch.tensor([[0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, 4]])
    return model, optimizer, scheduler, tokens, targets


def _realized_frame_condition(model: VFEModel, tokens: torch.Tensor) -> torch.Tensor:   # (N, H)
    with torch.no_grad():
        phi = model.prior_bank.phi_embed[tokens[0]]
        transport = build_factored_transport(phi, model.group, compact_blocks=True,
                                             right_phi=model._pos_phi_right(phi))
        return torch.linalg.cond(transport.exp_blocks.to(torch.float64))


def test_stretched_frame_is_bounded_after_one_training_step() -> None:
    kappa_max = 50.0
    model, optimizer, scheduler, tokens, targets = _train_case(phi_frame_condition_max=kappa_max)
    with torch.no_grad():
        stretch = torch.diag(torch.tensor([3.0, 0.0, -3.0]))     # traceless symmetric, kappa_2(e^X) = e^6
        model.prior_bank.phi_embed[0, :9] = stretch.reshape(-1)
    assert float(_realized_frame_condition(model, tokens)[0, 0]) > 5 * kappa_max

    train_module.train_step(model, optimizer, scheduler, tokens, targets)

    assert float(_realized_frame_condition(model, tokens).max()) <= kappa_max * (1 + 1e-4)


def test_option_off_never_calls_the_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(train_module, "bound_phi_frame_condition_", lambda *a, **k: calls.append(a))

    model, optimizer, scheduler, tokens, targets = _train_case()
    assert model.cfg.phi_frame_condition_max is None
    train_module.train_step(model, optimizer, scheduler, tokens, targets)
    assert calls == []

    model, optimizer, scheduler, tokens, targets = _train_case(phi_frame_condition_max=50.0)
    train_module.train_step(model, optimizer, scheduler, tokens, targets)
    assert len(calls) == 1


@pytest.mark.parametrize("overrides, message", [
    (dict(phi_frame_condition_max=1.0), "finite real > 1"),
    (dict(phi_frame_condition_max=True), "finite real > 1"),
    (dict(phi_frame_condition_max=10.0, m_phi_update_mode="pullback_group"), "m_phi_update_mode='adamw'"),
    (dict(phi_frame_condition_max=10.0, encode_mode="per_token_additive"), "encode_mode='per_token'"),
])
def test_config_refuses_unsupported_settings(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        test_config(**overrides)


def test_inert_groups_are_reported_inactive() -> None:
    assert "phi_frame_condition_max" in test_config(gauge_group="block_so",
                                                    phi_frame_condition_max=10.0).inactive_fields()
    assert "phi_frame_condition_max" not in test_config(phi_frame_condition_max=10.0).inactive_fields()
