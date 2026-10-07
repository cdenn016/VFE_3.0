r"""Phase 2 of the SO(N) mixed-tower plan: the ``gauge_tying`` axis.

Tied ``block_glk`` (``gauge_tying='shared'``): one GL(d_head) element per token acting identically on
every head, U = I_H (x) g, n_gen = d_head^2. Untied ``so_n`` (``gauge_tying='per_head'``): every irrep
block carries its own SO(N) element, n_gen = dim so(N) per nontrivial block. Pins, on CPU at tiny
size: the generator bases and their Grams; frames identical across heads (tied) or independent per
block (untied); the dense frame against the explicit exponential; the refusals (per-block metrics,
pullback_group, fisher_group and group_log for tied block_glk; CG coupling for untied so_n); the
per-block Killing preconditioner accepted for untied so_n; the native defaults and a saved config
without the key; and train steps, gauge equivariance and the run record on both new groups.
"""
import math
import warnings

import pytest
import torch

from tests.profiles import test_config, tiny_config
from vfe3.config import config_from_serialized, serialize_config
from vfe3.geometry.groups import get_group
from vfe3.geometry.irreps import direct_sum_generators
from vfe3.geometry.generators import generate_son
from vfe3.model.model import VFEModel, build_group

TIED_GLK = dict(embed_dim=6, n_heads=2, gauge_tying="shared")
UNTIED_SON = dict(gauge_group="so_n", group_n=3, irrep_spec=[("l1", 1), ("l2", 1)], embed_dim=8, n_heads=2,
                  gauge_tying="per_head")
GRADIENT = dict(attention_mode="projected_gradient", predictive_covariance="passthrough", attention_residual=False)


def _model(dtype: torch.dtype = torch.float32, **overrides: object) -> VFEModel:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(tiny_config(**overrides)).to(dtype)
    with torch.no_grad():
        model.prior_bank.phi_embed.normal_(mean=0.0, std=0.3)
    return model


def _blocks(matrix: torch.Tensor, dims: list) -> list:
    out, start = [], 0
    for d in dims:
        out.append(matrix[..., start:start + d, start:start + d])
        start += d
    return out


# ---------------------------------------------------------------- defaults and config

def test_native_tying_defaults_and_saved_configs_without_the_key() -> None:
    assert test_config().gauge_tying == "per_head"
    son = dict(gauge_group="so_n", group_n=3, irrep_spec=[("l1", 2)], embed_dim=6, n_heads=2)
    assert test_config(**son).gauge_tying == "shared"
    for cfg in (test_config(), test_config(**son)):
        payload = serialize_config(cfg)
        native = payload.pop("gauge_tying")
        assert config_from_serialized(payload, source="test").gauge_tying == native
    assert build_group(test_config()).tying == "per_head"
    assert build_group(test_config(**son)).tying == "shared"
    with pytest.raises(ValueError, match="gauge_tying"):
        test_config(gauge_tying="tied")
    tied = test_config(**TIED_GLK)
    assert config_from_serialized(serialize_config(tied), source="test").gauge_tying == "shared"


# ---------------------------------------------------------------- tied block_glk

def test_tied_block_glk_basis_and_gram() -> None:
    group = get_group("block_glk")(6, 2, tying="shared", dtype=torch.float64)
    G = group.generators
    assert G.shape == (9, 6, 6) and group.tying == "shared"
    E = torch.eye(9, dtype=torch.float64).reshape(9, 3, 3)
    torch.testing.assert_close(G, torch.stack([torch.kron(torch.eye(2, dtype=torch.float64), e) for e in E]))
    torch.testing.assert_close(torch.einsum("aij,bij->ab", G, G), 2.0 * torch.eye(9, dtype=torch.float64))
    assert group.frobenius_gram_uniform == 2.0 and group.phi_coordinate_layout is None
    assert group.elementary_block_shape is None and group.pullback_capability is None
    assert get_group("block_glk")(6, 2).pullback_capability == "pullback_per_block"


def test_tied_block_glk_frames_are_identical_across_heads() -> None:
    from vfe3.geometry.transport import build_factored_transport
    group = get_group("block_glk")(6, 2, tying="shared", dtype=torch.float64)
    phi = 0.4 * torch.randn(1, 4, 9, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    frame = build_factored_transport(phi, group, compact_blocks=True)
    U = frame.exp_phi                                                    # (1, N, 6, 6) dense route
    first, second = _blocks(U, [3, 3])
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    torch.testing.assert_close(first, torch.linalg.matrix_exp(phi.reshape(1, 4, 3, 3)), atol=1e-12, rtol=1e-12)
    assert float(U[..., :3, 3:].abs().max()) == 0.0 and float(U[..., 3:, :3].abs().max()) == 0.0
    torch.testing.assert_close(frame.exp_neg_phi @ U, torch.eye(6, dtype=torch.float64).expand_as(U),
                               atol=1e-12, rtol=0)


def test_tied_block_glk_refuses_pullback_fisher_and_group_log() -> None:
    from tests.test_phi_fisher import _ACTIVE_ROUTE
    pullback = dict(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1, pos_phi="none",
                    m_phi_update_mode="pullback_group",
                    transport_chart_max_norm=6.0, m_phi_lr=0.1)
    test_config(**pullback)
    test_config(**pullback, m_phi_group_retraction="group_log")
    test_config(**_ACTIVE_ROUTE)
    with pytest.raises(ValueError, match="accepts only untied block_glk"):
        test_config(**pullback, gauge_tying="shared")
    with pytest.raises(ValueError, match="gauge_tying='per_head'"):
        test_config(**_ACTIVE_ROUTE, gauge_tying="shared")
    fisher_group_log = {**_ACTIVE_ROUTE, "m_phi_group_retraction": "group_log"}
    with pytest.raises(ValueError, match="gauge_tying='per_head'"):
        test_config(**fisher_group_log, gauge_tying="shared")


# ---------------------------------------------------------------- untied so_n

def test_untied_so_n_basis_supports_each_generator_on_one_block() -> None:
    spec = [("l0", 1), ("l1", 1), ("l2", 1)]
    group = get_group("so_n")(9, group_n=3, irrep_spec=spec, tying="per_head", dtype=torch.float64)
    tied = get_group("so_n")(9, group_n=3, irrep_spec=spec, dtype=torch.float64)
    assert group.tying == "per_head" and tied.tying == "shared"
    assert group.generator_shape == (6, 9, 9) and tied.generator_shape == (3, 9, 9)  # l0 carries none
    G = group.generators
    for a in range(6):
        mass = [float(b.abs().sum()) for b in _blocks(G[a], [1, 3, 5])]
        assert sum(m > 0 for m in mass) == 1
    # Block-major: the l1 copy, then the l2 copy, each the tied block's generators.
    G_def = generate_son(3, dtype=torch.float64)
    G_tied, _ = direct_sum_generators(G_def, algebra="so", irrep_spec=spec)
    torch.testing.assert_close(G[:3], G_tied * (torch.arange(9) < 4).double().outer((torch.arange(9) < 4).double()))
    gram = torch.einsum("aij,bij->ab", G, G)
    torch.testing.assert_close(gram, torch.diag(gram.diagonal()))
    torch.testing.assert_close(gram.diagonal(), torch.tensor([2.0] * 3 + [10.0] * 3, dtype=torch.float64))
    assert group.frobenius_gram_diagonal and group.frobenius_gram_uniform is None
    assert (G + G.transpose(-1, -2)).abs().max() < 1e-12


def test_untied_so_n_frames_are_independent_per_block() -> None:
    from vfe3.geometry.transport import build_factored_transport
    spec = [("l1", 1), ("l2", 1)]
    group = get_group("so_n")(8, group_n=3, irrep_spec=spec, tying="per_head", dtype=torch.float64)
    phi = 0.5 * torch.randn(2, 6, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    phi[1, :3] = phi[0, :3]                                              # same l1 element, different l2
    U = build_factored_transport(phi, group).exp_phi                     # (2, 8, 8)
    l1, l2 = _blocks(U, [3, 5])
    torch.testing.assert_close(l1[0], l1[1], atol=1e-14, rtol=0)
    assert float((l2[0] - l2[1]).abs().max()) > 1e-3
    G = group.generators
    for b, (start, d, coords) in enumerate([(0, 3, slice(0, 3)), (3, 5, slice(3, 6))]):
        block_gens = G[coords, start:start + d, start:start + d]
        expected = torch.linalg.matrix_exp(torch.einsum("na,aij->nij", phi[:, coords], block_gens))
        torch.testing.assert_close(_blocks(U, [3, 5])[b], expected, atol=1e-12, rtol=1e-12)


def test_untied_so_n_refuses_cg() -> None:
    with pytest.raises(ValueError, match="gauge_tying='shared'"):
        test_config(**UNTIED_SON, use_cg_coupling=True)


# ---------------------------------------------------------------- model routes

@pytest.mark.parametrize("group", ["tied_glk", "untied_son"])
@pytest.mark.parametrize("route", ["predictive", "gradient"])
def test_train_step_and_gauge_equivariance(group: str, route: str) -> None:
    from vfe3.train import build_optimizer, train_step
    from vfe3.viz import extract
    overrides = {**(TIED_GLK if group == "tied_glk" else UNTIED_SON), **(GRADIENT if route == "gradient" else {})}
    model = _model(**overrides)
    n_gen = 9 if group == "tied_glk" else 6
    assert model.prior_bank.phi_embed.shape[-1] == n_gen
    residuals = extract.projected_gauge_equivariance(
        model.eval(), torch.tensor([[1, 4, 2]]), n_draws=2)["residuals"]
    assert residuals and all(max(v) < 1e-10 for v in residuals.values()), residuals
    model.train()
    optimizer = build_optimizer(model, model.cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    phi0 = model.prior_bank.phi_embed.detach().clone()
    loss = train_step(model, optimizer, scheduler, torch.tensor([[1, 4, 2]]), torch.tensor([[4, 2, 3]]))
    assert math.isfinite(loss)
    assert float((model.prior_bank.phi_embed.detach() - phi0).abs().max()) > 0.0


def test_run_record_carries_the_tying() -> None:
    import vfe3.run_artifacts as ra
    model = _model(**TIED_GLK)
    assert ra._cost_model_fields(model, model.cfg, 1, 0)["gauge_tying"] == "shared"
    assert ra._cost_model_fields(_model(), _model().cfg, 1, 0)["gauge_tying"] == "per_head"
