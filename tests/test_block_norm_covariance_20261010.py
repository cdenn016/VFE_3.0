r"""Covariance through the block LayerNorm, audit 2026-10-10 (item 9).

In local coordinates the block LayerNorm rescales the mean. ``block_norm_covariance='pushforward'`` also carries the
covariance to post-norm units at the pre-norm mean :math:`u`,

.. math::
    \Sigma \mapsto J_{\mathrm{LN}}\Sigma J_{\mathrm{LN}}^\top + fD\Sigma D,\qquad
    (J_{\mathrm{LN}})_{kl} = \frac{\gamma_k}{s}\Bigl(\delta_{kl} - \frac{1 + z_kz_l}{K}\Bigr),\qquad
    D = \operatorname{diag}(\gamma)/s,

plus the radial second-order term :math:`\rho gg^\top` (audit 2026-10-10 round 2, item 11), keeping its diagonal on
``gaussian_diagonal`` (closed form, O(K) per token) and its head blocks on ``gaussian_block``.
The same M-projection admits ``block_mlp_covariance='delta_full'`` on the diagonal family,
:math:`(J\circ J)\sigma + fD^2\sigma`.

Checks, float64 and tiny K unless stated: the closed form against the dense diagonal and its gradcheck, the float64
rescue of a saturated float32 norm, the passthrough default and the inert toggle under a delta_full MLP, finite model
forwards and the executed pre- and post-MLP seams on the three Gaussian families, the diagonal delta_full against the
full family's on ``diag_embed(sigma)``, the reporting, and a whole-model finite-difference gradient.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from tests.profiles import tiny_config
from vfe3.contracts import CanonicalFrameContext
from vfe3.families.gaussian import head_block_projection
from vfe3.geometry.norms import AffineLayerNorm, LayerNorm
from vfe3.metrics import covariance_frozen_at_encode
from vfe3.model.block_mlp import (
    BlockMLP, CanonicalFrameBlockMLP, GaugeGateBlockMLP, _layernorm_diagonal_pushforward_at,
    block_norm_covariance_pushforward, block_norm_jacobian, block_norm_linearization, block_norm_radial_term,
    layernorm_diagonal_pushforward, layernorm_scale, layernorm_trace_cap,
)
from vfe3.model.model import VFEModel

FLOOR = 1e-3
H, D = 2, 2
K = H * D


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _affine_norm(k: int, eps: float, g: torch.Generator) -> AffineLayerNorm:
    norm = AffineLayerNorm(k, eps=eps).double()
    with torch.no_grad():
        norm.weight.copy_(1.0 + 0.5 * torch.randn(k, generator=g, dtype=torch.float64))
        norm.bias.copy_(0.3 * torch.randn(k, generator=g, dtype=torch.float64))
    return norm


def _dense_diagonal(norm, u: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    r""":math:`\operatorname{diag}(J\operatorname{diag}(\sigma)J^\top) + fD^2\sigma + \rho g^2` from the dense Jacobian and the
    radial second-order term on the dense layout (audit 2026-10-10 round 2, item 11)."""
    J = block_norm_jacobian(norm, u)                                                  # (..., K, K)
    d = layernorm_scale(u, eps=norm.eps, weight=getattr(norm, "weight", None))         # (..., K)
    dense = J @ torch.diag_embed(sigma) @ J.transpose(-1, -2)
    radial = block_norm_radial_term(norm, u, torch.diag_embed(sigma))
    return (torch.diagonal(dense, dim1=-2, dim2=-1) + FLOOR * d.square() * sigma
            + radial.rho * radial.direction.square())


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach(), b.detach()
    return float((a - b).abs().max() / b.abs().max())


# ----------------------------------------------------------------------------------------------
# The closed form
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("eps", [1e-6, 0.25])
@pytest.mark.parametrize("affine", [False, True])
def test_closed_form_equals_the_dense_diagonal(affine: bool, eps: float) -> None:
    g = torch.Generator().manual_seed(0)
    k = 5
    norm = _affine_norm(k, eps, g) if affine else LayerNorm(k, eps=eps)
    u = 1.7 * torch.randn(3, 4, k, generator=g, dtype=torch.float64)
    sigma = 0.1 + torch.rand(3, 4, k, generator=g, dtype=torch.float64)
    got = layernorm_diagonal_pushforward(u, sigma, eps=eps, covariance_floor=FLOOR,
                                         weight=norm.weight if affine else None)
    assert got.shape == sigma.shape and got.dtype == torch.float64
    assert _relative(got, _dense_diagonal(norm, u, sigma)) < 1e-13
    # The layout dispatcher: the diagonal layout is the closed form, the full layout on diag_embed(sigma) is
    # J Sigma J^T + f D Sigma D + rho g g^T, whose diagonal it is; both are capped at the sphere bound, which binds
    # on one token at eps = 1e-6 with gamma (audit 2026-10-12 round 2, M3).
    got = got * layernorm_trace_cap(got, norm.weight if affine else None, True)
    assert torch.equal(block_norm_covariance_pushforward(norm, u, sigma, FLOOR), got)
    full = block_norm_covariance_pushforward(norm, u, torch.diag_embed(sigma), FLOOR).detach()
    assert _relative(torch.diagonal(full, dim1=-2, dim2=-1), got) < 1e-12
    assert float((full - torch.diag_embed(torch.diagonal(full, dim1=-2, dim2=-1))).abs().max()) > 1e-3   # dense


def test_closed_form_gradcheck() -> None:
    g = torch.Generator().manual_seed(1)
    u = torch.randn(2, K, generator=g, dtype=torch.float64).requires_grad_()
    sigma = (0.2 + torch.rand(2, K, generator=g, dtype=torch.float64)).requires_grad_()
    gamma = (1.0 + 0.3 * torch.randn(K, generator=g, dtype=torch.float64)).requires_grad_()

    def fn(u_: torch.Tensor, s_: torch.Tensor, w_: torch.Tensor) -> torch.Tensor:
        return layernorm_diagonal_pushforward(u_, s_, eps=0.1, covariance_floor=FLOOR, weight=w_)

    assert torch.autograd.gradcheck(fn, (u, sigma, gamma), eps=1e-6, atol=1e-8, rtol=1e-6)


def test_saturated_float32_norm_is_recomputed_in_float64() -> None:
    r"""At :math:`K = 2` the LayerNorm saturates (:math:`J = O(\epsilon/v)`), so with a tiny floor the signed float32
    bracket rounds to non-positive values; those entries are the float64 result rounded back."""
    g = torch.Generator().manual_seed(2)
    u = torch.randn(4096, 2, generator=g)
    sigma = 0.1 + torch.rand(4096, 2, generator=g)
    floor = 1e-12
    working = _layernorm_diagonal_pushforward_at(u, sigma, 1e-6, floor, None)
    failed = working <= 0
    assert bool(failed.any())                                                       # the rescue is exercised
    got = layernorm_diagonal_pushforward(u, sigma, eps=1e-6, covariance_floor=floor)
    high = _layernorm_diagonal_pushforward_at(u.double(), sigma.double(), 1e-6, floor, None)
    assert got.dtype == torch.float32 and bool((got > 0).all()) and bool((high > 0).all())
    assert torch.equal(got[failed], high.float()[failed]) and torch.equal(got[~failed], working[~failed])


# ----------------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------------

_DECODE = {"gaussian_diagonal": "diagonal_chunked", "gaussian_full": "full_chunked", "gaussian_block": "full_chunked"}


def _cfg(family: str = "gaussian_diagonal", **over: object):
    values = dict(embed_dim=K, n_heads=H, vocab_size=9, max_seq_len=5, n_layers=1, family=family,
                  decode_mode=_DECODE[family], attention_mode="projected_predictive",
                  predictive_covariance="moment_matched", content_link_chart="exponential",
                  norm_type_block="layernorm", layernorm_affine=True, use_block_mlp=True,
                  block_mlp_mode="coordinate", block_mlp_expansion=2, block_mlp_covariance_floor=FLOOR,
                  block_norm_covariance="pushforward")
    values.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tiny_config(**values)


def _model(cfg, seed: int = 0) -> VFEModel:
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg).double().eval()
    g = torch.Generator().manual_seed(seed + 50)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad or name.endswith("phi_embed"):
                p.add_((0.3 if name.endswith("phi_embed") else 0.2) * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return model


def _tokens(seed: int = 3) -> 'tuple[torch.Tensor, torch.Tensor]':
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 9, (2, 5), generator=g), torch.randint(0, 9, (2, 5), generator=g)


def _expected(model: VFEModel, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    pushed = block_norm_covariance_pushforward(model.block_norm, mu, sigma, FLOOR)
    return head_block_projection(pushed, [D] * H) if model.cfg.family == "gaussian_block" else pushed


def _positive(sigma: torch.Tensor, diagonal: bool) -> bool:
    return bool((sigma > 0).all()) if diagonal else bool((torch.linalg.eigvalsh(sigma)[..., 0] > 0).all())


def test_passthrough_default_keeps_the_covariance_object_and_pushforward_is_inert_under_delta_full() -> None:
    cfg = _cfg(block_norm_covariance="passthrough")
    model = _model(cfg)
    tokens, _ = _tokens()
    capture: dict = {}
    with torch.no_grad():
        model.forward_beliefs(tokens, capture=capture)
    assert capture["block_mlp_input"].sigma is capture["converged"].sigma
    # Under a delta_full MLP the norm pushforward is inert: bitwise the passthrough model.
    for family in ("gaussian_diagonal", "gaussian_full"):
        base = _model(_cfg(family, block_mlp_covariance="delta_full", block_norm_covariance="passthrough"))
        pushed = _model(_cfg(family, block_mlp_covariance="delta_full"))
        assert "block_norm_covariance" in pushed.cfg.inactive_fields()
        with torch.no_grad():
            assert torch.equal(base(tokens), pushed(tokens))


@pytest.mark.parametrize("family", ["gaussian_diagonal", "gaussian_full", "gaussian_block"])
def test_model_forward_is_finite_and_executes_the_pre_mlp_pushforward(family: str) -> None:
    model = _model(_cfg(family))
    tokens, targets = _tokens()
    capture: dict = {}
    with torch.no_grad():
        model.forward_beliefs(tokens, capture=capture)
    converged, seam = capture["converged"], capture["block_mlp_input"]
    diagonal = family == "gaussian_diagonal"
    assert seam.sigma.shape == converged.sigma.shape
    assert torch.allclose(seam.sigma, _expected(model, converged.mu, converged.sigma), rtol=1e-14, atol=0)
    assert float((seam.sigma - converged.sigma).abs().max()) > 1e-3                 # the covariance moved
    assert _positive(seam.sigma, diagonal)
    if family == "gaussian_block":
        assert float((seam.sigma - head_block_projection(seam.sigma, [D] * H)).abs().max()) == 0.0
    _, loss, _ = model(tokens, targets)
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    assert model.block_norm.weight.grad is not None and float(model.block_norm.weight.grad.abs().max()) > 0.0


# ----------------------------------------------------------------------------------------------
# delta_full on the diagonal family
# ----------------------------------------------------------------------------------------------

def _delta_full_mlp(mode: str, g: torch.Generator):
    if mode == "coordinate":
        mlp, kwargs = BlockMLP(K, 3, "gelu", 0.0, covariance_contract="delta_full", covariance_floor=FLOOR), {}
    else:
        mlp = CanonicalFrameBlockMLP(K, 3, "silu", 0.0, covariance_contract="delta_full", covariance_floor=FLOOR)
        U = torch.eye(K, dtype=torch.float64) + 0.3 * torch.randn(2, 3, K, K, generator=g, dtype=torch.float64)
        kwargs = {"frame_context": CanonicalFrameContext(U, torch.linalg.inv(U))}
    mlp = mlp.double().eval()
    with torch.no_grad():
        for p in mlp.parameters():
            p.copy_(0.5 * torch.randn(p.shape, generator=g, dtype=torch.float64))
    return mlp, kwargs


@pytest.mark.parametrize("through_norm", [False, True])
@pytest.mark.parametrize("mode", ["coordinate", "canonical_frame"])
def test_diagonal_delta_full_is_the_diagonal_of_the_full_delta_full(mode: str, through_norm: bool) -> None:
    g = torch.Generator().manual_seed(4)
    mlp, kwargs = _delta_full_mlp(mode, g)
    u = 1.5 * torch.randn(2, 3, K, generator=g, dtype=torch.float64)
    sigma = 0.2 + torch.rand(2, 3, K, generator=g, dtype=torch.float64)
    if through_norm:
        norm = _affine_norm(K, 1e-6, g)
        lin = block_norm_linearization(norm, u)
        x, extra = norm(u, sigma), {"input_jacobian": lin.jacobian, "input_scale": lin.scale}
        radial = {"input_radial": block_norm_radial_term(norm, u, sigma)}
        radial_full = {"input_radial": block_norm_radial_term(norm, u, torch.diag_embed(sigma))}
    else:
        x, extra, radial, radial_full = u, {}, {}, {}
    diag = mlp.forward_moments(x, sigma, **kwargs, **extra, **radial)
    full = mlp.forward_moments(x, torch.diag_embed(sigma), **kwargs, **extra, **radial_full)
    assert diag.sigma.shape == sigma.shape and torch.equal(diag.mu, full.mu)
    assert _relative(diag.sigma, torch.diagonal(full.sigma, dim1=-2, dim2=-1)) < 1e-12
    assert bool((diag.sigma > 0).all())


def test_gauge_gate_delta_full_still_refuses_the_diagonal_layout() -> None:
    gate = GaugeGateBlockMLP([2, 2], 2, "gelu", 0.0, covariance_contract="delta_full").double()
    with pytest.raises(ValueError, match="delta_full"):
        gate.forward_moments(torch.ones(1, K, dtype=torch.float64), torch.ones(1, K, dtype=torch.float64))


def test_diagonal_delta_full_model_runs_and_is_booked_and_labeled() -> None:
    from vfe3.run_artifacts import _pure_path_report

    cfg = _cfg(block_mlp_covariance="delta_full", block_norm_covariance="passthrough")
    model = _model(cfg)
    tokens, targets = _tokens()
    capture: dict = {}
    with torch.no_grad():
        model.forward_beliefs(tokens, capture=capture)
        _, loss, _ = model(tokens, targets)
    converged, mlp_out = capture["converged"], capture["block_mlp_output"]
    lin = block_norm_linearization(model.block_norm, converged.mu)
    radial = block_norm_radial_term(model.block_norm, converged.mu, torch.diag_embed(converged.sigma))
    from vfe3.model.block_mlp import block_norm_trace_cap   # the sphere-bound cap (audit 2026-10-12, item 1)
    cap = block_norm_trace_cap(model.block_norm, converged.mu, converged.sigma, cfg.block_mlp_covariance_floor)
    full = model.block_mlps[0].forward_moments(capture["block_mlp_input"].mu, torch.diag_embed(converged.sigma),
                                               input_jacobian=lin.jacobian, input_scale=lin.scale,
                                               input_radial=radial, input_cap=cap).sigma
    assert torch.allclose(mlp_out.sigma, torch.diagonal(full, dim1=-2, dim2=-1), rtol=1e-12, atol=0)
    assert bool(torch.isfinite(loss)) and bool((mlp_out.sigma > 0).all())
    # The diagonal delta_full books (J o J) sigma and its floor, no Cholesky.
    assert model.executable_build.block_mlp.flops_per_token == pytest.approx(
        cfg.n_layers * (4.0 * cfg.block_mlp_expansion * K * K + 3.0 * K * K + 4.0 * K))
    toggles = _pure_path_report(cfg, [], executable_build=model.executable_build)["config_toggles"]
    assert toggles["block_mlp_covariance_contract"] == \
        "diagonal_projection_of_delta_full_through_layernorm_plus_post_norm_scaled_floor"


def test_reporting_names_the_toggle_and_the_unfrozen_covariance() -> None:
    from vfe3.run_artifacts import _pure_path_report

    over = dict(predictive_covariance="passthrough")
    pushed, passthrough = _cfg(**over), _cfg(**over, block_norm_covariance="passthrough")
    assert _pure_path_report(pushed, [])["config_toggles"]["block_norm_covariance"] == "pushforward"
    assert _pure_path_report(passthrough, [])["config_toggles"]["block_norm_covariance"] == "passthrough"
    # With the predictive covariance frozen, only the norm pushforward makes Sigma context dependent.
    assert covariance_frozen_at_encode(passthrough) and not covariance_frozen_at_encode(pushed)


# ----------------------------------------------------------------------------------------------
# Whole-model finite differences
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["gaussian_diagonal", "gaussian_full"])
def test_whole_model_gradient_matches_finite_differences_under_pushforward(family: str) -> None:
    r"""Central differences of the training loss, float64, in the block-norm gain and one fc1 weight (two layers,
    so the pushed covariance reaches the second router and the Gaussian decoder)."""
    model = _model(_cfg(family, n_layers=2))
    tokens, targets = _tokens(5)
    # Negative control: the same parameters under passthrough give a different loss.
    plain = _model(_cfg(family, n_layers=2, block_norm_covariance="passthrough"))
    plain.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        assert abs(float(plain(tokens, targets)[1]) - float(model(tokens, targets)[1])) > 1e-6
    _, loss, _ = model(tokens, targets)
    loss.backward()
    h = 1e-6
    for param, index in ((model.block_norm.weight, (1,)), (model.block_mlps[0].fc1.weight, (0, 1))):
        analytic = float(param.grad[index])
        with torch.no_grad():
            original = float(param[index])
            param[index] = original + h
            up = float(model(tokens, targets)[1])
            param[index] = original - h
            down = float(model(tokens, targets)[1])
            param[index] = original
        numeric = (up - down) / (2.0 * h)
        assert abs(analytic - numeric) <= 1e-6 * max(1.0, abs(numeric)), (analytic, numeric)
        assert abs(analytic) > 1e-6
