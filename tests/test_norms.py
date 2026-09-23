import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from vfe3.geometry.norms import AffineLayerNorm, LayerNorm, MahalanobisNorm, _NORMS, get_norm


def test_mahalanobis_formula_diagonal():
    K = 4
    norm = MahalanobisNorm(K)
    mu = torch.randn(3, K); sigma = torch.rand(3, K) + 0.5
    out = norm(mu, sigma)
    s2 = (mu ** 2 / sigma).sum(-1, keepdim=True)
    assert torch.allclose(out, mu * torch.sqrt(K / s2), atol=1e-5)


def test_mahalanobis_is_gauge_invariant_scale():
    # The Mahalanobis scalar mu^T Sigma^-1 mu is invariant under mu->g mu, Sigma->g Sigma g^T,
    # so the norm SCALE sqrt(K/s2) is gauge-invariant; out transforms as a vector (out -> g out).
    # Seeded (suite convention) and g conditioned to kappa < 10: Sigma_g = g g^T has
    # kappa(Sigma_g) = kappa(g)^2, and the full-cov solve amplifies fp32 roundoff by that
    # factor, so an unconditioned g=randn+2I gives an O(1) residual ~1.5% of the time --
    # not a math failure but fp32 conditioning. Restricting to well-conditioned g keeps
    # the gauge-invariance claim a clean atol-1e-4 check rather than a flaky one.
    K = 3
    rng = torch.Generator().manual_seed(0)
    norm = MahalanobisNorm(K)
    while True:
        g = torch.randn(K, K, generator=rng) + 2 * torch.eye(K)  # invertible
        if torch.linalg.cond(g).item() < 10.0:                   # well-conditioned draw
            break
    mu = torch.randn(2, K, generator=rng)
    sigma_full = torch.eye(K).expand(2, K, K).contiguous()
    out = norm(mu, sigma_full)
    mu_g = mu @ g.T
    sig_g = g @ sigma_full @ g.T
    out_g = norm(mu_g, sig_g)
    assert torch.allclose(out_g, out @ g.T, atol=1e-4)


def test_layernorm_matches_torch_reference():
    # Parameter-free LayerNorm == torch.nn.functional.layer_norm over the last dim (no affine).
    K = 4
    norm = LayerNorm(K, eps=1e-5)
    mu = torch.randn(3, 5, K)
    sigma = torch.rand(3, 5, K) + 0.5                       # ignored by LN
    out = norm(mu, sigma)
    ref = F.layer_norm(mu, (K,), weight=None, bias=None, eps=1e-5)
    assert torch.allclose(out, ref, atol=1e-5)


def test_layernorm_standardizes_features():
    # Output has ~zero feature-mean and ~unit (biased) feature-variance over the belief dim.
    K = 5
    norm = LayerNorm(K)
    mu = torch.randn(7, K) * 3.0 + 2.0
    out = norm(mu, torch.ones(7, K))
    assert torch.allclose(out.mean(-1), torch.zeros(7), atol=1e-5)
    assert torch.allclose(out.var(-1, unbiased=False), torch.ones(7), atol=1e-4)


def test_layernorm_ignores_sigma():
    # LN acts on the point mu; sigma is inert (like the identity norm) and its shape is never read,
    # so a diagonal and a full-covariance sigma give byte-identical output.
    K = 3
    norm = LayerNorm(K)
    mu = torch.randn(4, K)
    out_diag = norm(mu, torch.rand(4, K) + 0.5)
    out_full = norm(mu, torch.rand(4, K, K))               # full-cov shape also ignored (no crash)
    assert torch.allclose(out_diag, out_full, atol=1e-6)


def test_layernorm_is_not_gauge_equivariant():
    # DOCUMENTED departure: standard LayerNorm is NOT gauge-equivariant. Pin LN(g mu) != g LN(mu)
    # for a generic (non-orthogonal) g in GL(K), so a future refactor cannot silently assume
    # equivariance here (mirrors the equivariance-break pins on head_mixer / regime_ii). The
    # gauge-pure norms remain none / mahalanobis.
    K = 3
    rng = torch.Generator().manual_seed(0)
    norm = LayerNorm(K)
    g = torch.randn(K, K, generator=rng) + 2 * torch.eye(K)  # invertible, generic (non-orthogonal)
    mu = torch.randn(2, K, generator=rng)
    ln_mu = norm(mu, torch.ones(2, K))
    ln_gmu = norm(mu @ g.T, torch.ones(2, K))
    assert not torch.allclose(ln_gmu, ln_mu @ g.T, atol=1e-3)


def test_layernorm_registered_and_buildable():
    # Registered under "layernorm" and buildable via get_norm; config validation reads _NORMS, so
    # registration alone makes "layernorm" a valid norm_type_block / norm_type_final value.
    assert "layernorm" in _NORMS
    built = get_norm("layernorm")(4, eps=1e-6)
    assert isinstance(built, LayerNorm)


def test_affine_layernorm_identity_at_init():
    # gamma=1, beta=0 at construction -> AffineLayerNorm output is byte-identical to the
    # parameter-free LayerNorm (the step-0 byte-identical contract the learned exceptions carry).
    K = 4
    aff = AffineLayerNorm(K, eps=1e-5)
    plain = LayerNorm(K, eps=1e-5)
    mu = torch.randn(3, 5, K)
    sigma = torch.rand(3, 5, K) + 0.5
    assert torch.equal(aff(mu, sigma), plain(mu, sigma))


def test_affine_layernorm_matches_torch_reference():
    # With learned gamma/beta set, AffineLayerNorm == torch F.layer_norm(mu, (K,), weight, bias, eps).
    torch.manual_seed(0)                                   # deterministic tolerance headroom (fp32)
    K = 4
    aff = AffineLayerNorm(K, eps=1e-5)
    with torch.no_grad():
        aff.weight.copy_(torch.randn(K))
        aff.bias.copy_(torch.randn(K))
    mu = torch.randn(3, 5, K)
    out = aff(mu, torch.rand(3, 5, K) + 0.5)               # sigma ignored
    ref = F.layer_norm(mu, (K,), weight=aff.weight, bias=aff.bias, eps=1e-5)
    assert torch.allclose(out, ref, atol=1e-5)


def test_affine_layernorm_is_module_with_gamma_beta():
    # nn.Module carrying exactly two learnable (K,) params: gamma (weight, ones) and beta (bias, zeros).
    K = 3
    aff = AffineLayerNorm(K)
    assert isinstance(aff, nn.Module)
    params = dict(aff.named_parameters())
    assert set(params) == {"weight", "bias"}
    assert params["weight"].shape == (K,) and params["bias"].shape == (K,)
    assert torch.equal(params["weight"], torch.ones(K))
    assert torch.equal(params["bias"], torch.zeros(K))
    assert params["weight"].requires_grad and params["bias"].requires_grad


def test_affine_layernorm_backward_reaches_gamma_beta():
    # Gradients flow to both gamma and beta (so the M-step can train them).
    K = 4
    aff = AffineLayerNorm(K)
    mu = torch.randn(2, K)
    aff(mu, torch.ones(2, K)).pow(2).sum().backward()
    assert aff.weight.grad is not None and torch.isfinite(aff.weight.grad).all()
    assert aff.bias.grad is not None and torch.isfinite(aff.bias.grad).all()


def test_affine_layernorm_builder_selects_variant():
    # affine=True -> AffineLayerNorm (nn.Module); affine=False / omitted -> parameter-free LayerNorm.
    aff = get_norm("layernorm")(4, eps=1e-6, affine=True)
    plain = get_norm("layernorm")(4, eps=1e-6, affine=False)
    assert isinstance(aff, AffineLayerNorm) and isinstance(aff, nn.Module)
    assert isinstance(plain, LayerNorm) and not isinstance(plain, nn.Module)


@pytest.mark.parametrize("dtype,values", [
    (torch.float16, [-300.0, 300.0]),
    (torch.bfloat16, [100.0, 100.5, 101.0, 101.5]),
    (torch.float16, [1.0, 1.0, 1.0, 1.0009765625]),
])
@pytest.mark.parametrize("affine", [False, True])
def test_layernorm_low_precision_matches_fp32(dtype: torch.dtype, values: list[float], affine: bool) -> None:
    mu = torch.tensor([values], dtype=dtype, requires_grad=True)
    ref_mu = mu.detach().clone().requires_grad_(True)
    K = mu.shape[-1]
    norm = AffineLayerNorm(K) if affine else LayerNorm(K)
    if affine:
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.5, 1.5, K))
            norm.bias.copy_(torch.linspace(-0.2, 0.3, K))
        weight = norm.weight.detach().clone().requires_grad_(True)
        bias = norm.bias.detach().clone().requires_grad_(True)
    else:
        weight = bias = None
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = norm(mu, torch.ones_like(mu))
    ref = F.layer_norm(ref_mu.float(), (K,), weight, bias, eps=1e-6)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)
    probe = torch.linspace(-0.7, 0.9, K)
    (out * probe).sum().backward()
    (ref * probe).sum().backward()
    torch.testing.assert_close(mu.grad, ref_mu.grad)
    if affine:
        torch.testing.assert_close(norm.weight.grad, weight.grad, atol=5e-5, rtol=1e-5)
        torch.testing.assert_close(norm.bias.grad, bias.grad)


def test_layernorm_preserves_double_and_passes_gradcheck() -> None:
    mu = torch.tensor([[-0.7, 0.1, 0.4, 1.3]], dtype=torch.float64, requires_grad=True)
    norm = LayerNorm(4)
    sigma = torch.ones_like(mu)
    assert norm(mu, sigma).dtype == torch.float64
    assert torch.autograd.gradcheck(lambda x: norm(x, sigma), (mu,))


def test_layernorm_predictive_bf16_runtime_and_gradients() -> None:
    from tests.test_relational_integration import _cfg, _tokens
    from vfe3.model.model import VFEModel

    torch.manual_seed(17)
    model = VFEModel(_cfg(attention_mode="projected_predictive",
                          norm_type_block="layernorm", layernorm_affine=True,
                          attention_residual=True))
    observed = []

    def capture(module: nn.Module, args: tuple, output: torch.Tensor) -> None:
        observed.append((args[0].dtype, output.dtype))

    hook = model.block_norm.register_forward_hook(capture)
    ids, targets = _tokens()
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _, logits = model.forward_beliefs(ids, return_logits=True)
        F.cross_entropy(logits.float().reshape(-1, 7), targets.reshape(-1)).backward()
    finally:
        hook.remove()
    assert observed == [(torch.bfloat16, torch.float32)]
    for param in model.block_norm.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
