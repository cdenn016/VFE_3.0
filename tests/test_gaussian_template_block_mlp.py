import pytest
import torch
from torch.distributions import MultivariateNormal, kl_divergence

from vfe3.config import VFE3Config
from vfe3.model.block_mlp import (
    GaussianTemplateBlockMLP,
    build_block_mlp,
)
from vfe3.model.model import VFEModel


def _random_template(embed_dim: int = 3, expansion: int = 2, seed: int = 0) -> GaussianTemplateBlockMLP:
    torch.manual_seed(seed)
    mlp = GaussianTemplateBlockMLP(embed_dim=embed_dim, expansion=expansion, dropout=0.0).double()
    with torch.no_grad():
        mlp.log_lambda.normal_(0.0, 0.5)
        mlp.log_pi.normal_(0.0, 0.5)
        mlp.nu.normal_(0.0, 1.0)
    return mlp


def _random_spd(batch: int, dim: int) -> torch.Tensor:
    a = torch.randn(batch, dim, dim, dtype=torch.float64)
    return a @ a.transpose(-1, -2) + 0.5 * torch.eye(dim, dtype=torch.float64)


def test_responsibilities_are_gibbs_weights_of_exact_kl():
    r"""r_ia = softmax_a(log pi_a - KL(q_i || p_a)) against torch's full-covariance KL."""
    mlp = _random_template()
    mu = torch.randn(4, 3, dtype=torch.float64)
    sigma = _random_spd(4, 3)

    r = mlp.responsibilities(mu, sigma)

    lam = mlp.log_lambda.exp()
    q = MultivariateNormal(mu.unsqueeze(1), covariance_matrix=sigma.unsqueeze(1))       # (4, 1)
    p = MultivariateNormal(mlp.nu.unsqueeze(0), covariance_matrix=torch.diag_embed(lam).unsqueeze(0))
    expected = torch.softmax(mlp.log_pi - kl_divergence(q, p), dim=-1)                  # (4, M)
    torch.testing.assert_close(r, expected, rtol=1e-10, atol=1e-12)


def test_shared_isotropic_templates_reduce_to_affine_logits():
    r"""With lambda_a = 1 the softmax logits are nu_a . mu - |nu_a|^2 / 2 + log pi_a (an MLP layer)."""
    mlp = _random_template()
    with torch.no_grad():
        mlp.log_lambda.zero_()
    mu = torch.randn(5, 3, dtype=torch.float64)
    sigma = 2.0 * torch.eye(3, dtype=torch.float64).expand(5, 3, 3)

    expected = torch.softmax(
        mu @ mlp.nu.T - 0.5 * mlp.nu.square().sum(-1) + mlp.log_pi, dim=-1)
    torch.testing.assert_close(mlp.responsibilities(mu, sigma), expected)


def test_full_and_diagonal_covariance_layouts_agree_and_covariance_passes_through():
    mlp = _random_template()
    mu = torch.randn(2, 4, 3, dtype=torch.float64)
    sigma_full = _random_spd(8, 3).reshape(2, 4, 3, 3)
    sigma_diag = torch.diagonal(sigma_full, dim1=-2, dim2=-1)

    full = mlp.forward_moments(mu, sigma_full)
    diag = mlp.forward_moments(mu, sigma_diag)

    torch.testing.assert_close(full.mu, diag.mu)
    assert full.sigma is sigma_full and full.jacobian is None
    expected_mu = mu + mlp.responsibilities(mu, sigma_diag) @ mlp.values
    torch.testing.assert_close(full.mu, expected_mu)


def test_finite_difference_gradients():
    mlp = _random_template(embed_dim=2, expansion=2)
    names = [name for name, _ in mlp.named_parameters()]
    mu = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    sigma = torch.rand(3, 2, dtype=torch.float64).add(0.5).requires_grad_(True)
    params = tuple(p.detach().clone().requires_grad_(True) for p in mlp.parameters())

    def functional(mu_in, sigma_in, *param_values):
        state = dict(zip(names, param_values))
        return torch.func.functional_call(mlp, state, (mu_in, sigma_in))

    assert torch.autograd.gradcheck(functional, (mu, sigma) + params)


def _tiny_cfg(**overrides) -> VFE3Config:
    config = dict(
        vocab_size=17,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=2,
        n_e_steps=1,
        e_q_mu_lr=0.1,
        e_q_sigma_lr=0.02,
        e_phi_lr=0.0,
        decode_mode="diagonal_chunked",
        norm_type_block="none",
        use_block_mlp=True,
        block_mlp_mode="gaussian_template",
        block_mlp_expansion=2,
    )
    config.update(overrides)
    return VFE3Config(**config)


@pytest.mark.parametrize("overrides", [
    {},
    {"family": "gaussian_full", "decode_mode": "full_chunked",
     "attention_mode": "projected_predictive"},
])
def test_model_builds_templates_and_every_parameter_gets_finite_gradients(overrides):
    model = VFEModel(_tiny_cfg(**overrides))
    assert all(type(mlp) is GaussianTemplateBlockMLP for mlp in model.block_mlps)

    _, loss, _ = model(torch.tensor([[1, 2, 3, 4]]), torch.tensor([[2, 3, 4, 5]]))
    loss.backward()

    for mlp in model.block_mlps:
        for name, parameter in mlp.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name


def test_reported_parameter_count_and_flops_match_the_module():
    cfg = _tiny_cfg()
    model = VFEModel(cfg)
    metadata = model.executable_build.block_mlp
    realized = sum(p.numel() for p in model.block_mlps.parameters())
    assert metadata.parameter_count == realized
    K, M = cfg.embed_dim, cfg.embed_dim * cfg.block_mlp_expansion
    assert metadata.flops_per_token == cfg.n_layers * 6.0 * M * K
    assert metadata.report_label == "gaussian_template_mean_only_nonintertwiner"


def test_delta_full_covariance_is_rejected():
    with pytest.raises(ValueError, match="block_mlp_covariance"):
        _tiny_cfg(block_mlp_covariance="delta_full")
    with pytest.raises(ValueError, match="covariance"):
        build_block_mlp(
            "gaussian_template", embed_dim=3, irrep_dims=(3,), expansion=1, activation="gelu",
            dropout=0.0, covariance_contract="delta_full", covariance_floor=1e-4)
