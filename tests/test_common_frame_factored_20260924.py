"""Square-root common-frame full-Gaussian KL (optimization_plan.md WS3), 2026-09-24.

``full_cov_kl_backend="common_frame_factored"`` scores KL(q_i || Omega_ij q_j) in the pulled-back
frame, KL(U_i^{-1} q_i || U_j^{-1} q_j), like ``"common_frame"``, but factors each pulled-back
covariance from its square root U^{-1} chol(Sigma) by CholeskyQR2 and certifies that factor,
instead of admitting only frames with cond(U_i)^2 cond(C_j) <= 1/(64 d eps).
"""
import pytest
import torch

from vfe3.families.full_gaussian_score import common_frame_kl_energy
from vfe3.families.gaussian import FullGaussian
from vfe3.geometry.transport import CompactFactoredTransport


def _conditioned(n: int, kappa: float, generator: torch.Generator, batch: tuple) -> torch.Tensor:
    r"""float64 (…, n, n) matrices with singular values geometric from 1 to 1/kappa."""
    left = torch.linalg.qr(torch.randn(*batch, n, n, generator=generator, dtype=torch.float64))[0]
    right = torch.linalg.qr(torch.randn(*batch, n, n, generator=generator, dtype=torch.float64))[0]
    singular = torch.logspace(0., -torch.log10(torch.tensor(kappa)).item(), n, dtype=torch.float64)
    return (left * singular) @ right.transpose(-1, -2)


def _case(kappa_U: float, kappa_Sigma: float, tokens: int = 6, H: int = 2, d: int = 4) -> tuple:
    r"""float64 (mu, Sigma, U) with per-token frame blocks of condition ``kappa_U``."""
    generator = torch.Generator().manual_seed(29)
    K = H * d
    U = _conditioned(d, kappa_U, generator, (1, tokens, H))
    root = _conditioned(K, kappa_Sigma ** .5, generator, (1, tokens))
    Sigma = root @ root.transpose(-1, -2)
    Sigma = Sigma / Sigma.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None]
    mu = torch.randn(1, tokens, K, generator=generator, dtype=torch.float64)
    return mu, Sigma, U


def _energy(case: tuple, backend: str, dtype: torch.dtype = torch.float32,
            requires_grad: bool = False) -> tuple:
    mu, Sigma, U = (x.to(dtype).detach().requires_grad_(requires_grad) for x in case)
    omega = CompactFactoredTransport(U, torch.linalg.inv(U), mu.shape[-1], same_frame_flat_cocycle=True)
    energy = common_frame_kl_energy(mu, Sigma, mu, Sigma, omega, kl_max=float("inf"),
                                    precision_policy="fp32_escalate", irrep_dims=[U.shape[-1]] * U.shape[-3],
                                    **({} if backend == "common_frame" else {"factored": True}))
    return energy, (mu, Sigma, U)


def test_factored_backend_admits_frames_the_squared_bound_rejects() -> None:
    case = _case(kappa_U=10., kappa_Sigma=30.)
    assert _energy(case, "common_frame")[0] is None             # cond(U)^2 cond(C) exceeds 1/(64 d eps)
    factored = _energy(case, "common_frame_factored")[0]
    assert factored is not None and factored.dtype == torch.float32
    reference = _energy(case, "common_frame", torch.float64)[0]
    error = (factored.double() - reference).abs().max().item()
    assert error <= 1e-4 * reference.abs().max().item()


def test_factored_backend_rejects_an_uncertified_factor() -> None:
    assert _energy(_case(kappa_U=3e3, kappa_Sigma=30.), "common_frame_factored")[0] is None


def test_factored_backend_float64_matches_common_frame_values_and_gradients() -> None:
    case = _case(kappa_U=4., kappa_Sigma=5., tokens=3)
    energies, inputs = [], []
    for backend in ("common_frame", "common_frame_factored"):
        energy, leaves = _energy(case, backend, torch.float64, requires_grad=True)
        energies.append(energy)
        inputs.append(leaves)
    torch.testing.assert_close(energies[1], energies[0], rtol=1e-10, atol=1e-10)
    weights = torch.linspace(.2, 1.1, energies[0].numel(), dtype=torch.float64).reshape_as(energies[0])
    grads = [torch.autograd.grad((weights * e).sum(), leaves) for e, leaves in zip(energies, inputs)]
    for got, want in zip(grads[1], grads[0]):
        torch.testing.assert_close(got, want, rtol=1e-8, atol=1e-8)


def test_factored_backend_gradient_matches_finite_differences() -> None:
    mu, Sigma, U = _case(kappa_U=3., kappa_Sigma=3., tokens=3, H=1, d=3)

    def energy(m, S, frames):
        S = .5 * (S + S.transpose(-1, -2))
        omega = CompactFactoredTransport(frames, torch.linalg.inv(frames), m.shape[-1],
                                         same_frame_flat_cocycle=True)
        return common_frame_kl_energy(m, S, m, S, omega, kl_max=float("inf"), precision_policy="fp64",
                                      irrep_dims=[3], factored=True)

    assert torch.autograd.gradcheck(energy, tuple(x.requires_grad_(True) for x in (mu, Sigma, U)))


@pytest.mark.cuda
def test_factored_backend_resolves_its_decisions_in_one_host_transfer() -> None:
    import warnings

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    mu, Sigma, U = (x.float().cuda() for x in _case(kappa_U=10., kappa_Sigma=30.))
    omega = CompactFactoredTransport(U, torch.linalg.inv(U), mu.shape[-1], same_frame_flat_cocycle=True)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            energy = common_frame_kl_energy(mu, Sigma, mu, Sigma, omega, kl_max=float("inf"),
                                            precision_policy="fp32_escalate", irrep_dims=[4, 4], factored=True)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    assert energy is not None
    assert sum("synchroniz" in str(w.message).lower() for w in caught) == 1


def test_coupling_energy_dispatches_the_factored_backend() -> None:
    mu, Sigma, U = (x.float() for x in _case(kappa_U=10., kappa_Sigma=30.))
    omega = CompactFactoredTransport(U, torch.linalg.inv(U), mu.shape[-1], same_frame_flat_cocycle=True)
    options = dict(irrep_dims=[4, 4], precision_policy="fp32_escalate", diagonal_out=False, kl_max=float("inf"))
    factored = FullGaussian.coupling_energy(mu, Sigma, mu, Sigma, omega,
                                            score_backend="common_frame_factored", **options)
    direct = common_frame_kl_energy(mu, Sigma, mu, Sigma, omega, kl_max=float("inf"),
                                    precision_policy="fp32_escalate", irrep_dims=[4, 4], factored=True)
    torch.testing.assert_close(factored, direct, rtol=0., atol=0.)


def test_config_accepts_the_factored_backend_and_the_model_runs() -> None:
    from tests.test_full_gaussian_score_config import _tiny_config
    from vfe3.model.model import VFEModel

    cfg = _tiny_config(full_cov_kl_backend="common_frame_factored")
    torch.manual_seed(3)
    model = VFEModel(cfg)
    tokens = torch.tensor([[0, 1]])
    _, loss, _ = model(tokens, torch.tensor([[1, 2]]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
