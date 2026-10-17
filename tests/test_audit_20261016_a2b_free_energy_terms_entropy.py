r"""Audit 2026-10-16 F92: the reported attention entropy of ``metrics.free_energy_terms`` is unfloored.

``free_energy_terms`` reports the raw entropy block
:math:`\tau\sum_{ij}\beta_{ij}(\log\beta_{ij} - \log\pi_{ij})` beside a ``total`` that the
hierarchical rows assemble with ``xlogy``. The reported block used ``log(beta.clamp(min=eps))``, the
floored form ``vfe3.free_energy`` documents as changing the scalar below the floor, so at
``lambda_beta = 1`` the reported blocks no longer summed to ``total``. A large ``eps`` makes the floor
visible; the uniform-prior ``log pi`` floor is inactive here because ``1/N > eps``.
"""
import torch

from vfe3.metrics import free_energy_terms


def test_reported_entropy_is_the_xlogy_block_the_total_carries() -> None:
    g = torch.Generator().manual_seed(0)
    N, eps, tau = 5, 0.1, 0.7
    energy = 4.0 * torch.rand(2, N, N, generator=g, dtype=torch.float64)
    beta = torch.softmax(-energy / tau, dim=-1)
    assert bool((beta < eps).any())                              # entries the floor would lift
    self_div = torch.rand(2, N, generator=g, dtype=torch.float64)
    alpha = torch.rand(2, N, generator=g, dtype=torch.float64)

    terms = free_energy_terms(self_div, energy, beta, alpha, tau=tau, eps=eps)

    log_pi = torch.log(torch.tensor(1.0 / N, dtype=torch.float64))
    expected = float((tau * (torch.special.xlogy(beta, beta) - beta * log_pi)).sum())
    assert abs(terms["attention_entropy"] - expected) < 1e-12
    blocks = terms["self_coupling"] + terms["belief_coupling"] + terms["attention_entropy"]
    assert abs(terms["total"] - blocks) < 1e-12


def test_reported_entropy_is_zero_safe_on_masked_keys() -> None:
    N, tau = 4, 1.0
    beta = torch.tensor([[0.5, 0.5, 0.0, 0.0],
                         [1.0, 0.0, 0.0, 0.0],
                         [0.25, 0.25, 0.25, 0.25],
                         [0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    energy = torch.zeros(N, N, dtype=torch.float64)
    terms = free_energy_terms(torch.zeros(N, dtype=torch.float64), energy, beta,
                              torch.ones(N, dtype=torch.float64), tau=tau)
    log_pi = torch.log(torch.tensor(1.0 / N, dtype=torch.float64))
    expected = float((torch.special.xlogy(beta, beta) - beta * log_pi).sum())
    assert abs(terms["attention_entropy"] - expected) < 1e-12
