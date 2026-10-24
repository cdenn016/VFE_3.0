"""Small CPU examples for the prototype generative-model investigation.

No model construction, training, dataset access, or accelerator use occurs.
Run with C:/anaconda/python.exe from any directory.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from vfe3.model.prototype_agents import (
    prototype_elbo,
    prototype_exact_posterior,
    prototype_mean_field_posterior,
    template_kl_logits,
)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_default_dtype(torch.float64)


def error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the maximum absolute discrepancy between two small tensors."""
    return float((a - b).abs().max())


def direct_mixture(
    xi: torch.Tensor,
    C: torch.Tensor,
    nu: torch.Tensor,
    log_lambda: torch.Tensor,
    log_pi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Compute component posteriors independently in precision coordinates."""
    C_inv = torch.linalg.inv(C)
    precisions = torch.diag_embed(torch.exp(-log_lambda))
    P = torch.linalg.inv(C_inv + precisions)
    h = C_inv @ xi + (precisions @ nu.unsqueeze(-1)).squeeze(-1)
    m = (P @ h.unsqueeze(-1)).squeeze(-1)
    logits = torch.log_softmax(log_pi, dim=-1) + torch.distributions.MultivariateNormal(
        nu, covariance_matrix=C + torch.diag_embed(log_lambda.exp())
    ).log_prob(xi)
    w = logits.softmax(-1)
    mean = (w[:, None] * m).sum(0)
    delta = m - mean
    covariance = (w[:, None, None] * (P + delta[:, :, None] * delta[:, None, :])).sum(0)
    return mean, covariance, w, torch.logsumexp(logits, dim=-1)


@torch.no_grad()
def main() -> None:
    """Execute fixed examples and save measured discrepancies and gaps."""
    xi = torch.tensor([0.2, -0.1])
    C = torch.tensor([[1.2, 0.4], [0.4, 0.8]])
    nu = torch.tensor([[-1.5, 0.7], [1.7, -0.8], [0.2, 1.8]])
    log_lambda = torch.tensor([[0.4, 0.9], [1.1, 0.3], [0.7, 1.4]]).log()
    log_pi = torch.tensor([-0.3, 0.2, 0.4])
    posterior = prototype_exact_posterior(xi, C, nu, log_lambda, log_pi)
    mean, covariance, weights, evidence = direct_mixture(xi, C, nu, log_lambda, log_pi)
    exact_errors = {
        "mean": error(posterior.xi_plus, mean),
        "covariance": error(posterior.C_plus, covariance),
        "labels": error(posterior.weights, weights),
        "log_evidence": error(posterior.log_evidence, evidence),
    }
    assert max(exact_errors.values()) < 1e-12

    epsilon = 1e-5
    basis = torch.eye(2)
    columns = []
    score = []
    for e in basis:
        plus = prototype_exact_posterior(xi + epsilon * e, C, nu, log_lambda, log_pi)
        minus = prototype_exact_posterior(xi - epsilon * e, C, nu, log_lambda, log_pi)
        columns.append((plus.xi_plus - minus.xi_plus) / (2 * epsilon))
        score.append((plus.log_evidence - minus.log_evidence) / (2 * epsilon))
    jacobian = torch.stack(columns, dim=-1)
    score_tensor = torch.stack(score)
    response_error = error(jacobian, covariance @ torch.linalg.inv(C))
    score_error = error(mean - xi, C @ score_tensor)
    assert max(response_error, score_error) < 1e-8

    diagonal = C.diagonal().clone()
    dense_diagonal = prototype_exact_posterior(xi, torch.diag(diagonal), nu, log_lambda, log_pi)
    diagonal_output = prototype_exact_posterior(xi, diagonal, nu, log_lambda, log_pi)
    projection_error = error(diagonal_output.C_plus, dense_diagonal.C_plus.diagonal())
    dropped_covariance = float(dense_diagonal.C_plus[0, 1])
    assert projection_error < 1e-12
    assert abs(dropped_covariance) > 1e-3

    cavi_rows = []
    for steps in (1, 2, 4, 8):
        q = prototype_mean_field_posterior(xi, C, nu, log_lambda, log_pi, cavi_steps=steps)
        elbo = prototype_elbo(xi, C, q.xi_plus, q.C_plus, q.responsibilities, nu, log_lambda, log_pi)
        refreshed = template_kl_logits(q.xi_plus, q.C_plus.diagonal(), nu, log_lambda, log_pi).softmax(-1)
        cavi_rows.append({
            "sweeps": steps,
            "elbo": float(elbo),
            "gap": float(evidence - elbo),
            "label_staleness_max": error(q.responsibilities, refreshed),
        })
    assert all(b["elbo"] >= a["elbo"] - 1e-12 for a, b in zip(cavi_rows, cavi_rows[1:]))

    x0 = torch.zeros(1)
    c0 = torch.ones(1)
    centers = torch.tensor([[-3.0], [3.0]])
    widths = torch.full((2, 1), math.log(0.25))
    priors = torch.zeros(2)
    p = prototype_exact_posterior(x0, c0, centers, widths, priors)
    exact_product_elbo = prototype_elbo(x0, c0, p.xi_plus, p.C_plus, p.weights, centers, widths, priors)
    q = prototype_mean_field_posterior(x0, c0, centers, widths, priors, cavi_steps=8)
    mf_elbo = prototype_elbo(x0, c0, q.xi_plus, q.C_plus, q.responsibilities, centers, widths, priors)
    exact_gap = float(p.log_evidence - exact_product_elbo)
    mf_gap = float(p.log_evidence - mf_elbo)
    assert abs(float(p.C_plus) - 5.96) < 1e-12
    assert abs(exact_gap - (28.8 - 0.5 * math.log(29.8))) < 1e-12
    assert abs(mf_gap - 14.4) < 1e-12

    results = {
        "scope": "Fixed tiny CPU float64 examples, K <= 2 and M <= 3. No training or GPU evidence.",
        "torch_version": torch.__version__,
        "threads": torch.get_num_threads(),
        "cuda_visible": torch.cuda.is_available(),
        "independent_exact_errors": exact_errors,
        "posterior_response_finite_difference_error": response_error,
        "evidence_score_finite_difference_error": score_error,
        "diagonal_projection_error": projection_error,
        "discarded_off_diagonal_covariance": dropped_covariance,
        "cavi": cavi_rows,
        "symmetric_example": {
            "input_mean": 0.0, "input_variance": 1.0,
            "template_means": [-3.0, 3.0], "template_variances": [0.25, 0.25],
            "exact_weights": p.weights.tolist(),
            "exact_output_mean": float(p.xi_plus),
            "exact_output_variance": float(p.C_plus),
            "factorized_exact_output_gap_nats": exact_gap,
            "mean_field_output_mean": float(q.xi_plus),
            "mean_field_output_variance": float(q.C_plus),
            "mean_field_gap_nats": mf_gap,
        },
        "all_checks_passed": True,
    }
    output = Path(__file__).with_name("check-results.json")
    output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
