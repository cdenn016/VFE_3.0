"""Tiny CPU probe for receiver-global minimal-agent precision fallback; root executes.

No production configuration is imported. K=2, H=1, N=3; all tensors are CPU.
"""
import json

import torch

from tests.profiles import make_relational_operators
from vfe3.contracts import CanonicalFrameContext
from vfe3.relational_scoring import set_projected_condition_recovery


def main() -> None:
    torch.set_num_threads(1)
    ops = make_relational_operators(
        [2], 0, attention_residual=False,
        predictive_readout="minimal_agent", minimal_agent_inference="exact",
        predictive_covariance="moment_matched", predictive_covariance_floor=1e-4,
    )
    with torch.no_grad():
        ops.raw_score_q.copy_(torch.tensor([[[1.0, 0.2], [-0.1, 1.0]]]))
        ops.T.copy_(torch.tensor([[[0.8, -0.3], [0.1, 1.1]]]))
        ops.channel_log_noise.fill_(-0.4)
    eye = torch.eye(2).expand(1, 3, 2, 2)
    frame = CanonicalFrameContext(eye, eye, block_diagonal=False)
    mu = torch.tensor([[[0.3, -0.7], [1.1, 0.2], [-0.5, 0.4]]])
    sigma = torch.tensor([[[0.7, 1.3], [1.2, 0.8], [0.9, 1.1]]])
    suffix = sigma.clone()
    suffix[:, 2, :] = torch.tensor([1e4, 1.0])
    prior = torch.ones_like(sigma)
    lp = torch.zeros(3, 3).masked_fill(torch.ones(3, 3, dtype=torch.bool).triu(1), -torch.inf)
    previous = set_projected_condition_recovery(True)
    try:
        rows = []
        outputs = []
        for name, covariance in (("baseline", sigma), ("extreme_suffix", suffix)):
            with torch.no_grad():
                floor = ops.floor_term(prior, covariance, frame, diagonal=True)
                raw_mean, raw_cov, _, status = ops._minimal_agent_body(
                    mu, covariance, frame, lp, True, False, floor)
                out_mean, out_cov = ops.minimal_agent_update(
                    mu, covariance, frame, lp, return_covariance=True, floor_anchor=prior)
            rows.append({
                "case": name,
                "fp32_status": status.tolist(),
                "raw_prefix_mean": raw_mean[:, :2].tolist(),
                "returned_prefix_mean": out_mean[:, :2].tolist(),
                "raw_prefix_covariance": raw_cov[:, :2].tolist(),
                "returned_prefix_covariance": out_cov[:, :2].tolist(),
            })
            outputs.append((raw_mean, raw_cov, out_mean, out_cov))
        a, b = outputs
        result = {
            "cases": rows,
            "raw_prefix_mean_equal": torch.equal(a[0][:, :2], b[0][:, :2]),
            "raw_prefix_covariance_equal": torch.equal(a[1][:, :2], b[1][:, :2]),
            "returned_prefix_mean_max_difference": (a[2][:, :2] - b[2][:, :2]).abs().max().item(),
            "returned_prefix_covariance_max_difference": (a[3][:, :2] - b[3][:, :2]).abs().max().item(),
        }
        print(json.dumps(result, indent=2))
    finally:
        set_projected_condition_recovery(previous)


if __name__ == "__main__":
    main()
