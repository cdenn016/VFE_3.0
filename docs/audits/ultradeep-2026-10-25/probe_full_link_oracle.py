"""Coordinator-only tiny CPU fixed-context cost/oracle check."""
from __future__ import annotations

import json
from pathlib import Path
import runpy
from typing import Any

scope = runpy.run_path(str(Path(__file__).with_name("probes_contracts_artifacts.py")))
torch, attempt = scope["torch"], scope["attempt"]
from vfe3.gradients.full_link import FullLinkContext, full_link_cost, full_link_cost_parts, full_link_gradients


def check(length: int) -> dict[str, Any]:
    mu = torch.zeros(1, length, 2, dtype=torch.float64)
    sigma = torch.tensor([2.0, 3.0], dtype=mu.dtype).expand_as(mu).clone().requires_grad_()
    Lambda = torch.eye(2, dtype=mu.dtype).expand(1, length, 2, 2)
    c = torch.tensor([1.0, 2.0], dtype=mu.dtype).expand_as(mu)
    ctx = FullLinkContext(Lambda, mu, c, torch.ones(1, 1, length, length) / length,
                          torch.zeros(1, 1, length, length), None)
    expected = .5 * (sigma - c * sigma.log()).sum()
    observed = attempt(lambda: float(full_link_cost(mu, sigma, ctx).detach()))
    return {"length": length, "expected": float(expected.detach()), "cost": observed,
            "parts": [float(v.detach()) for v in full_link_cost_parts(mu, sigma, ctx)]}


def main() -> None:
    torch.set_num_threads(1)
    result = [attempt(lambda n=n: check(n)) for n in (3, 1)]
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
