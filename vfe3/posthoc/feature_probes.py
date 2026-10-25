r"""End-of-run probes of the minimal-agent attention readout and the prototype-agent block MLP.

Both probes replay the trained model once on report token batches (eval mode, no gradients) and wrap the feature's
entry point on the module instance for that replay only, so the comparison runs at exactly the inputs the trained
forward produced.

* :func:`minimal_agent_gap`: every ``RelationalOperators.minimal_agent_update`` call is re-run under both
  ``minimal_agent_inference`` modes at the same input. Per layer and head it reports the total variation between
  the two label sets; per layer the input-whitened mean shift
  :math:`\bigl(\delta_i^\top\Sigma_i^{-1}\delta_i/K\bigr)^{1/2}`, :math:`\delta_i = \mu_i^{+,\mathrm{exact}} - \mu_i^{+,\mathrm{mf}}`,
  and the covariance ratio :math:`(\log\det\Sigma_i^{+,\mathrm{mf}} - \log\det\Sigma_i^{+,\mathrm{exact}})/K` (negative:
  the mean field is the more confident), together with the exact spread share
  :math:`\operatorname{tr}(AR)/\operatorname{tr}(A\bar\Lambda)` per head.
* :func:`prototype_probe`: per ``prototype_agents`` layer, the mean template weight ``(M,)``, the tokens each template
  wins most often (top-1 assignment counts, for the codebook), the template means and log variances, and, on at most
  ``gap_tokens`` tokens, the same exact-versus-mean-field gap for the block (label total variation, whitened mean
  shift, covariance ratio when the block returns a posterior covariance).

The probes are measurements of one trained model on one token population; they do not train or change it.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from vfe3.model.block_mlp import PrototypeAgentBlockMLP


def _whitened_shift(delta: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    r""":math:`(\delta^\top\Sigma^{-1}\delta/K)^{1/2}` per token for diagonal ``(..., K)`` or dense ``(..., K, K)`` Sigma."""
    delta, sigma = delta.double(), sigma.double()
    K = delta.shape[-1]
    if sigma.shape == delta.shape:
        return torch.sqrt((delta.square() / sigma).sum(-1) / K)
    if sigma.shape == (*delta.shape, K):
        solved = torch.linalg.solve(sigma, delta.unsqueeze(-1)).squeeze(-1)
        return torch.sqrt((delta * solved).sum(-1) / K)
    raise ValueError(f"unsupported covariance layout {tuple(sigma.shape)} for means {tuple(delta.shape)}")


def _logdet(sigma: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    r""":math:`\log\det\Sigma` per token, diagonal (shaped like ``mean``) or dense ``(..., K, K)``."""
    sigma = sigma.double()
    if sigma.shape == mean.shape:
        return sigma.log().sum(-1)
    return torch.linalg.slogdet(sigma)[1]


def _label_tv(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    r"""Mean over rows of :math:`\tfrac12\sum_j|p_{ij} - q_{ij}|`, per head: ``(..., H, N, N)`` -> ``(H,)``."""
    tv = 0.5 * (first.double() - second.double()).abs().sum(-1)                 # (..., H, N)
    return tv.reshape(-1, *tv.shape[-2:]).mean(dim=(0, 2)) if tv.dim() >= 3 else tv.mean(-1)


@torch.no_grad()
def minimal_agent_gap(
    model:         torch.nn.Module,
    token_batches: Iterable[torch.Tensor],      # (B, N) token ids
    device:        torch.device,
) -> Optional[Dict[str, Any]]:
    r"""Exact-versus-mean-field gap of every minimal-agent layer at the trained inputs; None without such a layer."""
    layers = list(getattr(model, "relational_operators_by_layer", None) or [])
    targets = [(index, ops) for index, ops in enumerate(layers)
               if getattr(ops, "predictive_readout", "kernel") == "minimal_agent"]
    if not targets:
        return None
    sums: Dict[int, Dict[str, Any]] = {}

    for layer, ops in targets:
        original = ops.minimal_agent_update

        def wrapper(mu, sigma, frame, log_prior=None, *, _layer=layer, _ops=ops, _original=original, **kwargs):
            result = _original(mu, sigma, frame, log_prior, **kwargs)
            configured = _ops.minimal_agent_inference
            outputs = {}
            try:
                for mode in ("exact", "mean_field"):
                    _ops.minimal_agent_inference = mode
                    capture: dict = {}
                    # The layer prior when the caller passed it (moment_matched), else the input covariance: the
                    # floor f Sigma_p is common to both modes and does not enter the mean.
                    mean, cov = _original(mu, sigma, frame, log_prior, return_covariance=True, capture=capture,
                                          floor_anchor=kwargs.get("floor_anchor", sigma))
                    outputs[mode] = (mean, cov, capture)
            finally:
                _ops.minimal_agent_inference = configured
            (m_ex, c_ex, cap_ex), (m_mf, c_mf, cap_mf) = outputs["exact"], outputs["mean_field"]
            K = mu.shape[-1]
            entry = sums.setdefault(_layer, {"tv": 0.0, "spread": 0.0, "shift": 0.0, "logdet": 0.0,
                                             "tokens": 0, "calls": 0})
            entry["tv"] = entry["tv"] + _label_tv(cap_ex["beta_pre"], cap_mf["beta_pre"]).cpu()
            entry["spread"] = entry["spread"] + cap_ex["relational_minimal_agent_spread_share"].double().cpu()
            shift = _whitened_shift(m_ex - m_mf, sigma)                         # (..., N)
            ratio = (_logdet(c_mf, mu) - _logdet(c_ex, mu)) / K                 # (..., N)
            entry["shift"] += float(shift.sum())
            entry["logdet"] += float(ratio.sum())
            entry["tokens"] += int(shift.numel())
            entry["calls"] += 1
            return result

        ops.minimal_agent_update = wrapper
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        for ids in token_batches:
            model.forward_beliefs(ids.to(device))
    finally:
        for module, was_training in modes:
            module.training = was_training
        for _, ops in targets:
            del ops.minimal_agent_update
    out: Dict[str, Any] = {"configured_inference": targets[0][1].minimal_agent_inference, "layers": []}
    for layer, entry in sorted(sums.items()):
        calls, tokens = max(1, entry["calls"]), max(1, entry["tokens"])
        out["layers"].append({
            "layer": layer,
            "label_tv": (entry["tv"] / calls).tolist(),
            "spread_share": (entry["spread"] / calls).tolist(),
            "whitened_mean_shift": entry["shift"] / tokens,
            "logdet_ratio_per_dim": entry["logdet"] / tokens,
            "tokens": entry["tokens"],
        })
    return out


@torch.no_grad()
def prototype_probe(
    model:         torch.nn.Module,
    token_batches: Iterable[torch.Tensor],      # (B, N) token ids
    device:        torch.device,

    *,
    top_tokens:    int = 8,
    gap_tokens:    int = 2048,
) -> Optional[Dict[str, Any]]:
    r"""Template usage, top-1 token codebook and exact-versus-mean-field gap of every ``prototype_agents`` layer.

    The gap reruns the block in the other inference mode, so it is limited to the first ``gap_tokens`` tokens (the
    exact path on a block or dense layout is an extreme opt-in cost).

    A bank shared across layers (``block_mlp_tie_layers=True``) is one object listed at every layer. It gets one
    recording wrapper, and each forward pass calls it once per layer in layer order (:mod:`vfe3.model.stack`),
    so its ``k``-th call of a pass is attributed to the ``k``-th layer that holds it; the wrapper is removed once.
    """
    blocks = getattr(model, "block_mlps", None)
    targets = ([(index, mlp) for index, mlp in enumerate(blocks) if isinstance(mlp, PrototypeAgentBlockMLP)]
               if blocks is not None else [])
    if not targets:
        return None
    current: Dict[str, Optional[torch.Tensor]] = {"ids": None}
    stats: Dict[int, Dict[str, Any]] = {}
    # Distinct bank objects, each with the layers that call it in order (one layer each when untied).
    banks: Dict[int, Tuple[PrototypeAgentBlockMLP, List[int]]] = {}
    for layer, mlp in targets:
        banks.setdefault(id(mlp), (mlp, []))[1].append(layer)
    calls: Dict[int, int] = {key: 0 for key in banks}                 # calls of each bank in the current pass

    for key, (mlp, layers) in banks.items():
        original = mlp.forward_moments

        def recording(mu, sigma, *args, _key=key, _layers=tuple(layers), _mlp=mlp, _original=original, **kwargs):
            call = calls[_key]
            if call >= len(_layers):
                raise RuntimeError(
                    f"prototype_probe: the block MLP of layers {list(_layers)} was called {call + 1} times in one "
                    "forward pass; its calls cannot be attributed to layers")
            calls[_key] = call + 1
            _layer = _layers[call]
            result = _original(mu, sigma, *args, **kwargs)
            weights, _ = _mlp.weights_and_gain(mu, sigma)                       # (B, N, M)
            M = weights.shape[-1]
            entry = stats.setdefault(_layer, {"usage": torch.zeros(M, dtype=torch.float64), "tokens": 0,
                                              "assign": [Counter() for _ in range(M)],
                                              "tv": 0.0, "shift": 0.0, "logdet": 0.0, "gap_tokens": 0,
                                              "logdet_available": True})
            flat = weights.double().reshape(-1, M).cpu()
            entry["usage"] += flat.sum(0)
            entry["tokens"] += flat.shape[0]
            ids = current["ids"]
            if ids is not None and ids.numel() == flat.shape[0]:
                for token, winner in zip(ids.reshape(-1).tolist(), flat.argmax(-1).tolist()):
                    entry["assign"][winner][token] += 1
            budget = gap_tokens - entry["gap_tokens"]
            if budget > 0:
                n = min(budget, flat.shape[0])
                lead = mu.shape[:-1]
                mu_n = mu.reshape(-1, mu.shape[-1])[:n]
                sig_n = sigma.reshape(-1, *sigma.shape[len(lead):])[:n]
                configured = _mlp.inference
                outs = {}
                try:
                    for mode in ("exact", "mean_field"):
                        _mlp.inference = mode
                        w_mode, _ = _mlp.weights_and_gain(mu_n, sig_n)
                        moments = _original(mu_n, sig_n, *args, **kwargs)
                        outs[mode] = (w_mode, moments)
                finally:
                    _mlp.inference = configured
                (w_ex, r_ex), (w_mf, r_mf) = outs["exact"], outs["mean_field"]
                entry["tv"] += float(0.5 * (w_ex.double() - w_mf.double()).abs().sum(-1).sum())
                if sig_n.dim() == mu_n.dim() or sig_n.shape == (*mu_n.shape, mu_n.shape[-1]):
                    entry["shift"] += float(_whitened_shift(r_ex.mu - r_mf.mu, sig_n).sum())
                    if r_ex.sigma is not sig_n and r_ex.sigma.shape == sig_n.shape:
                        K = mu_n.shape[-1]
                        entry["logdet"] += float(((_logdet(r_mf.sigma, mu_n) - _logdet(r_ex.sigma, mu_n)) / K).sum())
                    else:
                        entry["logdet_available"] = False
                else:
                    entry["shift"] = math.nan                                   # block layout: not whitened here
                    entry["logdet_available"] = False
                entry["gap_tokens"] += n
            return result

        mlp.forward_moments = recording
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        for ids in token_batches:
            current["ids"] = ids
            for key in calls:
                calls[key] = 0
            model.forward_beliefs(ids.to(device))
    finally:
        current["ids"] = None
        for module, was_training in modes:
            module.training = was_training
        for mlp, _ in banks.values():                                    # one wrapper per distinct bank
            del mlp.forward_moments
    out: Dict[str, Any] = {"configured_inference": targets[0][1].inference, "layers": []}
    for layer, mlp in targets:
        entry = stats.get(layer)
        if entry is None:
            continue
        usage = entry["usage"] / max(1, entry["tokens"])                        # (M,)
        order = torch.argsort(usage, descending=True).tolist()
        gap_n = max(1, entry["gap_tokens"])
        out["layers"].append({
            "layer": layer,
            "usage": usage.tolist(),
            "codebook": [{"template": a, "usage": float(usage[a]),
                          "top_tokens": entry["assign"][a].most_common(top_tokens)} for a in order],
            "nu": mlp.nu.detach().double().cpu().tolist(),
            "log_lambda_mean": mlp.log_lambda.detach().double().mean(-1).cpu().tolist(),
            "label_tv": entry["tv"] / gap_n,
            "whitened_mean_shift": entry["shift"] / gap_n,
            "logdet_ratio_per_dim": (entry["logdet"] / gap_n if entry["logdet_available"] else math.nan),
            "tokens": entry["tokens"],
            "gap_tokens": entry["gap_tokens"],
        })
    return out
