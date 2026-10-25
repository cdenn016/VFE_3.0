r"""One switch to the principled exact route (``principled_exact_route`` in ``train_vfe3.py``).

:func:`principled_exact_route_overrides` returns the launcher keys that put a configuration on the exact route,
leaving every other key as given:

* attention is exact Bayesian conditioning, the minimal agent's exact source posterior with its posterior covariance
  (``projected_predictive``, ``predictive_readout='minimal_agent'``, ``minimal_agent_inference='exact'``,
  ``predictive_covariance='moment_matched'``);
* the block MLP slot is the exact cue/result memory, one switching linear-Gaussian step with Bayesian regime
  selection, in reference coordinates, with no block norm between attention and the memory (``cue_result_memory``,
  ``block_mlp_inference='exact'``, ``block_mlp_covariance='posterior'``, no dropout, no EM term,
  ``norm_type_block='none'``, ``block_coordinates='reference'``);
* one bank serves every layer and its regime belief is carried through depth (``block_mlp_tie_layers=True``, and
  ``block_memory_regime_transition='sticky'`` unless 'sticky' or 'learned' is already chosen);
* the belief family is the covariant diagonal one, ``gaussian_frame_diagonal``, which stores the reference variances
  :math:`c_i = \operatorname{diag}(U_i^{-1}\Sigma_iU_i^{-\top})` (so every consumer, the minimal agent, the memory and the
  decoder, reads the reference moments and the diagonal truncation commutes with frame changes), with the Gaussian-KL
  ``diagonal_chunked`` decoder at unit temperature and an untied decode bank (``decode_tau=1``);
* everything :func:`vfe3.gauge_covariant.fully_gauge_covariant_overrides` requires on top of that (frame transport on,
  reference decoding, no fixed-coordinate caps or trust regions, the total-KL self term).

Each layer is then exact inference in a declared per-agent model, collapsed to the stored Gaussian once per step:
attention conditions the receiver on its peers' records, and the memory conditions the result on the cue record. The
collapses (assumed-density filtering over heads and to one Gaussian, and the diagonal M-projection of the family) are
the declared approximations; nothing else in the chain is a neural map. Training still uses the outer next-token
gradient. Not touched: the regression rank and result noise of the memory, the regime trust :math:`\beta` and stay
probability, the routing prior, the predictive covariance floor, the learning rates, the shape and the numerics.

The local-diagonal family ``gaussian_diagonal`` is not used: its diagonal projection in local coordinates does not
commute with frame changes, and the memory would need the dense :math:`C_i = U_i^{-1}\operatorname{diag}(\sigma_i)U_i^{-\top}`.
:func:`principled_exact_route_failures` re-derives every requirement from the resolved config, so a launcher can
refuse a run the overrides did not put on the route.
"""
from typing import Any, Dict, List, Mapping

# Keys the exact route fixes, with their values (the regime transition is handled separately: 'learned' is kept).
EXACT_ROUTE_SETTINGS: Mapping[str, Any] = {
    "attention_mode":          "projected_predictive",
    "predictive_readout":      "minimal_agent",
    "minimal_agent_inference": "exact",
    "predictive_covariance":   "moment_matched",
    "family":                  "gaussian_frame_diagonal",
    "decode_mode":             "diagonal_chunked",
    "untie_decode_bank":       True,
    "decode_tau":              1.0,
    "decode_second_order":     False,
    "norm_type_block":         "none",
    "block_coordinates":       "reference",
    "use_block_mlp":           True,
    "block_mlp_mode":          "cue_result_memory",
    "block_mlp_inference":     "exact",
    "block_mlp_covariance":    "posterior",
    "block_mlp_dropout":       0.0,
    "block_mlp_em_weight":     0.0,
    "block_mlp_tie_layers":    True,
    "content_link_edges":      "off",
    "use_cg_coupling":         False,
}
_REGIME_TRANSITIONS = ("sticky", "learned")


def principled_exact_route_overrides(
    config: Mapping[str, Any],    # the launcher's config dictionary (VFE3Config keyword arguments)
) -> Dict[str, Any]:              # keys whose value must change; apply as {**config, **overrides}
    r"""The launcher keys to change for the principled exact route (module docstring), including the fully
    gauge-covariant overrides of the resulting configuration."""
    from vfe3.gauge_covariant import _launcher_lookup, fully_gauge_covariant_overrides

    get = _launcher_lookup(config)
    wanted: Dict[str, Any] = dict(EXACT_ROUTE_SETTINGS)
    if get("block_memory_regime_transition") not in _REGIME_TRANSITIONS:
        wanted["block_memory_regime_transition"] = "sticky"
    if get("rope_mode") == "comparison_invariant":
        # The kernel router's projected comparison covariance; the minimal agent applies the same comparison-space
        # rotation as a record channel by exact conditioning ('comparison').
        wanted["rope_mode"] = "comparison"
    overrides = {key: value for key, value in wanted.items() if get(key) != value}
    gauge = fully_gauge_covariant_overrides({**config, **overrides})
    clash = {key for key in gauge if key in EXACT_ROUTE_SETTINGS and gauge[key] != EXACT_ROUTE_SETTINGS[key]}
    if clash:                                     # the gauge switch never changes these on the exact route
        raise ValueError(f"principled_exact_route: the gauge-covariant overrides would change {sorted(clash)}")
    return {**overrides, **gauge}


def describe_principled_exact_route_overrides(
    config:    Mapping[str, Any],    # the launcher's config dictionary
    overrides: Mapping[str, Any],    # principled_exact_route_overrides(config)
) -> str:                            # "key=before -> after, ..." (empty when nothing changes)
    r"""The launcher's override log line: each key's value before the switch (the launcher's literal, else the
    ``VFE3Config`` default) and after."""
    from vfe3.gauge_covariant import _launcher_lookup

    get = _launcher_lookup(config)
    return ", ".join(f"{key}={get(key)!r} -> {value!r}" for key, value in overrides.items())


def principled_exact_route_failures(cfg: Any) -> List[str]:   # cfg: a resolved VFE3Config
    r"""Requirements of the exact route that ``cfg`` does not meet (empty when it is on the route): every fixed key,
    a carried regime belief, and every gauge flag of the pure-path report."""
    from vfe3.gauge_covariant import gauge_covariance_failures

    failures = [f"{key}={getattr(cfg, key)!r} (requires {value!r})"
                for key, value in EXACT_ROUTE_SETTINGS.items() if getattr(cfg, key) != value]
    if cfg.block_memory_regime_transition not in _REGIME_TRANSITIONS:
        failures.append(f"block_memory_regime_transition={cfg.block_memory_regime_transition!r} "
                        f"(requires one of {_REGIME_TRANSITIONS})")
    failures.extend(f"gauge flag {name}" for name in gauge_covariance_failures(cfg))
    return failures
