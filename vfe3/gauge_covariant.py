r"""One switch to the fully gauge-covariant model (``fully_gauge_covariant`` in ``train_vfe3.py``).

:func:`fully_gauge_covariant_overrides` returns the smallest set of launcher keys that puts a configuration on the
gauge facet of the pure-path report (every ``gauge_flags`` entry of
:func:`vfe3.run_artifacts._pure_path_report` true), leaving every other key as given. Under a common gauge
transformation :math:`U_i \to g_iU_i` the model then transforms covariantly end to end:

* the learned frames transport (``gauge_transport='on'``) and the family can represent :math:`g\Sigma g^\top`
  (``gaussian_full`` unless ``gaussian_block`` or ``gaussian_frame_diagonal`` is chosen; the latter stores
  :math:`c_i = \operatorname{diag}(U_i^{-1}\Sigma_iU_i^{-\top})`, which the action leaves fixed);
* the block norm and block MLP act as :math:`x \mapsto U_iN(U_i^{-1}x)` (``block_coordinates='reference'``, on
  ``projected_predictive`` only), or are covariant on their own (``canonical_frame``, ``gauge_gate`` with an
  equivariant norm, or no block MLP under the equivariant ``none`` or ``mahalanobis`` norm, where
  ``block_coordinates`` is left as given);
* the decoder reads the reference moments :math:`(\xi_i, C_i)` (``decode_coordinates='reference'``);
* no fixed-coordinate cap or trust region rewrites the covariance or the frames (``sigma_max``,
  ``phi_mstep_max_matrix_norm``, ``phi_frame_condition_max``, ``m_phi_update_mode``);
* the encoder has no additive per-coordinate control, RoPE (if any) acts in comparison space, and the self term
  uses the frame-invariant total KL (``lambda_alpha_mode='state_dependent'``).

The content links are not touched: ``content_link_chart``, ``content_link_edges`` (the Lie-algebra links
:math:`L_{ij}`) and the latent-link settings stay as configured. Neither are the routing, the attention prior, the
readout, the predictive covariance and its floor, the decode covariance, the learning rates or the numerics. Both
floors are covariant under the declared action: the prior floor :math:`f\Sigma_{p,i}` moves with the prior, and the
relative floor :math:`f\tfrac{\operatorname{tr}C_i}{K}U_iU_i^\top` has an invariant :math:`\operatorname{tr}C_i`
and so maps to :math:`g_i(\cdot)g_i^\top`; the relative floor fails only under a non-orthogonal global right
action :math:`U_i \to U_ia`, which the switch does not declare (audit 2026-10-17, item 43). Under
``decode_coordinates='reference'`` the decode covariance ``"constant"`` is :math:`\sigma_{\mathrm{init}}I` in
reference coordinates, so it is invariant as well (audit 2026-10-17, item 25).

:func:`gauge_covariance_failures` re-derives the gauge flags from the resolved config, so a launcher can refuse a
run the overrides did not make covariant.
"""
import warnings
from typing import Any, Callable, Dict, List, Mapping

# Block-MLP modes covariant in local coordinates; block_coordinates='reference' refuses them.
_SELF_COVARIANT_MLP_MODES = ("canonical_frame", "gauge_gate")
# Block norms block_coordinates='reference' admits (it conjugates them by the realized frame).
_REFERENCE_NORMS = ("none", "layernorm")


def _launcher_lookup(
    config: Mapping[str, Any],    # the launcher's config dictionary (VFE3Config keyword arguments)
) -> Callable[[str], Any]:        # key -> the launcher's literal, else the VFE3Config default
    r"""The value each key resolves to before the switch: the launcher's literal, else the ``VFE3Config`` default."""
    from vfe3.config import VFE3Config

    defaults = VFE3Config()
    return lambda key: config.get(key, getattr(defaults, key))


def fully_gauge_covariant_overrides(
    config: Mapping[str, Any],    # the launcher's config dictionary (VFE3Config keyword arguments)
) -> Dict[str, Any]:              # keys whose value must change; apply as {**config, **overrides}
    r"""The launcher keys to change for the fully gauge-covariant model (module docstring).

    Raises ``ValueError`` for a combination with no covariant reading that keeps the chosen block or decoder: a
    block MLP that is covariant only in local coordinates (``canonical_frame``, ``gauge_gate``) under a
    non-equivariant block norm; a block that needs reference coordinates (a LayerNorm, or a block MLP that is not
    covariant on its own) under a norm they do not admit or on an attention route without them (only
    ``projected_predictive`` has them); or a decoder with no reference-coordinate reading (``decode_mode`` outside
    ``linear``, ``diagonal_chunked``, ``full_chunked``).
    """
    from vfe3.config import _REFERENCE_DECODE_MODES
    from vfe3.geometry.norms import get_norm_registration

    get = _launcher_lookup(config)
    decode_mode = get("decode_mode")
    if decode_mode not in _REFERENCE_DECODE_MODES:
        raise ValueError(
            f"fully_gauge_covariant: decode_mode={decode_mode!r} has no reference-coordinate reading "
            f"(decode_coordinates='reference' admits {_REFERENCE_DECODE_MODES}); choose one of those decoders")
    wanted: Dict[str, Any] = {
        "gauge_transport":           "on",
        "decode_coordinates":        "reference",
        "sigma_max":                 None,
        "phi_mstep_max_matrix_norm": None,
        "phi_frame_condition_max":   None,
        "lambda_alpha_mode":         "state_dependent",
    }
    # gaussian_frame_diagonal keeps its family: its stored reference variances c_i are invariant under the action, and
    # Sigma_i = U_i diag(c_i) U_i^T moves to g_i Sigma_i g_i^T with the frame.
    if get("family") not in ("gaussian_full", "gaussian_block", "gaussian_frame_diagonal"):
        wanted["family"] = "gaussian_full"
        # The decoder kernel must match the covariance rank; the second-order linear term is diagonal-only.
        if get("decode_mode") == "diagonal_chunked":
            wanted["decode_mode"] = "full_chunked"
        wanted["decode_second_order"] = False
    if get("encode_mode") == "per_token_additive":
        wanted["encode_mode"] = "per_token"
    if get("m_phi_update_mode") in ("pullback_group", "fisher_group"):
        wanted["m_phi_update_mode"] = "adamw"
    if get("rope_mode") in ("frame", "frame_value"):
        # Frame-space RoPE folds a rotation into the content frame, so the reference decoder has no frame to read;
        # comparison-space RoPE applies the same rotations to the routing moments only.
        wanted["rope_mode"] = "comparison"

    norm = get("norm_type_block")
    equivariant_norm = get_norm_registration(norm).gauge_equivariant
    mlp_mode = get("block_mlp_mode") if get("use_block_mlp") else None
    if mlp_mode in _SELF_COVARIANT_MLP_MODES:
        if not equivariant_norm:
            raise ValueError(
                f"fully_gauge_covariant: block_mlp_mode={mlp_mode!r} is covariant only in local coordinates, where "
                f"norm_type_block={norm!r} is not gauge-equivariant; set norm_type_block to 'none' or "
                "'mahalanobis', or use a block MLP mode that block_coordinates='reference' conjugates")
        wanted["block_coordinates"] = "local"
    elif mlp_mode is None and equivariant_norm:
        # Nothing frame-dependent runs in the block (no block MLP; no norm or an equivariant one), so it is covariant
        # in local coordinates and block_coordinates is left as given (audit 2026-10-16 F31).
        pass
    else:
        attention_mode = get("attention_mode")
        if attention_mode != "projected_predictive":
            block = f"block_mlp_mode={mlp_mode!r}" if mlp_mode is not None else "no block MLP"
            raise ValueError(
                f"fully_gauge_covariant: this block (norm_type_block={norm!r}, {block}) is covariant "
                f"only in reference coordinates, which attention_mode={attention_mode!r} does not have "
                "(block_coordinates='reference' requires 'projected_predictive'); set use_block_mlp=False with "
                "norm_type_block 'none' or 'mahalanobis', or use block_mlp_mode 'canonical_frame' or 'gauge_gate' "
                "under such a norm")
        if norm not in _REFERENCE_NORMS:
            raise ValueError(
                f"fully_gauge_covariant: block_coordinates='reference' admits norm_type_block in {_REFERENCE_NORMS}, "
                f"got {norm!r}")
        wanted["block_coordinates"] = "reference"
        wanted["use_cg_coupling"] = False        # CG coupling acts in local coordinates before the norm
    family = wanted.get("family", get("family"))
    if (family in ("gaussian_full", "gaussian_block") and norm == "layernorm"
            and get("block_norm_covariance") == "pushforward"):
        # A warning, not an override: the switch never changes the covariance estimator (audit 2026-10-16 F2).
        capped = get("block_norm_trace_cap")
        warnings.warn(
            "fully_gauge_covariant: block_norm_covariance='pushforward' with the full or head-block family keeps the "
            + ("capped" if capped else "uncapped (block_norm_trace_cap=False)")
            + " delta-method LayerNorm pushforward, which is near rank one (condition number about 1e7, "
            "float32-indefinite) where Sigma/s^2 >> 1, as at the launcher's initialization; "
            "block_norm_covariance='pushforward_moment' is the moment estimate there (audit 2026-10-16 F2"
            + ("" if capped else "; without the cap the delta method is unbounded there, audit 2026-10-17 item 7")
            + ")",
            stacklevel=2)
    return {key: value for key, value in wanted.items() if get(key) != value}


def describe_fully_gauge_covariant_overrides(
    config:    Mapping[str, Any],    # the launcher's config dictionary
    overrides: Mapping[str, Any],    # fully_gauge_covariant_overrides(config)
) -> str:                            # "key=before -> after, ..." (empty when nothing changes)
    r"""The launcher's override log line. Each key shows the value the switch compared before overriding it, the
    launcher's literal or, for a key the launcher omits, the ``VFE3Config`` default (audit 2026-10-16 F33)."""
    get = _launcher_lookup(config)
    return ", ".join(f"{key}={get(key)!r} -> {value!r}" for key, value in overrides.items())


def gauge_covariance_failures(cfg: Any) -> List[str]:   # cfg: a resolved VFE3Config
    r"""Names of the pure-path report's ``gauge_flags`` that are false for ``cfg`` (empty when fully covariant)."""
    from vfe3.run_artifacts import _pure_path_report

    flags = _pure_path_report(cfg, [])["gauge_flags"]
    return sorted(name for name, value in flags.items() if value is not True and value != "not_applicable")
