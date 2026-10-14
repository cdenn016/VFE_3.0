r"""One switch to the fully gauge-covariant model (``fully_gauge_covariant`` in ``train_vfe3.py``).

:func:`fully_gauge_covariant_overrides` returns the smallest set of launcher keys that puts a configuration on the
gauge facet of the pure-path report (every ``gauge_flags`` entry of
:func:`vfe3.run_artifacts._pure_path_report` true), leaving every other key as given. Under a common gauge
transformation :math:`U_i \to g_iU_i` the model then transforms covariantly end to end:

* the learned frames transport (``gauge_transport='on'``) and the family can represent :math:`g\Sigma g^\top`
  (``gaussian_full`` unless ``gaussian_block`` is chosen);
* the block norm and block MLP act as :math:`x \mapsto U_iN(U_i^{-1}x)` (``block_coordinates='reference'``), or
  are covariant on their own (``canonical_frame``, ``gauge_gate`` with an equivariant norm);
* the decoder reads the reference moments :math:`(\xi_i, C_i)` (``decode_coordinates='reference'``);
* no fixed-coordinate cap or trust region rewrites the covariance or the frames (``sigma_max``,
  ``phi_mstep_max_matrix_norm``, ``phi_frame_condition_max``, ``m_phi_update_mode``);
* the encoder has no additive per-coordinate control, RoPE (if any) acts in comparison space, and the self term
  uses the frame-invariant total KL (``lambda_alpha_mode='state_dependent'``).

The content links are not touched: ``content_link_chart``, ``content_link_edges`` (the Lie-algebra links
:math:`L_{ij}`) and the latent-link settings stay as configured. Neither are the routing, the attention prior, the
readout, the predictive covariance, the learning rates or the numerics.

:func:`gauge_covariance_failures` re-derives the gauge flags from the resolved config, so a launcher can refuse a
run the overrides did not make covariant.
"""
from typing import Any, Dict, List, Mapping

# Block-MLP modes covariant in local coordinates; block_coordinates='reference' refuses them.
_SELF_COVARIANT_MLP_MODES = ("canonical_frame", "gauge_gate")
# Block norms block_coordinates='reference' admits (it conjugates them by the realized frame).
_REFERENCE_NORMS = ("none", "layernorm")


def fully_gauge_covariant_overrides(
    config: Mapping[str, Any],    # the launcher's config dictionary (VFE3Config keyword arguments)
) -> Dict[str, Any]:              # keys whose value must change; apply as {**config, **overrides}
    r"""The launcher keys to change for the fully gauge-covariant model (module docstring).

    Raises ``ValueError`` for a combination with no covariant reading that keeps the chosen block: a block MLP that
    is covariant only in local coordinates (``canonical_frame``, ``gauge_gate``) under a non-equivariant block norm,
    or a block MLP that needs reference coordinates under a norm they do not admit.
    """
    from vfe3.config import VFE3Config
    from vfe3.geometry.norms import get_norm_registration

    defaults = VFE3Config()
    get = lambda key: config.get(key, getattr(defaults, key))       # noqa: E731
    wanted: Dict[str, Any] = {
        "gauge_transport":           "on",
        "decode_coordinates":        "reference",
        "sigma_max":                 None,
        "phi_mstep_max_matrix_norm": None,
        "phi_frame_condition_max":   None,
        "lambda_alpha_mode":         "state_dependent",
    }
    if get("family") not in ("gaussian_full", "gaussian_block"):
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
    mlp_mode = get("block_mlp_mode") if get("use_block_mlp") else None
    if mlp_mode in _SELF_COVARIANT_MLP_MODES:
        if not get_norm_registration(norm).gauge_equivariant:
            raise ValueError(
                f"fully_gauge_covariant: block_mlp_mode={mlp_mode!r} is covariant only in local coordinates, where "
                f"norm_type_block={norm!r} is not gauge-equivariant; set norm_type_block to 'none' or "
                "'mahalanobis', or use a block MLP mode that block_coordinates='reference' conjugates")
        wanted["block_coordinates"] = "local"
    else:
        if norm not in _REFERENCE_NORMS:
            raise ValueError(
                f"fully_gauge_covariant: block_coordinates='reference' admits norm_type_block in {_REFERENCE_NORMS}, "
                f"got {norm!r}")
        wanted["block_coordinates"] = "reference"
        wanted["use_cg_coupling"] = False        # CG coupling acts in local coordinates before the norm
    return {key: value for key, value in wanted.items() if get(key) != value}


def gauge_covariance_failures(cfg: Any) -> List[str]:   # cfg: a resolved VFE3Config
    r"""Names of the pure-path report's ``gauge_flags`` that are false for ``cfg`` (empty when fully covariant)."""
    from vfe3.run_artifacts import _pure_path_report

    flags = _pure_path_report(cfg, [])["gauge_flags"]
    return sorted(name for name, value in flags.items() if value is not True and value != "not_applicable")
