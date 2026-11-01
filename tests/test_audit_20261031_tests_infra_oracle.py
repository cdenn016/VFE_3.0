r"""Validation and the inert-field oracle away from the live route (audit 2026-10-31 gap-route-admissibility-fuzz-9).

The remediation tests pinned the live route and single deviations, while every admissibility failure the October 31
fuzz found needed two or more moved factors. Eight rows of that fuzz's covering array (seed of the lane's cover
probe, rows 0, 1, 3, 4, 5, 6, 7 and 8, every row that ran among the first nine) are fixed here as deviations from the
frozen K = 8 reference route (:func:`tests.profiles.reference_k8_values`), so no launcher value is read. Each row
moves between 12 and 44 factors: families, decoders, gauge groups (so(n) towers, block CO, SL and Borel), frame
transport, RoPE modes, readouts, routings and block MLP modes. On each row, with one ``train_step``, one
``evaluate`` and one evaluation forward from seed 0:

* validation: ``__post_init__`` admits the row, the model builds, every value is finite, and a second run is
  bitwise the first;
* oracle, marked direction: every field ``inactive_fields()`` marks, moved to the first admitted alternative at
  which it is still marked (a value-dependent mark such as ``attention_window`` makes no claim at a value where it
  is not marked), leaves the run bitwise unchanged, or within roundoff when the field's reason says so;
* oracle, unmarked direction: every field the run never reads (construction, optimizer, step, evaluation, eval
  forward), other than the fields only the training loop, the launchers and the artifact writers read, is marked.

Known defects whose fixes are in other remediation packages are listed in :data:`_PENDING`; empty it once those
patches are merged.
"""
import math
import re
import warnings
from dataclasses import fields
from typing import Any, Dict, List, Optional, Set

import pytest
import torch

from tests.profiles import reference_k8_values
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer, evaluate, train_step

_FIELD_NAMES = frozenset(f.name for f in fields(VFE3Config))
# Read only by train(), the launchers or the artifact writers, never by construction, build_optimizer, train_step,
# evaluate or an eval forward (the lane's list).
_LOOP_ONLY = frozenset({
    "batch_size", "max_steps", "warmup_steps", "min_lr_frac", "grad_accum_steps", "seed", "deterministic",
    "log_interval", "eval_interval", "checkpoint_interval", "checkpoint_keep_last", "checkpoint_at_end",
    "block_variance_at_log", "val_diagnostic_batches", "diagnose_canonical_free_energy", "eval_stride",
    "generate_embedding_figures", "resume_from", "trust_resume_checkpoint", "max_failed_step_skips",
    "max_consecutive_skips", "loss_spike_skip_nats", "weight_ema_decay", "weight_ema_start_frac",
    "weight_ema_update_interval", "weight_ema_device", "vocab_size", "max_seq_len", "embed_dim", "n_heads",
    "fp32_matmul_precision",
})
# Lane defects that the config-oracle remediation fixes, held here until it was merged: block_coordinates under
# identity frames holds only to roundoff while its reason claimed coincidence (gap-route-admissibility-fuzz-4, A083
# rewords the reason), and the irrep draw fields are never read under an explicit tower yet were unmarked
# (gap-route-admissibility-fuzz-5, A022 marks them). With that patch merged both sets are empty.
_PENDING: Dict[str, frozenset] = {
    "roundoff": frozenset(),          # config-oracle (A083) is merged
    "unread": frozenset(),            # config-oracle (A022) is merged
}

_BASE: Dict[str, Any] = dict(warmup_steps=0, fp32_matmul_precision="highest", amp_dtype=None)
_ROWS: Dict[str, Dict[str, Any]] = {
    "row0": dict(
        gauge_tying="per_head", attention_residual=True, relational_routing="noisy_key",
        predictive_covariance="passthrough", predictive_covariance_spread_squares=False, block_mlp_inference="exact",
        block_memory_regression_rank=1, block_mlp_tie_layers=True, block_memory_regime_transition="sticky",
        block_memory_triton=True, phi_frame_condition_max=20, phi_type_dropout_freq_scale=1e-05),
    "row1": dict(
        n_layers=1, decode_mode="linear", decode_covariance="encoded", decode_unigram_prior=False,
        untie_decode_bank=False, gauge_group="so_n", gauge_transport="off", group_n=3,
        irrep_spec=[("l1", 1), ("l2", 1)], pos_phi="frozen", pos_phi_project_slk=True, rope_mode="comparison",
        content_link_chart="first_order", relational_receiver_measurement="adjoint",
        relational_receiver_update="euler", tie_comparison_maps=True, minimal_agent_inference="exact",
        predictive_covariance_floor_mode="relative", predictive_covariance_ceiling=50.0,
        predictive_covariance_checkpoint=True, predictive_covariance_fp64_escalation=True,
        predictive_covariance_precision="fp64", predictive_covariance_symmetric_pairs=True,
        lambda_alpha_mode="state_dependent_per_coord", mstep_self_coupling_weight=0.5,
        learn_attention_temperature=True, beta_attention_prior="causal_noself", norm_type_block="none",
        block_coordinates="reference", block_mlp_mode="gaussian_template", block_memory_regime_transition="sticky",
        layernorm_affine=False, block_norm_trace_cap=False, block_norm_cap_moment_fallback=True,
        skip_belief_sigma_update=True, m_table_update_mode="adamw", m_phi_sparse_rows=True,
        m_phi_group_retraction="group_log", routing_kl_max_reference="query_shifted_v1", amp_dtype="bf16",
        phi_type_dropout=0.3, diagnose_agent_elbo=True),
    "row3": dict(
        n_layers=1, n_e_steps=2, family="gaussian_full", decode_mode="full_chunked", decode_covariance="constant",
        untie_decode_bank=False, gauge_group="so_n", gauge_transport="frozen", group_n=4, irrep_spec=[("l1", 2)],
        gauge_tying="shared", cg_covariance_mode="delta_full", pos_phi="learned", pos_phi_project_slk=True,
        rope_mode="frame_value", attention_mode="projected_gradient", attention_residual=True,
        content_link_chart="affine", relational_receiver_measurement="adjoint", relational_receiver_update="euler",
        predictive_covariance="passthrough", predictive_covariance_floor_mode="prior",
        predictive_covariance_checkpoint=True, predictive_covariance_spread_squares=False,
        beta_attention_prior="causal", norm_type_block="none", block_mlp_mode="canonical_frame",
        block_mlp_covariance="delta_full", block_mlp_template_init="absolute", block_mlp_tie_layers=True,
        block_memory_regime_transition="learned", block_memory_triton=True, per_type_hidden_bias=True,
        block_norm_covariance="pushforward_moment", skip_belief_sigma_update=True, m_table_update_mode="adamw",
        m_phi_group_retraction="group_log", m_phi_fisher_completion="stabilizer",
        routing_kl_max_reference="absolute", projected_factorization="gram_certified", phi_frame_condition_max=20,
        phi_type_dropout_freq_scale=1e-05),
    "row4": dict(
        family="gaussian_block", prior_covariance_table="block_cholesky", decode_mode="full_chunked",
        decode_covariance="encoded", decode_unigram_prior=False, decode_second_order=True, gauge_transport="off",
        gauge_tying="per_head", pos_phi="frozen", attention_residual=True,
        relational_receiver_update="closed_form_profiled", relational_routing="noisy_query", tie_comparison_maps=True,
        predictive_covariance="passthrough", predictive_covariance_ceiling=50.0,
        predictive_covariance_fp64_escalation=True, predictive_covariance_symmetric_pairs=True,
        mstep_self_coupling_weight=0.5, learn_attention_temperature=True, beta_attention_prior="causal_noself",
        use_block_mlp=False, block_mlp_mode="cue_result_memory", block_mlp_covariance="posterior",
        block_mlp_inference="exact", block_mlp_template_init="absolute", block_mlp_tie_layers=True,
        layernorm_affine=False, block_norm_covariance="pushforward_moment", block_norm_trace_cap=False,
        block_norm_cap_moment_fallback=True, skip_belief_sigma_update=True, m_phi_update_mode="fisher_group",
        m_table_update_mode="adamw", m_phi_fisher_completion="stabilizer", routing_kl_max_reference="row_min",
        phi_type_dropout=0.3, diagnose_agent_elbo=True),
    "row5": dict(
        n_layers=1, family="gaussian_full", encode_mode="per_token_additive", decode_mode="linear",
        decode_coordinates="reference", decode_covariance="encoded", untie_decode_bank=False,
        gauge_group="block_co", gauge_tying="per_head", cg_covariance_mode="delta_full", pos_phi_project_slk=True,
        rope_mode="none", content_link_chart="first_order", relational_receiver_measurement="adjoint",
        relational_receiver_update="closed_form_profiled", relational_routing="mean_distance",
        tie_comparison_maps=True, predictive_covariance="passthrough", predictive_covariance_floor_mode="prior",
        predictive_covariance_ceiling=50.0, predictive_covariance_fp64_escalation=True,
        predictive_covariance_precision="fp64", predictive_covariance_symmetric_pairs=True,
        lambda_alpha_mode="state_dependent_per_coord", learn_attention_temperature=True,
        beta_attention_prior="causal_windowed", block_coordinates="reference", block_mlp_covariance="delta_full",
        block_mlp_inference="exact", block_mlp_template_init="absolute", block_memory_regression_rank=1,
        block_memory_regime_transition="learned", block_memory_triton=True, per_type_hidden_bias=True,
        layernorm_affine=False, block_norm_covariance="pushforward", block_norm_trace_cap=False,
        m_phi_sparse_rows=True, m_phi_fisher_completion="stabilizer", routing_kl_max_reference="row_min",
        amp_dtype="bf16", projected_factorization="gram_certified", phi_type_dropout_freq_scale=1e-05),
    "row6": dict(
        n_layers=1, n_e_steps=2, decode_mode="expected_likelihood_chunked", decode_unigram_prior=False,
        decode_second_order=True, gauge_group="so_n", gauge_transport="frozen", group_n=3,
        irrep_spec=[("l1", 1), ("l2", 1)], gauge_tying="shared", pos_phi="learned", rope_mode="frame_value",
        attention_mode="projected_gradient", content_link_chart="affine", relational_receiver_measurement="adjoint",
        predictive_covariance="spread_corrected", predictive_covariance_floor_mode="relative",
        predictive_covariance_ceiling=50.0, predictive_covariance_precision="fp64",
        predictive_covariance_symmetric_pairs=True, lambda_alpha_mode="state_dependent_per_coord",
        mstep_self_coupling_weight=0.5, learn_attention_temperature=True, beta_attention_prior="causal_alibi_noself",
        norm_type_block="mahalanobis", block_mlp_mode="canonical_frame", block_mlp_inference="exact",
        block_mlp_template_init="absolute", block_memory_regression_rank=1, block_memory_regime_transition="sticky",
        per_type_hidden_bias=True, block_norm_covariance="pushforward", block_norm_cap_moment_fallback=True,
        skip_belief_sigma_update=True, m_phi_update_mode="pullback_group", m_decode_update_mode="fisher",
        projected_factorization="gram_certified", phi_type_dropout=0.3, diagnose_agent_elbo=True),
    "row7": dict(
        family="gaussian_full", decode_mode="full_chunked", decode_coordinates="reference", gauge_group="block_sl",
        cg_covariance_mode="delta_full", pos_phi="learned", rope_mode="comparison", content_link_edges="lowrank",
        content_link_latent=True, content_link_loop_weight=0.5, relational_receiver_measurement="adjoint",
        relational_routing="noisy_query", predictive_covariance_floor_mode="prior",
        predictive_covariance_ceiling=50.0, predictive_covariance_checkpoint=True,
        predictive_covariance_fp64_escalation=True, predictive_covariance_spread_squares=False,
        mstep_self_coupling_weight=0.5, learn_attention_temperature=True, beta_attention_prior="causal_windowed",
        block_coordinates="reference", block_mlp_mode="prototype_agents", block_mlp_covariance="posterior",
        block_mlp_inference="exact", block_mlp_template_init="absolute", block_mlp_tie_layers=True,
        block_memory_regime_transition="learned", block_norm_covariance="pushforward_moment",
        block_norm_cap_moment_fallback=True, m_decode_update_mode="fisher", m_phi_group_retraction="group_log",
        amp_dtype="bf16", projected_factorization="gram_certified", phi_frame_condition_max=20,
        phi_type_dropout=0.3, phi_type_dropout_freq_scale=1e-05, diagnose_agent_elbo=True),
    "row8": dict(
        family="gaussian_frame_diagonal", decode_unigram_prior=False, gauge_group="block_borel",
        gauge_transport="frozen", gauge_tying="per_head", rope_mode="none", content_link_chart="first_order",
        relational_receiver_update="euler", relational_routing="noisy_query", predictive_readout="minimal_agent",
        predictive_covariance="spread_corrected", mstep_self_coupling_weight=0.5, beta_attention_prior="causal",
        norm_type_block="none", use_block_mlp=False, block_mlp_mode="canonical_frame",
        block_mlp_covariance="delta_full", block_memory_regression_rank=1, block_memory_triton=True,
        layernorm_affine=False, block_norm_cap_moment_fallback=True, m_decode_update_mode="fisher",
        m_phi_sparse_rows=True, m_phi_group_retraction="group_log", m_phi_fisher_completion="stabilizer",
        # phi_type_dropout stays at the reference's 0.0: frame-dropout (tables-families-1) refuses the table Fisher
        # step of gaussian_frame_diagonal under frame dropout, so the fuzz row's 0.3 is no longer admitted.
        routing_kl_max_reference="absolute", projected_factorization="cholesky_qr2_certified",
        diagnose_agent_elbo=True),
}
_COUNTS = torch.tensor([40.0, 3.0, 17.0, 8.0, 25.0, 1.0, 11.0, 6.0, 30.0, 2.0, 9.0])   # a unigram table


class _TracedConfig(VFE3Config):
    r"""A :class:`VFE3Config` whose dataclass-field reads are recorded in ``reads``."""

    reads: Set[str] = set()

    def __getattribute__(self, name: str) -> Any:
        if name in _FIELD_NAMES:
            _TracedConfig.reads.add(name)
        return object.__getattribute__(self, name)


def _values(row: str, **overrides: Any) -> Dict[str, Any]:
    return reference_k8_values(**{**_BASE, **_ROWS[row], **overrides})


def _config(values: Dict[str, Any]) -> VFE3Config:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFE3Config(**values)


def _batch(cfg: VFE3Config) -> tuple:
    gen = torch.Generator().manual_seed(1)
    shape = (cfg.batch_size, cfg.max_seq_len)
    x = torch.randint(0, cfg.vocab_size, shape, generator=gen)
    y = torch.randint(0, cfg.vocab_size, shape, generator=gen)
    y[-1, -1] = -100
    return x, y


def _run(values: Dict[str, Any], *, trace: bool = False) -> Dict[str, Any]:
    r"""Build from seed 0, calibrate templates where the route draws them, one ``train_step`` at the base rates,
    one ``evaluate`` and one evaluation forward; the loss, CE, logits, parameters and pre-step gradients."""
    cfg = _config(values)
    x, y = _batch(cfg)
    if trace:
        object.__setattr__(cfg, "__class__", _TracedConfig)
        _TracedConfig.reads = set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.manual_seed(0)
        model = VFEModel(cfg)
        if cfg.decode_unigram_prior:
            model.prior_bank.set_unigram_log_prior(_COUNTS[:cfg.vocab_size].clone())
        if (cfg.use_block_mlp and cfg.block_mlp_mode in ("prototype_agents", "cue_result_memory")
                and cfg.block_mlp_template_init == "working_variance"):
            model.calibrate_prototype_templates(x, pull=float(cfg.block_mlp_template_init_pull), seed=int(cfg.seed))
        optimizer = build_optimizer(model, cfg)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        model.train()
        torch.manual_seed(5)
        status: Dict[str, Any] = {}
        loss = train_step(model, optimizer, scheduler, x, y, grad_clip=cfg.grad_clip,
                          grad_clip_scope=cfg.grad_clip_scope, grad_accum_steps=cfg.grad_accum_steps,
                          status_out=status)
        names = {id(p): name for name, p in model.named_parameters()}
        grads = {names[key]: g.detach().clone() for key, g in status.get("grads", {}).items()
                 if key in names and g is not None}
        ce = evaluate(model, [(x, y)])["ce"]
        model.eval()
        with torch.no_grad():
            logits = model(x).detach().clone()
    reads = set(_TracedConfig.reads) if trace else set()
    if trace:
        object.__setattr__(cfg, "__class__", VFE3Config)
    return {"loss": float(loss), "ce": float(ce), "logits": logits, "grads": grads, "reads": reads,
            "params": {name: p.detach().clone() for name, p in model.named_parameters()}}


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / max(float(a.norm()), 1e-30))


def _compare(a: Dict[str, Any], b: Dict[str, Any]) -> str:
    r"""'bitwise', 'roundoff' (loss and evaluation within 1e-5, gradients and parameters within 1e-3, relative)
    or 'changed'."""
    if a["params"].keys() != b["params"].keys() or a["grads"].keys() != b["grads"].keys() or any(
            a["params"][k].shape != b["params"][k].shape for k in a["params"]):
        return "changed"
    if (a["loss"] == b["loss"] and a["ce"] == b["ce"] and torch.equal(a["logits"], b["logits"])
            and all(torch.equal(a["grads"][k], b["grads"][k]) for k in a["grads"])
            and all(torch.equal(a["params"][k], b["params"][k]) for k in a["params"])):
        return "bitwise"
    rel_value = max(abs(a["loss"] - b["loss"]) / max(abs(a["loss"]), 1e-30),
                    abs(a["ce"] - b["ce"]) / max(abs(a["ce"]), 1e-30), _relative(a["logits"], b["logits"]))
    rel_state = max([_relative(a["grads"][k], b["grads"][k]) for k in a["grads"]]
                    + [_relative(a["params"][k], b["params"][k]) for k in a["params"]], default=0.0)
    return "roundoff" if rel_value <= 1e-5 and rel_state <= 1e-3 else "changed"


def _finite(run: Dict[str, Any]) -> List[str]:
    bad = [name for name in ("loss", "ce") if not math.isfinite(run[name])]
    bad += ["logits"] if not bool(torch.isfinite(run["logits"]).all()) else []
    bad += [name for name, p in run["params"].items() if not bool(torch.isfinite(p).all())]
    return bad + ["grad:" + name for name, g in run["grads"].items() if not bool(torch.isfinite(g).all())]


def _options(values: Dict[str, Any], key: str) -> List[str]:
    r"""The validator's option set for a string field, read from its refusal of a probe value."""
    try:
        _config({**values, key: "__probe__"})
    except Exception as exc:  # noqa: BLE001: every refusal names the options it admits
        return [option for option in re.findall(r"'([^']*)'", str(exc)) if option != "__probe__"]
    return []


def _alternatives(values: Dict[str, Any], key: str) -> List[Any]:
    value = values[key]
    if isinstance(value, bool):
        return [not value]
    if isinstance(value, str):
        return [option for option in _options(values, key) if option != value]
    if value is None:
        return [2.0, 0.5, 64, 4]
    if isinstance(value, int):
        return [value + 1, 2 * value, value - 1] if value != 0 else [1, 2]
    if isinstance(value, float):
        return [1.5 * value, 0.5 * value, value + 0.1] if value != 0.0 else [0.1, 0.5, 1.0]
    return []


def _marked_neighbor(values: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    r"""The first admitted alternative at which ``key`` is still marked, as complete values."""
    for candidate in _alternatives(values, key):
        moved = {**values, key: candidate}
        try:
            cfg = _config(moved)
        except (ValueError, TypeError):
            continue
        if key in cfg.inactive_fields():
            return moved
    return None


@pytest.mark.usefixtures("restore_model_process_globals")
@pytest.mark.parametrize("row", sorted(_ROWS))
def test_validation_and_the_inert_oracle_hold_on_a_cover_row(row: str, record_property: Any) -> None:
    torch.set_num_threads(min(2, torch.get_num_threads()))
    values = _values(row)
    cfg = _config(values)                                         # validation admits the row
    marked = cfg.inactive_fields()
    reference = _run(values, trace=True)
    assert _finite(reference) == []
    assert _compare(reference, _run(values)) == "bitwise"        # the probe is deterministic

    disagreements: Dict[str, str] = {}
    checked = 0
    for key in sorted(marked):
        moved = _marked_neighbor(values, key)
        if moved is None:
            continue
        checked += 1
        verdict = _compare(reference, _run(moved))
        roundoff_stated = "roundoff" in marked[key] or key in _PENDING["roundoff"]
        if verdict == "changed" or (verdict == "roundoff" and not roundoff_stated):
            disagreements[f"{key}={moved[key]!r}"] = f"marked ({marked[key]}) but {verdict}"
    unread = _FIELD_NAMES - reference["reads"] - _LOOP_ONLY - set(marked) - _PENDING["unread"]
    for key in sorted(unread):
        disagreements[key] = "never read by the run, yet not marked"
    record_property("marked_fields_checked", checked)
    assert checked >= 20, f"only {checked} marked fields had a marked neighbor"
    assert disagreements == {}, disagreements
