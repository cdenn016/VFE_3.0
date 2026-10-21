r"""Audit 2026-10-17, lane L5: configuration, gauge certification and the tests of the config.

* item 23: ``prior_handoff_sigma`` is live only where the prior covariance is read (the prior floor, or a ceiling);
* item 24: a snapshot of every ``VFE3Config`` field default, and an oracle check that ``inactive_fields`` agrees with
  two executed optimizer steps on the tiny live route;
* item 25 and 43: the ``fully_gauge_covariant`` switch keeps the decode covariance and the floor mode (both are
  covariant under the declared local action; the passive-gauge harness measures it);
* item 51: the era pin of ``block_norm_trace_cap``;
* item 53: ``ablation`` reports the fields its baseline leaves to defaults that the launcher sets differently;
* item 54: ``train()`` resolves its run keywords from ``cfg``;
* item 55: ``PriorBank``'s config-mirroring keywords are keyword-only and required;
* item 57: the switch's pushforward warning names the trace-cap state.

Run as a script (``python -m tests.test_audit_20261017_l5_config``) to rewrite the default snapshot after a default
change has been handled as the snapshot test's failure message asks.
"""
from __future__ import annotations

import dataclasses
import inspect
import json
import math
import re
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
import torch

from tests.profiles import config_values, live_values, test_config
from vfe3.config import (_HISTORICAL_DEFAULT_PINS, VFE3Config, config_from_serialized, historical_default_pins,
                         serialize_config)
from vfe3.gauge_covariant import fully_gauge_covariant_overrides, gauge_covariance_failures

_SNAPSHOT = Path(__file__).parent / "data" / "vfe3config_defaults_caa0114.json"
_LAUNCHERS = ("train_vfe3.config", "ablation.BASELINE_CONFIG", "scaling.config", "parameter_optimization.config")


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# ------------------------------------------------------------------------------------- item 24: default snapshot

def _jsonable(value: Any) -> Any:
    r"""A JSON-comparable form of a default: tuples become lists, non-finite floats strings."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def field_defaults() -> Dict[str, Any]:
    r"""Every ``VFE3Config`` field default (a ``default_factory`` is called), in JSON-comparable form."""
    out: Dict[str, Any] = {}
    for f in dataclasses.fields(VFE3Config):
        if f.default is not dataclasses.MISSING:
            out[f.name] = _jsonable(f.default)
        elif f.default_factory is not dataclasses.MISSING:          # type: ignore[misc]
            out[f.name] = _jsonable(f.default_factory())            # type: ignore[misc]
    return out


def _launcher_dicts() -> Dict[str, Dict[str, Any]]:
    import ablation
    import parameter_optimization
    import scaling
    import train_vfe3
    return dict(zip(_LAUNCHERS, (train_vfe3.config, ablation.BASELINE_CONFIG, scaling.config,
                                 parameter_optimization.config)))


def test_no_field_default_changes_without_a_pin_row() -> None:
    r"""A changed default reaches every record saved without the field and every launcher that omits it.

    ``_HISTORICAL_DEFAULT_PINS`` says to add a row whenever a default flips; this test enforces it against a
    snapshot of the defaults at caa0114. Fields added after the snapshot are not checked.
    """
    snapshot: Dict[str, Any] = json.loads(_SNAPSHOT.read_text(encoding="utf-8"))
    current = field_defaults()
    removed = sorted(name for name in snapshot if name not in current)
    changed = {name: (snapshot[name], current[name]) for name in snapshot
               if name in current and snapshot[name] != current[name]}
    if not (removed or changed):
        return
    launchers = _launcher_dicts()
    lines = []
    for name, (old, new) in sorted(changed.items()):
        omitting = [label for label, values in launchers.items() if name not in values]
        pinned = (f"pinned to {_HISTORICAL_DEFAULT_PINS[name]!r}" if name in _HISTORICAL_DEFAULT_PINS
                  else "NO pin row")
        lines.append(f"  {name}: default {old!r} -> {new!r} ({pinned}; omitted by {omitting or 'no launcher'})")
    for name in removed:
        lines.append(f"  {name}: removed (config_from_serialized needs a migration that drops or refuses it)")
    pytest.fail(
        "VFE3Config field defaults changed since the snapshot:\n" + "\n".join(lines) + "\n"
        "For each changed default: add a _HISTORICAL_DEFAULT_PINS row in vfe3/config.py pinning the OLD value (or an "
        "era rule in historical_default_pins) so that records saved without the field reload as they ran, or show "
        "that every loadable record carries the field; spell the intended value in every launcher listed as "
        "omitting it; then rewrite the snapshot with `python -m tests.test_audit_20261017_l5_config`.")


def test_the_snapshot_covers_the_pinned_fields() -> None:
    snapshot = json.loads(_SNAPSHOT.read_text(encoding="utf-8"))
    assert set(_HISTORICAL_DEFAULT_PINS) <= set(snapshot) <= set(field_defaults()) | set(snapshot)
    assert {f.name for f in dataclasses.fields(VFE3Config)} >= set(_HISTORICAL_DEFAULT_PINS)


# --------------------------------------------------------------------- item 24: inactive_fields against execution

# Knobs whose effect on two tiny CPU steps is legitimately nil, so an unmarked field may still leave the run
# bitwise unchanged. Inert-marked fields are checked for identity whatever their category.
_TRAIN_LOOP = {"log_interval", "eval_interval", "checkpoint_interval", "max_steps", "warmup_steps", "min_lr_frac",
               "eval_stride", "val_diagnostic_batches", "resume_from", "trust_resume_checkpoint",
               "generate_embedding_figures", "diagnose_canonical_free_energy", "diagnose_agent_elbo", "seed",
               "grad_accum_steps", "grad_clip_scope", "checkpoint_interval",
               # lane L3 (audit 2026-10-17): output policy and failure handling, inert on a healthy two-step run
               "checkpoint_at_end", "checkpoint_keep_last", "block_variance_at_log", "max_failed_step_skips",
               "max_consecutive_skips", "loss_spike_skip_nats",
               # audit 2026-10-20 item 8: the weight EMA shadows the weights and never feeds back into training
               "weight_ema_decay"}
_PRECISION = {"deterministic", "fp32_matmul_precision", "amp_dtype", "decode_av_precision",
              "predictive_covariance_precision", "projected_factorization", "decode_ce_triton", "decode_logits_amp",
              "decode_chunk_size", "relational_tile_mib", "decode_ce_checkpoint", "predictive_covariance_checkpoint",
              "pair_kl_gemm_tol", "relational_diag_moments_kernel"}
# Thresholds that act only when crossed (clamps, caps, escalations, trust radii), and the corpus unigram tempering,
# which the synthetic batches here do not populate.
_RARE_TRIGGER = {"predictive_covariance_ceiling", "phi_frame_condition_max", "frame_pair_escalation_tol", "kl_max",
                 "sigma_max", "transport_chart_max_norm", "phi_mstep_max_matrix_norm", "eps", "grad_clip",
                 "e_mu_q_trust", "e_sigma_q_trust", "predictive_covariance_fp64_escalation",
                 "decode_ranking_fp64_escalation", "unigram_kappa", "block_norm_trace_cap",
                 "m_table_fisher_trust_radius", "m_phi_group_trust_radius", "attention_window"}
_ALLOWLIST = _TRAIN_LOOP | _PRECISION | _RARE_TRIGGER
# Shape and run fields the tiny profile sets; perturbing them changes the problem, not a knob.
_SHAPE = {"vocab_size", "embed_dim", "n_heads", "max_seq_len", "batch_size", "n_layers", "block_mlp_expansion"}


def _two_steps(cfg: VFE3Config) -> Tuple[List[float], Dict[str, torch.Tensor]]:
    r"""Losses and final state of two optimizer steps of :func:`vfe3.train.train` from a fixed initialization."""
    from vfe3.model.model import VFEModel
    from vfe3.train import train

    torch.manual_seed(0)
    model = VFEModel(cfg)
    gen = torch.Generator().manual_seed(1)
    shape = (cfg.batch_size, cfg.max_seq_len)
    batches = [(torch.randint(0, cfg.vocab_size, shape, generator=gen),
                torch.randint(0, cfg.vocab_size, shape, generator=gen)) for _ in range(2)]
    losses = train(model, batches, cfg, n_steps=2, grad_clip=cfg.grad_clip, log_interval=0, eval_interval=0,
                   generate_samples=False)
    return losses, {k: v.detach().clone() for k, v in model.state_dict().items()}


def _identical(a: Tuple[List[float], Dict[str, torch.Tensor]], b: Tuple[List[float], Dict[str, torch.Tensor]]) -> bool:
    return (a[0] == b[0] and a[1].keys() == b[1].keys()
            and all(torch.equal(a[1][k], b[1][k]) for k in a[1]))


def _options(values: Dict[str, Any], key: str) -> List[str]:
    r"""The validator's option set for a string field, read from its refusal of a probe value."""
    try:
        VFE3Config(**{**values, key: "__probe__"})
    except Exception as exc:                                    # every refusal names the options it admits
        return [option for option in re.findall(r"'([^']*)'", str(exc)) if option != "__probe__"]
    return []


def _perturbations(values: Dict[str, Any], key: str) -> List[Any]:
    r"""Candidate values for ``key`` other than its current one, in the order they are tried."""
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


_PROFILES: Dict[str, Callable[[], Dict[str, Any]]] = {
    "launcher": lambda: live_values(),
    # the relative floor, where the 2026-10-17 audit found the one disagreement (item 23)
    "relative_floor": lambda: live_values(predictive_covariance_floor_mode="relative"),
}


@pytest.mark.parametrize("profile", sorted(_PROFILES))
def test_inactive_fields_agree_with_two_executed_steps(profile: str) -> None:
    r"""For each launcher key, a valid perturbation leaves two optimizer steps bitwise identical when
    ``inactive_fields`` marks the key inert (at the base and the perturbed value), and changes them when the key is
    unmarked and not allowlisted (a train-loop, precision or rare-trigger knob)."""
    torch.set_num_threads(min(2, torch.get_num_threads()))
    values = _PROFILES[profile]()
    base_cfg = VFE3Config(**values)
    reference = _two_steps(base_cfg)
    assert _identical(reference, _two_steps(base_cfg))            # the probe is deterministic
    base_inert = base_cfg.inactive_fields()
    keys = sorted(key for key in values if key not in _SHAPE)     # every field: the launcher may omit a live one
    disagreements, checked = [], 0
    for key in keys:
        for candidate in _perturbations(values, key):
            try:
                cfg = VFE3Config(**{**values, key: candidate})
            except (ValueError, TypeError):
                continue
            try:
                run = _two_steps(cfg)
            except (ValueError, RuntimeError):                    # a runtime requirement this probe cannot meet
                continue
            checked += 1
            inert = key in base_inert and key in cfg.inactive_fields()
            same = _identical(reference, run)
            if inert and not same:
                disagreements.append(f"{key}={candidate!r}: marked inert ({base_inert[key]}) but the run changed")
            elif not inert and same and key not in _ALLOWLIST:
                disagreements.append(f"{key}={candidate!r}: unmarked but the run is bitwise identical "
                                     "(mark it in inactive_fields, or allowlist a train-loop, precision or "
                                     "rare-trigger knob here)")
            break
    assert checked >= 40, f"only {checked} launcher keys had a valid perturbation"
    assert disagreements == []


# --------------------------------------------------------------------------------- item 23: prior_handoff_sigma

def _handoff_values(readout: str, **overrides: Any) -> Dict[str, Any]:
    return live_values(attention_mode="projected_predictive", predictive_readout=readout,
                       predictive_covariance="moment_matched", minimal_agent_inference="exact",
                       mstep_self_coupling_weight=0.0, cg_energy_weight=0.0, diagnose_canonical_free_energy=False,
                       n_layers=2, prior_handoff_sigma=0.0, **overrides)


@pytest.mark.parametrize("readout", ["kernel", "minimal_agent"])
@pytest.mark.parametrize("floor_mode, ceiling, live", [("relative", None, False), ("prior", None, True),
                                                      ("relative", 2e-4, True)])
def test_prior_handoff_sigma_is_live_only_where_the_prior_covariance_is_read(
        readout: str, floor_mode: str, ceiling: Optional[float], live: bool) -> None:
    r"""Under the relative floor without a ceiling nothing reads the layer prior covariance, so the variance handoff
    is inert and two steps are bitwise identical; the prior floor and a binding ceiling read it."""
    torch.set_num_threads(min(2, torch.get_num_threads()))
    values = _handoff_values(readout, predictive_covariance_floor_mode=floor_mode,
                             predictive_covariance_ceiling=ceiling)
    frozen, handed_off = VFE3Config(**values), VFE3Config(**{**values, "prior_handoff_sigma": 0.5})
    assert ("prior_handoff_sigma" in frozen.inactive_fields()) is (not live)
    assert ("prior_handoff_sigma" in handed_off.inactive_fields()) is (not live)
    assert _identical(_two_steps(frozen), _two_steps(handed_off)) is (not live)


# ---------------------------------------------------------------------------- items 25, 43 and 57: the switch

def _switch_base(**overrides: Any) -> Dict[str, Any]:
    return config_values(attention_mode="projected_predictive", predictive_readout="kernel",
                         predictive_covariance="moment_matched", lambda_alpha_mode="state_dependent_per_coord",
                         **overrides)


@pytest.mark.parametrize("decode_covariance", ["belief", "encoded", "constant"])
@pytest.mark.parametrize("floor_mode", ["prior", "relative"])
def test_the_switch_keeps_the_floor_mode_and_the_decode_covariance(decode_covariance: str, floor_mode: str) -> None:
    r"""Both floors and every decode covariance are covariant under :math:`U_i \to g_iU_i` (the passive-gauge
    harness measures it), so the switch overrides neither and still certifies the result."""
    raw = _switch_base(family="gaussian_full", decode_mode="full_chunked", decode_covariance=decode_covariance,
                       predictive_covariance_floor_mode=floor_mode)
    overrides = fully_gauge_covariant_overrides(raw)
    assert "predictive_covariance_floor_mode" not in overrides and "decode_covariance" not in overrides
    assert gauge_covariance_failures(VFE3Config(**{**raw, **overrides})) == []


@pytest.mark.parametrize("cap", [True, False])
def test_the_switch_warning_names_the_trace_cap_state(cap: bool) -> None:
    raw = _switch_base(family="gaussian_full", decode_mode="full_chunked", block_norm_covariance="pushforward",
                       block_norm_trace_cap=cap)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fully_gauge_covariant_overrides(raw)
    text = " ".join(str(w.message) for w in caught if "pushforward" in str(w.message))
    assert ("uncapped" in text) is (not cap) and ("capped" in text)


# ------------------------------------------------------------------------------- item 51: trace-cap era pin

def _payload(**overrides: Any) -> Dict[str, Any]:
    return serialize_config(test_config(block_norm_covariance="pushforward", norm_type_block="layernorm",
                                        **overrides))


def test_a_record_from_before_the_trace_cap_reloads_uncapped() -> None:
    old = {k: v for k, v in _payload().items() if k not in ("block_norm_trace_cap", "phi_frame_condition_max")}
    assert historical_default_pins(old)["block_norm_trace_cap"] is False
    assert config_from_serialized(old, source="pre-5f5d484 record").block_norm_trace_cap is False


def test_a_record_from_the_cap_era_keeps_the_default() -> None:
    capped_era = {k: v for k, v in _payload().items() if k != "block_norm_trace_cap"}
    assert "block_norm_trace_cap" not in historical_default_pins(capped_era)
    assert config_from_serialized(capped_era, source="5f5d484..3aeede1 record").block_norm_trace_cap is True
    for value in (True, False):                                   # a record carrying the toggle keeps its value
        payload = {k: v for k, v in _payload(block_norm_trace_cap=value).items() if k != "phi_frame_condition_max"}
        assert config_from_serialized(payload, source="record").block_norm_trace_cap is value


# ------------------------------------------------------------------------------- item 53: ablation divergence

def test_ablation_reports_baseline_defaults_the_launcher_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    import ablation

    default = VFE3Config.__dataclass_fields__["kl_max"].default
    monkeypatch.setattr(ablation, "BASELINE_CONFIG", {"embed_dim": 8})
    launcher = {"embed_dim": 16, "kl_max": 128.0, "eps": VFE3Config.__dataclass_fields__["eps"].default}
    before = (dict(ablation.BASELINE_CONFIG), dict(launcher))
    assert ablation.baseline_default_divergence(launcher) == {"kl_max": (default, 128.0)}
    assert (dict(ablation.BASELINE_CONFIG), dict(launcher)) == before    # neither dictionary is changed


def test_ablation_divergence_against_the_real_launcher_is_well_formed() -> None:
    import ablation
    import train_vfe3

    divergence = ablation.baseline_default_divergence()
    for name, (default, value) in divergence.items():
        assert name not in ablation.BASELINE_CONFIG and train_vfe3.config[name] == value != default


# ------------------------------------------------------------------------------- item 54: train() keywords

def test_train_keywords_resolve_from_the_config(monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.train as T
    from vfe3.model.model import VFEModel

    cfg = test_config(max_steps=2, warmup_steps=1, grad_clip=0.5, log_interval=0, eval_interval=0)
    for name in ("n_steps", "grad_clip", "log_interval", "eval_interval"):
        assert inspect.signature(T.train).parameters[name].default is None
    seen: List[Any] = []
    original = T.train_step

    def spy(*args: Any, **kwargs: Any) -> float:
        seen.append(kwargs.get("grad_clip"))
        return original(*args, **kwargs)

    monkeypatch.setattr(T, "train_step", spy)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    gen = torch.Generator().manual_seed(1)
    shape = (cfg.batch_size, cfg.max_seq_len)
    batches = [(torch.randint(0, cfg.vocab_size, shape, generator=gen),
                torch.randint(0, cfg.vocab_size, shape, generator=gen))]
    losses = T.train(model, batches, cfg, generate_samples=False)
    assert len(losses) == cfg.max_steps and seen == [cfg.grad_clip] * cfg.max_steps


# ------------------------------------------------------------------------------- item 55: PriorBank keywords

_PRIOR_BANK_CONFIG_KEYWORDS = ("mu_init_std", "sigma_init", "phi_scale", "decode_tau", "eps", "family", "encode_mode",
                               "decode_mode", "decode_chunk_size", "decode_av_precision", "decode_ce_checkpoint",
                               "decode_ce_triton", "decode_ranking_fp64_escalation", "unigram_kappa",
                               "decode_unigram_prior", "untie_decode_bank", "decode_coordinates",
                               "prior_covariance_table")


def test_prior_bank_config_keywords_are_keyword_only_and_required() -> None:
    from vfe3.model.prior_bank import PriorBank

    parameters = inspect.signature(PriorBank.__init__).parameters
    config_fields = {f.name for f in dataclasses.fields(VFE3Config)}
    for name in _PRIOR_BANK_CONFIG_KEYWORDS:
        assert name in config_fields
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        PriorBank(7, 4, 3)                                         # type: ignore[call-arg]


if __name__ == "__main__":
    _SNAPSHOT.write_text(json.dumps(field_defaults(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {_SNAPSHOT}")
