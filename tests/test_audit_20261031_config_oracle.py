r"""Audit 2026-10-31, remediation package config-oracle: the inactivity oracle, the route switches and validation.

Every configuration comes from ``tests.profiles`` (never a launcher's click-to-run values), every model is tiny
(single-digit widths, layers, heads and vocabulary) and runs on the CPU. Pinned:

* A009 the principled exact route requires the belief covariance at the decoder and no covariance ceiling;
* A022, A314 ``validate_sweeps`` refuses arms that resolve to one effective configuration (a coerced, inert or
  coinciding value), and ``inactive_fields`` marks the coerced frame fields and the irrep draw fields of an explicit
  spec, which the joint search's inert gate then refuses too (scaling-harness-2);
* A023 the gauge switch unties every prior-bank decoder, and a tied one is not certified covariant;
* A036 the carried regime knobs are inert at depth one;
* A063 no float field takes a bool;
* A065 validation no longer refuses settings the oracle calls inert;
* A077, A082, A083, A360 the oracle marks agree with execution;
* A084 ``__post_init__`` refuses the per-type hidden width and a tower with no Clebsch-Gordan path, so the
  paper-arms capacity arm on a shallow base resolves invalid (paper-arms-harness-6);
* A188 ``gaussian_full`` admits the diagonal Gaussian-KL decoder, and the switch keeps it;
* A258 the block-MLP validator messages name every admitted mode;
* A259 the exact-route docstring carries no tab;
* A263 the replay ladder skips a rung with the same CUDA matmul flag;
* A333 ``n_e_steps`` admits no zero-depth control;
* A352 the retired-field migrations read the frozen introduction values.
"""
from __future__ import annotations

import inspect
import math
import warnings
from dataclasses import fields
from types import ModuleType
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest
import torch

import vfe3.config as config_module
from tests.profiles import config_values, test_config
from vfe3.config import VFE3Config, config_from_serialized, serialize_config
from vfe3.exact_route import principled_exact_route_failures, principled_exact_route_overrides
from vfe3.gauge_covariant import fully_gauge_covariant_overrides, gauge_covariance_failures

torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _predictive(**overrides: Any) -> Dict[str, Any]:
    r"""The test profile on the predictive kernel route with a prior-bank decoder and passthrough covariances."""
    values = dict(attention_mode="projected_predictive", predictive_readout="kernel",
                  predictive_covariance="passthrough", attention_residual=True, content_link_chart="exponential",
                  decode_mode="diagonal_chunked",
                  embed_dim=8, n_heads=2, max_seq_len=4, n_layers=2, block_mlp_expansion=2, vocab_size=7,
                  fp32_matmul_precision=None, beta_attention_prior="causal", decode_ranking_fp64_escalation=False,
                  decode_av_precision="fp32", min_lr_frac=0.0)
    values.update(overrides)
    return config_values(**values)


def _cfg(values: Dict[str, Any]) -> VFE3Config:
    return VFE3Config(**values)


def _model(cfg: VFE3Config, seed: int = 0) -> torch.nn.Module:
    from vfe3.model.model import VFEModel
    torch.manual_seed(seed)
    return VFEModel(cfg)


def _two_steps(cfg: VFE3Config) -> Tuple[List[float], Dict[str, torch.Tensor]]:
    r"""Losses and final state of two optimizer steps of :func:`vfe3.train.train` from a fixed initialization."""
    from vfe3.train import train

    model = _model(cfg)
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


# ----------------------------------------------------------------------------- A009: the exact route's decoder

def _exact(values: Dict[str, Any]) -> VFE3Config:
    return VFE3Config(**{**values, **principled_exact_route_overrides(values)})


@pytest.mark.parametrize("decode_covariance", ["encoded", "constant"])
def test_exact_route_requires_the_belief_covariance_at_the_decoder(decode_covariance: str) -> None:
    values = config_values()
    on_route = _exact(values)
    assert principled_exact_route_failures(on_route) == []
    off = VFE3Config(**{**values, **principled_exact_route_overrides(values), "decode_covariance": decode_covariance})
    assert any(f.startswith("decode_covariance=") for f in principled_exact_route_failures(off))
    # The switch puts a launcher that decodes another covariance back on the route.
    assert principled_exact_route_overrides({**values, "decode_covariance": decode_covariance})[
        "decode_covariance"] == "belief"


def test_exact_route_refuses_the_covariance_ceiling() -> None:
    values = config_values()
    clamped = VFE3Config(**{**values, **principled_exact_route_overrides(values), "predictive_covariance_ceiling": 2.0})
    assert any(f.startswith("predictive_covariance_ceiling=") for f in principled_exact_route_failures(clamped))
    overrides = principled_exact_route_overrides({**values, "predictive_covariance_ceiling": 2.0})
    assert overrides["predictive_covariance_ceiling"] is None
    assert principled_exact_route_failures(VFE3Config(**{**values, "predictive_covariance_ceiling": 2.0,
                                                         **overrides})) == []


# ------------------------------------------------------- A022, A314: arms that resolve to one configuration

@pytest.fixture
def ablation_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    import ablation
    monkeypatch.setattr(ablation, "principled_exact_route", False)
    monkeypatch.setattr(ablation, "fully_gauge_covariant", False)
    return ablation


_SWEEP = "audit_20261031_config_oracle"


def _validate(ablation: Any, monkeypatch: pytest.MonkeyPatch, baseline: Dict[str, Any], sweep: Dict[str, Any]) -> None:
    monkeypatch.setattr(ablation, "BASELINE_CONFIG", baseline)
    monkeypatch.setitem(ablation.SWEEPS, _SWEEP, {"include_baseline": False, **sweep})
    ablation.validate_sweeps([_SWEEP])


@pytest.mark.parametrize("sweep", [
    {"param": "pos_phi", "values": ["none", "learned", "frozen"], "requires": {"gauge_transport": "off"}},
    {"param": "phi_scale", "values": [0.001, 0.3], "requires": {"gauge_transport": "off"}},
    {"param": "m_phi_lr", "values": [0.002, 0.05], "requires": {"gauge_transport": "frozen"}},
], ids=["pos_phi_off", "phi_scale_off", "m_phi_lr_frozen"])
def test_validate_sweeps_refuses_a_coerced_contrast(ablation_module: Any, monkeypatch: pytest.MonkeyPatch,
                                                    sweep: Dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="one effective configuration|INERT on EVERY arm"):
        _validate(ablation_module, monkeypatch, _predictive(), sweep)


def test_validate_sweeps_admits_a_live_frame_contrast(ablation_module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _validate(ablation_module, monkeypatch, _predictive(),
              {"param": "phi_scale", "values": [0.001, 0.3], "requires": {"gauge_transport": "on"}})


def test_coerced_frame_fields_are_marked_under_identity_and_frozen_frames() -> None:
    off = _cfg(_predictive(gauge_transport="off")).inactive_fields()
    assert {"phi_scale", "pos_phi", "m_phi_lr"} <= set(off)
    frozen = _cfg(_predictive(gauge_transport="frozen")).inactive_fields()
    assert "m_phi_lr" in frozen and "phi_scale" not in frozen and "pos_phi" not in frozen
    # The per-type hidden bias inherits the DECLARED m_phi_lr, so the declared rate is read there.
    biased = _cfg(_predictive(gauge_transport="frozen", per_type_hidden_bias=True, block_mlp_expansion=8,
                              n_layers=4)).inactive_fields()
    assert "m_phi_lr" not in biased
    assert "m_phi_lr" not in _cfg(_predictive()).inactive_fields()


def test_irrep_draw_fields_are_inert_under_an_explicit_spec() -> None:
    explicit = test_config(gauge_group="so_n", group_n=3, irrep_spec=[("l1", 1), ("l2", 1)], embed_dim=8)
    assert {"irrep_spec_seed", "irrep_min_rank", "irrep_max_rank"} <= set(explicit.inactive_fields())
    drawn = test_config(gauge_group="so_n", group_n=4, embed_dim=64, n_heads=8, irrep_spec="random",
                        irrep_spec_seed=1)
    assert not ({"irrep_spec_seed", "irrep_min_rank", "irrep_max_rank"} & set(drawn.inactive_fields()))
    # A reloaded drawn run carries the list, which reads the draw fields no more.
    reloaded = config_from_serialized(serialize_config(drawn), source="test")
    assert reloaded == drawn and "irrep_spec_seed" in reloaded.inactive_fields()


def test_seeds_that_draw_one_tower_share_a_fingerprint() -> None:
    configs = [test_config(gauge_group="so_n", group_n=4, embed_dim=8, n_heads=2, irrep_spec="random",
                           irrep_spec_seed=seed) for seed in range(4)]
    for a in configs:
        for b in configs:
            assert (a.route_fingerprint() == b.route_fingerprint()) is (a.irrep_spec == b.irrep_spec)


def test_belief_and_encoded_decoding_coincide_on_a_passthrough_route() -> None:
    passthrough = [_cfg(_predictive(decode_covariance=value)) for value in ("belief", "encoded")]
    assert passthrough[0].route_fingerprint() == passthrough[1].route_fingerprint()
    assert "decode_covariance" not in passthrough[0].inactive_fields()          # 'constant' still differs
    assert (_cfg(_predictive(decode_covariance="constant")).route_fingerprint()
            != passthrough[0].route_fingerprint())
    matched = [_cfg(_predictive(decode_covariance=value, predictive_covariance="moment_matched"))
               for value in ("belief", "encoded")]
    assert matched[0].route_fingerprint() != matched[1].route_fingerprint()
    # Executed: the logits of the two readings are bitwise equal on the passthrough route.
    tokens = torch.randint(0, 7, (2, 4), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        logits = [_model(cfg).eval()(tokens) for cfg in passthrough]
    assert torch.equal(logits[0], logits[1])


def test_validate_sweeps_refuses_coinciding_decode_covariances(ablation_module: Any,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    sweep = {"configs": [{"label": "belief", "decode_covariance": "belief"},
                         {"label": "encoded", "decode_covariance": "encoded"}]}
    with pytest.raises(ValueError, match="one effective configuration"):
        _validate(ablation_module, monkeypatch, _predictive(), sweep)
    _validate(ablation_module, monkeypatch, _predictive(predictive_covariance="moment_matched"), sweep)


def test_the_search_gate_refuses_frame_parameters_that_gauge_transport_overwrites() -> None:
    r"""scaling-harness-2 (duplicate of A022): the joint search refuses an enabled parameter that ``inactive_fields``
    marks on its baseline, so the coerced frame rate and scale are refused under frozen and identity frames."""
    import parameter_optimization as po

    spec = {"bounds": [0.001, 0.1], "scale": "log"}
    with pytest.raises(ValueError, match="m_phi_lr"):
        po._refuse_inert_parameters([("m_phi_lr", spec)], _predictive(gauge_transport="frozen"), {})
    with pytest.raises(ValueError, match="phi_scale"):
        po._refuse_inert_parameters([("phi_scale", spec)], _predictive(gauge_transport="off"), {})
    po._refuse_inert_parameters([("m_phi_lr", spec), ("phi_scale", spec)], _predictive(), {})


# --------------------------------------------------------------------- A023: the tied bank under the switch

def _switch_values(**overrides: Any) -> Dict[str, Any]:
    return config_values(attention_mode="projected_predictive", predictive_readout="kernel",
                         predictive_covariance="moment_matched", **overrides)


@pytest.mark.parametrize("family, decode_mode", [("gaussian_full", "full_chunked"),
                                                 ("gaussian_block", "diagonal_chunked"),
                                                 ("gaussian_diagonal", "diagonal_chunked")])
def test_the_switch_unties_every_prior_bank_decoder(family: str, decode_mode: str) -> None:
    values = _switch_values(family=family, decode_mode=decode_mode, untie_decode_bank=False)
    overrides = fully_gauge_covariant_overrides(values)
    assert overrides.get("untie_decode_bank") is True
    switched = VFE3Config(**{**values, **overrides})
    assert gauge_covariance_failures(switched) == []
    tied = VFE3Config(**{**values, **overrides, "untie_decode_bank": False})
    assert "decoder_gauge_compatible" in gauge_covariance_failures(tied)


def test_a_tied_bank_is_not_gauge_compatible_but_a_linear_decoder_is() -> None:
    from vfe3.run_artifacts import _pure_path_report

    def flag(**overrides: Any) -> Any:
        return _pure_path_report(_cfg(_switch_values(decode_coordinates="reference", **overrides)),
                                 [])["gauge_flags"]["decoder_gauge_compatible"]

    assert flag(family="gaussian_full", decode_mode="full_chunked", untie_decode_bank=False) is False
    assert flag(family="gaussian_full", decode_mode="full_chunked", untie_decode_bank=True) is True
    assert flag(decode_mode="linear", untie_decode_bank=False) is True
    assert "untie_decode_bank" not in fully_gauge_covariant_overrides(_switch_values(decode_mode="linear"))


# ------------------------------------------------------------------- A036: the regime knobs at depth one

_REGIME = ("block_memory_regime_transition", "block_memory_regime_beta", "block_memory_regime_stay")


def _exact_values(**overrides: Any) -> Dict[str, Any]:
    values = config_values(embed_dim=4, n_heads=2, max_seq_len=4, block_mlp_expansion=2,
                           block_memory_regression_rank=1, fp32_matmul_precision=None, **overrides)
    return {**values, **principled_exact_route_overrides(values)}


def test_the_regime_knobs_are_inert_at_depth_one() -> None:
    assert set(_REGIME) <= set(VFE3Config(**_exact_values(n_layers=1)).inactive_fields())
    assert not (set(_REGIME) & set(VFE3Config(**_exact_values(n_layers=2)).inactive_fields()))
    # The tied-bank requirement concerns two settings a single layer leaves inert, so it applies from depth two.
    untied = VFE3Config(**{**_exact_values(n_layers=1), "block_mlp_tie_layers": False})
    assert {"block_mlp_tie_layers", "block_memory_regime_transition"} <= set(untied.inactive_fields())
    with pytest.raises(ValueError, match="tied banks"):
        VFE3Config(**{**_exact_values(n_layers=2), "block_mlp_tie_layers": False})


def _loss_and_grads(cfg: VFE3Config) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
    model = _model(cfg).train()
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len), generator=torch.Generator().manual_seed(9))
    _, loss, _ = model(tokens, tokens)
    loss.backward()
    return loss.detach(), {name: (None if p.grad is None else p.grad.clone()) for name, p in model.named_parameters()}


def _same_loss_and_grads(a: Tuple[torch.Tensor, Dict[str, Any]], b: Tuple[torch.Tensor, Dict[str, Any]]) -> bool:
    return torch.equal(a[0], b[0]) and a[1].keys() == b[1].keys() and all(
        (a[1][k] is None) == (b[1][k] is None) and (a[1][k] is None or torch.equal(a[1][k], b[1][k])) for k in a[1])


def test_regime_trust_changes_nothing_at_depth_one_and_something_at_depth_two() -> None:
    one = [_loss_and_grads(VFE3Config(**_exact_values(n_layers=1, block_memory_regime_beta=beta)))
           for beta in (0.0, 2.0)]
    assert _same_loss_and_grads(*one)
    assert one[0][1]["block_mlps.0.regime_stay_logit"] is None
    two = [_loss_and_grads(VFE3Config(**_exact_values(n_layers=2, block_memory_regime_beta=beta)))
           for beta in (0.0, 2.0)]
    assert not _same_loss_and_grads(*two)


def test_validate_sweeps_refuses_a_depth_one_regime_sweep(ablation_module: Any,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    sweep = {"param": "block_memory_regime_beta", "values": [0.0, 1.0, 2.0], "exact_route": True}
    with pytest.raises(ValueError, match="INERT on EVERY arm"):
        _validate(ablation_module, monkeypatch, _exact_values(n_layers=1), sweep)
    _validate(ablation_module, monkeypatch, _exact_values(n_layers=2), sweep)


# ------------------------------------------------------------------------------- A063: no float takes a bool

_FLOAT_FIELDS = [f.name for f in fields(VFE3Config)
                 if f.type in (float, Optional[float], 'float | List[float]')]


@pytest.mark.parametrize("name", _FLOAT_FIELDS)
def test_a_float_field_refuses_a_bool(name: str) -> None:
    assert len(_FLOAT_FIELDS) >= 70
    with pytest.raises((TypeError, ValueError), match=name):
        test_config(**{name: True})


@pytest.mark.parametrize("name", ["kappa_beta", "b0", "c0"])
def test_a_list_valued_float_field_refuses_a_bool_entry(name: str) -> None:
    extra = {"lambda_alpha_mode": "state_dependent_per_coord"} if name in ("b0", "c0") else {}
    length = 4 if name in ("b0", "c0") else 2
    with pytest.raises((TypeError, ValueError), match=name):
        test_config(**{name: [True] + [1.0] * (length - 1)}, **extra)


# ------------------------------------------------------------- A065: validation against the oracle's marks

def test_an_inert_frequency_scale_does_not_need_the_unigram_table() -> None:
    values = _predictive(phi_type_dropout=0.0, phi_type_dropout_freq_scale=1e-5, decode_unigram_prior=False)
    cfg = _cfg(values)
    assert "phi_type_dropout_freq_scale" in cfg.inactive_fields()
    with pytest.raises(ValueError, match="decode_unigram_prior"):
        _cfg({**values, "phi_type_dropout": 0.1})


@pytest.mark.parametrize("name", ["predictive_covariance_spread_squares", "predictive_covariance_symmetric_pairs"])
def test_an_inert_head_pair_setting_admits_edge_links(name: str) -> None:
    edges = dict(content_link_edges="lowrank", content_link_edge_rank=2)
    cfg = _cfg(_predictive(**{name: True}, **edges))
    assert name in cfg.inactive_fields()
    with pytest.raises(ValueError, match="content_link_edges"):
        _cfg(_predictive(**{name: True}, predictive_covariance="moment_matched", **edges))


# ---------------------------------------------------------------- A077: identity and frozen frames, executed

def test_identity_and_frozen_frames_mark_the_frame_knobs() -> None:
    off = _cfg(_predictive(gauge_transport="off")).inactive_fields()
    assert {"phi_type_dropout", "phi_type_dropout_freq_scale", "transport_chart_max_norm",
            "phi_mstep_max_matrix_norm"} <= set(off)
    frozen = _cfg(_predictive(gauge_transport="frozen")).inactive_fields()
    assert "phi_mstep_max_matrix_norm" in frozen
    assert not ({"phi_type_dropout", "transport_chart_max_norm"} & set(frozen))
    # A dropout draw shifts the masks of the block-MLP dropout, which shares the global generator.
    shared = _cfg(_predictive(gauge_transport="off", block_mlp_dropout=0.5)).inactive_fields()
    assert "phi_type_dropout" not in shared and "phi_type_dropout_freq_scale" in shared


@pytest.mark.parametrize("transport, name, values", [
    ("off", "phi_type_dropout", (0.0, 0.5)),
    ("off", "transport_chart_max_norm", (50.0, 1.0)),
    ("frozen", "phi_mstep_max_matrix_norm", (None, 0.05)),
])
def test_marked_frame_knobs_leave_two_steps_bitwise_unchanged(transport: str, name: str, values: tuple) -> None:
    runs = [_two_steps(_cfg(_predictive(gauge_transport=transport, phi_scale=0.5, **{name: value})))
            for value in values]
    assert _identical(*runs)


def test_frame_dropout_under_identity_frames_moves_a_dropout_run() -> None:
    runs = [_two_steps(_cfg(_predictive(gauge_transport="off", block_mlp_dropout=0.5, phi_type_dropout=value)))
            for value in (0.0, 0.5)]
    assert not _identical(*runs)


# ------------------------------------------------------------- A082: handoff, self clamp and the router clamp

_SO3_CG = dict(gauge_group="so_n", group_n=3, irrep_spec=[("l1", 1), ("l2", 1)], embed_dim=8, n_heads=2,
               use_cg_coupling=True, use_block_mlp=False, norm_type_block="none")


def test_the_cg_energy_reads_no_prior_mean() -> None:
    values = _predictive(**_SO3_CG, cg_energy_weight=0.5)
    cfg = _cfg(values)
    assert "prior_handoff_rho" in cfg.inactive_fields()
    assert _identical(_two_steps(cfg), _two_steps(_cfg({**values, "prior_handoff_rho": 0.3})))


def test_the_canonical_diagnostic_reads_self_kl_max() -> None:
    assert "self_kl_max" in _cfg(_predictive()).inactive_fields()
    assert "self_kl_max" not in _cfg(_predictive(diagnose_canonical_free_energy=True)).inactive_fields()


def test_kl_max_is_inert_on_the_minimal_agent_when_every_reader_takes_self_kl_max() -> None:
    agent = dict(predictive_readout="minimal_agent", minimal_agent_inference="exact", mstep_self_coupling_weight=0.5)
    assert "kl_max" in _cfg(_predictive(**agent, self_kl_max=10.0)).inactive_fields()
    assert "kl_max" in _cfg(_predictive(**agent, self_kl_max=10.0,
                                        diagnose_canonical_free_energy=True)).inactive_fields()
    assert "kl_max" not in _cfg(_predictive(**agent)).inactive_fields()


# ------------------------------------------------------- A083: tolerances and identity-frame coordinates

def test_pair_kl_gemm_tol_is_live_on_the_gradient_receiver() -> None:
    assert "pair_kl_gemm_tol" not in test_config().inactive_fields()                 # projected_gradient
    assert "pair_kl_gemm_tol" in _cfg(_predictive(predictive_readout="minimal_agent")).inactive_fields()
    assert "pair_kl_gemm_tol" in _cfg(_predictive(relational_routing="mean_distance")).inactive_fields()
    assert "pair_kl_gemm_tol" not in _cfg(_predictive()).inactive_fields()


def test_the_frame_pair_tolerance_is_marked_on_identity_frames_only_above_d_eps32() -> None:
    from vfe3.geometry import transport

    d_eps = 4 * 2.0 ** -23                                                  # d = 4 (K = 8, two heads)
    for tol, marked in ((None, True), (1e-3, True), (1.05 * d_eps, True), (0.9 * d_eps, False), (1e-8, False)):
        cfg = _cfg(_predictive(gauge_transport="off", frame_pair_escalation_tol=tol))
        assert ("frame_pair_escalation_tol" in cfg.inactive_fields()) is marked, tol
    previous = (transport.frame_pair_fp64_escalation(), transport._FRAME_PAIR_CONDITION_TOL)
    tokens = torch.randint(0, 7, (2, 4), generator=torch.Generator().manual_seed(5))
    try:
        for tol, escalates in ((0.9 * d_eps, True), (1.05 * d_eps, False)):
            model = _model(_cfg(_predictive(gauge_transport="off", frame_pair_escalation_tol=tol))).eval()
            transport.reset_frame_pair_fp64_escalations()
            with torch.no_grad():
                model(tokens)
            assert (transport.frame_pair_fp64_escalations() > 0) is escalates, tol
    finally:
        transport.set_frame_pair_fp64_escalation(previous[0])
        transport.set_frame_pair_condition_tol(previous[1])
        transport.reset_frame_pair_fp64_escalations()


def test_the_frame_pair_tolerance_bound_on_unequal_heads_is_never_tripped_when_marked() -> None:
    r"""Unequal heads (so(3) blocks of 3 and 5): the mark takes the bound K = 8, which no stored block exceeds, so a
    marked tolerance escalates no call, and a tolerance below the 5 x 5 block's d eps_32 is not marked."""
    from vfe3.geometry import transport

    eps = 2.0 ** -23
    tower = dict(gauge_group="so_n", group_n=3, irrep_spec=[("l1", 1), ("l2", 1)], gauge_transport="off")
    previous = (transport.frame_pair_fp64_escalation(), transport._FRAME_PAIR_CONDITION_TOL)
    tokens = torch.randint(0, 7, (2, 4), generator=torch.Generator().manual_seed(5))
    try:
        for tol, marked in ((0.9 * 5 * eps, False), (1.05 * 8 * eps, True)):
            cfg = _cfg(_predictive(**tower, frame_pair_escalation_tol=tol))
            assert ("frame_pair_escalation_tol" in cfg.inactive_fields()) is marked, tol
            model = _model(cfg).eval()
            transport.reset_frame_pair_fp64_escalations()
            with torch.no_grad():
                model(tokens)
            assert (transport.frame_pair_fp64_escalations() > 0) is not marked, tol
    finally:
        transport.set_frame_pair_fp64_escalation(previous[0])
        transport.set_frame_pair_condition_tol(previous[1])
        transport.reset_frame_pair_fp64_escalations()


def test_the_identity_frame_coordinate_reason_states_roundoff() -> None:
    reason = _cfg(_predictive(gauge_transport="off", block_coordinates="reference")).inactive_fields()[
        "block_coordinates"]
    assert "roundoff" in reason


# -------------------------------------------- A084: construction refusals that __post_init__ now raises

def test_the_per_type_hidden_width_is_refused_at_configuration() -> None:
    with pytest.raises(ValueError, match="per_type_hidden_bias"):
        test_config(embed_dim=64, n_heads=2, n_layers=1, block_mlp_expansion=8, per_type_hidden_bias=True)
    test_config(embed_dim=64, n_heads=2, n_layers=1, block_mlp_expansion=32, per_type_hidden_bias=True)
    with pytest.raises(ValueError, match="per_type_hidden_bias"):
        test_config(embed_dim=8, n_heads=2, n_layers=1, block_mlp_expansion=2, per_type_hidden_bias=True)
    _model(test_config(embed_dim=8, n_heads=2, n_layers=2, block_mlp_expansion=2, per_type_hidden_bias=True,
                       vocab_size=7, max_seq_len=3))


@pytest.mark.parametrize("group", [
    dict(),
    dict(gauge_tying="shared"),
    dict(gauge_group="block_so"),
    dict(gauge_group="block_co"),
    dict(gauge_group="block_glmc"),
    dict(gauge_group="block_sl"),
    dict(gauge_group="block_borel"),
    dict(gauge_group="block_unipotent"),
    dict(gauge_group="so_n", group_n=3, irrep_spec=[("l1", 1), ("l2", 1)]),
    dict(gauge_group="so_n", group_n=3, irrep_spec=[("l1", 1), ("l0", 1), ("l1", 1), ("l0", 1), ("l0", 1)],
         gauge_tying="per_head", n_heads=5, embed_dim=9),
    dict(gauge_group="su_n", group_n=3, irrep_spec=[("pq1_0", 2)], embed_dim=12),
    dict(gauge_group="su_n", group_n=3, irrep_spec=[("pq1_0", 2)], embed_dim=12, gauge_tying="per_head"),
], ids=lambda g: "-".join(f"{v}" for v in g.values() if isinstance(v, str)) or "block_glk")
def test_the_generator_count_matches_the_built_group(group: Dict[str, Any]) -> None:
    from vfe3.model.model import build_group

    cfg = test_config(**{"embed_dim": 8, "n_heads": 2, **group})
    assert cfg._generator_count() == build_group(cfg).generator_shape[0]


def test_the_capacity_arm_on_a_shallow_base_resolves_invalid_with_parameter_counts() -> None:
    r"""paper-arms-harness-6 (duplicate of A084): the per-type width now fails the arm's configuration, which
    ``resolve_arm`` reports as an invalid row, instead of model construction during parameter counting."""
    from pathlib import Path

    import paper_arms

    base_cfg = test_config(attention_mode="projected_predictive", embed_dim=8, n_heads=2, n_layers=1,
                           block_mlp_expansion=2, use_block_mlp=True, block_mlp_mode="coordinate", vocab_size=7,
                           max_seq_len=4)
    base = paper_arms.Base(name="tiny", run_dir=Path("."), cfg=base_cfg, dataset="wikitext-103")
    label = paper_arms.DECISION_CAPACITY
    plan = paper_arms.resolve_arm(base, label, paper_arms.ARMS[label], budget="base_protocol", seeds=[(0, 0)],
                                  counts=True)
    assert plan.status == "invalid" and "per_type_hidden_bias" in plan.reason


def test_a_tower_without_a_clebsch_gordan_path_is_refused_at_configuration() -> None:
    with pytest.raises(ValueError, match="Clebsch-Gordan"):
        test_config(gauge_group="so_n", group_n=4, irrep_spec=[("l1", 2)], embed_dim=8, n_heads=2,
                    use_cg_coupling=True)
    _model(test_config(**{**_SO3_CG, "vocab_size": 7, "max_seq_len": 3}))


# --------------------------------------------------------------- A188: the diagonal decoder on gaussian_full

def test_gaussian_full_admits_the_diagonal_decoder_and_the_switch_keeps_it() -> None:
    _cfg(_predictive(family="gaussian_full", decode_mode="diagonal_chunked"))
    values = _switch_values(decode_mode="diagonal_chunked", lambda_alpha_mode="state_dependent_per_coord")
    overrides = fully_gauge_covariant_overrides(values)
    assert overrides["family"] == "gaussian_full" and "decode_mode" not in overrides
    assert gauge_covariance_failures(VFE3Config(**{**values, **overrides})) == []


@pytest.mark.parametrize("coordinates", ["local", "reference"])
def test_the_diagonal_decoder_on_gaussian_full_has_the_full_decoders_cross_entropy(coordinates: str) -> None:
    values = _predictive(family="gaussian_full", predictive_covariance="moment_matched", decode_mode="full_chunked",
                         decode_coordinates=coordinates, phi_scale=0.3)
    full = _model(_cfg(values)).double().eval()
    diagonal = _model(_cfg({**values, "decode_mode": "diagonal_chunked"})).double().eval()
    diagonal.load_state_dict(full.state_dict(), strict=True)
    tokens = torch.randint(0, 7, (2, 4), generator=torch.Generator().manual_seed(4))
    targets = torch.randint(0, 7, (2, 4), generator=torch.Generator().manual_seed(6))
    with torch.no_grad():
        lp_full, lp_diag = (torch.log_softmax(m(tokens), dim=-1) for m in (full, diagonal))
        loss_full, loss_diag = full(tokens, targets)[1], diagonal(tokens, targets)[1]
    torch.testing.assert_close(lp_diag, lp_full, rtol=0.0, atol=1e-10)
    assert math.isclose(float(loss_diag), float(loss_full), rel_tol=1e-11)


# ----------------------------------------------------------------- A258: the validator messages name each mode

def test_the_block_mlp_messages_name_every_admitted_mode() -> None:
    with pytest.raises(ValueError) as frame_diagonal:
        test_config(attention_mode="projected_predictive", family="gaussian_frame_diagonal",
                    block_mlp_mode="gauge_gate", norm_type_block="none")
    message = str(frame_diagonal.value)
    assert "'coordinate'" in message and "'cue_result_memory'" in message
    with pytest.raises(ValueError) as reference:
        test_config(attention_mode="projected_predictive", block_coordinates="reference",
                    block_mlp_mode="gauge_gate", norm_type_block="none")
    message = str(reference.value)
    for mode in ("coordinate", "gaussian_template", "prototype_agents", "cue_result_memory"):
        assert f"'{mode}'" in message, mode


# ------------------------------------------------------------------------- A259: the exact-route docstring

def test_the_exact_route_docstring_has_no_tab_and_names_the_chart_optimizer() -> None:
    import vfe3.exact_route as exact_route

    assert "\t" not in exact_route.__doc__
    assert "chart" in exact_route.__doc__ and "AdamW" in exact_route.__doc__


# ------------------------------------------------------------------- A263: the replay ladder on the device

def test_the_replay_ladder_skips_a_rung_with_the_same_cuda_flag() -> None:
    from vfe3 import train as T

    assert T._replay_policies("medium", torch.device("cpu")) == ("high", "highest")
    assert T._replay_policies("high", torch.device("cpu")) == ("highest",)
    assert T._replay_policies("medium", None) == ("high", "highest")
    try:
        flags = {policy: T._cuda_matmul_flag(policy) for policy in ("medium", "high", "highest")}
    except AttributeError:
        pytest.skip("this torch build has no per-backend fp32 matmul flag")
    cuda = T._replay_policies("medium", torch.device("cuda"))
    sequence = ("medium", *cuda)
    assert len({flags[policy] for policy in sequence}) == len(sequence)
    assert cuda[-1] == "highest"
    if flags["medium"] == flags["high"]:
        assert cuda == ("highest",)
    assert T._replay_policies("high", torch.device("cuda")) == ("highest",)


# ----------------------------------------------------------------------- A333: no zero-depth control

def test_n_e_steps_has_no_zero_depth_control() -> None:
    for route in ("projected_gradient", "projected_predictive"):
        for steps in (0, -1):
            with pytest.raises(ValueError, match="n_e_steps"):
                test_config(attention_mode=route, n_e_steps=steps)
    # The field comment states the domain the validation enforces, with no zero-depth control.
    field_line = next(line for line in inspect.getsource(config_module.VFE3Config).splitlines()
                      if line.strip().startswith("n_e_steps:"))
    assert "zero-depth" not in field_line and "1 on projected_predictive" in field_line


# ------------------------------------------------------- A352: migrations read the frozen introduction values

def _flip_defaults(monkeypatch: pytest.MonkeyPatch, flips: Dict[str, Any]) -> None:
    constructor = VFE3Config.__init__
    names = list(inspect.signature(constructor).parameters)[1:]
    defaults = list(constructor.__defaults__)
    for name, value in flips.items():
        defaults[names.index(name)] = value
        monkeypatch.setattr(VFE3Config, name, value)
        monkeypatch.setattr(VFE3Config.__dataclass_fields__[name], "default", value)
    monkeypatch.setattr(constructor, "__defaults__", tuple(defaults))


def test_a_migration_reads_the_introduction_value_of_a_field_the_record_predates(
        monkeypatch: pytest.MonkeyPatch) -> None:
    payload = serialize_config(test_config(attention_mode="projected_predictive"))
    del payload["predictive_readout"]
    payload["relational_output_init"] = "depth_scaled"
    with pytest.raises(ValueError, match="relational_output_init"):
        config_from_serialized(dict(payload), source="record without predictive_readout")
    _flip_defaults(monkeypatch, {"predictive_readout": "minimal_agent"})
    with pytest.raises(ValueError, match="relational_output_init"):
        config_from_serialized(dict(payload), source="record without predictive_readout")


# ---------------------------------------------------------------- A360: the family on the reference stack

def _reference_stack(**overrides: Any) -> Dict[str, Any]:
    return _predictive(**{"family": "gaussian_block", "block_coordinates": "reference",
                          "decode_coordinates": "reference", "phi_scale": 0.3, **overrides})


def test_the_family_of_a_passthrough_reference_stack_is_marked_inert() -> None:
    assert "family" in _cfg(_reference_stack()).inactive_fields()
    assert "family" in _cfg(_reference_stack(decode_mode="linear")).inactive_fields()
    assert "family" not in _cfg(_reference_stack(decode_mode="full_chunked")).inactive_fields()
    assert "family" not in _cfg(_reference_stack(predictive_covariance="moment_matched")).inactive_fields()
    assert "roundoff" in _cfg(_reference_stack()).inactive_fields()["family"]


def test_the_reference_stack_families_agree_to_roundoff() -> None:
    block = _model(_cfg(_reference_stack())).double().eval()
    diagonal = _model(_cfg(_reference_stack(family="gaussian_diagonal"))).double().eval()
    diagonal.load_state_dict(block.state_dict(), strict=True)
    tokens = torch.randint(0, 7, (2, 4), generator=torch.Generator().manual_seed(8))
    with torch.no_grad():
        torch.testing.assert_close(block(tokens), diagonal(tokens), rtol=0.0, atol=1e-10)
