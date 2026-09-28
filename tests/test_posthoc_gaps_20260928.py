r"""Post-hoc gap analyses: residual ratios (v2:899), routing-covariance gap and hidden temperature
(v2:872), word-normalized perplexity (``tab:placeholder_calibration``), and their wiring into
:func:`vfe3.posthoc.report.run_posthoc` (including the lazily imported M4 module).

Tiny CPU models only (vocabulary 24, K=4, two heads, two layers, 8-token windows); the end-to-end tests
reuse the fake run directory of ``tests/test_posthoc_20260928.py``.
"""

from __future__ import annotations

import json
import math
import sys
import types
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from tests.test_posthoc_20260928 import _cfg, _fake_run, _model, _stream
from vfe3.posthoc import calibration as cal
from vfe3.posthoc import covariance as cv
from vfe3.posthoc import evaluate as ev
from vfe3.posthoc import residual as rs
from vfe3.posthoc import streams as st

DEVICE = torch.device("cpu")


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


# ----------------------------------------------------------------------------------------------
# Residual ratios (v2:899)
# ----------------------------------------------------------------------------------------------

def test_residual_observer_equals_the_training_probe_on_the_same_windows() -> None:
    from vfe3.train import _residual_and_attention_rows

    cfg = _cfg(attention_residual=True, norm_type_block="layernorm", eval_stride=8)
    model = _model(cfg)
    tokens = _stream(4 * 8 + 1, 21)                                  # four windows, every position scored
    dataset = st.eval_windows(tokens, 8, 8)
    indices = np.arange(len(dataset))
    observer = rs.ResidualObserver(model)
    result = ev.nll_pass(model, dataset, indices, batch_size=2, device=DEVICE, observers=[observer])
    assert result.n == 32
    windows = torch.stack([dataset[int(i)][0] for i in indices])     # (4, 8) the same input windows
    probe, _ = _residual_and_attention_rows(model, windows, 0, DEVICE)
    got = observer.result()
    assert got["status"] == "ok" and got["ln_change"]["defined"] and got["residual_additive"]
    assert len(got["layers"]) == len(probe) == cfg.n_layers
    for layer, (mine, theirs) in enumerate(zip(got["layers"], probe)):
        assert mine["rho_n"] == 32 and mine["ln_change_n"] == 32
        for key in ("q10", "q50", "q90"):
            assert math.isclose(mine[f"rho_{key}"], theirs[key], rel_tol=1e-5), key
        # The probe's LN median is torch.median, the LOWER middle value for an even count; the per-token
        # values are the same, and the report's q50 interpolates like the rho quantiles.
        changes = np.sort(observer.layer_values(layer)[1].astype(np.float64))
        assert math.isclose(changes[(changes.size - 1) // 2], theirs["ln_rel_change_q50"], rel_tol=1e-5)
        assert math.isclose(mine["ln_change_q50"], float(np.quantile(changes, 0.5)), rel_tol=1e-12)
        assert mine["rho_min"] <= mine["rho_q10"] <= mine["rho_q50"] <= mine["rho_q90"] <= mine["rho_max"]
    assert got["rho_min_all_layers"] == min(r["rho_min"] for r in got["layers"])
    assert got["ln_change_max_all_layers"] == max(r["ln_change_max"] for r in got["layers"])


def test_residual_observer_counts_only_scored_targets_and_records_undefined_ln_change() -> None:
    cfg = _cfg(attention_residual=True)                             # no block norm
    model = _model(cfg)
    dataset = st.eval_windows(_stream(61, 22), 8, 4)
    observer = rs.ResidualObserver(model)
    result = ev.nll_pass(model, dataset, np.arange(len(dataset)), batch_size=2, device=DEVICE,
                         observers=[observer])
    got = observer.result()
    assert all(r["rho_n"] == result.n == 60 for r in got["layers"])  # scored targets, not all window rows
    assert not got["ln_change"]["defined"] and "normalization" in got["ln_change"]["reason"]
    assert all(r["ln_change_n"] == 0 and not np.isfinite(r["ln_change_q50"]) for r in got["layers"])
    no_residual = _model(_cfg(attention_residual=False, norm_type_block="layernorm"))
    defined, reason = rs.ln_change_applicability(no_residual)
    assert not defined and "no additive residual" in reason
    assert not rs.residual_additive(no_residual)


def test_self_weight_summary_from_attention_rows() -> None:
    heads = [{"layer": 0, "head": 0, "self_weight": 0.1}, {"layer": 0, "head": 1, "self_weight": 0.3}]
    out = rs.self_weight_summary(heads, [{"layer": 0, "self_weight": 0.2, "n_rows": 10}])
    assert out["layers"] == [{"layer": 0, "self_weight_mean": 0.2, "self_weight_head_min": 0.1,
                              "self_weight_head_max": 0.3, "n_rows": 10}]


# ----------------------------------------------------------------------------------------------
# Routing covariance gap and hidden temperature (v2:872)
# ----------------------------------------------------------------------------------------------

def test_covariance_gap_vanishes_for_scalar_covariance_under_non_orthogonal_frames() -> None:
    r"""With :math:`\Sigma = cI`, :math:`C_i = c(U_i^\top U_i)^{-1}` for ANY frame: gap 0, fitted scale c."""
    cfg = _cfg(sigma_init=15.0)
    model = _model(cfg)
    scale = 2.5
    with torch.no_grad():
        model.prior_bank.sigma_log_embed.fill_(math.log(scale))
    dataset = st.eval_windows(_stream(61, 23), 8, 4)
    inverse, _, _ = cv.routing_frame_and_covariance(model, dataset[0][0][None])
    G = inverse.double() @ inverse.double().transpose(-1, -2)                     # (1, N, Hb, d, d)
    eye = torch.eye(G.shape[-1], dtype=torch.float64)
    assert float((G - eye).abs().max()) > 0.1                                      # frames far from orthogonal
    out = cv.routing_covariance_gap(model, dataset, stream_tokens=dataset.tokens, device=DEVICE,
                                    n_windows=4, n_uniform_windows=3)
    assert out["status"] == "ok" and out["covariance_fixed_through_stack"]
    for name, sample in out["samples"].items():
        assert math.isclose(sample["c"], scale, rel_tol=1e-9), name
        assert sample["frobenius_gap_max"] < 1e-6 and sample["logdet_gap_max"] < 1e-9
        assert math.isclose(sample["c_logdet"], scale, rel_tol=1e-9)
        assert all(r["frobenius_gap_max"] < 1e-6 for r in sample["per_block"])
    assert out["samples"]["occurrence"]["n_pairs"] == 4 * 8
    temperature = cv.hidden_temperature(model)
    assert math.isclose(temperature["factor"], 15.0 / scale, rel_tol=1e-6)
    assert math.isclose(temperature["per_coordinate_factor_q50"], 15.0 / scale, rel_tol=1e-6)


def test_covariance_is_the_routers_and_logdet_gap_sees_only_the_stored_scale() -> None:
    r"""Layer-0 ``projected_key_covariance_pre`` equals :math:`S_K C_i S_K^\top` with this module's :math:`C_i`."""
    cfg = _cfg()
    model = _model(cfg)
    dataset = st.eval_windows(_stream(61, 24), 8, 4)
    inputs = torch.stack([dataset[0][0], dataset[1][0]])                           # (2, 8)
    capture: Dict[str, Any] = {"diagnostic": {}}
    with torch.no_grad():
        model.forward_beliefs(inputs, capture=capture)
    G_k = capture["diagnostic"]["layer_conditional_captures"][0]["projected_key_covariance_pre"].double()
    inverse, sigma, block_diagonal = cv.routing_frame_and_covariance(model, inputs)
    assert block_diagonal and sigma.shape == (2, 8, cfg.embed_dim)
    Hb, d = inverse.shape[-3], inverse.shape[-1]
    U_inv = torch.zeros(2, 8, Hb * d, Hb * d, dtype=torch.float64)
    for b in range(Hb):
        U_inv[..., b * d:(b + 1) * d, b * d:(b + 1) * d] = inverse[..., b, :, :].double()
    C = U_inv @ torch.diag_embed(sigma.double()) @ U_inv.transpose(-1, -2)         # (2, 8, K, K)
    operators = model.relational_operators_by_layer[0]
    _, S_k = operators.comparison_pair()
    S_k = S_k.double()
    if operators.input_scope == "full":
        expected = torch.einsum("hrk,bnkl,hsl->bnhrs", S_k, C, S_k)
    else:
        blocks = torch.stack([C[..., h * d:(h + 1) * d, h * d:(h + 1) * d] for h in range(Hb)], dim=-3)
        expected = torch.einsum("hrk,bnhkl,hsl->bnhrs", S_k, blocks, S_k)
    torch.testing.assert_close(G_k, expected, rtol=1e-4, atol=1e-5)
    stats = cv.pair_statistics(inverse, sigma, torch.ones(2, 8, dtype=torch.bool))
    stats["_d"], stats["_d_block"] = float(Hb * d), float(d)
    c = cv.fit_common_scale(stats)
    _, logdet = cv.gaps(stats, c)
    mean_log_var = torch.log(sigma.double()).mean(-1).reshape(-1).numpy()
    np.testing.assert_allclose(logdet, np.abs(mean_log_var - math.log(c)), rtol=1e-6, atol=1e-8)


def test_covariance_not_applicable_and_temperature_without_table() -> None:
    transported = types.SimpleNamespace(cfg=types.SimpleNamespace(attention_mode="transported"),
                                        relational_operators=None)
    assert "no projected router" in cv.gap_applicability(transported)
    frame_diag = types.SimpleNamespace(cfg=types.SimpleNamespace(family="gaussian_frame_diagonal"),
                                       relational_operators=object())
    assert "intrinsic" in cv.gap_applicability(frame_diag)
    no_table = types.SimpleNamespace(prior_bank=types.SimpleNamespace(), cfg=types.SimpleNamespace(sigma_init=1.0))
    assert cv.hidden_temperature(no_table)["status"] == "not_applicable"


def test_hidden_temperature_definition_on_a_known_table() -> None:
    cfg = _cfg(sigma_init=15.0)
    model = _model(cfg)
    variance = torch.tensor([1.0, 3.0]).repeat(cfg.vocab_size, cfg.embed_dim // 2)   # mean 2
    with torch.no_grad():
        model.prior_bank.sigma_log_embed.copy_(torch.log(variance))
    counts = np.zeros(cfg.vocab_size)
    counts[0] = 1.0
    out = cv.hidden_temperature(model, type_counts=counts)
    assert math.isclose(out["factor"], 7.5, rel_tol=1e-6)
    assert math.isclose(out["factor_inverse_variance_mean"], 15.0 * (1.0 + 1.0 / 3.0) / 2.0, rel_tol=1e-6)
    assert math.isclose(out["fraction_within_10pct_of_unit"], 0.5)
    assert math.isclose(out["factor_occurrence_weighted"], 7.5, rel_tol=1e-6)


# ----------------------------------------------------------------------------------------------
# Word-normalized perplexity
# ----------------------------------------------------------------------------------------------

WORD_ROWS = [" = Alpha = \n", "", " The cat sat on the <unk> . \n", "",
             " A 1 @,@ 000 year @-@ old tree . \n"]
LEGACY_TEXT = " = Alpha = \n\n\n The cat sat on the mat . \n\n\n A 1,000 year-old tree . \n"


def _bytes_stream(text: str) -> torch.Tensor:
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.int64)


def _decode(ids: Any) -> bytes:
    return bytes(int(i) for i in ids)


def test_text_audit_and_word_counts_on_synthetic_streams() -> None:
    stream = cal.WordStream(WORD_ROWS, "synthetic")
    counts = cal.word_counts(stream)
    assert counts["words_only"]["count"] == 19 and counts["words_plus_eos_per_line"]["count"] == 24
    audit = cal.text_audit(LEGACY_TEXT, stream.text)
    assert audit["same_text"] and audit["char_match_fraction"] == 1.0
    assert audit["unk_wildcards"] == 1 and audit["unk_wildcard_chars"] == 3                 # "mat"
    assert audit["raw_counts"]["word"]["@-@"] == 1 and audit["raw_counts"]["bpe"]["@-@"] == 0
    v1_style = cal.text_audit("".join(WORD_ROWS), stream.text)                              # the <unk> text itself
    assert v1_style["same_text"] and v1_style["unk_wildcard_chars"] == len("<unk>")
    changed = cal.text_audit(LEGACY_TEXT.replace("cat", "dog"), stream.text)
    assert not changed["same_text"] and changed["char_match_fraction"] < 1.0
    assert changed["mismatched_segments"] >= 1 and changed["mismatch_examples"]
    longer = cal.text_audit(LEGACY_TEXT + " extra words \n", stream.text)
    assert not longer["same_text"] and longer["unmatched_bpe_chars"] == len("extrawords")


def test_word_normalized_perplexity_with_a_known_word_count() -> None:
    tokens = _bytes_stream(LEGACY_TEXT)
    T = int(tokens.numel())
    nll = np.full(T - 1, 0.5)
    target_pos = np.arange(1, T)
    result = cal.calibration_for_run("synthetic", tokens, {"final_model.pt": (nll, target_pos)}, seq_len=8,
                                     stride=4, decode_bytes=_decode, word_stream=cal.WordStream(WORD_ROWS, "syn"))
    assert result["status"] == "ok" and result["audit"]["same_text"]
    entry = result["checkpoints"]["final_model.pt"]
    assert entry["coverage"]["complete"]
    assert math.isclose(entry["conventions"]["words_plus_eos_per_line"]["ppl"], math.exp(0.5 * (T - 1) / 24))
    assert math.isclose(entry["conventions"]["words_only"]["ppl"], math.exp(0.5 * (T - 1) / 19))
    assert math.isclose(entry["bpe_ppl"], math.exp(0.5))
    text = cal.calibration_table(result)
    assert f"{math.exp(0.5 * (T - 1) / 24):.2f}" in text and "same text = True" in text
    partial = cal.calibration_for_run("synthetic", tokens, {"final_model.pt": (nll[1:], target_pos[1:])},
                                      seq_len=8, stride=4, decode_bytes=_decode,
                                      word_stream=cal.WordStream(WORD_ROWS, "syn"))
    assert partial["status"] == "incomplete_coverage" and "conventions" not in partial["checkpoints"]["final_model.pt"]
    other = cal.calibration_for_run("synthetic", _bytes_stream(LEGACY_TEXT.replace("cat", "dog")),
                                    {"final_model.pt": (nll, target_pos)}, seq_len=8, stride=4,
                                    decode_bytes=_decode, word_stream=cal.WordStream(WORD_ROWS, "syn"))
    assert other["status"] == "audit_failed"


def test_calibration_skips_with_reasons_and_reads_a_supplied_word_file(tmp_path: Path) -> None:
    tokens = _bytes_stream(LEGACY_TEXT)
    passes = {"final_model.pt": (np.zeros(3), np.arange(1, 4))}
    assert cal.calibration_for_run("synthetic-period3", tokens, passes, seq_len=8,
                                   stride=4)["status"] == "not_applicable"
    assert "word-level run" in cal.calibration_for_run("wikitext-103-word", tokens, passes, seq_len=8,
                                                       stride=4)["reason"]
    missing = cal.calibration_for_run("synthetic", tokens, passes, seq_len=8, stride=4, decode_bytes=_decode,
                                      word_stream_path=tmp_path / "absent.tokens")
    assert missing["status"] == "skipped" and "word stream unavailable" in missing["reason"]
    path = tmp_path / "wiki.test.tokens"
    path.write_text("".join(row if row else "\n" for row in WORD_ROWS), encoding="utf-8")
    stream, reason = cal.load_word_stream("test", path=path)
    assert reason is None and stream.n_rows == 5 and stream.n_words == 19 and stream.n_blank_rows == 2


def test_target_coverage() -> None:
    assert cal.target_coverage(np.arange(1, 10), 10)["complete"]
    duplicated = cal.target_coverage(np.array([1, 2, 2, 3]), 5)
    assert not duplicated["complete"] and duplicated["duplicates"] == 1 and duplicated["missing"] == 1


def test_ce_by_context() -> None:
    from vfe3.posthoc.figures import ce_by_context

    rows = ce_by_context(np.array([1, 1, 2, 2, 2]), np.array([1.0, 3.0, 2.0, 2.0, 2.0]))
    assert rows[0] == {"context": 1, "n": 2, "ce": 2.0, "se": pytest.approx(1.0)}
    assert rows[1]["ce"] == 2.0 and rows[1]["se"] == 0.0


# ----------------------------------------------------------------------------------------------
# End to end
# ----------------------------------------------------------------------------------------------

def _light_options(cache_dir: Path, **over: Any) -> Any:
    from vfe3.posthoc.report import PosthocOptions

    values = dict(device="cpu", cache_dir=str(cache_dir), splits=("validation", "test"), interventions=(),
                  maps=False, resources=False, figures=False, save_nll=False, latency_batches=0,
                  bootstrap_resamples=0, klcap_splits=(), cross_check=False)
    values.update(over)
    return PosthocOptions(**values)


def test_run_posthoc_gap_sections_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.posthoc.report import PosthocOptions, run_posthoc

    monkeypatch.setattr(st, "FALLBACK_BLOCK_TOKENS", 16)
    run_dir, cache_dir = _fake_run(tmp_path)
    options = PosthocOptions(device="cpu", cache_dir=str(cache_dir), bootstrap_resamples=100, latency_batches=0,
                             interventions=())
    run_posthoc(run_dir, options=options, log=lambda message: None)
    out = run_dir / "posthoc"
    saved = json.loads((out / "posthoc.json").read_text(encoding="utf-8"))
    for checkpoint in ("final_model.pt", "best_model.pt"):
        residual = saved["residual"][checkpoint]
        assert residual["status"] == "ok" and len(residual["layers"]) == 2
        assert residual["layers"][0]["rho_n"] == 60                     # every validation target once
        assert not residual["ln_change"]["defined"] and residual["ln_change"]["reason"]
        assert residual["self_weight"]["status"] == "ok" and len(residual["self_weight"]["layers"]) == 2
        covariance = saved["covariance"][checkpoint]
        assert covariance["routing_covariance"]["status"] == "ok"
        assert covariance["hidden_temperature"]["status"] == "ok"
        assert "arrays" not in json.dumps(covariance)                   # per-pair arrays live in the .npz
    final_sw = saved["residual"]["final_model.pt"]["self_weight"]["layers"]
    assert [r["self_weight_mean"] for r in final_sw] == pytest.approx(
        [r["self_weight"] for r in saved["attention"]["layers"]])
    assert saved["calibration"]["status"] == "not_applicable" and saved["calibration"]["reason"]
    assert saved["routing"]["status"] in ("ok", "skipped", "failed")
    if saved["routing"]["status"] != "ok":
        assert saved["routing"].get("reason") or saved["routing"].get("error")
    assert set(saved["ce_by_context"]["series"]) == {"validation, final weights", "test, final weights"}
    for name in ("residual_final", "residual_best", "covariance_final", "covariance_best", "calibration"):
        assert (out / "tables" / f"{name}.md").is_file(), name
    for name in ("s1a_residual_ratio_final", "v2_872_covariance_final", "ce_by_context"):
        assert (out / "figures" / f"{name}.pdf").is_file(), name
    pairs = np.load(out / "csv" / "covariance_pairs_final.npz")
    assert pairs["occurrence_frobenius_gap"].shape == pairs["occurrence_type"].shape


def test_run_posthoc_routing_module_ok_failing_and_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.posthoc as package
    from vfe3.posthoc.report import run_posthoc

    run_dir, cache_dir = _fake_run(tmp_path)
    calls: List[Any] = []

    def decomposition(model: Any, dataset: Any, indices: Any, *, batch_size: int, device: Any) -> Dict[str, Any]:
        calls.append((len(indices), batch_size, str(device)))
        return {"status": "ok", "variance": torch.tensor([1.0, 2.0])}

    def writer(result: Any, out_dir: Path) -> List[Path]:
        path = Path(out_dir) / "tables" / "routing_decomposition.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("M4\n", encoding="utf-8")
        return [path]

    fake = types.ModuleType("vfe3.posthoc.routing")
    fake.routing_decomposition, fake.write_routing_outputs = decomposition, writer
    monkeypatch.setitem(sys.modules, "vfe3.posthoc.routing", fake)
    monkeypatch.setattr(package, "routing", fake, raising=False)
    report = run_posthoc(run_dir, out_dir=tmp_path / "ok", options=_light_options(cache_dir, covariance=False),
                         log=lambda message: None)
    assert report["routing"]["status"] == "ok" and report["routing"]["result"]["variance"] == [1.0, 2.0]
    assert calls == [(len(st.eval_windows(_stream(61, 11), 8, 4)), 2, "cpu")]
    assert (tmp_path / "ok" / "tables" / "routing_decomposition.md").is_file()

    def broken(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError("boom")

    fake.routing_decomposition = broken
    report = run_posthoc(run_dir, out_dir=tmp_path / "failed", options=_light_options(cache_dir, covariance=False),
                         log=lambda message: None)
    assert report["routing"]["status"] == "failed" and "boom" in report["routing"]["error"]
    assert report["passes"] and all(p["status"] == "ok" for p in report["passes"])

    monkeypatch.setitem(sys.modules, "vfe3.posthoc.routing", None)
    monkeypatch.delattr(package, "routing", raising=False)
    report = run_posthoc(run_dir, out_dir=tmp_path / "missing", options=_light_options(cache_dir, covariance=False),
                         log=lambda message: None)
    assert report["routing"]["status"] == "skipped" and "unavailable" in report["routing"]["reason"]
    off = run_posthoc(run_dir, out_dir=tmp_path / "off", options=_light_options(
        cache_dir, covariance=False, routing=False, calibration=False, residual_split=None),
        log=lambda message: None)
    assert off["routing"]["status"] == "disabled" and off["calibration"]["status"] == "disabled"
    assert off["residual"] == {} and any("residual" in note for note in off["notes"])


def test_run_posthoc_transported_route_records_reasons(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""A run without a projected router: residual ratios still defined, C_i gap and self-weights skipped."""
    import tests.test_posthoc_20260928 as base
    from vfe3.model.model import VFEModel
    from vfe3.posthoc.report import run_posthoc

    def transported_cfg(**over: Any) -> Any:
        return _cfg(attention_mode="transported")

    def transported_model(cfg: Any, seed: int = 0) -> VFEModel:
        torch.manual_seed(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VFEModel(cfg).eval()
        with torch.no_grad():
            model.prior_bank.sigma_log_embed.normal_(0.0, 0.5)
            model.prior_bank.phi_embed.normal_(0.0, 0.3)
        return model

    monkeypatch.setattr(base, "_cfg", transported_cfg)
    monkeypatch.setattr(base, "_model", transported_model)
    run_dir, cache_dir = _fake_run(tmp_path)
    report = run_posthoc(run_dir, out_dir=tmp_path / "out", options=_light_options(cache_dir, figures=True),
                         log=lambda message: None)
    assert all(p["status"] == "ok" for p in report["passes"])
    residual = report["residual"]["final_model.pt"]
    assert residual["status"] == "ok" and residual["residual_additive"]
    assert residual["self_weight"]["status"] == "unavailable" and "no projected router" in residual["self_weight"]["reason"]
    covariance = report["covariance"]["final_model.pt"]
    assert covariance["routing_covariance"]["status"] == "not_applicable"
    assert "no projected router" in covariance["routing_covariance"]["reason"]
    assert covariance["hidden_temperature"]["status"] == "ok"
    assert report["calibration"]["status"] == "not_applicable"
    assert report["routing"]["status"] == "ok" or report["routing"].get("reason") or report["routing"].get("error")
    assert (tmp_path / "out" / "figures" / "s1a_residual_ratio_final.pdf").is_file()
    assert (tmp_path / "out" / "figures" / "v2_872_covariance_final.pdf").is_file()   # temperature panel only


def test_run_posthoc_calibration_on_a_wikitext_named_synthetic_run(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    r"""The report computes the calibration from its full test pass, with the byte decoder and word file."""
    from vfe3.posthoc.report import run_posthoc

    run_dir, cache_dir = _fake_run(tmp_path)
    table = {i: bytes([97 + i]) + (b" " if i % 3 == 0 else b"") for i in range(24)}      # 'a', 'b ', ...
    decode = lambda ids: b"".join(table[int(i)] for i in ids)                          # noqa: E731
    from vfe3.data import datasets as data

    monkeypatch.setattr(data, "get_tiktoken_byte_decoder", lambda dataset, **kw: decode)
    monkeypatch.setattr(st, "_byte_decoder", lambda dataset: None)                  # articles: token blocks
    tokens = torch.load(cache_dir / "synthetic-period3_test_tiktoken_tokens.pt")
    text = decode(tokens.tolist()).decode("utf-8")
    words = tmp_path / "wiki.test.tokens"
    words.write_text(text.strip() + "\n", encoding="utf-8")
    monkeypatch.setattr(cal, "WIKITEXT_PREFIXES", ("synthetic",))
    report = run_posthoc(run_dir, out_dir=tmp_path / "cal", options=_light_options(
        cache_dir, covariance=False, routing=False, word_stream_path=str(words)), log=lambda message: None)
    result = report["calibration"]
    assert result["status"] == "ok" and result["audit"]["same_text"]
    entry = result["checkpoints"]["final_model.pt"]
    assert entry["coverage"]["complete"] and entry["n_targets"] == 52
    n_words = len(text.split())
    assert entry["conventions"]["words_only"]["count"] == n_words
    assert entry["conventions"]["words_plus_eos_per_line"]["count"] == n_words + 1
    test_pass = next(p for p in report["passes"] if p["checkpoint"] == "final_model.pt" and p["split"] == "test")
    assert math.isclose(entry["conventions"]["words_only"]["nats_per_word"], test_pass["ce"] * 52 / n_words,
                        rel_tol=1e-9)
    assert (tmp_path / "cal" / "tables" / "calibration.md").is_file()
