r"""Audit 2026-10-08 remediation: config, run artifacts, training loop and data lifecycle.

Tiny CPU models only (K <= 8, two heads, N <= 6, vocab <= 11). Covers the inactive-field corrections,
the additive-encoder validation, historical-default pins on reload, int/float-stable fingerprints,
settings-blind driver identity, the comparison-RoPE run-record labels, the git line-ending probe, the
run_vfe resume reset, the ablation train generator under DATA_SEED=None, finalized-run selection in
generate_text, the word-table digest check, the fully clamped router-row count, the train-sample cap
record, and the matmul-precision replay ladder inside train().
"""
from tests.profiles import test_config
import csv
import json
import logging
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.run_artifacts as ra
import vfe3.train as T
from vfe3.config import VFE3Config, config_from_serialized, serialize_config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel

B, N, V = 2, 4, 7


@pytest.fixture(autouse=True)
def _one_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    torch.set_num_threads(1)


def _cfg(**overrides: Any) -> VFE3Config:
    values: Dict[str, Any] = dict(
        vocab_size=V, embed_dim=4, n_heads=2, batch_size=B, max_seq_len=N, eval_stride=2,
        n_layers=2, n_e_steps=1, warmup_steps=0, max_steps=6, log_interval=0, eval_interval=0,
        checkpoint_interval=0, skip_belief_sigma_update=True, generate_embedding_figures=False,
        use_block_mlp=True, block_mlp_expansion=1, beta_attention_prior="causal",
        family="gaussian_diagonal", gauge_group="block_glk", attention_mode="projected_predictive",
        predictive_readout="kernel", attention_residual=True, norm_type_block="layernorm",
        layernorm_affine=True, decode_mode="linear", pos_phi="none")
    values.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


def _rope(**overrides: Any) -> VFE3Config:
    return _cfg(rope_mode="comparison", rope_base=100.0, **overrides)


def _loaders(cfg: VFE3Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = TokenWindows(torch.arange(41) % V, N)
    val_ds = TokenWindows((torch.arange(23) * 3) % V, N, pad_final=True, stride=cfg.eval_stride)
    test_ds = TokenWindows((torch.arange(19) * 5 + 1) % V, N, pad_final=True, stride=cfg.eval_stride)
    return (DataLoader(train_ds, batch_size=B, shuffle=True, generator=torch.Generator().manual_seed(1)),
            DataLoader(val_ds, batch_size=2, shuffle=False),
            DataLoader(test_ds, batch_size=2, shuffle=False))


def _rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ----------------------------------------------------------------------------------------------
# P2-5 / P2-6: inactive_fields
# ----------------------------------------------------------------------------------------------

def test_pos_phi_scale_and_passthrough_mlp_floor_are_inactive() -> None:
    inactive = _cfg(block_mlp_covariance="passthrough").inactive_fields()
    assert "pos_phi_scale" in inactive and "block_mlp_covariance_floor" in inactive
    live = _cfg(pos_phi="learned").inactive_fields()
    assert "pos_phi_scale" not in live


def test_mean_distance_routing_marks_the_projected_factorization_inactive() -> None:
    inactive = _cfg(relational_routing="mean_distance", projected_factorization="gram_certified").inactive_fields()
    assert "projected_factorization" in inactive
    assert "projected_factorization" not in _cfg(projected_factorization="gram_certified").inactive_fields()


def test_comparison_rope_keeps_the_factor_route_live() -> None:
    # P2-6 is moot after P1-2: the rotated route keeps its source tuples, so cholesky_qr2 runs there.
    inactive = _rope(projected_factorization="cholesky_qr2_certified").inactive_fields()
    assert "projected_factorization" not in inactive


def test_block_coordinates_mark_no_longer_claims_bit_identity() -> None:
    inactive = _cfg(block_coordinates="reference", norm_type_block="none", norm_type_final="none",
                    use_block_mlp=False).inactive_fields()
    assert "roundoff" in inactive["block_coordinates"]


@pytest.mark.parametrize("transport", ["off", "frozen"])
def test_additive_encoder_requires_trainable_frames(transport: str) -> None:
    with pytest.raises(ValueError, match="per_token_additive"):
        _cfg(encode_mode="per_token_additive", gauge_transport=transport)
    assert _cfg(encode_mode="per_token_additive").encode_mode == "per_token_additive"


# ----------------------------------------------------------------------------------------------
# P2-3 / P2-4: reload pins and fingerprints
# ----------------------------------------------------------------------------------------------

def test_reload_pins_introduction_time_values_of_flipped_defaults() -> None:
    payload = serialize_config(_cfg())
    for name in ("content_link_chart", "predictive_covariance", "m_decode_mu_lr", "m_decode_sigma_lr"):
        payload.pop(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        restored = config_from_serialized(payload, source="pre-flip record")
    assert (restored.content_link_chart, restored.predictive_covariance) == ("affine", "passthrough")
    assert restored.m_decode_mu_lr is None and restored.m_decode_sigma_lr is None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        current = config_from_serialized(serialize_config(_cfg()), source="current record")
    assert current.content_link_chart == _cfg().content_link_chart


def test_fingerprint_ignores_int_versus_float_spelling() -> None:
    a = serialize_config(_rope(rope_fraction=1, sigma_init=15))
    b = serialize_config(_rope(rope_fraction=1.0, sigma_init=15.0))
    assert a["rope_fraction"] == 1 and type(a["rope_fraction"]) is int
    assert ra.semantic_config_fingerprint(a) == ra.semantic_config_fingerprint(b)
    assert ra.semantic_config_fingerprint({**a, "embed_dim": 4}) != ra.semantic_config_fingerprint(
        {**a, "embed_dim": 8})


# ----------------------------------------------------------------------------------------------
# P2-2 / gap-data-lifecycle#8: driver identity
# ----------------------------------------------------------------------------------------------

def test_driver_identity_ignores_settings_and_tracks_logic() -> None:
    base = b"import os\nconfig = dict(a=1)\nDATA_SEED = 3\ndef run():\n    return 1\n"
    settings = b"import os\nconfig = dict(a=1, resume_from='x.pt')\nconfig['b'] = 2\nDATA_SEED = None\n" \
               b"def run():\n    return 1\n"
    logic = b"import os\nconfig = dict(a=1)\nDATA_SEED = 3\ndef run():\n    return 2\n"
    assert ra._driver_logic_bytes(base) == ra._driver_logic_bytes(settings)
    assert ra._driver_logic_bytes(base) == ra._driver_logic_bytes(base.replace(b"\n", b"\r\n"))
    assert ra._driver_logic_bytes(base) != ra._driver_logic_bytes(logic)
    broken = b"config = {\n"
    assert ra._driver_logic_bytes(broken) == broken              # unparsable: raw bytes still hashed


def test_package_identity_covers_the_baseline_and_paper_arm_launchers(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: List[str] = []
    real = ra._driver_logic_bytes
    monkeypatch.setattr(ra, "_driver_logic_bytes", lambda source: (seen.append(source), real(source))[1])
    ra._package_code_identity()
    repository = Path(ra.__file__).resolve().parent.parent
    for name in ("train_baseline.py", "paper_arms.py", "train_vfe3.py"):
        if (repository / name).is_file():
            assert (repository / name).read_bytes() in seen, name


# ----------------------------------------------------------------------------------------------
# Run-record labels and provenance
# ----------------------------------------------------------------------------------------------

def test_pure_path_report_labels_comparison_rope_and_records_its_settings() -> None:
    toggles = ra._pure_path_report(_rope(rope_fraction=1), [])["config_toggles"]
    assert toggles["rope_pair_energy_exactness"] == "comparison_space_rotated_projected_moments"
    assert (toggles["rope_mode"], toggles["rope_fraction"], toggles["rope_base"]) == ("comparison", 1.0, 100.0)
    plain = ra._pure_path_report(_cfg(), [])["config_toggles"]
    assert plain["rope_pair_energy_exactness"] == "not_applicable" and plain["rope_mode"] == "none"


def test_git_probe_carries_the_system_line_ending_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ra, "_git_line_ending_config", lambda git: (("core.autocrlf", "true"),))
    env = ra._git_environment("git")
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert (env["GIT_CONFIG_COUNT"], env["GIT_CONFIG_KEY_0"], env["GIT_CONFIG_VALUE_0"]) == (
        "1", "core.autocrlf", "true")
    monkeypatch.setattr(ra, "_git_line_ending_config", lambda git: ())
    assert "GIT_CONFIG_COUNT" not in ra._git_environment("git")


# ----------------------------------------------------------------------------------------------
# P2-9, P2-7, generate_text, word cache
# ----------------------------------------------------------------------------------------------

def test_run_vfe_clears_an_inherited_resume_path() -> None:
    from vfe3.experiment import _without_resume

    cfg = _cfg(resume_from="vfe3_runs/base/checkpoints/step_4.pt")
    cleared = _without_resume(cfg, "arm", logging.getLogger("resume-test"))
    assert cleared.resume_from is None and not cleared.trust_resume_checkpoint
    assert serialize_config(cleared) == {**serialize_config(cfg), "resume_from": None}
    plain = _cfg()
    assert _without_resume(plain, "arm", logging.getLogger("resume-test")) is plain


def test_ablation_train_loader_always_has_a_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    import ablation

    built: Dict[str, Any] = {}

    def fake(dataset: str, split: str, *args: Any, generator: Any = None, **kwargs: Any) -> object:
        built[split] = generator
        return object()

    monkeypatch.setattr(ablation, "make_dataloader", fake)
    monkeypatch.setattr(ablation, "DATA_SEED", None)
    monkeypatch.setattr(ablation, "_LOADER_CACHE", {})
    ablation.get_loader("tiny", N, B, "train")
    ablation.get_loader("tiny", N, B, "validation")
    assert isinstance(built["train"], torch.Generator) and built["validation"] is None
    assert ablation._effective_data_seed(7) == 7                # run_single re-pins it to cfg.seed


def test_generate_text_skips_runs_that_never_finalized(tmp_path: Path) -> None:
    import os
    import generate_text

    def _run(name: str, stamp: int, finalized: bool) -> Path:
        run = tmp_path / name
        run.mkdir()
        for artifact in ("config.json", "best_model.pt", *(("summary.json",) if finalized else ())):
            (run / artifact).write_text("x", encoding="utf-8")
            os.utime(run / artifact, ns=(stamp, stamp))
        return run

    finalized = _run("100.00_tiny_s1", 10**18, True)
    _run("20261008-120000_tiny_s2", 2 * 10**18, False)            # still training: newer, no summary
    assert generate_text.newest_run_dir(tmp_path) == finalized


def test_word_cache_digest_catches_a_reordered_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.data import datasets
    from vfe3.data.word_vocab import WordVocabulary

    dataset = "wikitext-103-word"
    words = ["<unk>", "a", "b", "c"]
    meta_path = Path(str(datasets.cache_path(dataset, "train", suffix="pt", cache_dir=tmp_path)) + ".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"vocab_size": 4, "min_count": 3, "vocabulary_sha256": datasets.word_vocabulary_sha256(words)}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    table = {"words": words}
    monkeypatch.setattr(datasets, "_load_word_vocabulary",
                        lambda *args, **kwargs: WordVocabulary(table["words"], min_count=3))
    datasets._require_word_cache_matches_vocabulary(dataset, "train", cache_dir=tmp_path)
    table["words"] = ["<unk>", "b", "a", "c"]                      # same size and cutoff, new id order
    with pytest.raises(ValueError, match="vocabulary_sha256"):
        datasets._require_word_cache_matches_vocabulary(dataset, "train", cache_dir=tmp_path)


# ----------------------------------------------------------------------------------------------
# Training loop: clamped rows, train-sample cap, replay ladder
# ----------------------------------------------------------------------------------------------

def _diagnostic(model: VFEModel) -> Dict[str, Any]:
    ids = (torch.arange(2 * N) % V).reshape(2, N)
    capture: dict = {"diagnostic": {}}
    with torch.no_grad():
        model.forward_beliefs(ids, capture=capture)
    return capture["diagnostic"]


def test_fully_clamped_rows_are_counted_per_layer_and_head() -> None:
    cfg = _cfg()
    torch.manual_seed(0)
    diagnostic = _diagnostic(VFEModel(cfg))
    rows_out: List[List[Tuple[int, int]]] = []
    cap, _ = T._capture_numerics_counts(diagnostic, -1.0, rows_out=rows_out)   # every cost is capped
    # causal prior, N = 4: rows 1..3 have >= 2 sources in each of the 2 windows
    assert rows_out == [[(6, 6)] * cfg.n_heads] * cfg.n_layers
    assert all(c == e for layer in cap for c, e in layer)
    free: List[List[Tuple[int, int]]] = []
    T._capture_numerics_counts(diagnostic, math.inf, rows_out=free)
    assert free == [[(0, 6)] * cfg.n_heads] * cfg.n_layers
    accumulator: Dict[str, Any] = {}
    T._merge_numerics_counts(accumulator, cap, [(0, 1)] * cfg.n_layers, rows_out)
    rows = T._cap_clamped_row_rows(accumulator, 5)
    assert [list(row) for row in rows][0] == list(T.CAP_CLAMPED_ROW_FIELDS)
    assert {(r["layer"], r["head"]): (r["clamped_rows"], r["rows"]) for r in rows} == {
        (layer, head): (6, 6) for layer in range(cfg.n_layers) for head in range(cfg.n_heads)}


@pytest.fixture
def _high_precision() -> Any:
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    yield
    torch.set_float32_matmul_precision(previous)


def _train(cfg: VFE3Config, artifacts: Any = None) -> Tuple[VFEModel, List[float], Tuple[DataLoader, ...]]:
    loaders = _loaders(cfg)
    torch.manual_seed(5)
    model = VFEModel(cfg)
    torch.manual_seed(11)
    losses = T.train(model, loaders[0], cfg, n_steps=cfg.max_steps, grad_clip=1.0, log_interval=1,
                     eval_interval=cfg.max_steps, val_loader=loaders[1], device=torch.device("cpu"),
                     logger=logging.getLogger("replay-test"), artifacts=artifacts, generate_samples=False)
    return model, losses, loaders


def test_replay_ladder_inside_train_records_counts_and_reasons(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _high_precision: Any) -> None:
    cfg = _cfg(matmul_precision_retry=True, fp32_matmul_precision="high")
    _, reference, _ = _train(cfg)

    real_step = T.train_step
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> float:
        calls["n"] += 1
        if calls["n"] in (3, 6):                   # the first attempt of loop steps 2 and 4 fails pre-update
            raise ValueError(f"forced certificate failure {calls['n']}")
        return real_step(*args, **kwargs)

    monkeypatch.setattr(T, "train_step", flaky)
    run_dir = tmp_path / "run"
    artifacts = ra.RunArtifacts(run_dir, cfg, VFEModel(cfg), save_model_artifacts=False)
    model, losses, loaders = _train(cfg, artifacts)
    assert losses == reference                     # a replay reproduces the unreplayed step bitwise
    rows = _rows(run_dir / "metrics.csv")
    by_step = {int(row["step"]): float(row["matmul_precision_replays"]) for row in rows}
    # metrics rows are numbered by completed steps; the failing steps are loop steps 2 and 4
    assert (by_step[2], by_step[3], by_step[4], by_step[5], by_step[cfg.max_steps]) == (0.0, 1.0, 1.0, 2.0, 2.0)
    assert artifacts.realized_updates["matmul_precision_replays"] == 2
    records = [json.loads(line) for line in
               (run_dir / "diagnostics" / T.MATMUL_PRECISION_REPLAY_LOG).read_text(encoding="utf-8").splitlines()]
    assert [(r["policy"], r["replay_policy"], r["exception"]) for r in records] == [
        ("high", "highest", "ValueError")] * 2
    assert [r["step"] for r in records] == [2, 4]
    assert [r["message"] for r in records] == ["forced certificate failure 3", "forced certificate failure 6"]
    assert (run_dir / "diagnostics" / "cap_clamped_rows.csv").is_file()

    fields = ra._best_weight_and_train_sample_evaluation(
        model, artifacts, cfg, logging.getLogger("replay-test"), final_step=cfg.max_steps,
        final_test={"test_ce": 1.0, "test_ppl": math.e}, train_loader=loaders[0], val_loader=loaders[1],
        test_loader=loaders[2], tokens_per_char=None, device=torch.device("cpu"))
    assert fields["train_sample_max_targets"] == T.TRAIN_SAMPLE_MAX_TARGETS
    assert fields["train_sample_seed"] == T.TRAIN_SAMPLE_SEED
    assert isinstance(fields["train_sample_scored_targets"], int) and fields["train_sample_scored_targets"] > 0
