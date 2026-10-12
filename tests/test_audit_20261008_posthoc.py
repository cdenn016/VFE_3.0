r"""Regression tests for the post-hoc findings of the 2026-10-08 audit (hotspot-2-vfe3-posthoc, ml-engineer#3).

Tiny CPU models and synthetic token streams only.
"""

from __future__ import annotations

from tests.profiles import test_config

import math
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from vfe3.config import VFE3Config, serialize_config
from vfe3.model.model import VFEModel
from vfe3.posthoc import covariance as cv
from vfe3.posthoc import evaluate as ev
from vfe3.posthoc import interventions as iv
from vfe3.posthoc import report as rp
from vfe3.posthoc import streams as st

V = 24
DEVICE = torch.device("cpu")


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _cfg(**over: Any) -> VFE3Config:
    values = dict(vocab_size=V, embed_dim=4, n_heads=2, max_seq_len=8, eval_stride=4, n_layers=1,
                  batch_size=2, attention_mode="projected_predictive", block_mlp_expansion=2,
                  norm_type_block="none", layernorm_affine=False, skip_belief_sigma_update=False,
                  beta_attention_prior="causal_alibi")
    values.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


def _model(cfg: VFE3Config, seed: int = 0) -> VFEModel:
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg).eval()
    with torch.no_grad():
        model.prior_bank.sigma_log_embed.normal_(0.0, 0.5)
        model.prior_bank.phi_embed.normal_(0.0, 0.3)
        for op in model.relational_operators_by_layer:
            for p in op.parameters():
                p.add_(0.3 * torch.randn_like(p))
    return model


def _stream(n: int, seed: int) -> torch.Tensor:
    return torch.randint(0, V, (n,), generator=torch.Generator().manual_seed(seed))


def _nll(model: VFEModel, dataset: Any) -> np.ndarray:
    return ev.nll_pass(model, dataset, np.arange(len(dataset)), batch_size=2, device=DEVICE).nll


# ----------------------------------------------------------------------------------------------
# P2-17: train-sample reproduction compares the training loop's own sample
# ----------------------------------------------------------------------------------------------

def test_training_loop_sample_is_located_inside_the_larger_posthoc_sample() -> None:
    from vfe3.data.datasets import TokenWindows
    from vfe3.train import TRAIN_SAMPLE_MAX_TARGETS, _build_fixed_train_sample

    seq_len, stride = 8, 4
    tokens = torch.randint(0, V, (200_000,), generator=torch.Generator().manual_seed(0))
    n_val = 100_000                                                    # above the training cap of 32,768
    train_loader = SimpleNamespace(dataset=TokenWindows(tokens, seq_len))
    val_loader = SimpleNamespace(dataset=SimpleNamespace(tokens=torch.zeros(n_val + 1, dtype=torch.long)),
                                 batch_size=4)
    training, reason = _build_fixed_train_sample(train_loader, val_loader,
                                                 SimpleNamespace(eval_stride=stride, batch_size=4))
    assert reason is None
    posthoc = st.fixed_train_sample(tokens, seq_len, stride, n_val)
    assert len(posthoc) > len(training.indices) == math.ceil(TRAIN_SAMPLE_MAX_TARGETS / stride)
    windows, cap = st.training_loop_sample_windows(posthoc, tokens, n_val)
    assert cap == TRAIN_SAMPLE_MAX_TARGETS
    assert posthoc.starts[windows].tolist() == [seq_len * i for i in training.indices]


def test_subsample_reproduction_scores_only_the_logged_population() -> None:
    nll = np.array([1.0, 1.0, 3.0, 3.0, 5.0, 5.0])
    result = ev.TargetNLL(nll=nll, target_pos=np.arange(6), token=np.zeros(6, dtype=np.int64),
                          window=np.array([0, 0, 1, 1, 2, 2]), context=np.ones(6, dtype=np.int32),
                          expected=6, excluded=0, seconds=0.0)
    reference = {"source": "summary.json train_sample_ce_final", "ce": 2.0, "scored_targets": None,
                 "kind": "in-training evaluation"}
    rep = rp._subsample_reproduction(result, np.arange(3), np.array([0, 1]), reference, matched=True)
    assert rep["status"] == "ok" and rep["ce"] == 2.0 and rep["n_scored"] == 4 and rep["n_windows"] == 2
    whole = rp._reproduction({"ce": float(nll.mean()), "n_scored": 6}, reference, full=True, matched=True)
    assert whole["status"] == "mismatch"                               # the old, unlike-for-like comparison
    partial = rp._subsample_reproduction(result, np.array([0, 2]), np.array([0, 1]), reference, matched=True)
    assert partial["status"] == "subset"
    summary = rp._reproduction_summary({"passes": [{"checkpoint": "final_model.pt", "split": "train_sample",
                                                    "ce": float(nll.mean()), "reproduction": rep}]},
                                       SimpleNamespace(dropped_config_fields={}))
    assert summary["final_weights_reproduced"] is True and summary["checks"][0]["ce"] == 2.0


# ----------------------------------------------------------------------------------------------
# P2-18: cl100k unassigned ids; max_tokens-capped train stream
# ----------------------------------------------------------------------------------------------

def test_article_ids_never_decodes_ids_absent_from_the_stream() -> None:
    table = {0: b" = Alpha = \n", 1: b"x", 2: b" ", 3: b"\n", 4: b" = Beta = \n"}

    def decode(seq: List[int]) -> bytes:
        return b"".join(table[int(t)] for t in seq)                     # KeyError on an unassigned id

    tokens = torch.tensor([0, 1, 2, 1, 3, 4, 1, 1, 3])
    found = st.article_ids(tokens, decode_bytes=decode)
    assert found is not None
    ids, titles, info = found
    assert titles == ["Alpha", "Beta"] and ids.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1]
    units, unit_info = st.resampling_units(tokens, None, decode_bytes=decode)
    assert unit_info["unit"] == "article" and units.tolist() == ids.tolist()


def test_article_ids_on_cl100k_with_unassigned_ids_below_n_vocab() -> None:
    tiktoken = pytest.importorskip("tiktoken")
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        pytest.skip(f"cl100k_base encoding unavailable: {exc}")
    tokens = torch.tensor(enc.encode(" = Alpha = \n\n Text .\n = Beta = \n\n More .\n"))
    found = st.article_ids(tokens, decode_bytes=lambda seq: enc.decode_bytes([int(t) for t in seq]),
                           vocab_size=enc.n_vocab)
    assert found is not None and found[1] == ["Alpha", "Beta"]


def test_resolve_stream_loads_the_train_split_at_the_recorded_cap(tmp_path: Path) -> None:
    dataset, tokens, cap = "synthetic-period3", _stream(200, 4), 120
    tmp_path.mkdir(parents=True, exist_ok=True)
    for split, stream in (("train", tokens), ("validation", tokens)):
        torch.save(stream.to(torch.int64).contiguous(), tmp_path / f"{dataset}_{split}_tiktoken_tokens.pt")
    info = st.RunInfo(run_dir=tmp_path, cfg=_cfg(), dataset=dataset,
                      provenance={"max_tokens": cap, "train_data_sha256": st.token_content_sha256(tokens[:cap]),
                                  "train_data_n_tokens": cap,
                                  "val_data_sha256": st.token_content_sha256(tokens), "val_data_n_tokens": 200})
    loaded, record = st.resolve_stream(info, "train", cache_dir=tmp_path)
    assert record["status"] == "verified" and record["max_tokens"] == cap and loaded.numel() == cap
    validation, record = st.resolve_stream(info, "validation", cache_dir=tmp_path)
    assert record["status"] == "verified" and record["max_tokens"] is None and validation.numel() == 200


# ----------------------------------------------------------------------------------------------
# M5 rows 3, 8, 9-12
# ----------------------------------------------------------------------------------------------

def test_prior_flattened_refuses_a_row_constant_prior_and_runs_on_alibi() -> None:
    with pytest.raises(iv.NotApplicable, match="constant on the support"):
        with iv.prior_flattened(_model(_cfg(beta_attention_prior="causal"))):
            pass
    model = _model(_cfg())
    dataset = st.eval_windows(_stream(41, 1), 8, 4)
    base = _nll(model, dataset)
    with iv.prior_flattened(model) as record:
        arm = _nll(model, dataset)
    assert record["intercepted_calls"] > 0 and np.abs(arm - base).max() > 0.0


def test_positional_frames_identity_removes_the_frame_space_rope_fold() -> None:
    cfg = _cfg(rope_mode="frame_value", pos_phi="none")
    model = _model(cfg)
    dataset = st.eval_windows(_stream(41, 2), 8, 4)
    assert model._rope_rotation(8, DEVICE) is not None
    base = _nll(model, dataset)
    with iv.positional_frames_identity(model) as record:
        assert model._rope_rotation(8, DEVICE) is None
        arm = _nll(model, dataset)
    assert "frame fold" in record["table"] and np.abs(arm - base).max() > 0.0
    assert model._rope_rotation(8, DEVICE) is not None and np.array_equal(_nll(model, dataset), base)


def test_frame_roles_are_not_refused_for_comparison_rope() -> None:
    model = _model(_cfg(rope_mode="comparison"))
    iv._require_normal_form(model)                                        # no frame carries the rotation
    dataset = st.eval_windows(_stream(41, 3), 8, 4)
    base = _nll(model, dataset)
    with iv.frames_removed_input_moments(model):
        arm = _nll(model, dataset)
    assert np.abs(arm - base).max() > 0.0 and np.array_equal(_nll(model, dataset), base)
    with pytest.raises(iv.NotApplicable, match="rope_mode='frame_value'"):
        iv._require_normal_form(_model(_cfg(rope_mode="frame_value")))


# ----------------------------------------------------------------------------------------------
# Covariance analyses on the mean_distance arm; the real-input mask
# ----------------------------------------------------------------------------------------------

def test_covariance_analyses_refuse_the_covariance_blind_route() -> None:
    model = _model(_cfg(relational_routing="mean_distance"))
    assert cv.hidden_temperature(model)["status"] == "not_applicable"
    assert "mean_distance" in cv.gap_applicability(model)
    assert cv.gap_applicability(_model(_cfg())) is None


def test_real_input_mask_excludes_the_first_padded_input() -> None:
    from vfe3.posthoc.routing import _batch_masks

    tokens = torch.arange(30) % V
    dataset = st.eval_windows(tokens, 8, 4)
    final = len(dataset) - 1
    inputs, _ = dataset[final]
    start = ev.window_start(dataset, final)
    n_real = int(tokens.numel()) - start - 1                              # inputs window[:-1] of the stream tail
    mask = _batch_masks(dataset, np.array([final]), 8, DEVICE)[0]
    assert mask.tolist() == [True] * n_real + [False] * (8 - n_real)
    keep = next(cv._window_batches(dataset, np.array([final]), 1))[1][0]
    assert keep.tolist() == mask.tolist()

    seen: Dict[str, Any] = {}

    class Observer:
        def update(self, capture: Any, valid: Any, meta: Dict[str, Any]) -> None:
            seen["input_valid"] = meta["input_valid"][0].tolist()

    ev.nll_pass(_model(_cfg()), dataset, np.array([final]), batch_size=1, device=DEVICE, observers=[Observer()])
    assert seen["input_valid"] == mask.tolist()


# ----------------------------------------------------------------------------------------------
# posthoc.json carries no M4 per-row samples; the fp32 replay is IEEE fp32
# ----------------------------------------------------------------------------------------------

def test_routing_section_drops_the_per_row_samples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.posthoc import routing as rt

    def decomposition(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"status": "ok", "variance": torch.tensor([1.0]), "samples": {"L0H0": {"epsilon": np.zeros(5)}}}

    def writer(result: Dict[str, Any], out_dir: Path) -> List[Path]:
        assert "samples" in result                                       # the sidecar writer still sees them
        path = Path(out_dir) / "m4_routing_samples.npz"
        np.savez_compressed(path, L0H0_epsilon=result["samples"]["L0H0"]["epsilon"])
        return [path]

    monkeypatch.setattr(rt, "routing_decomposition", decomposition)
    monkeypatch.setattr(rt, "write_routing_outputs", writer)
    section = rp._routing(None, None, np.arange(2), batch_size=1, device=DEVICE, out=tmp_path,
                          checkpoint="final_model.pt", split="validation", subset=False, log=lambda m: None)
    assert section["status"] == "ok" and "samples" not in section["result"]
    assert section["result"]["variance"] == [1.0]
    assert section["samples_file"].endswith("m4_routing_samples.npz")


def test_fp32_replay_pins_ieee_matmul_precision(tmp_path: Path) -> None:
    cfg = _cfg(fp32_matmul_precision="high")
    torch.save({"model_state": _model(cfg).state_dict(), "config": serialize_config(cfg)},
               tmp_path / "final_model.pt")
    before = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        loaded = ev.load_model(tmp_path, "final_model.pt", device=DEVICE, precision="fp32")
        assert torch.get_float32_matmul_precision() == "highest"
        assert loaded.cfg.fp32_matmul_precision == "highest"
        assert any("fp32_matmul_precision='highest'" in note for note in loaded.notes)
        run = ev.load_model(tmp_path, "final_model.pt", device=DEVICE, precision="run")
        assert run.cfg.fp32_matmul_precision == "high"
    finally:
        torch.set_float32_matmul_precision(before)
