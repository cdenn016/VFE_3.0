r"""Post-hoc analyses (:mod:`vfe3.posthoc`, :mod:`vfe3.run_resources`) on tiny CPU models.

Every model here has vocabulary 24, K=4, two heads, two layers and 8-token windows, so each test runs in
well under a second on one thread; the end-to-end test builds a complete fake run directory (checkpoints,
token caches, provenance, summary, metrics, map snapshots) and runs :func:`vfe3.posthoc.report.run_posthoc`.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from vfe3.config import VFE3Config, serialize_config
from vfe3.model.model import VFEModel
from vfe3.posthoc import evaluate as ev
from vfe3.posthoc import interventions as iv
from vfe3.posthoc import maps as mp
from vfe3.posthoc import streams as st
from vfe3.posthoc.attention import AttentionStatsObserver
from vfe3.posthoc.klcap import KLCapObserver
from vfe3.run_resources import optimizer_state_bytes, parameter_breakdown, throughput_summary

V = 24
DEVICE = torch.device("cpu")


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _cfg(**over: Any) -> VFE3Config:
    values = dict(vocab_size=V, embed_dim=4, n_heads=2, max_seq_len=8, eval_stride=4, n_layers=2,
                  batch_size=2, attention_mode="projected_predictive", e_phi_lr=0.0, block_mlp_expansion=2)
    values.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFE3Config(**values)


def _model(cfg: VFE3Config, seed: int = 0) -> VFEModel:
    r"""A tiny model moved off its initialization so every intervention changes something."""
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg).eval()
    with torch.no_grad():
        model.prior_bank.sigma_log_embed.normal_(0.0, 0.5)
        model.prior_bank.phi_embed.normal_(0.0, 0.3)
        model.pos_phi_free.normal_(0.0, 0.3)
        for op in model.relational_operators_by_layer:
            for p in op.parameters():
                p.add_(0.3 * torch.randn_like(p))
    return model


def _stream(n: int, seed: int) -> torch.Tensor:
    return torch.randint(0, V, (n,), generator=torch.Generator().manual_seed(seed))


# ----------------------------------------------------------------------------------------------
# run_resources
# ----------------------------------------------------------------------------------------------

def test_parameter_breakdown_partitions_parameters_and_counts_reads() -> None:
    cfg = _cfg()
    model = _model(cfg)
    named = dict(model.named_parameters())
    out = parameter_breakdown(named, cfg)
    assert out["total"] == sum(p.numel() for p in named.values())
    assert sum(out[s]["total"] for s in ("per_type_tables", "positional_frames", "context_processing",
                                          "other")) == out["total"]
    pb = model.prior_bank
    assert out["per_type_tables"]["phi_embed"] == pb.phi_embed.numel()
    assert out["per_type_tables"]["decode"] == pb.output_proj_weight.numel() + pb.output_proj_bias.numel()
    assert out["positional_frames"]["pos_phi"] == model.pos_phi_free.numel()
    relational = sum(p.numel() for n, p in named.items() if n.startswith("relational_operators"))
    assert out["context_processing"]["relational_score"] + out["context_processing"]["relational_content"] == relational
    # One row of each encoder table + the whole linear decoder + one positional row + context processing.
    expected = (pb.mu_embed.shape[1] + pb.sigma_log_embed.shape[1] + pb.phi_embed.shape[1]
                + pb.output_proj_weight.numel() + pb.output_proj_bias.numel()
                + model.pos_phi_free.shape[1] + out["context_processing"]["total"])
    assert out["parameters_read_per_token"] == expected
    # A state dict gives the same counts; its buffers are listed, not counted.
    from_state = parameter_breakdown(model.state_dict(), cfg)
    assert from_state["total"] == out["total"]
    assert "prior_bank.unigram_log_prior" in from_state["excluded_buffers"]
    wrapped = parameter_breakdown({f"_orig_mod.{n}": p for n, p in named.items()}, cfg)
    assert wrapped["parameters_read_per_token"] == out["parameters_read_per_token"]
    assert all(wrapped[s]["total"] == out[s]["total"] for s in ("per_type_tables", "context_processing"))


def test_optimizer_state_bytes_and_throughput(tmp_path: Path) -> None:
    cfg = _cfg()
    model = _model(cfg)
    names = [n for n, _ in model.named_parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.param_groups[0]["role"] = "mu"
    sum(p.sum() for p in model.parameters()).backward()
    optimizer.step()
    bundle = {"step": 10, "optimizer_state": optimizer.state_dict(), "config": serialize_config(cfg),
              "optimizer_parameter_manifest": {"parameter_groups": [
                  [{"parameter_id": i, "name": n} for i, n in enumerate(names)]]}}
    path = tmp_path / "step_10.pt"
    torch.save(bundle, path)
    out = optimizer_state_bytes(path)
    n_float = sum(p.numel() for p in model.parameters())
    assert out["by_slot"]["exp_avg"] == 4 * n_float and out["by_slot"]["exp_avg_sq"] == 4 * n_float
    assert out["optimizer_state_bytes"] == sum(out["by_slot"].values())
    assert out["by_parameter_section"]["per_type_tables"] > 0 and out["by_group_role"] == {
        "mu": out["optimizer_state_bytes"]}
    (tmp_path / "metrics.csv").write_text(
        "step,tokens_per_s,peak_mem_mb,train_step_ms_mean\n10,100,50,20\n20,300,70,10\n20,,,\n",
        encoding="utf-8")
    (tmp_path / "summary.json").write_text(json.dumps({"wall_time_s": 7200.0}), encoding="utf-8")
    thr = throughput_summary(tmp_path)
    assert thr["peak_mem_mb"] == 70.0 and thr["wall_time_hours"] == 2.0
    assert thr["throughput_tokens_per_s"]["tokens_per_s"]["median"] == 200.0
    assert math.isclose(thr["train_only_hours_estimate"], (10 * 0.020 + 10 * 0.010) / 3600.0)


# ----------------------------------------------------------------------------------------------
# maps
# ----------------------------------------------------------------------------------------------

def test_map_invariants_are_invariant_under_the_symmetries() -> None:
    cfg = _cfg()
    model = _model(cfg)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    rows = mp.map_invariants(state, cfg)
    assert len(rows) == cfg.n_layers * cfg.n_heads
    torch.manual_seed(3)
    moved = dict(state)
    for prefix in ("relational_operators", "relational_operators_extra.0"):
        S_q, S_k = state[f"{prefix}.raw_score_q"], state[f"{prefix}.raw_score_k"]     # (H, r, w)
        T, O = state[f"{prefix}.T"], state[f"{prefix}.O"]                             # (H, s, w), (H, K, s)
        A = torch.randn(S_q.shape[0], S_q.shape[1], S_q.shape[1]) + 3 * torch.eye(S_q.shape[1])
        Bm = torch.randn(T.shape[0], T.shape[1], T.shape[1]) + 3 * torch.eye(T.shape[1])
        moved[f"{prefix}.raw_score_q"], moved[f"{prefix}.raw_score_k"] = A @ S_q, A @ S_k
        moved[f"{prefix}.T"], moved[f"{prefix}.O"] = Bm @ T, O @ torch.linalg.inv(Bm)
    after = mp.map_invariants(moved, cfg)
    for a, b in zip(rows, after):
        for key in ("gain_min", "gain_median", "gain_max", "angle_min_deg", "angle_median_deg",
                    "angle_max_deg", "M_sigma_max", "M_fro"):
            assert math.isclose(a[key], b[key], rel_tol=1e-4, abs_tol=1e-4), key
        assert not math.isclose(a["scale_rms"], b["scale_rms"], rel_tol=1e-3)     # moved along the symmetry
    norms = mp.head_block_norms(state, cfg)
    for layer, per_map in norms.items():
        prefix = "relational_operators" if layer == 0 else f"relational_operators_extra.{layer - 1}"
        for name, leaf in (("S_Q", "raw_score_q"), ("T", "T"), ("O", "O")):
            total = state[f"{prefix}.{leaf}"].double().square().sum(dim=(1, 2)).numpy()   # (H,)
            assert per_map[name].shape == (cfg.n_heads, 2)
            np.testing.assert_allclose(np.square(per_map[name]).sum(axis=1), total, rtol=1e-10)


def test_conserved_quantities_vanish_at_initialization_and_snapshot_curves(tmp_path: Path) -> None:
    cfg = _cfg()
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg)
    rows = mp.map_invariants(model.state_dict(), cfg)
    assert max(r["conserved_S_drift"] for r in rows) < 1e-5
    assert max(r["conserved_TO_drift"] for r in rows) < 1e-5
    directory = tmp_path / "map_snapshots"
    directory.mkdir()
    for step in (5, 10):
        state = {k: v for k, v in model.state_dict().items()
                 if k.startswith("relational_operators") or k == "pos_phi_free"}
        torch.save({"step": step, "state": state}, directory / f"step_{step:07d}.pt")
    curves = mp.snapshot_curves(directory, cfg)
    assert sorted({r["step"] for r in curves}) == [5, 10] and len(curves) == 2 * len(rows)


# ----------------------------------------------------------------------------------------------
# streams
# ----------------------------------------------------------------------------------------------

def test_article_ids_from_level_one_headings() -> None:
    pieces = [b" = Alpha = \n", b"\n", b" text one .\n", b" = = Sub = = \n", b" more .\n",
              b" = Beta = \n", b" text two .\n"]
    table = {i: piece for i, piece in enumerate(pieces)}
    tokens = torch.tensor([0, 1, 2, 3, 4, 5, 6])
    ids, titles, info = st.article_ids(tokens, decode_bytes=lambda seq: b"".join(table[int(t)] for t in seq),
                                       vocab_size=len(pieces))
    assert titles == ["Alpha", "Beta"]
    assert ids.tolist() == [0, 0, 0, 0, 0, 1, 1]          # the level-2 heading does not start an article
    assert info["n_articles"] == 2 and info["n_preamble_tokens"] == 0


def test_article_ids_on_real_gpt2_tokens() -> None:
    tiktoken = pytest.importorskip("tiktoken")
    try:
        enc = tiktoken.get_encoding("gpt2")
    except Exception as exc:
        pytest.skip(f"gpt2 encoding unavailable: {exc}")
    text = (" = Homarus = \n\n Homarus is a lobster .\n\n = = Description = = \n\n It is blue .\n"
            " = Second Article = \n\n Another text .\n")
    tokens = torch.tensor(enc.encode(text))
    found = st.article_ids(tokens, decode_bytes=lambda seq: enc.decode_bytes([int(t) for t in seq]),
                           vocab_size=enc.n_vocab)
    assert found is not None
    ids, titles, _ = found
    assert titles == ["Homarus", "Second Article"]
    second = int(np.argmax(ids == 1))
    assert enc.decode(tokens[second:].tolist()).lstrip("\n").startswith(" = Second Article")


def test_fixed_train_sample_is_deterministic_disjoint_and_masked() -> None:
    tokens = _stream(1000, 5)
    a = st.fixed_train_sample(tokens, 8, 4, n_targets=40)
    b = st.fixed_train_sample(tokens, 8, 4, n_targets=40)
    assert np.array_equal(a.starts, b.starts) and len(a) == 10
    assert np.all(np.diff(a.starts) >= 8) and np.all(a.starts % 8 == 0)
    x, y = a[0]
    s = int(a.starts[0])
    assert torch.equal(x, tokens[s:s + 8]) and (y[:4] == -100).all() and torch.equal(y[4:], tokens[s + 5:s + 9])
    assert len(st.fixed_train_sample(tokens, 8, 4, n_targets=10 ** 9, max_targets=12)) == 3


def test_fixed_train_sample_matches_the_in_training_sample() -> None:
    from types import SimpleNamespace

    from vfe3.data.datasets import TokenWindows
    from vfe3.train import _build_fixed_train_sample

    tokens, val_tokens = _stream(1000, 5), _stream(81, 5)
    train_loader = SimpleNamespace(dataset=TokenWindows(tokens, 8))
    val_loader = SimpleNamespace(dataset=SimpleNamespace(tokens=val_tokens), batch_size=4)
    sample, reason = _build_fixed_train_sample(train_loader, val_loader, SimpleNamespace(eval_stride=4, batch_size=4))
    assert reason is None
    posthoc = st.fixed_train_sample(tokens, 8, 4, n_targets=int(val_tokens.numel()) - 1)
    assert [8 * i for i in sample.indices] == posthoc.starts.tolist()
    for k, index in enumerate(sample.indices):
        x_train, y_train = sample._window(index)
        x_post, y_post = posthoc[k]
        assert torch.equal(x_train, x_post) and torch.equal(y_train, y_post)


def _write_cache(cache_dir: Path, dataset: str, split: str, tokens: torch.Tensor) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tokens.to(torch.int64).contiguous(), cache_dir / f"{dataset}_{split}_tiktoken_tokens.pt")


def test_resolve_stream_verifies_and_refuses_a_mismatch(tmp_path: Path) -> None:
    from vfe3.data.datasets import cache_source_identity

    dataset, tokens = "synthetic-period3", _stream(61, 1)
    _write_cache(tmp_path, dataset, "validation", tokens)
    file_sha = cache_source_identity(dataset, "validation", cache_dir=tmp_path)["sha256"]
    info = st.RunInfo(run_dir=tmp_path, cfg=_cfg(), dataset=dataset,
                      provenance={"val_data_sha256": st.token_content_sha256(tokens), "val_data_n_tokens": 61},
                      selection_identity={"split": "validation", "source": {"sha256": file_sha}})
    _, record = st.resolve_stream(info, "validation", cache_dir=tmp_path)
    assert record["status"] == "verified" and len(record["checks"]) == 3
    info.provenance["val_data_n_tokens"] = 60
    with pytest.raises(st.StreamIdentityError):
        st.resolve_stream(info, "validation", cache_dir=tmp_path)
    _, record = st.resolve_stream(info, "validation", cache_dir=tmp_path, allow_mismatch=True)
    assert record["status"] == "mismatch_allowed"


# ----------------------------------------------------------------------------------------------
# evaluate
# ----------------------------------------------------------------------------------------------

def test_nll_pass_matches_vfe3_train_evaluate_and_bootstrap() -> None:
    cfg = _cfg()
    model = _model(cfg)
    dataset = st.eval_windows(_stream(61, 2), 8, 4)
    indices = np.arange(len(dataset))
    result = ev.nll_pass(model, dataset, indices, batch_size=2, device=DEVICE)
    reference = ev.protocol_evaluate(model, dataset, indices, batch_size=2, device=DEVICE)
    assert result.n == 60 == int(reference["scored_targets"])            # every transition exactly once
    assert abs(result.ce - float(reference["ce"])) < ev.CROSS_CHECK_TOL
    assert sorted(result.target_pos.tolist()) == list(range(1, 61))
    units = np.arange(61)[result.target_pos] // 10
    interval = ev.ce_interval(result.nll, units, n_resamples=500)
    assert math.isclose(interval["point"], result.ce, rel_tol=1e-12) and interval["lo"] < result.ce < interval["hi"]
    zero = ev.paired_delta_interval(result.nll, result.nll, units, n_resamples=200)
    assert zero["point"] == 0.0 and zero["lo"] == 0.0 and zero["hi"] == 0.0


# ----------------------------------------------------------------------------------------------
# interventions
# ----------------------------------------------------------------------------------------------

def test_interventions_change_the_output_and_restore_it_exactly() -> None:
    cfg = _cfg(attention_residual=True)
    model = _model(cfg)
    dataset = st.eval_windows(_stream(61, 3), 8, 4)
    indices = np.arange(len(dataset))
    base = ev.nll_pass(model, dataset, indices, batch_size=2, device=DEVICE)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    for intervention in iv.INTERVENTIONS[1:]:
        with intervention.apply(model) as record:
            arm = ev.nll_pass(model, dataset, indices, batch_size=2, device=DEVICE)
        assert np.array_equal(arm.target_pos, base.target_pos)
        assert np.abs(arm.nll - base.nll).max() > 1e-6, intervention.key
        if "intercepted_calls" in record:
            assert record["intercepted_calls"] == cfg.n_layers * math.ceil(len(dataset) / 2)
    after = ev.nll_pass(model, dataset, indices, batch_size=2, device=DEVICE)
    assert np.array_equal(after.nll, base.nll)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(op.attention_residual for op in model.relational_operators_by_layer)
    assert iv.NOT_IMPLEMENTED == () and [i.row for i in iv.INTERVENTIONS] == list(range(1, 13))


def test_identity_covariance_score_is_exact_when_covariances_are_already_identity() -> None:
    r"""With :math:`\Sigma_p = I` and every frame the identity, :math:`C_i = I`: row 6 is then a no-op."""
    cfg = _cfg()
    model = _model(cfg)
    with torch.no_grad():
        model.prior_bank.sigma_log_embed.zero_()
        model.prior_bank.phi_embed.zero_()
        model.pos_phi_free.zero_()
    dataset = st.eval_windows(_stream(61, 4), 8, 4)
    indices = np.arange(len(dataset))
    base = ev.nll_pass(model, dataset, indices, batch_size=2, device=DEVICE)
    with iv.routing_covariances_identity(model):
        arm = ev.nll_pass(model, dataset, indices, batch_size=2, device=DEVICE)
    np.testing.assert_allclose(arm.nll, base.nll, rtol=0, atol=1e-5)


def test_not_applicable_arms() -> None:
    cfg = _cfg(attention_residual=False)
    model = _model(cfg)
    with pytest.raises(iv.NotApplicable):
        with iv.residual_removed(model):
            pass
    with pytest.raises(ValueError):
        iv.by_key(["no_such_row"])
    assert [i.key for i in iv.by_key(["token_frames_identity"])] == ["baseline", "token_frames_identity"]


# ----------------------------------------------------------------------------------------------
# observers
# ----------------------------------------------------------------------------------------------

def _capture_batch(model: VFEModel, dataset: Any, batch: Tuple[int, ...]) -> Tuple[Dict[str, Any], torch.Tensor]:
    _, inputs, targets = next(ev.iter_batches(dataset, np.asarray(batch), len(batch)))
    capture = {"diagnostic": {}}
    with torch.no_grad():
        model.forward_beliefs(inputs, return_logits=True, capture=capture)
    return capture["diagnostic"], targets != -100


def test_klcap_observer_counts_and_weights_capped_entries() -> None:
    cfg = _cfg(kl_max=1.0)
    model = _model(cfg)
    dataset = st.eval_windows(_stream(61, 6), 8, 4)
    observer = KLCapObserver(cfg.kl_max)
    diagnostic, valid = _capture_batch(model, dataset, (0, 1))
    input_valid = torch.ones_like(valid)
    input_valid[1, 6:] = False                                                    # pretend final-window padding
    meta = {"windows": np.array([0, 1]), "starts": np.array([0, 4]), "input_valid": input_valid}
    observer.update(diagnostic, valid, meta)
    cap = diagnostic["layer_conditional_captures"][0]
    score, beta = cap["score_energy_pre"], cap["beta_pre"]                         # (B, H, N, N)
    causal = torch.tril(torch.ones(8, 8, dtype=torch.bool))
    for population, rows in (("scored", valid), ("computed", input_valid)):
        rows = rows & (torch.arange(8) >= 1)[None, :]                             # >= 2 causal sources
        entries = causal & rows[:, None, :, None]
        capped = entries & (score >= 1.0)
        row = [r for r in observer.rows()
               if r["population"] == population and r["layer"] == 0 and r["head"] == 0][0]
        assert row["n_capped"] == int(capped[:, 0].sum()) > 0
        assert row["n_entries"] == int(entries[:, 0].sum())
        expected_mass = float((beta[:, 0] * capped[:, 0]).sum()) / int(rows.sum())
        assert math.isclose(row["capped_beta_mass_mean"], expected_mass, rel_tol=1e-5)
        assert row["capped_at_window_start"] == int(capped[:, 0, :, 0].sum())
    assert observer.summary()["cap_active"] and observer.summary()["cap_active_scored"]
    unbound = KLCapObserver(1e9)
    unbound.update(diagnostic, valid, meta)
    assert not unbound.summary()["cap_active"]


def test_attention_observer_matches_direct_definitions() -> None:
    cfg = _cfg()
    model = _model(cfg)
    dataset = st.eval_windows(_stream(61, 7), 8, 4)
    observer = AttentionStatsObserver(np.zeros(61, dtype=np.int64))
    diagnostic, valid = _capture_batch(model, dataset, (0, 1))
    observer.update(diagnostic, valid, {"windows": np.array([0, 1]), "starts": np.array([0, 4])})
    beta = diagnostic["layer_conditional_captures"][1]["beta_pre"][:, 1]           # layer 1, head 1: (B, N, N)
    prior = torch.softmax(diagnostic["log_prior"][1].float(), dim=-1)            # (N, N)
    rows = valid & (torch.arange(8) >= 1)[None, :]                                 # >= 2 causal sources
    i = torch.arange(8)
    entropy = -torch.special.xlogy(beta, beta).sum(-1) / torch.log((i + 1).clamp_min(2).float())
    distance = (beta * (i[:, None] - i[None, :]).abs()).sum(-1)
    got = [r for r in observer.rows(n_resamples=0) if r["layer"] == 1 and r["head"] == 1][0]
    assert got["n_rows"] == int(rows.sum())
    assert math.isclose(got["entropy_norm"], float(entropy[rows].mean()), rel_tol=1e-5)
    assert math.isclose(got["distance_mean"], float(distance[rows].mean()), rel_tol=1e-5)
    assert math.isclose(got["self_weight"], float(torch.diagonal(beta, dim1=-2, dim2=-1)[rows].mean()),
                        rel_tol=1e-5)
    assert math.isclose(got["prior_overlap"], float(torch.minimum(beta, prior).sum(-1)[rows].mean()),
                        rel_tol=1e-5)
    assert got["min_entropy_norm"] == pytest.approx(float(entropy[rows].min()), rel=1e-5)


# ----------------------------------------------------------------------------------------------
# End to end
# ----------------------------------------------------------------------------------------------

def _fake_run(tmp_path: Path, *, retired_field: bool = False) -> Tuple[Path, Path]:
    from vfe3.data.datasets import cache_source_identity
    from vfe3.train import evaluate

    dataset, run_dir, cache_dir = "synthetic-period3", tmp_path / "run", tmp_path / "cache"
    run_dir.mkdir()
    streams = {"validation": _stream(61, 11), "test": _stream(53, 12), "train": _stream(400, 13)}
    for split, tokens in streams.items():
        _write_cache(cache_dir, dataset, split, tokens)
    cfg = _cfg(attention_residual=True)
    final, best = _model(cfg, seed=0), _model(cfg, seed=1)
    payload = serialize_config(cfg)
    if retired_field:
        payload = dict(payload, lambda_twohop=0.0)
    identity = {"dataset": dataset, "split": "validation",
                "source": {"sha256": cache_source_identity(dataset, "validation", cache_dir=cache_dir)["sha256"]},
                "iterator": {"seq_len": 8, "stride": 4, "n_windows": len(st.eval_windows(streams["validation"], 8, 4)),
                             "batch_size": 2}}
    for name, model in (("final_model.pt", final), ("best_model.pt", best)):
        torch.save({"model_state": model.state_dict(), "config": payload, "selection_data_identity": identity},
                   run_dir / name)

    def ce(model: VFEModel, tokens: torch.Tensor) -> float:
        loader = DataLoader(st.eval_windows(tokens, 8, 4), batch_size=2, shuffle=False)
        return float(evaluate(model, loader, device=DEVICE)["ce"])

    summary = {"n_steps": 20, "best_step": 10, "evaluation_weights": "final", "evaluated_step": 20,
               "final_val_ce": ce(final, streams["validation"]), "final_val_scored_targets": 60,
               "test_ce": ce(final, streams["test"]), "test_scored_targets": 52,
               "best_val_ppl": math.exp(ce(best, streams["validation"])), "wall_time_s": 36.0}
    provenance = {"device_name": "cpu", "selection_data_identity": identity}
    for split, prefix in (("validation", "val"), ("test", "test"), ("train", "train")):
        provenance[f"{prefix}_data_sha256"] = st.token_content_sha256(streams[split])
        provenance[f"{prefix}_data_n_tokens"] = int(streams[split].numel())
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    (run_dir / "metrics.csv").write_text("step,tokens_per_s,peak_mem_mb,train_step_ms_mean\n10,100,5,2\n20,120,6,2\n",
                                         encoding="utf-8")
    optimizer = torch.optim.Adam(final.parameters(), lr=1e-3)
    sum(p.sum() for p in final.parameters()).backward()
    optimizer.step()
    names = [n for n, _ in final.named_parameters()]
    (run_dir / "checkpoints").mkdir()
    torch.save({"step": 20, "optimizer_state": optimizer.state_dict(), "config": payload,
                "optimizer_parameter_manifest": {"parameter_groups": [
                    [{"parameter_id": i, "name": n} for i, n in enumerate(names)]]}},
               run_dir / "checkpoints" / "step_20.pt")
    (run_dir / "map_snapshots").mkdir()
    for step, model in ((10, best), (20, final)):
        state = {k: v for k, v in model.state_dict().items()
                 if k.startswith("relational_operators") or k == "pos_phi_free"}
        torch.save({"step": step, "state": state}, run_dir / "map_snapshots" / f"step_{step:07d}.pt")
    return run_dir, cache_dir


def test_run_posthoc_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.posthoc.report import PosthocOptions, run_posthoc

    monkeypatch.setattr(st, "FALLBACK_BLOCK_TOKENS", 16)           # several resampling units in 61 tokens
    run_dir, cache_dir = _fake_run(tmp_path)
    options = PosthocOptions(device="cpu", cache_dir=str(cache_dir), bootstrap_resamples=200, latency_batches=1)
    report = run_posthoc(run_dir, options=options, log=lambda message: None)
    out = run_dir / "posthoc"
    saved = json.loads((out / "posthoc.json").read_text(encoding="utf-8"))
    assert saved["reproduction"]["final_weights_reproduced"] is True
    assert {(c["checkpoint"], c["split"]) for c in saved["reproduction"]["checks"]} == {
        ("final_model.pt", "validation"), ("final_model.pt", "test"), ("best_model.pt", "validation")}
    passes = {(p["checkpoint"], p["split"]): p for p in saved["passes"]}
    assert len(passes) == 6 and all(p["status"] == "ok" and p["cross_check"]["ok"] for p in passes.values())
    assert passes[("final_model.pt", "train_sample")]["n_scored"] == 60
    assert saved["streams"]["validation"]["status"] == "verified"
    assert saved["streams"]["validation"]["protocol"]["iterator_ok"] is True
    rows = {r["key"]: r for r in saved["interventions"]["rows"]}
    assert all(rows[i.key]["status"] == "ok" for i in iv.INTERVENTIONS)
    assert rows["restoration_check"]["status"] == "ok"
    assert [rows[i.key]["row"] for i in iv.INTERVENTIONS] == list(range(1, 13))
    assert saved["klcap"]["validation"]["summary"]["per_layer"] and saved["attention"]["rows"]
    assert saved["resources"]["parameter_breakdown"]["total"] == sum(
        p.numel() for p in _model(_cfg()).parameters())
    assert saved["resources"]["optimizer_state"]["optimizer_state_bytes"] > 0
    assert saved["maps"]["snapshot_curves"]["n_snapshots"] == 2
    for name in ("interventions", "relational_run", "resources", "klcap_validation", "attention_stats",
                 "map_invariants_final"):
        assert (out / "tables" / f"{name}.md").is_file(), name
    for name in ("m5_interventions", "s2b_attention_stats", "s2a_head_block_shares_final",
                 "s1b_map_invariants_final", "s1b_map_invariant_curves"):
        assert (out / "figures" / f"{name}.pdf").is_file(), name
    dump = np.load(out / "nll" / "interventions_validation.npz")
    assert dump["nll_baseline"].shape == dump["unit"].shape and "nll_token_frames_identity" in dump.files
    assert report["alibi_slopes"] is not None and len(report["alibi_slopes"]) == 2


def test_run_posthoc_legacy_config_fields(tmp_path: Path) -> None:
    from vfe3.posthoc.report import PosthocOptions, run_posthoc

    run_dir, cache_dir = _fake_run(tmp_path, retired_field=True)
    options = PosthocOptions(device="cpu", cache_dir=str(cache_dir), splits=("validation",), checkpoints=(
        "final_model.pt",), interventions=(), maps=False, resources=False, figures=False, save_nll=False,
        latency_batches=0, bootstrap_resamples=0, klcap_splits=(), attention_split=None)
    with pytest.raises(ValueError, match="unknown field"):
        run_posthoc(run_dir, options=options, log=lambda message: None)
    options.drop_unknown_config_fields = True
    report = run_posthoc(run_dir, options=options, log=lambda message: None)
    assert report["reproduction"]["dropped_config_fields"] == {"lambda_twohop": 0.0}
    assert report["reproduction"]["final_weights_reproduced"] is True
