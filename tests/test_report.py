r"""Single-run publication-figure driver (vfe3.viz.report) + the converged_state extractor.

These pin the WIRING the user found missing: the figure generators and extract runners existed
and were unit-tested in isolation, but nothing drove them end-to-end against a real model, so a
trained run produced only one of the publication figures. The proof is PNG files on disk, so the
integration test asserts the figure set actually appears when the driver runs the real model.
"""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pytest
import torch
from torch.utils.data import DataLoader

from vfe3 import run_artifacts
from vfe3.config import VFE3Config, serialize_config
from vfe3.data.datasets import TokenWindows
from vfe3.geometry.transport import CompactFactoredTransport
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts, semantic_config_fingerprint
from vfe3.train import _loader_data_identity, train
from vfe3.viz.extract import converged_state
from vfe3.viz.figures import register_figure
from vfe3.viz.report import generate_figures, plan_single_run_figures, vocab_comparison_figures
from vfe3.viz.specs import FigureSpec, emit_registered_figures


def _loader(seed=0, n=600, seq_len=8, bs=8):
    g = torch.Generator().manual_seed(seed)
    base = torch.arange(3).repeat(n // 3 + 2)                  # period-3 stream over {0,1,2}
    ds = TokenWindows(base[:n].long(), seq_len)
    return DataLoader(ds, batch_size=bs, shuffle=False, drop_last=True, generator=g)


def _cfg(**kw):
    base = dict(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=2, e_q_mu_lr=0.1, e_phi_lr=0.05)
    base.update(kw)
    return VFE3Config(**base)


def _model(**kw):
    torch.manual_seed(0)
    return VFEModel(_cfg(**kw))


def _bind_offline_selection_contract(
    artifacts: RunArtifacts,
) -> None:
    identity = _loader_data_identity(_loader(), artifacts.cfg.vocab_size)
    artifacts.bind_selection_data_identity(identity)
    artifacts.save_json("provenance.json", {
        "code_identity_sha256":    artifacts.code_identity_sha256,
        "selection_data_identity": identity,
    })


def test_converged_state_shapes_and_finite():
    model = _model(n_layers=2)
    tok = torch.randint(0, 6, (2, 8))                          # only seq 0 is used
    st = converged_state(model, tok)
    n, k = 8, 4
    assert st["mu"].shape == (n, k)
    assert st["phi"].shape[0] == n
    assert st["exp_phi"].shape == (n, k, k)
    assert isinstance(st["omega"], CompactFactoredTransport)
    assert st["omega"].exp_blocks.shape == (n, 2, 2, 2)
    assert st["omega"].inv_blocks.shape == (n, 2, 2, 2)
    assert torch.isfinite(st["omega"].exp_blocks).all()
    assert torch.isfinite(st["omega"].inv_blocks).all()
    assert st["energy"].shape[-2:] == (n, n)
    assert st["beta"].shape[-2:] == (n, n)
    assert st["self_div"].shape[0] == n
    for key in ("mu", "sigma", "phi", "exp_phi", "energy", "beta", "self_div"):
        assert torch.isfinite(st[key]).all(), key


def test_plan_single_run_figures_skips_english_taxonomies_for_japanese():
    planned = plan_single_run_figures("wiki-ja", {
        "decode_readout":             True,
        "vocab_calibration":          True,
        "unknown_figure":             True,
        "vocab_probability_heatmap":  True,
        "vocab_confusion":            True,
        "belief_category_separation": True,
    }, generate_embedding_figures=True)
    written = set(planned)
    assert "belief_category_separation.png" not in written
    assert "vocab_confusion.png" not in written
    assert {"vocab_probability_heatmap.png", "vocab_calibration.png", "decode_readout.png"} <= written
    assert "unknown_figure.png" not in written
    assert planned == (
        "vocab_probability_heatmap.png",
        "vocab_calibration.png",
        "decode_readout.png",
    )


def test_reliability_bins_come_from_current_report_bank():
    from vfe3.viz import report

    reliability = report._reliability_from_ce_bank({
        "conf":    torch.tensor([0.1, 0.4, 0.6, 0.9]),
        "correct": torch.tensor([1.0, 0.0, 1.0, 1.0]),
    }, n_bins=2)

    assert reliability == [
        {"conf": pytest.approx(0.25), "acc": pytest.approx(0.5), "frac": pytest.approx(0.5)},
        {"conf": pytest.approx(0.75), "acc": pytest.approx(1.0), "frac": pytest.approx(0.5)},
    ]


def test_saved_reliability_requires_matching_split_provenance(tmp_path):
    from vfe3.viz import report

    saved = [{"conf": 0.8, "acc": 0.5, "frac": 1.0}]
    (tmp_path / "research.json").write_text(json.dumps({
        "reliability_split": "test",
        "reliability": saved,
    }), encoding="utf-8")
    logger = logging.getLogger("test_saved_reliability")

    assert report._saved_reliability_for_split(tmp_path, "validation", logger) is None
    assert report._saved_reliability_for_split(tmp_path, "test", logger) == saved


def test_generate_figures_reuses_one_same_token_snapshot(tmp_path, monkeypatch):
    from vfe3.viz import extract, report

    model = _model()
    built = []
    seen = []
    real_build = model.build_diagnostic_snapshot

    def build_snapshot(tokens):
        snapshot = real_build(tokens)
        built.append(snapshot)
        return snapshot

    monkeypatch.setattr(model, "build_diagnostic_snapshot", build_snapshot)

    for name in ("e_step_belief_trace", "across_layer_belief_trace", "converged_state",
                 "numerical_health"):
        original = getattr(extract, name)

        def wrapper(*args, _name=name, _original=original, **kwargs):
            seen.append((_name, kwargs.get("snapshot")))
            return _original(*args, **kwargs)

        monkeypatch.setattr(extract, name, wrapper)

    for name in ("attention_maps", "diagnostics_per_layer"):
        original = getattr(model, name)

        def wrapper(*args, _name=name, _original=original, **kwargs):
            seen.append((_name, kwargs.get("snapshot")))
            return _original(*args, **kwargs)

        monkeypatch.setattr(model, name, wrapper)

    class _NoopUMAPWorker:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

    umap_calls = []

    def plot_umap(
        _bank:   dict,
        channel: str = "mu",
        **kwargs: object,
    ) -> object:
        kind = str(kwargs.get("kind", "Belief"))
        path = str(kwargs["path"])
        umap_calls.append((channel, kind, path))
        figure = plt.figure()
        figure.savefig(path)
        return figure

    monkeypatch.setattr(report.figs, "UMAPWorker", _NoopUMAPWorker)
    monkeypatch.setattr(report.figs, "plot_belief_umap", plot_umap)

    paths = generate_figures(
        tmp_path / "run",
        model=model,
        loader=_loader(),
        max_sequences=1,
        generate_embedding_figures=True,
    )

    assert len(built) == 1
    expected = {"e_step_belief_trace", "across_layer_belief_trace", "converged_state",
                "numerical_health", "attention_maps",
                "diagnostics_per_layer"}
    assert {name for name, _ in seen} == expected
    # attention_maps is called with snapshot=None: it scores at "entry" now, which a snapshot
    # cannot serve, so it REPLAYS from the model (audit 2026-08-06 F19 -- the figure and the CSV
    # column had been measuring different objects under the same name). Every OTHER consumer still
    # shares the one snapshot, and nothing builds a second, which is what this test pins.
    assert all(snapshot is built[0] for name, snapshot in seen if name != "attention_maps")
    assert all(snapshot is None for name, snapshot in seen if name == "attention_maps")
    assert len(built) == 1
    written = {path.name for path in paths}
    assert {
        "reliability_diagram.png",
    } <= written
    belief_umap_calls = [call for call in umap_calls if call[1] == "Belief"]
    assert {(channel, kind) for channel, kind, _ in belief_umap_calls} == {
        ("mu", "Belief"),
        ("sigma", "Belief"),
        ("phi", "Belief"),
    }
    for channel, _, rendered_path in belief_umap_calls:
        rendered = Path(rendered_path)
        assert rendered.parent == tmp_path / "run" / "figures"
        assert rendered.name.startswith(f".belief_umap_{channel}.")
        assert rendered.name.endswith(".tmp.png")
        assert not rendered.exists()
        assert (tmp_path / "run" / "figures" / f"belief_umap_{channel}.png").exists()


def test_generate_figures_renders_belief_umap_below_default_controlled_population(
    tmp_path, monkeypatch, caplog,
):
    r"""A loader shorter than the 16,384-token controlled default must still render the belief
    UMAP figures (2026-09-23): the exact-match ``belief_controlled_ready`` gate previously marked
    ``belief_umap_mu/sigma/phi`` entirely unavailable whenever the loader could not fill the full
    controlled population, even with ``generate_embedding_figures=True`` and a nonempty bank.
    """
    from vfe3.viz import report

    model = _model()

    class _NoopUMAPWorker:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

    umap_calls = []

    def plot_umap(_bank: dict, channel: str = "mu", **kwargs: object) -> object:
        umap_calls.append((channel, kwargs.get("controlled")))
        figure = plt.figure()
        figure.savefig(str(kwargs["path"]))
        return figure

    monkeypatch.setattr(report.figs, "UMAPWorker", _NoopUMAPWorker)
    monkeypatch.setattr(report.figs, "plot_belief_umap", plot_umap)

    paths = generate_figures(
        tmp_path / "run",
        model=model,
        loader=_loader(),                 # 600-token base stream, far under the controlled cap
        generate_embedding_figures=True,  # max_tokens/max_sequences both omitted -> default 16,384
    )

    written = {path.name for path in paths}
    assert {"belief_umap_mu.png", "belief_umap_sigma.png", "belief_umap_phi.png"} <= written
    assert {channel for channel, _ in umap_calls} == {"mu", "sigma", "phi"}
    assert all(controlled is True for _, controlled in umap_calls)
    assert "controlled belief UMAP population short: requested 16384 tokens but loader supplied" \
        in caplog.text


def test_generate_figures_forwards_model_owned_full_gaussian_policy(tmp_path, monkeypatch):
    """The report driver forwards the live model policy to its gauge-residual measurement."""
    from vfe3 import metrics
    from vfe3.viz import report

    model = _model(
        family="gaussian_full",
        full_cov_kl_precision="fp64",
        full_cov_congruence_precision="fp64",
        e_phi_lr=0.0,
        n_e_steps=1,
    )
    seen = []

    def observed_residual(*_args, full_cov_kl_precision, **_kwargs):
        seen.append(full_cov_kl_precision)
        return {
            "energy_in_group": torch.tensor([0.0]),
            "energy_out_group": torch.tensor([1.0]),
            "beta_in_group": torch.tensor([0.0]),
            "beta_out_group": torch.tensor([1.0]),
        }

    monkeypatch.setattr(metrics, "gauge_equivariance_residual", observed_residual)
    paths = generate_figures(tmp_path / "run", model=model, loader=_loader(), max_sequences=1)

    assert seen == ["fp64"]
    assert (tmp_path / "run" / "figures" / "gauge_equivariance.png") in paths


def test_generate_figures_rejects_corrupt_best_bundle_fingerprint(tmp_path):
    cfg = _cfg()
    model = _model()
    art = RunArtifacts(tmp_path / "run", cfg, model, dataset="synthetic-period3")
    _bind_offline_selection_contract(art)
    art.maybe_save_best(1, model, 1.0)
    bundle = torch.load(art.best_path, map_location="cpu", weights_only=True)
    bundle["config_fingerprint"] = "corrupt"
    torch.save(bundle, art.best_path)

    with pytest.raises(ValueError, match="fingerprint"):
        generate_figures(art.run_dir, loader=_loader(), max_sequences=16)


def test_generate_figures_rejects_empty_best_bundle_state(tmp_path):
    cfg = _cfg()
    model = _model()
    art = RunArtifacts(tmp_path / "run", cfg, model, dataset="synthetic-period3")
    _bind_offline_selection_contract(art)
    config = serialize_config(cfg)
    torch.save({
        "model_state":             {},
        "config":                  config,
        "config_fingerprint":      semantic_config_fingerprint(config),
        "code_identity_sha256":    art.code_identity_sha256,
        "selection_data_identity": art.selection_data_identity,
    }, art.best_path)

    with pytest.raises(ValueError, match="model_state keys"):
        generate_figures(art.run_dir, loader=_loader(), max_sequences=16)


def test_vocab_comparison_rejects_semantically_mismatched_best_bundle(tmp_path):
    cfg = _cfg()
    model = _model()
    art = RunArtifacts(tmp_path / "run", cfg, model, dataset="synthetic-period3")
    _bind_offline_selection_contract(art)
    mismatched = serialize_config(_cfg(n_e_steps=3))
    torch.save({
        "model_state":             model.state_dict(),
        "config":                  mismatched,
        "config_fingerprint":      semantic_config_fingerprint(mismatched),
        "code_identity_sha256":    art.code_identity_sha256,
        "selection_data_identity": art.selection_data_identity,
    }, art.best_path)

    with pytest.raises(ValueError, match="selection config"):
        vocab_comparison_figures([art.run_dir], tmp_path / "comparison")


class _MemoryGuardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(
            vocab_size=50257,
            max_seq_len=1024,
            batch_size=32,
            n_e_steps=1,
            embed_dim=4,
            decode_mode="family_chunked",
            diagonal_covariance=True,
            family="gaussian_diagonal",
            decode_chunk_size=4096,
        )


def test_generate_figures_memory_guard_uses_materialized_batch_peak(tmp_path, monkeypatch, caplog):
    from vfe3.viz import extract

    calls = []
    monkeypatch.setattr(extract, "e_step_belief_trace", lambda *args, **kwargs: calls.append("trace"))
    monkeypatch.setattr(extract, "belief_ce_bank", lambda *args, **kwargs: calls.append("ce_bank"))
    monkeypatch.setattr(extract, "vocab_prediction_stats", lambda *args, **kwargs: calls.append("vocab"))
    small_loader = [torch.zeros((1, 2), dtype=torch.long)]

    generate_figures(tmp_path / "small", model=_MemoryGuardModel(), loader=small_loader)

    assert "trace" in calls
    assert {"ce_bank", "vocab"} <= set(calls)
    assert "full-vocab" not in caplog.text

    calls.clear()
    caplog.clear()
    large_tokens = torch.zeros(1, dtype=torch.long).expand(1, 20_000)
    generate_figures(tmp_path / "guarded", model=_MemoryGuardModel(), loader=[large_tokens])

    assert "ce_bank" not in calls
    assert "vocab" not in calls
    assert "full-vocab" in caplog.text



def test_on_demand_probe_preparation_reuses_retained_batch_and_preserves_existing(tmp_path, monkeypatch):
    from vfe3.viz import report

    tokens = torch.tensor([[0, 1, 2]])
    targets = torch.tensor([[1, 2, 0]])
    calls = []

    def collect_phi(model, seen_tokens):
        calls.append(("phi", model, seen_tokens.clone()))
        return {"probe": "phi"}

    def collect_depth(model, seen_tokens, seen_targets, depths):
        calls.append(("depth", model, seen_tokens.clone(), seen_targets.clone(), tuple(depths)))
        return {"probe": "depth"}

    monkeypatch.setattr(report, "collect_phi_numerics", collect_phi)
    monkeypatch.setattr(report, "collect_estep_depth_sensitivity", collect_depth)
    model = object()
    logger = logging.getLogger("test_on_demand_probes")

    report._prepare_missing_saved_probes(
        model,
        [(tokens, targets)],
        tmp_path,
        torch.device("cpu"),
        logger,
    )

    assert json.loads((tmp_path / "phi_numerics.json").read_text(encoding="utf-8")) == {
        "probe": "phi",
    }
    assert json.loads((tmp_path / "estep_depth_sensitivity.json").read_text(encoding="utf-8")) == {
        "probe": "depth",
    }
    assert [call[0] for call in calls] == ["phi", "depth"]
    assert torch.equal(calls[0][2], tokens)
    assert torch.equal(calls[1][2], tokens)
    assert torch.equal(calls[1][3], targets)
    assert calls[1][4] == (0, 1, 2, 3, 5, 8)

    report._prepare_missing_saved_probes(
        model,
        [(tokens + 1, targets + 1)],
        tmp_path,
        torch.device("cpu"),
        logger,
    )
    assert [call[0] for call in calls] == ["phi", "depth"]


def test_report_worker_rebuilds_persisted_figures_without_second_report_replay(
    tmp_path,
    monkeypatch,
):
    from vfe3.viz import figure_worker, report

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result_path = run_dir / ".figure_result.json"
    result_path.touch()
    request_path = run_dir / ".figure_request.json"
    request_path.write_text(json.dumps({
        "mode": "report",
        "run_dir": str(run_dir),
        "result_path": str(result_path),
        "device": "cpu",
        "split": "validation",
        "max_sequences": 8,
        "n_e_steps": None,
        "generate_embedding_figures": False,
    }), encoding="utf-8")
    monkeypatch.setenv("VFE3_FIGURE_REQUEST", str(request_path))

    publication_path = run_dir / "figures" / "publication.png"
    calls = []

    def generate_figures(*args, **kwargs):
        calls.append(("report", args, kwargs))
        return [publication_path]

    cfg = object()
    monkeypatch.setattr(report, "generate_figures", generate_figures)
    monkeypatch.setattr(figure_worker, "_load_worker_config", lambda path: cfg)
    monkeypatch.setattr(
        figure_worker,
        "_render_persisted_run_figures",
        lambda *args: calls.append(("persisted", args)),
    )

    assert figure_worker.main() == 0
    assert [call[0] for call in calls] == ["report", "persisted"]
    assert calls[0][2]["prepare_saved_probes"] is True
    assert calls[1][1] == (run_dir.resolve(), cfg, None, calls[0][2]["logger"])
    assert json.loads(result_path.read_text(encoding="utf-8")) == {
        "paths": [str(publication_path.resolve())],
    }


def test_figure_worker_snapshot_caps_actual_loader_shapes():
    tokens = torch.zeros((64, 1024), dtype=torch.long)
    targets = torch.ones_like(tokens)

    batches = run_artifacts._snapshot_report_batches([(tokens, targets)])

    assert len(batches) == 1
    assert batches[0][0].numel() <= run_artifacts._CONTROLLED_FIGURE_BANK_TOKENS
    assert batches[0][0].shape == batches[0][1].shape
    assert batches[0][0].untyped_storage().data_ptr() != tokens.untyped_storage().data_ptr()
    assert batches[0][1].untyped_storage().data_ptr() != targets.untyped_storage().data_ptr()
    tokens.fill_(7)
    targets.fill_(9)
    assert torch.equal(batches[0][0], torch.zeros_like(batches[0][0]))
    assert torch.equal(batches[0][1], torch.ones_like(batches[0][1]))


def test_vocab_comparison_rejects_mixed_tokenizers(tmp_path):
    cfg = _cfg()
    model = _model()
    gpt2 = RunArtifacts(tmp_path / "gpt2", cfg, model, dataset="wikitext-103")
    cl100k = RunArtifacts(tmp_path / "cl100k", cfg, model, dataset="wiki-en")

    with pytest.raises(ValueError, match="mixed tokenizer tags") as exc:
        vocab_comparison_figures(
            [gpt2.run_dir, cl100k.run_dir],
            tmp_path / "comparison",
        )

    message = str(exc.value)
    assert "tiktoken" in message
    assert "tiktoken_cl100k" in message
    assert not (tmp_path / "comparison").exists()


def test_metrics_csv_logs_at_log_cadence(tmp_path):
    # metrics.csv gets a row every log_interval (denser than eval_interval), but the validation
    # columns are EVAL-CADENCE: a value only on an eval step, a BLANK cell on the log-interval rows
    # in between (NOT carried forward).
    import csv
    import math
    cfg = _cfg()
    model = _model()
    art = RunArtifacts(tmp_path / "run", cfg, model, dataset="synthetic-period3")
    train(model, _loader(), cfg, n_steps=8, log_interval=2, eval_interval=4,
          val_loader=_loader(seed=1), artifacts=art)
    rows = list(csv.DictReader(open(tmp_path / "run" / "metrics.csv")))
    assert [r["step"] for r in rows] == ["2", "4", "6", "8"]          # a row every log_interval
    assert rows[0]["val_ce"] == ""                                    # blank before the first eval (step 4)
    assert math.isfinite(float(rows[1]["val_ce"]))                    # fresh val at the step-4 eval
    assert rows[2]["val_ce"] == ""                                    # blank between evals (NOT carried forward)
    assert math.isfinite(float(rows[3]["val_ce"]))                    # fresh again at the step-8 eval


# ---------------------------------------------------------------------------
# vfe3.viz.specs: declarative registered-report dispatch seam (PB-07).
# ---------------------------------------------------------------------------

def _one_axis_figure(path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    if path is not None:
        fig.savefig(path)
    return fig


def test_emit_registered_figure_uses_registry(tmp_path, monkeypatch):
    seen = {}

    @register_figure("report_probe", override=True)
    def probe(*, value, path=None):
        seen["value"] = value
        return _one_axis_figure(path)

    spec = FigureSpec("report_probe", "probe.png", lambda ctx: {"value": ctx["value"]})
    written = emit_registered_figures([spec], {"value": 7}, tmp_path)
    assert [p.name for p in written] == ["probe.png"]
    assert seen == {"value": 7}


def test_emit_registered_figure_atomically_replaces_existing_target_on_success(tmp_path):
    @register_figure("report_probe_replace", override=True)
    def probe(*, path=None):
        return _one_axis_figure(path)

    target = tmp_path / "replace.png"
    target.write_bytes(b"SENTINEL")
    before = set(plt.get_fignums())
    spec = FigureSpec("report_probe_replace", "replace.png", lambda ctx: {})
    written = emit_registered_figures([spec], {}, tmp_path)
    assert written == [target]
    assert target.read_bytes() != b"SENTINEL"
    assert target.stat().st_size > 0
    assert not list(tmp_path.glob(".replace.*.tmp*"))
    assert set(plt.get_fignums()) == before


def test_emit_registered_figure_skips_when_adapter_returns_none(tmp_path, caplog):
    target = tmp_path / "skip.png"
    target.write_bytes(b"SENTINEL")
    before = set(plt.get_fignums())
    spec = FigureSpec("report_probe", "skip.png", lambda ctx: None)
    with caplog.at_level(logging.WARNING):
        written = emit_registered_figures([spec], {}, tmp_path)
    assert written == []
    assert target.read_bytes() == b"SENTINEL"
    assert not list(tmp_path.glob(".skip.*.tmp*"))
    assert set(plt.get_fignums()) == before
    assert caplog.records == []                                      # intentional skip logs no warning


def test_emit_registered_figure_closes_figure_when_builder_raises_after_creating_one(tmp_path, caplog):
    @register_figure("report_probe_raise_after_create", override=True)
    def probe(*, path=None):
        plt.subplots()                                                # leaked figure the sweep must close
        raise RuntimeError("boom after create")

    target = tmp_path / "raise.png"
    target.write_bytes(b"SENTINEL")
    before = set(plt.get_fignums())
    spec = FigureSpec("report_probe_raise_after_create", "raise.png", lambda ctx: {})
    with caplog.at_level(logging.WARNING):
        written = emit_registered_figures([spec], {}, tmp_path)
    assert written == []
    assert target.read_bytes() == b"SENTINEL"
    assert not list(tmp_path.glob(".raise.*.tmp*"))
    assert set(plt.get_fignums()) == before
    assert len(caplog.records) == 1
    assert "report_probe_raise_after_create" in caplog.text


def test_emit_registered_figure_flags_builder_that_returns_without_writing(tmp_path, caplog):
    @register_figure("report_probe_no_write", override=True)
    def probe(*, path=None):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        return fig                                                    # never saves to `path`

    target = tmp_path / "nowrite.png"
    target.write_bytes(b"SENTINEL")
    before = set(plt.get_fignums())
    spec = FigureSpec("report_probe_no_write", "nowrite.png", lambda ctx: {})
    with caplog.at_level(logging.WARNING):
        written = emit_registered_figures([spec], {}, tmp_path)
    assert written == []
    assert target.read_bytes() == b"SENTINEL"
    assert not list(tmp_path.glob(".nowrite.*.tmp*"))
    assert set(plt.get_fignums()) == before
    assert "did not write its temporary output" in caplog.text


def test_emit_registered_figure_flags_missing_registry_key(tmp_path, caplog):
    target = tmp_path / "missing.png"
    target.write_bytes(b"SENTINEL")
    before = set(plt.get_fignums())
    spec = FigureSpec("report_probe_does_not_exist", "missing.png", lambda ctx: {})
    with caplog.at_level(logging.WARNING):
        written = emit_registered_figures([spec], {}, tmp_path)
    assert written == []
    assert target.read_bytes() == b"SENTINEL"
    assert not list(tmp_path.glob(".missing.*.tmp*"))
    assert set(plt.get_fignums()) == before
    assert "report_probe_does_not_exist" in caplog.text


def test_emit_registered_figures_rejects_duplicate_output_names_before_dispatch(tmp_path):
    target = tmp_path / "dup.png"
    target.write_bytes(b"SENTINEL")
    calls = []

    def adapter(ctx):
        calls.append(1)
        return {}

    spec_a = FigureSpec("report_probe", "dup.png", adapter)
    spec_b = FigureSpec("report_probe", "dup.png", adapter)
    with pytest.raises(ValueError, match="unique"):
        emit_registered_figures([spec_a, spec_b], {}, tmp_path)
    assert calls == []
    assert target.read_bytes() == b"SENTINEL"
    assert not list(tmp_path.glob(".dup.*.tmp*"))
