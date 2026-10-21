r"""Internal worker for end-of-run figures.

The parent process supplies one request path through ``VFE3_FIGURE_REQUEST``.  This module has no
user-facing argument parser; it exists only so training can contain native plotting-runtime failure
inside a disposable interpreter.
"""

import csv
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping

import torch

from vfe3.config import VFE3Config
from vfe3.path_utils import (
    path_is_reparse_point,
    portable_path_component_key,
    prepare_owned_output_child,
)
from vfe3.run_artifacts import (
    _atomic_replace,
    _save_figures,
    _unique_sibling_output_temp,
    _write_json_atomic,
)


_FINALIZE_REQUEST_FIELDS = (
    "run_dir",
    "losses",
    "losses_start_step",
    "generate_publication",
    "report_batches_path",
    "model_bundle_path",
    "device",
    "max_tokens",
    "generate_embedding_figures",
)


def _require_finalize_fields(request: Mapping[str, object]) -> None:
    """Reject a finalize request that omits any declared contract field."""
    missing = [field for field in _FINALIZE_REQUEST_FIELDS if field not in request]
    if missing:
        raise ValueError(
            "finalize figure request is missing required fields: " + ", ".join(missing)
        )


def _required_finalize_string(request: Mapping[str, object], field: str) -> str:
    """Return one required nonempty JSON string from a finalize request."""
    value = request.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"finalize figure {field} must be a non-empty string")
    return value


def _nullable_finalize_string(request: Mapping[str, object], field: str) -> str | None:
    """Return one nullable JSON string from a finalize request."""
    value = request.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"finalize figure {field} must be a non-empty string or null")
    return value


def _validated_finalize_request(
    request: Mapping[str, object],
) -> tuple[str, List[float] | None, int, bool, str | None, str | None, str, int, bool]:
    """Validate the exact JSON fields emitted by ``_run_figures_isolated``."""
    _require_finalize_fields(request)
    run_dir = _required_finalize_string(request, "run_dir")
    losses_value = request.get("losses")
    if losses_value is None:
        losses = None
    elif (
        type(losses_value) is list
        and all(type(value) is float for value in losses_value)
    ):
        losses = losses_value
    else:
        raise ValueError("finalize figure losses must be a list of floats or null")
    # Absolute optimizer steps before losses[0]; a resumed run's list holds only the continuation.
    losses_start_step = request.get("losses_start_step")
    if type(losses_start_step) is not int or losses_start_step < 0:
        raise ValueError("finalize figure losses_start_step must be an exact nonnegative integer")

    generate_publication = request.get("generate_publication")
    if type(generate_publication) is not bool:
        raise ValueError("finalize figure generate_publication must be an exact boolean")
    report_batches_path = _nullable_finalize_string(request, "report_batches_path")
    model_bundle_path = _nullable_finalize_string(request, "model_bundle_path")
    device = _required_finalize_string(request, "device")
    max_tokens = request.get("max_tokens")
    if type(max_tokens) is not int:
        raise ValueError("finalize figure max_tokens must be an exact integer")
    generate_embedding_figures = request.get("generate_embedding_figures")
    if type(generate_embedding_figures) is not bool:
        raise ValueError(
            "finalize figure generate_embedding_figures must be an exact boolean")
    return (
        run_dir,
        losses,
        losses_start_step,
        generate_publication,
        report_batches_path,
        model_bundle_path,
        device,
        max_tokens,
        generate_embedding_figures,
    )


def _history_from_csv(path: Path) -> List[Dict[str, object]]:
    """Rebuild history, migrating retired diagnostic columns at the loader boundary."""
    if not path.is_file():
        return []
    history: List[Dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed: Dict[str, object] = {}
            for key, raw in row.items():
                if raw is None or raw == "":
                    parsed[key] = float("nan")
                    continue
                try:
                    parsed[key] = float(raw)
                except ValueError:
                    parsed[key] = raw
            for old_key, new_key in {
                "cos_nat_phi": "phi_ridge_direction_cosine_mean",
                "pullback_cond_median": "phi_pullback_damped_gen_cond_median",
                "pullback_cond_max": "phi_pullback_damped_gen_cond_max",
            }.items():
                if new_key not in parsed and old_key in parsed:
                    parsed[new_key] = parsed[old_key]
            history.append(parsed)
    return history


def _load_worker_config(run_dir: Path) -> 'VFE3Config | SimpleNamespace':
    """Load one run's serialized config for history-only figure reconstruction.

    No forward is replayed, so a run that saved a value this build refuses is read from its raw saved
    values, as the manuscript assemblers read it (:func:`vfe3.paper.runs.load_config`).
    """
    from vfe3.paper.runs import load_config

    metadata_path = run_dir / "config.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping) or not isinstance(metadata.get("config"), Mapping):
        raise ValueError(f"run metadata {metadata_path} has no config mapping")
    cfg, load_mode, _ = load_config(metadata["config"], source=str(metadata_path))
    if cfg is None:
        logging.getLogger(__name__).warning(
            "%s: config could not be built (%s); raw saved values used", metadata_path, load_mode)
        return SimpleNamespace(**metadata["config"])
    return cfg


def _render_saved_probe_figures(run_dir: Path, logger: logging.Logger) -> List[Path]:
    """Render every persisted depth/phi probe, isolating one failed plot from the next."""
    from vfe3.viz.figures import (
        plot_beta_channel_decomposition,
        plot_context_sensitivity,
        plot_phi_numerics_reference,
        set_publication_style,
    )

    set_publication_style()

    saved_probes = (
        (
            "beta_channel_decomposition.json",
            "beta_channel_decomposition.png",
            plot_beta_channel_decomposition,
        ),
        ("context_sensitivity.json", "context_sensitivity.png", plot_context_sensitivity),
        ("phi_numerics.json", "phi_numerics_reference.png", plot_phi_numerics_reference),
    )
    written: List[Path] = []
    for source_name, figure_name, plotter in saved_probes:
        source_path = run_dir / source_name
        if not source_path.is_file():
            continue
        try:
            record = json.loads(source_path.read_text(encoding="utf-8"))
            final_path = run_dir / figure_name
            with _unique_sibling_output_temp(final_path) as temporary_path:
                figure = plotter(record, path=str(temporary_path))
                if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
                    raise RuntimeError(f"saved probe figure {figure_name!r} was not written")
                _atomic_replace(final_path, temporary_path)
            if figure is not None:
                from matplotlib import pyplot as plt
                plt.close(figure)
            written.append(final_path)
        except Exception as exc:
            from vfe3.viz.figures import FigureSkipped

            if isinstance(exc, FigureSkipped):
                logger.info("saved probe figure %s skipped (%s)", figure_name, exc)
                continue
            logger.warning(
                "saved probe figure %s failed (%s); remaining figures continue",
                figure_name,
                exc,
            )
    return written


def _render_persisted_run_figures(
    run_dir: Path,
    cfg:     'VFE3Config | SimpleNamespace',
    losses:  'List[float] | None',
    logger:  logging.Logger,

    *,
    losses_start_step: int = 0,
) -> None:
    r"""Rebuild history dashboards and saved probes from durable run artifacts.

    ``losses`` is available during end-of-training finalization. On-demand report mode passes
    ``None`` because historical strict-opt-out runs persist only cadence-sampled ``train_loss`` rows,
    not the original full-resolution per-step loss list; every other history dashboard remains
    reproducible from ``metrics.csv``. ``losses_start_step`` is the absolute step count before
    ``losses[0]`` (nonzero on a resumed run), so the loss curve shares the history's step axis.
    """
    history = _history_from_csv(run_dir / "metrics.csv")
    artifacts = SimpleNamespace(run_dir=run_dir, cfg=cfg, history=history)
    _render_saved_probe_figures(run_dir, logger)
    _save_figures(artifacts, losses, logger, losses_start_step=losses_start_step)


def _render_attention_request(request: Mapping[str, object], run_dir: Path) -> None:
    """Render one periodic beta map bundle entirely inside this worker."""
    channel = request.get("channel")
    if channel != "beta":
        raise ValueError(f"unsupported attention channel {channel!r}")
    step = request.get("step")
    if type(step) is not int or step < 0:
        raise ValueError(f"attention step must be a nonnegative integer, got {step!r}")
    maps_path_value = request.get("maps_path")
    if not isinstance(maps_path_value, str):
        raise ValueError("attention request maps_path must be a string")
    maps = torch.load(
        Path(maps_path_value).resolve(strict=True),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(maps, torch.Tensor):
        raise ValueError("attention request payload must be a tensor")
    if maps.ndim != 4:
        raise ValueError(f"{channel} attention request must have rank 4, got {maps.ndim}")

    from vfe3.viz import figures as figs

    figs.set_publication_style()
    # Claim the output directory ONCE. prepare_owned_output_child re-establishes ownership of the
    # child on every call, which sweeps in-flight temporaries -- calling it a second time while a
    # sibling figure was mid-write deleted that temp out from under the writer.
    attention_dir = prepare_owned_output_child(run_dir, "attention", role="attention figure")
    positive = maps[maps > 0]
    vmax = float(positive.max()) if positive.numel() else 1.0
    vmin = float(positive.min()) if positive.numel() else vmax * 1e-3
    n_layers, n_heads = maps.shape[0], maps.shape[1]
    for layer in range(n_layers):
        for head in range(n_heads):
            filename = f"step_{step}_layer{layer}_head{head}.png"
            title = f"Attention (step {step:,}), layer {layer} head {head}"
            matrix = maps[layer, head]
            final_path = attention_dir / filename
            with _unique_sibling_output_temp(final_path) as temporary_path:
                figure = figs.plot_attention_heatmap(
                    matrix,
                    log=True,
                    vmin=vmin,
                    vmax=vmax,
                    title=title,
                    path=str(temporary_path),
                )
                if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
                    raise RuntimeError(f"attention figure {filename!r} was not written")
                _atomic_replace(final_path, temporary_path)
            figs.plt.close(figure)


def _legacy_ablation_scope_files(scope: str) -> set[str]:
    """Return every pre-manifest filename owned by one ablation figure scope."""
    if scope == "__sensitivity__":
        return {"sensitivity_summary.png"}
    suffixes = {
        "",
        "_seed_aggregate",
        "_rank_collapse",
        "_wallclock_convergence",
        "_gauge_bars",
        "_ppl_equiv",
        "_kappa_dispersion",
        "_residual_drift",
        "_mu_precond",
        "_renyi_saturation",
        "_extrapolation",
        "_ablation_forest",
        "_lr_grid_heatmap",
    }
    return (
        {f"{scope}{suffix}.png" for suffix in suffixes}
        | {"ablation_forest.png", "lr_grid_heatmap.png"}
    )


def _validated_ablation_manifest(path: Path) -> Dict[str, Dict[str, object]]:
    """Load one manifest whose entries can authorize only known ablation figure names."""
    if not path.exists():
        return {}
    if path_is_reparse_point(path) or not path.is_file():
        raise ValueError("ablation figure manifest must be a regular file")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("ablation figure manifest is unreadable") from exc
    if (not isinstance(manifest, dict)
            or type(manifest.get("schema_version")) is not int
            or manifest.get("schema_version") != 1
            or not isinstance(manifest.get("scopes"), dict)):
        raise ValueError("ablation figure manifest has an unsupported schema")

    validated: Dict[str, Dict[str, object]] = {}
    for stored_scope, entry in manifest["scopes"].items():
        if not isinstance(stored_scope, str):
            raise ValueError("ablation figure manifest scope names must be strings")
        portable_path_component_key(stored_scope, field="ablation figure manifest scope")
        if not isinstance(entry, dict) or not isinstance(entry.get("files"), list):
            raise ValueError(f"ablation figure manifest scope {stored_scope!r} is malformed")
        allowed = _legacy_ablation_scope_files(stored_scope)
        files = entry["files"]
        if (any(not isinstance(name, str) or name not in allowed for name in files)
                or len(files) != len(set(files))):
            raise ValueError(
                f"ablation figure manifest scope {stored_scope!r} contains an unowned filename"
            )
        validated[stored_scope] = {
            "files": list(files),
            **({"updated_at": entry["updated_at"]}
               if isinstance(entry.get("updated_at"), str) else {}),
        }
    return validated


def _render_ablation_request(request: Mapping[str, object], output_dir: Path) -> None:
    r"""Render and stage one ablation scope before replacing its manifest-owned outputs."""
    scope = request.get("scope")
    if not isinstance(scope, str) or not scope or Path(scope).name != scope:
        raise ValueError(f"invalid ablation figure scope {scope!r}")
    portable_path_component_key(scope, field="ablation figure scope")
    invalidate = request.get("invalidate", False)
    if type(invalidate) is not bool:
        raise ValueError("ablation figure invalidate must be an exact boolean")

    import ablation

    figure_dir = ablation._prepare_owned_output_child(
        output_dir,
        "figures",
        role="ablation figure",
    )
    stage = Path(tempfile.mkdtemp(prefix=".ablation_figure_stage_", dir=str(output_dir)))
    try:
        if invalidate:
            produced = []
        elif scope == "__sensitivity__":
            cohort_identity = request.get("cohort_identity")
            if not isinstance(cohort_identity, Mapping):
                raise ValueError("sensitivity figure request requires a comparison cohort")
            ablation._plot_sensitivity(
                output_dir,
                stage,
                cohort_identity=cohort_identity,
            )
        else:
            sweep_dir = (output_dir / scope).resolve(strict=True)
            if sweep_dir.parent != output_dir or not sweep_dir.is_dir():
                raise ValueError(f"ablation sweep scope escapes output directory: {scope!r}")
            ablation._render_sweep_figures(sweep_dir, stage)

        if not invalidate:
            produced = sorted(
                path.name
                for path in stage.iterdir()
                if path.is_file() and path.suffix.lower() == ".png"
            )
        allowed_current = _legacy_ablation_scope_files(scope)
        if any(name not in allowed_current for name in produced):
            raise ValueError(
                f"ablation renderer produced a filename not owned by scope {scope!r}"
            )
        manifest_path = figure_dir / "ablation_figure_manifest.json"
        scopes = _validated_ablation_manifest(manifest_path)
        prior_entry = scopes.get(scope, {})
        prior_files = set(prior_entry.get("files", []))
        # The first manifest-backed render/invalidation must also retire every deterministic output
        # name from pre-manifest releases, including specialized figures and sensitivity summaries.
        prior_files.update(_legacy_ablation_scope_files(scope))
        other_owned = set()
        for other_scope, entry in scopes.items():
            if other_scope == scope:
                continue
            other_owned.update(entry["files"])

        for name in produced:
            os.replace(stage / name, figure_dir / name)
        for name in prior_files - set(produced) - other_owned:
            (figure_dir / name).unlink(missing_ok=True)
        scopes[scope] = {
            "files": produced,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _write_json_atomic(manifest_path, {
            "schema_version": 1,
            "scopes": scopes,
        })
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main() -> int:
    """Execute one validated figure request and return a process exit code."""
    raw_request = os.environ.get("VFE3_FIGURE_REQUEST")
    if not raw_request:
        raise RuntimeError("VFE3_FIGURE_REQUEST is required")
    request_path = Path(raw_request).resolve(strict=True)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(request, Mapping):
        raise ValueError("figure request must be a JSON object")

    # PDF font subsetting emits per-glyph INFO chatter; keep library warnings visible.
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)

    mode = request.get("mode", "finalize")
    if mode == "finalize":
        (
            run_dir_value,
            losses,
            losses_start_step,
            generate_publication,
            batches_path,
            model_bundle_path,
            device,
            max_tokens,
            generate_embedding_figures,
        ) = _validated_finalize_request(request)
        run_dir = Path(run_dir_value).resolve(strict=True)
        cfg = _load_worker_config(run_dir)

        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logger = logging.getLogger(__name__)
        _render_persisted_run_figures(
            run_dir, cfg, losses, logger, losses_start_step=losses_start_step)

        if generate_publication:
            from vfe3.viz.report import generate_figures

            loader = (
                torch.load(Path(batches_path), map_location="cpu", weights_only=True)
                if batches_path is not None
                else None
            )
            generate_figures(
                run_dir,
                checkpoint_path=(Path(model_bundle_path) if model_bundle_path is not None else None),
                split="test",
                max_tokens=max_tokens,
                generate_embedding_figures=generate_embedding_figures,
                loader=loader,
                device=torch.device(device),
                logger=logger,
            )
        return 0

    run_dir_value = request.get("run_dir")
    if not isinstance(run_dir_value, str):
        raise ValueError("figure request run_dir must be a string")
    run_dir = Path(run_dir_value).resolve(strict=True)
    if mode == "report":
        result_path_value = request.get("result_path")
        if not isinstance(result_path_value, str):
            raise ValueError("report figure request requires a result_path string")
        result_path = Path(result_path_value).resolve(strict=True)
        if result_path.parent != run_dir:
            raise ValueError("report figure result path must be owned by the run directory")
        split = request.get("split")
        device = request.get("device")
        max_sequences = request.get("max_sequences")
        n_e_steps = request.get("n_e_steps")
        generate_embedding_figures = request.get("generate_embedding_figures")
        if not isinstance(split, str) or not split:
            raise ValueError("report figure split must be a non-empty string")
        if not isinstance(device, str) or not device:
            raise ValueError("report figure device must be a non-empty string")
        if max_sequences is not None and (type(max_sequences) is not int or max_sequences <= 0):
            raise ValueError("report figure max_sequences must be a positive integer or null")
        if n_e_steps is not None and (type(n_e_steps) is not int or n_e_steps <= 0):
            raise ValueError("report figure n_e_steps must be a positive integer or null")
        if type(generate_embedding_figures) is not bool:
            raise ValueError(
                "report figure generate_embedding_figures must be an exact boolean")

        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logger = logging.getLogger(__name__)
        from vfe3.viz.report import generate_figures

        try:
            paths = generate_figures(
                run_dir,
                device=torch.device(device),
                split=split,
                max_sequences=max_sequences,
                generate_embedding_figures=generate_embedding_figures,
                prepare_saved_probes=True,
                n_e_steps=n_e_steps,
                logger=logger,
            )
        finally:
            # The history dashboards and saved probes need only metrics.csv and the lenient config, so they are
            # rebuilt even when the strict model replay above refuses this run's saved config; after a
            # successful replay they also render the probes it persisted (audit 2026-10-16 F84).
            _render_persisted_run_figures(
                run_dir,
                _load_worker_config(run_dir),
                None,
                logger,
            )
        _write_json_atomic(result_path, {"paths": [str(path.resolve()) for path in paths]})
        return 0
    if mode == "attention":
        _render_attention_request(request, run_dir)
        return 0
    if mode == "ablation":
        _render_ablation_request(request, run_dir)
        return 0
    raise ValueError(f"unsupported figure worker mode {mode!r}")


if __name__ == "__main__":
    raise SystemExit(main())
