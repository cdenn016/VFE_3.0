r"""Training-time diagnostics remediation (audit 2026-10-31, package train-diagnostics).

Tiny CPU models only (K <= 8, two heads, at most three layers, vocabulary at most 11, one thread). Covers the S1 panel
D factorization denominator off the factorizing routes (training-diagnostics-1), the wrappers of a tied prototype
bank (block-mlp-norm-1), the per-token eigenvalue-collapse alarm (training-diagnostics-2), the location columns of a
multi-batch held-out average (training-diagnostics-4), the eval-cadence attention maps read from the held-out snapshot
(training-diagnostics-7), the block-variance pass without the relational capture (training-diagnostics-8), the
routing-holonomy prior reference formed once per evaluation (training-diagnostics-9), the window formulas of the
empty-epoch message (data-pipeline-5), and a bitwise check that the logged and evaluated diagnostics leave the tiny
live route's training trajectory unchanged.
"""
from __future__ import annotations

import csv
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.relational_scoring as rs
import vfe3.run_artifacts as ra
import vfe3.train as T
from tests.profiles import live_config, test_config
from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import DiagnosticSnapshot, VFEModel

CPU = torch.device("cpu")
LOGGER = logging.getLogger("audit-20261031-train-diagnostics")
# A location of an extreme (the layer, head or token position attaining it), not a magnitude.
LOCATION_COLUMN = re.compile(r"_arg(?:max|min)_(?:layer|head|position)$")


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _live(**overrides: Any) -> VFE3Config:
    r"""The tiny launcher route with the switches these tests read pinned, so a launcher edit cannot move them: the
    projected predictive kernel readout on the diagonal comparison route (comparison_invariant RoPE at fraction 1,
    noisy-key routing, the gram factorization)."""
    values: Dict[str, Any] = dict(
        vocab_size=11, max_seq_len=6, n_layers=2, attention_mode="projected_predictive", predictive_readout="kernel",
        rope_mode="comparison_invariant", rope_fraction=1.0, relational_routing="noisy_key", relational_key_noise=1.0,
        projected_factorization="gram")
    values.update(overrides)
    return live_config(**values)


def _model(cfg: VFE3Config, seed: int = 0) -> VFEModel:
    torch.manual_seed(seed)
    return VFEModel(cfg)


def _windows(cfg: VFE3Config, count: int = 8, seed: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.vocab_size, (count, cfg.max_seq_len), generator=generator)     # (W, N)


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _same_rows(rows: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]]) -> bool:
    return len(rows) == len(expected) and all(
        list(row) == list(other) and all(_same(row[key], other[key]) for key in row)
        for row, other in zip(rows, expected))


# ---------------------------------------------------------------------------------- training-diagnostics-1 (A004)

def _counted_capture(
    model:       VFEModel,
    windows:     torch.Tensor,                       # (W, N)
    monkeypatch: pytest.MonkeyPatch,
) -> Tuple[Dict[str, int], int]:
    r"""The per-evaluation numerics columns of ``windows`` and the ``_cholesky_status`` calls of their capture pass."""
    calls = [0]
    real = rs._cholesky_status

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(rs, "_cholesky_status", counting)
    accumulator: Dict[str, Any] = {}
    T._residual_and_attention_rows(model, windows, 1, CPU, counters_out=accumulator)
    monkeypatch.setattr(rs, "_cholesky_status", real)
    columns, _ = T._numerics_eval_columns(accumulator, 1)
    return columns, calls[0]


def _static_route_verdicts(model: VFEModel) -> List[bool]:
    r"""Per layer, whether :func:`vfe3.gradients.relational.factorizes_projected_covariances` says its router factors."""
    from vfe3.gradients.relational import factorizes_projected_covariances
    return [factorizes_projected_covariances(ops) for ops in model.relational_operators_by_layer]


def test_diagonal_comparison_route_counts_no_factorization(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _live()
    model = _model(cfg)
    columns, status_calls = _counted_capture(model, _windows(cfg), monkeypatch)
    assert status_calls == 0                                                 # the diagonal route factors nothing
    assert columns["factorization_count"] == 0 and columns["factorization_fallback_count"] == 0
    assert 0 <= columns["cap_active_count"] <= columns["cap_entry_count"] and columns["cap_entry_count"] > 0
    reason = T._numerics_counter_skip_reasons(model)["factorization"]
    assert reason is not None and "diagonal comparison route" in reason
    assert _static_route_verdicts(model) == [False] * cfg.n_layers


def test_dense_route_counts_every_query_and_key_factorization(monkeypatch: pytest.MonkeyPatch) -> None:
    # The one-knob control of the audit probe: the certified factorization keeps the dense route.
    cfg = _live(projected_factorization="gram_certified", relational_routing="gaussian_kl")
    model = _model(cfg)
    windows = _windows(cfg)
    columns, status_calls = _counted_capture(model, windows, monkeypatch)
    assert status_calls > 0
    W, N, H, L = windows.shape[0], cfg.max_seq_len, cfg.n_heads, cfg.n_layers
    assert columns["factorization_count"] == 2 * W * N * H * L               # query and key role per layer
    assert 0 <= columns["factorization_fallback_count"] <= columns["factorization_count"]
    assert T._numerics_counter_skip_reasons(model)["factorization"] is None
    assert _static_route_verdicts(model) == [True] * cfg.n_layers


def test_mean_distance_route_counts_no_factorization(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _live(relational_routing="mean_distance")
    model = _model(cfg)
    columns, status_calls = _counted_capture(model, _windows(cfg), monkeypatch)
    assert status_calls == 0                                                 # the cost reads no covariance
    assert columns["factorization_count"] == 0 and columns["factorization_fallback_count"] == 0
    assert columns["cap_entry_count"] > 0
    reason = T._numerics_counter_skip_reasons(model)["factorization"]
    assert reason is not None and "mean_distance" in reason
    assert _static_route_verdicts(model) == [False] * cfg.n_layers


# The minimal agent applies the comparison rotation by exact conditioning, so it takes rope_mode='comparison'.
MINIMAL_AGENT = dict(predictive_readout="minimal_agent", relational_routing="gaussian_kl", attention_residual=False,
                     rope_mode="comparison")


def test_minimal_agent_counts_every_recorded_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    # The minimal agent factors its query and record covariances on every route; its capture carries no router count.
    cfg = _live(**MINIMAL_AGENT)
    model = _model(cfg)
    windows = _windows(cfg)
    columns, _ = _counted_capture(model, windows, monkeypatch)
    W, N, H, L = windows.shape[0], cfg.max_seq_len, cfg.n_heads, cfg.n_layers
    assert columns["factorization_count"] == 2 * W * N * H * L
    assert T._numerics_counter_skip_reasons(model)["factorization"] is None
    assert _static_route_verdicts(model) == [True] * cfg.n_layers


# ---------------------------------------------------------------------------------- block-mlp-norm-1 (A008)

B_P, N_P, V_P, L_P = 2, 4, 7, 3


def _prototype_cfg(tied: bool) -> VFE3Config:
    return test_config(vocab_size=V_P, embed_dim=4, n_heads=2, max_seq_len=N_P, n_layers=L_P, batch_size=B_P,
                       attention_mode="projected_predictive", use_block_mlp=True, block_mlp_mode="prototype_agents",
                       block_mlp_tie_layers=tied, block_mlp_expansion=2, block_mlp_dropout=0.0,
                       warmup_steps=0, max_steps=2, log_interval=1, eval_interval=0, checkpoint_interval=0)


@pytest.mark.parametrize("tied", [False, True])
def test_prototype_wrappers_attribute_each_call_to_its_layer(tied: bool) -> None:
    model = _model(_prototype_cfg(tied)).eval()
    tokens = torch.randint(0, V_P, (B_P, N_P), generator=torch.Generator().manual_seed(1))
    capture: dict = {"diagnostic": {}}
    with torch.no_grad():
        model.forward_beliefs(tokens, capture=capture)
    inputs = capture["diagnostic"]["layer_block_mlp_inputs"]                # per layer, the belief the MLP reads
    features: Dict[str, Any] = {}
    wrapped, begin_pass = T._wrap_prototype_blocks(model, features)
    try:
        assert len(wrapped) == len({id(mlp) for mlp in wrapped}) == (1 if tied else L_P)
        with torch.no_grad():
            for _ in range(2):                                              # the call count restarts per pass
                begin_pass()
                model.forward_beliefs(tokens)
    finally:
        for mlp in wrapped:                                                 # one wrapper per distinct bank
            del mlp.forward_moments
    assert all("forward_moments" not in vars(mlp) for mlp in model.block_mlps)
    sums = features["prototype"]
    assert sorted(sums) == list(range(L_P))
    expected = []
    for layer, belief in enumerate(inputs):
        with torch.no_grad():
            weights = model.block_mlps[layer].weights_gain_and_staleness(belief.mu, belief.sigma)[0]   # (B, N, M)
        expected.append(2 * weights.double().reshape(-1, weights.shape[-1]).sum(dim=0))
        assert sums[layer]["tokens"] == 2 * B_P * N_P
        assert torch.allclose(sums[layer]["weights"], expected[layer], rtol=1e-6, atol=1e-9), layer
    assert any(not torch.allclose(expected[0], other) for other in expected[1:])   # the layers' inputs differ


def test_tied_prototype_bank_trains_through_its_log_steps(tmp_path: Path) -> None:
    cfg = _prototype_cfg(True)
    model = _model(cfg, seed=5)
    loader = DataLoader(TokenWindows(torch.arange(41) % V_P, N_P), batch_size=B_P, shuffle=False, drop_last=True)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model, save_model_artifacts=False)
    torch.manual_seed(11)
    T.train(model, loader, cfg, n_steps=2, log_interval=1, eval_interval=0, device=CPU, logger=LOGGER,
            artifacts=artifacts, generate_samples=False)
    with (artifacts.run_dir / "diagnostics" / "prototype_templates_train.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(int(r["step"]), int(r["layer"])) for r in rows] == [(s, l) for s in (1, 2) for l in range(L_P)]
    assert all(int(r["n_tokens"]) == B_P * N_P for r in rows)
    assert all("forward_moments" not in vars(mlp) for mlp in model.block_mlps)


# ---------------------------------------------------------------------------------- training-diagnostics-2 (A068)

def _eigenvalue_records(model: VFEModel, tokens: torch.Tensor) -> Tuple[dict, List[Dict[str, Any]]]:
    d = model.diagnostics(tokens)
    records: List[Dict[str, Any]] = []
    T._warn_eigenvalue_collapse(d, set(), LOGGER, belief_layer=model.cfg.n_layers - 1, sink=records.append,
                                layer_extremes=model.last_belief_layer_extremes)
    return d, [record for record in records if record["kind"] == "belief_eigenvalue_ratio"]


def test_eigenvalue_alarm_reads_each_covariance_not_the_batch_range() -> None:
    # Passthrough covariance: every layer's converged covariance is the input token's prior variance row.
    cfg = _live(family="gaussian_diagonal", prior_covariance_table="diagonal", predictive_covariance="passthrough",
                block_norm_covariance="passthrough", block_mlp_covariance="passthrough")
    model = _model(cfg)
    K = cfg.embed_dim
    tokens = torch.tensor([[3, 1, 2, 4, 5, 6], [7, 8, 3, 9, 10, 1]])
    table = model.prior_bank.sigma_log_embed                                 # (V, K) log variances
    with torch.no_grad():
        table.fill_(math.log(15.0))
        table[3].fill_(math.log(1e-3))                  # every covariance isotropic, scales four decades apart
    d, records = _eigenvalue_records(model, tokens)
    extremes = model.last_belief_layer_extremes
    assert len(extremes["cond_max"]) == cfg.n_layers
    assert max(extremes["cond_max"]) == pytest.approx(d["belief_cond_max_layer_max"])
    assert d["belief_cond_max_layer_max"] == pytest.approx(1.0, rel=1e-5)
    assert min(extremes["lam_min"]) / max(extremes["lam_max"]) < T.EIGENVALUE_COLLAPSE_RATIO   # the old trigger
    assert records == []
    with torch.no_grad():
        table[3].fill_(math.log(15.0))
        table[3, : K // 2] = math.log(1e-3)             # one anisotropic covariance, condition 1.5e4
    d, records = _eigenvalue_records(model, tokens)
    assert sorted(record["layer"] for record in records) == list(range(cfg.n_layers))
    assert all(record["value"] == pytest.approx(1e-3 / 15.0, rel=1e-2) for record in records)
    assert all(record["value"] < T.EIGENVALUE_COLLAPSE_RATIO for record in records)


# ---------------------------------------------------------------------------------- training-diagnostics-4 (A101)

def test_multi_batch_held_out_locations_are_not_averaged() -> None:
    for seed in range(12):
        cfg = _live(n_layers=3, phi_scale=0.3)
        model = _model(cfg, seed)
        generator = torch.Generator().manual_seed(100 + seed)
        loader = [(torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len), generator=generator),
                   torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len), generator=generator)) for _ in range(3)]
        singles = [T._val_diagnostics(model, [batch], CPU, batches=1).metrics for batch in loader]
        keys = sorted(key for key in singles[0] if LOCATION_COLUMN.search(key))
        if any(len({s[key] for s in singles if math.isfinite(s[key])}) > 1 for key in keys):
            break
    else:
        pytest.fail("no seed gave batches whose extremes sit at different locations")
    averaged = T._val_diagnostics(model, loader, CPU, batches=3).metrics
    assert keys and set(keys) <= set(averaged)
    for key in keys:
        finite = [s[key] for s in singles if math.isfinite(s[key])]
        if not finite:
            assert math.isnan(averaged[key]), key
        else:                                        # the first batch that measured it, an integer location
            assert averaged[key] == finite[0] and averaged[key] == int(averaged[key]), key
    magnitude = "val_relational_projected_cov_min_eigenvalue_layer_min"   # magnitudes stay batch means
    values = [s[magnitude] for s in singles if math.isfinite(s[magnitude])]
    assert values and averaged[magnitude] == pytest.approx(sum(values) / len(values))


# ---------------------------------------------------------------------------------- training-diagnostics-7 (A158)

class _MapArtifacts:
    r"""The ``save_attention_maps`` seam of :class:`vfe3.run_artifacts.RunArtifacts`, recorded (no figure work)."""

    def __init__(self) -> None:
        self.saved: List[Tuple[int, torch.Tensor]] = []

    def save_attention_maps(self, step: int, maps: torch.Tensor, logger: Any = None) -> None:
        self.saved.append((step, maps))


def test_eval_attention_maps_read_the_held_out_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _live()
    model = _model(cfg)
    generator = torch.Generator().manual_seed(5)
    batch = (torch.randint(0, cfg.vocab_size, (3, cfg.max_seq_len), generator=generator),
             torch.randint(0, cfg.vocab_size, (3, cfg.max_seq_len), generator=generator))
    held_out = T._val_diagnostics(model, [batch], CPU, batches=1)
    replay = model.attention_maps(held_out.token_ids, score_at="entry")       # what the replay computed
    calls = [0]
    real = VFEModel.forward_beliefs

    def counting(self: VFEModel, *args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        return real(self, *args, **kwargs)

    monkeypatch.setattr(VFEModel, "forward_beliefs", counting)
    artifacts = _MapArtifacts()
    T._save_eval_attention_maps(held_out.token_ids, model, artifacts, LOGGER, step=3, snapshot=held_out.snapshot)
    assert calls[0] == 0                                                     # no second held-out forward
    (step, maps), = artifacts.saved
    assert step == 3
    assert torch.equal(maps, held_out.snapshot.beta_maps) and torch.equal(maps, replay)


def test_training_eval_hands_the_held_out_snapshot_to_the_maps(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _live()
    model = _model(cfg)
    seen: List[Tuple[torch.Tensor, Any]] = []
    monkeypatch.setattr(T, "_save_eval_attention_maps",
                        lambda token_ids, *args, **kwargs: seen.append((token_ids, kwargs.get("snapshot"))))
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    N = cfg.max_seq_len
    train_loader = DataLoader(TokenWindows(torch.arange(60) % cfg.vocab_size, N), batch_size=cfg.batch_size,
                              shuffle=False, drop_last=True)
    val_loader = DataLoader(TokenWindows((torch.arange(40) * 3) % cfg.vocab_size, N, pad_final=True),
                            batch_size=2, shuffle=False)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model, save_model_artifacts=False)
    torch.manual_seed(11)
    T.train(model, train_loader, cfg, n_steps=2, log_interval=0, eval_interval=2, val_loader=val_loader,
            device=CPU, logger=LOGGER, artifacts=artifacts, generate_samples=False)
    (token_ids, snapshot), = seen
    assert isinstance(snapshot, DiagnosticSnapshot) and torch.equal(snapshot.token_ids, token_ids)


# ---------------------------------------------------------------------------------- training-diagnostics-8 (A159)

class _RowArtifacts:
    r"""The ``append_diagnostic_rows`` seam of :class:`vfe3.run_artifacts.RunArtifacts`, recorded."""

    def __init__(self) -> None:
        self.written: List[Tuple[str, List[str], List[Dict[str, Any]]]] = []

    def append_diagnostic_rows(self, name: str, fields: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
        self.written.append((name, list(fields), list(rows)))


@pytest.mark.parametrize("route", [{}, MINIMAL_AGENT], ids=["kernel", "minimal_agent"])
def test_block_variance_pass_skips_the_relational_capture(route: Dict[str, Any],
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _live(**route)
    model = _model(cfg)
    windows = _windows(cfg)
    expected: Dict[Tuple[int, str], List[Any]] = {}
    model.eval()
    with torch.no_grad():
        for chunk in windows.split(T.DIAGNOSTIC_CHUNK_WINDOWS):
            capture: dict = {"diagnostic": {}}
            model.forward_beliefs(chunk, capture=capture)                    # with the relational capture
            assert capture["diagnostic"]["layer_conditional_captures"]
            T._collect_block_variances(capture["diagnostic"], expected)
    model.train()
    flags: List[bool] = []
    real = VFEModel.forward_beliefs

    def spy(self: VFEModel, token_ids: torch.Tensor, **kwargs: Any) -> Any:
        flags.append(bool(kwargs.get("_relational_capture", True)))
        return real(self, token_ids, **kwargs)

    monkeypatch.setattr(VFEModel, "forward_beliefs", spy)
    artifacts = _RowArtifacts()
    T._append_block_variance(model, artifacts, windows, 4, CPU)
    assert flags == [False] * math.ceil(windows.shape[0] / T.DIAGNOSTIC_CHUNK_WINDOWS)
    (name, fields, rows), = artifacts.written
    assert name == "block_variance.csv" and fields == list(T.BLOCK_VARIANCE_FIELDS)
    assert _same_rows(rows, T._block_variance_rows(expected, 4))
    assert model.training


# ---------------------------------------------------------------------------------- training-diagnostics-9 (A160)

def test_holonomy_prior_reference_is_formed_once_per_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _live()
    model = _model(cfg)
    windows = _windows(cfg)                                                  # two chunks of four windows
    real = T.routing_triangle_defect_sums
    shapes: List[Tuple[int, ...]] = []

    def spy(beta: torch.Tensor, support: Any = None) -> torch.Tensor:
        shapes.append(tuple(beta.shape))
        return real(beta, support)

    monkeypatch.setattr(T, "routing_triangle_defect_sums", spy)
    holonomy: Dict[Any, Any] = {}
    T._residual_and_attention_rows(model, windows, 1, CPU, holonomy_out=holonomy)
    assert "error" not in holonomy
    chunks = math.ceil(windows.shape[0] / T.DIAGNOSTIC_CHUNK_WINDOWS)
    L, H, N = cfg.n_layers, cfg.n_heads, cfg.max_seq_len
    assert len(shapes) == chunks * L + 1                     # the router per (chunk, layer), the prior once
    assert shapes.count((1, H, N, N)) == 1
    # The prior's sums as the former per-(chunk, layer) evaluation formed them, over every window of the chunk.
    prior_logits = model._attention_log_prior(N, CPU).float()
    while prior_logits.dim() < 4:
        prior_logits = prior_logits.unsqueeze(0)                            # (1, 1|H, N, N)
    expected = torch.zeros(H, 3, dtype=torch.float64)
    for chunk in windows.split(T.DIAGNOSTIC_CHUNK_WINDOWS):
        shape = (chunk.shape[0], H, N, N)
        support = torch.isfinite(prior_logits).expand(shape)
        prior = torch.softmax(prior_logits, dim=-1).expand(shape)
        baseline = torch.where(support, prior, torch.zeros_like(prior))
        baseline = baseline / baseline.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        expected += real(baseline, support)
    assert float(expected[:, 0].min()) > 0.0                                 # the prior has strict triangles
    for layer in range(L):
        for head in range(H):
            got = torch.tensor(holonomy[(layer, head)][3:6], dtype=torch.float64)
            assert torch.allclose(got, expected[head], rtol=1e-12, atol=0.0), (layer, head)


# ---------------------------------------------------------------------------------- the training trajectory

def test_diagnostics_leave_the_live_training_trajectory_bitwise(tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Log rows (training-batch diagnostics, eigenvalue alarm, block-variance pass) and eval rows (held-out maps
    from the snapshot, counters, holonomy) on the tiny live route change neither the losses nor the weights."""
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    maps: List[int] = []
    monkeypatch.setattr(ra, "_run_attention_maps_isolated", lambda maps_, run_dir, logger, **kwargs: maps.append(
        int(kwargs["step"])))                                                # no figure child process
    cfg = _live(block_variance_at_log=True)
    N = cfg.max_seq_len

    def loaders() -> Tuple[DataLoader, DataLoader]:
        return (DataLoader(TokenWindows(torch.arange(60) % cfg.vocab_size, N), batch_size=cfg.batch_size,
                           shuffle=False, drop_last=True),
                DataLoader(TokenWindows((torch.arange(40) * 3) % cfg.vocab_size, N, pad_final=True), batch_size=2,
                           shuffle=False))

    silent_model = _model(cfg)
    torch.manual_seed(11)
    silent = T.train(silent_model, loaders()[0], cfg, n_steps=4, log_interval=0, eval_interval=0, device=CPU,
                     logger=LOGGER, generate_samples=False)
    logged_model = _model(cfg)
    train_loader, val_loader = loaders()
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, logged_model, save_model_artifacts=False)
    torch.manual_seed(11)
    logged = T.train(logged_model, train_loader, cfg, n_steps=4, log_interval=1, eval_interval=2,
                     val_loader=val_loader, device=CPU, logger=LOGGER, artifacts=artifacts, generate_samples=False)
    artifacts.wait_background()
    assert logged == silent                                                  # bitwise loss history
    for (name, a), (_, b) in zip(silent_model.state_dict().items(), logged_model.state_dict().items()):
        assert torch.equal(a, b), name
    assert sorted(maps) == [2, 4]                                            # one map set per evaluation
    with (artifacts.run_dir / "diagnostics" / "block_variance.csv").open(newline="") as handle:
        steps = {int(row["step"]) for row in csv.DictReader(handle)}
    assert steps == {1, 2, 3, 4}                                             # log-only rows 1, 3; eval rows 2, 4
    with (artifacts.run_dir / "metrics.csv").open(newline="") as handle:
        rows = {int(row["step"]): row for row in csv.DictReader(handle)}
    assert all(float(rows[step]["factorization_count"]) == 0.0 for step in (2, 4))   # the diagonal route factors none
    assert all(float(rows[step]["cap_entry_count"]) > 0.0 for step in (2, 4))


# ---------------------------------------------------------------------------------- data-pipeline-5 (A226)

def test_per_epoch_window_count_matches_the_closed_form() -> None:
    r"""Per-epoch origins fit :math:`\lfloor (T - L)/s\rfloor` complete windows once :math:`T \ge L + s` (one below
    that) at every origin, with distinct starts, and the count is tight (the grid of the audit probe)."""
    failures: List[Tuple[Any, ...]] = []
    for L in (1, 2, 3, 5, 8):
        for s in (1, 2, 3, 5, 8, 13):
            for n_tokens in range(L + 1, L + 6 * s + 3):
                ds = TokenWindows(torch.arange(n_tokens, dtype=torch.long), L, stride=s)
                ds.use_per_epoch_offsets()
                n, bound = len(ds), ds.offset_bound
                if n != ((n_tokens - L) // s if n_tokens >= L + s else 1):
                    failures.append(("closed_form", n_tokens, L, s, n))
                for origin in range(bound):
                    ds.window_offset = origin
                    starts = []
                    for i in range(n):
                        x, y = ds[i]
                        if (x.numel() != L or y.numel() != L or not torch.equal(y[:-1], x[1:])
                                or int(y[-1]) != int(x[-1]) + 1):
                            failures.append(("window", n_tokens, L, s, origin, i))
                        starts.append(int(x[0]))
                    if starts != [origin + i * s for i in range(n)]:
                        failures.append(("starts", n_tokens, L, s, origin))
                if (bound - 1) + n * s + L + 1 <= n_tokens:          # one more window would fit at the last origin
                    failures.append(("not_tight", n_tokens, L, s))
    assert failures == []


def test_empty_epoch_message_states_the_exact_minimum_stream() -> None:
    def stream(n_tokens: int, L: int, s: int, mode: str) -> TokenWindows:
        ds = TokenWindows(torch.arange(n_tokens, dtype=torch.long), L, stride=s)
        if mode == "per_epoch_random":
            ds.use_per_epoch_offsets()
        return ds

    checked = 0
    for L in (2, 3, 5, 8):
        for s in (1, 2, 3, 5, 8):
            for B in (1, 2, 3, 4):
                for mode in ("fixed", "per_epoch_random"):
                    for n_tokens in range(L + 1, L + (B + 2) * s + 3):
                        loader = DataLoader(stream(n_tokens, L, s, mode), batch_size=B, shuffle=False,
                                            drop_last=True)
                        if len(loader):
                            continue
                        message = T._empty_training_epoch_message(loader)
                        found = re.search(r"needs at least (\d+) tokens", message)
                        assert found is not None, message
                        minimum = int(found.group(1))
                        case = (mode, n_tokens, L, s, B, minimum)
                        assert n_tokens < minimum, case
                        assert len(DataLoader(stream(minimum, L, s, mode), batch_size=B, drop_last=True)) > 0, case
                        assert len(DataLoader(stream(minimum - 1, L, s, mode), batch_size=B,
                                              drop_last=True)) == 0, case         # the threshold is exact
                        assert f"seq_len={L}, stride={s}" in message and f"batch_size={B}" in message
                        checked += 1
    assert checked > 0
    plain = T._empty_training_epoch_message(DataLoader(list(range(1)), batch_size=2, drop_last=True))
    assert plain.endswith("give it at least one full batch")
