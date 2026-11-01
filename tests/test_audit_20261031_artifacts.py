r"""Remediation of the October 31, 2026 audit, package ``artifacts`` (run artifacts, resume and resources).

Global ids and lane ids: A002 artifacts-finalize-1 (the diagonal-moments kernel record), A005 artifacts-finalize-2
(source drift and cohort comparability from the executable sources), A056 artifacts-finalize-3 (the scheduler block
of the table Fisher and decode groups), A057 artifacts-finalize-4 (unique-storage byte counts), A058
artifacts-resume-1 and A079 gap-certified-factorization-routes-4 (the factorization-fallback contract string),
A062 artifacts-finalize-5 (``test_best_bits_per_token``), A074 artifacts-resume-2 (int and float list spellings),
A147 artifacts-finalize-6 (``final_model.pt`` feeds the figure worker), A148 artifacts-resume-3 with block-mlp-norm-8
(checkpoint embedding and alias-preserving host copies), A264 gap-mixed-precision-routes-7 (a missing scaler state),
A278 artifacts-finalize-9 (module docstring), A300 posthoc-measurement-10 (unique parameters read per token), A309
artifacts-finalize-7 (the theory-pure conjunction), A310 artifacts-finalize-8 (the FD check's scope), A327
artifacts-resume-4 (failed saves in the failure record), A328 artifacts-resume-5 (resume drift fingerprints) and A331
artifacts-finalize-10 (EMA evaluation in the validation-only finalizer).

Tiny CPU models and JSON fixtures only. No GPU and no Triton: the fused diagonal-moments dispatch of a CUDA run is
emulated by a stand-in that counts as the fused kernel and computes the torch moments.
"""
from __future__ import annotations

import json
import logging
import math
import subprocess
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest
import torch
from torch.utils.data import DataLoader

import multiseed_analysis as ms
import vfe3.gradients.relational as grel
import vfe3.relational_scoring as rs
import vfe3.run_artifacts as ra
import vfe3.train as T
from tests.conftest import model_process_globals_restored
from tests.profiles import tiny_config
from tests.test_multiseed import _write_full_run, _write_request
from vfe3.config import VFE3Config, serialize_config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.run_resources import (
    frozen_parameter_names, optimizer_state_bytes, parameter_breakdown, tied_block_mlp_aliases,
)

CPU = torch.device("cpu")
LOG = logging.getLogger("audit-20261031-artifacts")
_CLEAN_GIT = {"git_sha": "0" * 40, "git_dirty": False, "git_dirty_fingerprint": None}
# Comparison-invariant RoPE at rope_fraction 1 on heads of width 2: P(G) is diagonal, the closed-form diagonal route.
_DIAGONAL_ROUTE = dict(rope_mode="comparison_invariant", rope_fraction=1.0)
_TIED = dict(n_layers=2, use_block_mlp=True, block_mlp_tie_layers=True, norm_type_block="layernorm",
             layernorm_affine=True)


def _no_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ra, "_git_code_identity", lambda *args, **kwargs: dict(_CLEAN_GIT))


def _loaders(cfg: VFE3Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    V, N, B = cfg.vocab_size, cfg.max_seq_len, cfg.batch_size
    train = TokenWindows((torch.arange(61) * 5 + 2) % V, N)
    val = TokenWindows((torch.arange(29) * 3 + 1) % V, N, pad_final=True)
    test = TokenWindows((torch.arange(31) * 2 + 3) % V, N, pad_final=True)
    return (DataLoader(train, batch_size=B, shuffle=False, drop_last=True),
            DataLoader(val, batch_size=2, shuffle=False),
            DataLoader(test, batch_size=2, shuffle=False))


def _batch(cfg: VFE3Config) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(7)
    shape = (2, cfg.max_seq_len)
    return (torch.randint(0, cfg.vocab_size, shape, generator=g),
            torch.randint(0, cfg.vocab_size, shape, generator=g))


def _unique_state_bytes(model: torch.nn.Module) -> int:
    r"""Bytes of the parameters (each once) and the persistent buffers: the model state without its aliases."""
    persistent = set(model.state_dict())
    return (sum(p.numel() * p.element_size() for p in model.parameters())
            + sum(b.numel() * b.element_size() for name, b in model.named_buffers() if name in persistent))


@pytest.fixture
def _restore_diag_moments() -> Any:
    previous = rs.diagonal_moments_kernel()
    counts = (rs._DIAG_MOMENTS_TRITON_CALLS, rs._DIAG_MOMENTS_TORCH_CALLS, rs._PROJECTION_TILE_LOOPS)
    yield
    rs.set_diagonal_moments_kernel(previous)
    rs._DIAG_MOMENTS_TRITON_CALLS, rs._DIAG_MOMENTS_TORCH_CALLS, rs._PROJECTION_TILE_LOOPS = counts


def _emulate_fused_diag_moments(mp: pytest.MonkeyPatch) -> None:
    r"""Make the CPU model take the fused branch a CUDA run takes under ``relational_diag_moments_kernel='triton'``.

    The dispatch ignores the device, and the stand-in counts as the fused kernel while it returns the torch moments.
    """
    real_moments = rs.projected_moments

    def fused(mu_h: torch.Tensor, Sigma_h: torch.Tensor, inverse_blocks: torch.Tensor, S: torch.Tensor,
              dtype: torch.dtype, batch: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        previous = rs.set_diagonal_moments_kernel("torch")
        torch_calls = rs._DIAG_MOMENTS_TORCH_CALLS
        try:
            out = real_moments(mu_h, Sigma_h, inverse_blocks, S, diagonal_only=True)
        finally:
            rs.set_diagonal_moments_kernel(previous)
        rs._DIAG_MOMENTS_TORCH_CALLS = torch_calls
        rs._DIAG_MOMENTS_TRITON_CALLS += 1
        return out

    mp.setattr(rs, "_diagonal_moments_triton_active", lambda *tensors: rs._DIAG_MOMENTS_KERNEL == "triton")
    mp.setattr(rs, "_fused_diagonal_moments", fused)


# The finalized tiny run shared by the end-to-end checks: the live route's switches that the findings read (noisy_key
# on the diagonal comparison route with the fused moments kernel, the untied prior-bank decode, Fisher tables, a block
# MLP tied across two layers with an affine block LayerNorm), a checkpoint at step 2 and at the end, and no evaluation
# during training, so finalize's final validation selects the final weights.
_E2E = dict(**_DIAGONAL_ROUTE, relational_routing="noisy_key", relational_diag_moments_kernel="triton",
            decode_mode="diagonal_chunked", untie_decode_bank=True, m_table_update_mode="fisher", **_TIED,
            max_steps=3, warmup_steps=0, min_lr_frac=0.25, checkpoint_interval=2, checkpoint_at_end=True,
            log_interval=0, eval_interval=0)


@pytest.fixture(scope="module")
def finalized(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    previous_kernel = rs.diagonal_moments_kernel()
    saved_counts = (rs._DIAG_MOMENTS_TRITON_CALLS, rs._DIAG_MOMENTS_TORCH_CALLS)
    figure_calls: List[Dict[str, Any]] = []
    # A module-scoped fixture runs outside the per-test restore of tests/conftest.py, so it restores the process
    # globals that model construction and train() set itself; otherwise later modules in the worker inherit them.
    with pytest.MonkeyPatch.context() as mp, model_process_globals_restored():
        mp.setattr(ra, "_git_code_identity", lambda *args, **kwargs: dict(_CLEAN_GIT))
        mp.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
        mp.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: figure_calls.append(kwargs) or True)
        _emulate_fused_diag_moments(mp)
        try:
            cfg = tiny_config(**_E2E)
            torch.manual_seed(5)
            model = VFEModel(cfg)
            run_dir = tmp_path_factory.mktemp("finalized") / "run"
            artifacts = ra.RunArtifacts(run_dir, cfg, model)
            train_loader, val_loader, test_loader = _loaders(cfg)
            torch.manual_seed(11)
            losses = T.train(model, train_loader, cfg, n_steps=3, grad_clip=cfg.grad_clip, log_interval=0,
                             eval_interval=0, val_loader=val_loader, device=CPU, logger=LOG, artifacts=artifacts,
                             generate_samples=False)
            after_train = ra._realized_diag_moments_kernel()
            results = ra.finalize_run(model, artifacts, cfg, final_validation=T.evaluate(model, val_loader, device=CPU),
                                      train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                      losses=losses, device=CPU, logger=LOG)
            peak_table_lrs = T._table_fisher_learning_rates(T.build_optimizer(model, cfg))   # lr_t / lr_0 = 1
        finally:
            rs.set_diagonal_moments_kernel(previous_kernel)
            rs._DIAG_MOMENTS_TRITON_CALLS, rs._DIAG_MOMENTS_TORCH_CALLS = saved_counts

    def read(name: str) -> Dict[str, Any]:
        return json.loads((run_dir / name).read_text(encoding="utf-8"))

    return SimpleNamespace(cfg=cfg, model=model, artifacts=artifacts, run_dir=run_dir, after_train=after_train,
                           results=results, figure_calls=figure_calls, peak_table_lrs=peak_table_lrs,
                           summary=read("summary.json"), test_results=read("test_results.json"),
                           resources=read("resources.json"), research=read("research.json"))


# ------------------------------------------------------------------------------------------------ A002

def test_measured_forward_flops_leave_the_kernel_record_unchanged(_restore_diag_moments: Any) -> None:
    cfg = tiny_config(**_DIAGONAL_ROUTE, relational_routing="noisy_key")
    torch.manual_seed(0)
    model = VFEModel(cfg)
    rs.reset_diagonal_moments_triton_calls()
    with torch.no_grad():
        model(torch.zeros(1, cfg.max_seq_len, dtype=torch.long))
    assert rs.diagonal_moments_torch_calls() > 0                      # the route runs the diagonal moments
    rs._DIAG_MOMENTS_TRITON_CALLS, rs._DIAG_MOMENTS_TORCH_CALLS = 6, 0   # a run that trained on the fused kernels
    rs._PROJECTION_TILE_LOOPS = 3                                       # and ran three scorer tile loops
    ra._measured_forward_flops(model, cfg, tokens_seen=10)
    assert ra._realized_diag_moments_kernel() == {
        "diag_moments_kernel": "triton", "diag_moments_kernel_calls": {"triton": 6, "torch": 0},
        "relational_tile_loops": 3}
    assert rs.diagonal_moments_kernel() == "torch"                    # the selection is restored too


def test_summary_kernel_record_is_the_kernel_that_trained(finalized: SimpleNamespace) -> None:
    assert finalized.after_train["diag_moments_kernel"] == "triton"
    assert finalized.after_train["diag_moments_kernel_calls"]["torch"] == 0
    summary = finalized.summary
    assert summary["diag_moments_kernel"] == finalized.after_train["diag_moments_kernel"]
    assert summary["diag_moments_kernel_calls"]["torch"] == 0
    assert summary["scaling_point"]["flops_per_token_forward_measured"] > 0     # the measurement still ran


# ------------------------------------------------------------------------------------------------ A005

def _dirty(fingerprint: str) -> Dict[str, object]:
    return {"git_sha": "a" * 40, "git_dirty": True, "git_dirty_fingerprint": fingerprint * 64}


def _provenance_with(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, git_identities: List[Dict[str, object]],
                     code_identities: List[str]) -> Dict[str, Any]:
    r"""``provenance.json`` of a run whose Git and executable-source identities change as listed (start, end)."""
    git_calls, code_calls = iter(git_identities), iter(code_identities)
    monkeypatch.setattr(ra, "STRICT_CODE_IDENTITY", False)
    monkeypatch.setattr(ra, "_PROCESS_PACKAGE_CODE_IDENTITY", code_identities[0])
    monkeypatch.setattr(ra, "_git_code_identity", lambda *args, **kwargs: dict(next(git_calls)))
    monkeypatch.setattr(ra, "_package_code_identity", lambda: next(code_calls))
    cfg = tiny_config()
    model = VFEModel(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model, save_model_artifacts=False)
    ra._write_provenance(artifacts, cfg, model, LOG)
    return json.loads((tmp_path / "run" / "provenance.json").read_text(encoding="utf-8"))


def test_a_non_source_edit_during_the_run_keeps_the_source_identity_stable(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # An untracked notes file or a tracked log changed between the two probes; the executable sources did not.
    prov = _provenance_with(tmp_path, monkeypatch, [_dirty("a"), _dirty("b")], ["c" * 64, "c" * 64])
    assert prov["git_identity_status"] == "drifted" and prov["git_identity_drift"] is True    # descriptive
    assert prov["code_identity_drift"] is False
    assert prov["source_identity_status"] == "stable" and prov["source_identity_drift"] is False
    assert prov["source_identity_basis"] == "code_identity_sha256"


def test_an_executable_source_edit_still_drifts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prov = _provenance_with(tmp_path, monkeypatch, [dict(_CLEAN_GIT), dict(_CLEAN_GIT)], ["c" * 64, "d" * 64])
    assert prov["git_identity_status"] == "stable"
    assert prov["source_identity_status"] == "drifted" and prov["source_identity_drift"] is True


def test_paper_source_row_reads_the_executable_source_drift() -> None:
    from vfe3.paper.tables import _source_items

    def labels(provenance: Dict[str, object]) -> List[str]:
        run = SimpleNamespace(provenance=provenance, config_record={}, dropped_config_fields={})
        return [label for label, _ in _source_items(run)]

    base = {"git_sha": "a" * 40, "git_dirty": True, "final_git_identity": {"git_sha": "b" * 40}}
    row = "source drifted; final revision"
    # A record written before the fix: the Git tree changed, the executable sources did not.
    assert row not in labels({**base, "source_identity_drift": True, "code_identity_drift": False})
    assert row in labels({**base, "source_identity_drift": True, "code_identity_drift": True})
    assert row in labels({**base, "source_identity_drift": True})       # a record without the code field


def _two_seed_cohort(root: Path, code_identities: Tuple[str, str],
                     process_identities: Tuple[str, str] = ("c" * 64, "c" * 64)) -> None:
    for seed, code, process in zip((1, 2), code_identities, process_identities):
        run = root / f"run_s{seed}"
        _write_full_run(run, ppl=10.0 + seed, seed=seed)
        provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
        # Each seed ran on a different dirty tree (an untracked file changed between them).
        provenance.update(git_dirty=True, git_dirty_fingerprint=str(seed) * 64, code_identity_sha256=code,
                          process_code_identity_sha256=process)
        (run / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    _write_request(root, [1, 2])


def test_multiseed_cohort_compares_the_executable_source_identity(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(ms.CONFIG, "allow_mixed_code_identity", False)
    _two_seed_cohort(tmp_path / "same", ("c" * 64, "c" * 64))
    design = ms._requested_seed_design(tmp_path / "same")
    assert design["complete"] is True and design["accepted_seeds"] == [1, 2]
    assert design["code_identity"]["mixed"] is False and design["code_identity"]["n_distinct"] == 1
    per_seed = design["code_identity"]["per_seed"]
    assert per_seed["2"]["git_dirty_fingerprint"] == "2" * 64 and per_seed["2"]["code_identity_sha256"] == "c" * 64
    # Different executable sources, or a seed whose process ran code imported before a source edit, still split it.
    for name, codes, processes in (("different", ("c" * 64, "d" * 64), ("c" * 64, "d" * 64)),
                                   ("mixed_generation", ("c" * 64, "c" * 64), ("c" * 64, "e" * 64))):
        _two_seed_cohort(tmp_path / name, codes, processes)
        design = ms._requested_seed_design(tmp_path / name)
        assert design["complete"] is False and design["accepted_seeds"] == []
        assert design["code_identity"]["mixed"] is True


def test_multiseed_digest_names_the_executable_identity_of_a_mixed_cohort(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(ms.CONFIG, "allow_mixed_code_identity", True)
    _two_seed_cohort(tmp_path / "mixed", ("c" * 64, "d" * 64), ("c" * 64, "d" * 64))
    design = ms._requested_seed_design(tmp_path / "mixed")
    assert design["complete"] is True and design["code_identity"]["mixed"] is True   # admitted by the toggle
    path = tmp_path / "digest.md"
    ms._write_md(path, tmp_path / "mixed", [1, 2], {}, None, code_identity=design["code_identity"])
    text = path.read_text(encoding="utf-8")
    assert "2 distinct executable-source identities" in text and "code=`dddddddddddd`" in text


# ------------------------------------------------------------------------------------------------ A056

def test_scheduler_block_records_the_fisher_step_rate_and_the_decode_roles(finalized: SimpleNamespace) -> None:
    cfg = finalized.cfg
    groups = finalized.summary["scheduler"]["groups"]
    by_role = {group["role"]: group for group in groups}
    assert {"mu", "sigma", "phi", "decode_mu", "decode_sigma"} <= set(by_role)
    mean_table = by_role["mu"]
    assert mean_table["update"] == "table_fisher" and mean_table["table_fisher"] == "mu"
    assert mean_table["base_lr"] == pytest.approx(cfg.m_p_mu_lr)               # the AdamW rate schedules only
    assert mean_table["step_lr_base"] == pytest.approx(cfg.m_table_fisher_lr)
    assert mean_table["step_lr_base"] == pytest.approx(finalized.peak_table_lrs["mu"])   # metrics lr_mu at the peak
    assert mean_table["step_lr_floor"] == pytest.approx(cfg.min_lr_frac * cfg.m_table_fisher_lr)
    for role in ("decode_mu", "decode_sigma"):
        decode = by_role[role]
        assert decode["update"] == "adamw" and decode["table_fisher"] is None
        assert decode["step_lr_base"] == decode["base_lr"] and decode["step_lr_floor"] == decode["effective_floor"]
    assert by_role["decode_mu"]["base_lr"] == pytest.approx(cfg.resolved_hyperparameter("m_decode_mu_lr"))


def test_scheduler_metadata_validates_its_markers() -> None:
    with pytest.raises(ValueError, match="table_fisher"):
        ra._scheduler_metadata(min_lr_frac=0.0, base_lrs=(0.1, 0.2), group_roles=("mu", "sigma"),
                               table_fisher=("mu",), table_fisher_lr=0.5)
    frozen = ra._scheduler_metadata(min_lr_frac=0.5, base_lrs=(0.0,), group_roles=("mu",),
                                    table_fisher=("mu",), table_fisher_lr=0.5)["groups"][0]
    assert frozen["frozen"] and frozen["step_lr_base"] == 0.0 and frozen["step_lr_floor"] == 0.0


# ------------------------------------------------------------------------------------------------ A057

def test_resources_model_state_bytes_count_each_storage_once(finalized: SimpleNamespace) -> None:
    model = finalized.model
    per_key = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    expected = _unique_state_bytes(model)
    assert per_key > expected                                          # the tied MLP has one key per layer
    assert finalized.resources["model_state_bytes"] == expected


def test_checkpoint_member_bytes_count_each_storage_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_git(monkeypatch)
    cfg = tiny_config(**_TIED)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model)
    path = artifacts.save_checkpoint(1, model, T.build_optimizer(model, cfg), cfg)
    assert optimizer_state_bytes(path)["other_member_bytes"]["model_state"] == _unique_state_bytes(model)


# ------------------------------------------------------------------------------------------------ A058, A079

def _fallback(**overrides: Any) -> str:
    return ra._relational_artifact_fields(tiny_config(**overrides))["relational_operators"]["factorization_fallback"]


@pytest.mark.parametrize("routing", ("gaussian_kl", "noisy_query", "noisy_key"))
@pytest.mark.parametrize("factorization", ("gram", "gram_certified", "cholesky_qr2_certified"))
def test_fallback_string_reads_the_router_predicate(routing: str, factorization: str,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    overrides = dict(**_DIAGONAL_ROUTE, relational_routing=routing, projected_factorization=factorization)
    cfg = tiny_config(**overrides)
    verdicts: List[bool] = []
    real = grel._diagonal_comparison_route

    def spy(*args: Any, **kwargs: Any) -> bool:
        verdicts.append(bool(real(*args, **kwargs)))
        return verdicts[-1]

    monkeypatch.setattr(grel, "_diagonal_comparison_route", spy)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    with torch.no_grad():
        model(torch.zeros(1, cfg.max_seq_len, dtype=torch.long))
    assert verdicts and len(set(verdicts)) == 1
    assert (_fallback(**overrides) == "none_closed_form_diagonal_comparison_route") is verdicts[0]


def test_live_noisy_key_route_records_the_closed_form_diagonal_route() -> None:
    for routing in ("noisy_key", "noisy_query"):
        assert _fallback(**_DIAGONAL_ROUTE, relational_routing=routing) == "none_closed_form_diagonal_comparison_route"


@pytest.mark.parametrize("factorization", ("gram", "gram_certified", "cholesky_qr2_certified"))
def test_noisy_key_dense_route_records_query_only_rebuilds(factorization: str,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    fallback = _fallback(relational_routing="noisy_key", projected_factorization=factorization)
    assert fallback.endswith("_source_root_shifted_cholesky_qr3_fp32_then_qr_fp64_query_entries_only_"
                             "broadened_keys_not_rebuilt")
    assert "source_root_qr_fp64" not in fallback
    # The executed route: noisy_key passes no key source tuple, so the factor route never runs, while gaussian_kl
    # under cholesky_qr2_certified takes it.
    calls: List[str] = []
    real_status = rs._factor_status
    monkeypatch.setattr(rs, "_factor_status", lambda *args, **kwargs: calls.append("factor") or real_status(
        *args, **kwargs))
    for routing in ("noisy_key", "gaussian_kl"):
        calls.clear()
        cfg = tiny_config(relational_routing=routing, projected_factorization=factorization)
        torch.manual_seed(0)
        with torch.no_grad():
            VFEModel(cfg)(torch.zeros(1, cfg.max_seq_len, dtype=torch.long))
        took_factor_route = bool(calls)
        assert took_factor_route is (routing == "gaussian_kl" and factorization == "cholesky_qr2_certified")
    assert _fallback(projected_factorization="cholesky_qr2_certified") == (
        "certificate_flagged_or_failed_entries_source_root_qr_fp64")


# ------------------------------------------------------------------------------------------------ A062

def test_best_equals_final_fills_every_test_best_metric(finalized: SimpleNamespace) -> None:
    results, summary = finalized.test_results, finalized.summary
    assert results["test_best_status"] == "best weights are the final weights"
    assert results["test_best_bits_per_token"] == results["test_bits_per_token"] is not None
    assert summary["test_best_bits_per_token"] == summary["test_bits_per_token"]


def test_every_test_best_branch_carries_the_bits_per_token_key() -> None:
    final_test = {"test_ce": 1.0, "test_ppl": math.e, "test_bits_per_token": 1.0 / math.log(2.0)}
    for best_step, status in ((5, "best weights are the final weights"), (None, "no validation-selected weights")):
        artifacts = SimpleNamespace(best_step=best_step, train_sample_by_step={}, save_model_artifacts=False,
                                    best_path=Path("unused"))
        out = ra._best_weight_and_train_sample_evaluation(
            torch.nn.Linear(1, 1), artifacts, tiny_config(), LOG, final_step=5, final_test=final_test,
            train_loader=None, val_loader=None, test_loader=None, tokens_per_char=None, device=CPU)
        assert out["test_best_status"] == status and "test_best_bits_per_token" in out
        assert out["test_best_bits_per_token"] == (final_test["test_bits_per_token"] if best_step else None)


# ------------------------------------------------------------------------------------------------ A074

def test_int_and_float_list_spellings_fingerprint_alike() -> None:
    ints, floats = tiny_config(kappa_beta=[1, 2]), tiny_config(kappa_beta=[1.0, 2.0])
    assert ints == floats
    assert (ra.semantic_config_fingerprint(serialize_config(ints))
            == ra.semantic_config_fingerprint(serialize_config(floats)))
    assert (ra.semantic_config_fingerprint(ra._selection_semantic_config(ints))
            == ra.semantic_config_fingerprint(ra._selection_semantic_config(floats)))
    # Only float-typed fields are canonicalized: an integer list of another field keeps its spelling.
    assert ra.semantic_config_fingerprint({"irrep_spec": [[2, 1]]}) != ra.semantic_config_fingerprint(
        {"irrep_spec": [[2.0, 1.0]]})


# ------------------------------------------------------------------------------------------------ A147

def test_finalize_hands_final_model_to_the_figure_worker(finalized: SimpleNamespace) -> None:
    (call,) = finalized.figure_calls
    assert call["generate_publication"] is True
    assert call["model_bundle_path"] == finalized.artifacts.final_path
    assert finalized.artifacts.final_path.is_file()


def test_figure_worker_reads_the_given_bundle_without_a_second_snapshot(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_git(monkeypatch)
    cfg = tiny_config()
    model = VFEModel(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model)
    seen: Dict[str, Any] = {}

    def fake_tree(command: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
        seen["request"] = json.loads(Path(kwargs["env"]["VFE3_FIGURE_REQUEST"]).read_text(encoding="utf-8"))
        seen["snapshots"] = [(p.name, p.stat().st_size) for p in artifacts.run_dir.glob(".figure_live_model.pt.*")]
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(ra, "run_process_tree", fake_tree)
    assert ra._run_figures_isolated(model, artifacts, None, LOG, generate_publication=True,
                                    model_bundle_path=artifacts.final_path)
    assert seen["request"]["model_bundle_path"] == str(artifacts.final_path)
    assert seen["snapshots"] == []
    # Without a bundle on disk (save_model_artifacts=False) the worker still gets a temporary live snapshot.
    assert ra._run_figures_isolated(model, artifacts, None, LOG, generate_publication=True)
    (snapshot,) = seen["snapshots"]
    assert seen["request"]["model_bundle_path"].endswith(snapshot[0]) and snapshot[1] > 0
    assert not list(artifacts.run_dir.glob(".figure_live_model.pt.*"))


# ------------------------------------------------------------------------------------------------ A148

def _selection_artifacts(run_dir: Path, cfg: VFE3Config, model: torch.nn.Module) -> ra.RunArtifacts:
    artifacts = ra.RunArtifacts(run_dir, cfg, model)
    artifacts.bind_selection_data_identity(T._loader_data_identity(_loaders(cfg)[1], cfg.vocab_size))
    return artifacts


def test_background_best_snapshot_keeps_the_tied_aliases_on_one_storage(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_git(monkeypatch)
    cfg = tiny_config(**_TIED)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model)
    assert artifacts.maybe_save_best(2, model, 9.0, background=True)
    artifacts.wait_background()
    saved = torch.load(artifacts.best_path, weights_only=True)["model_state"]
    aliases = tied_block_mlp_aliases(saved, cfg)
    assert aliases
    for alias, first in aliases.items():
        assert saved[alias].data_ptr() == saved[first].data_ptr()
    live = model.state_dict()
    assert len({t.data_ptr() for t in saved.values()}) == len({t.data_ptr() for t in live.values()})
    assert all(torch.equal(saved[name], tensor) for name, tensor in live.items())


def test_checkpoint_at_the_selected_step_embeds_the_live_storages_once(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_git(monkeypatch)
    cfg = tiny_config(**_TIED)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    artifacts = _selection_artifacts(tmp_path / "A", cfg, model)
    assert artifacts.maybe_save_best(2, model, 9.0, background=True)
    loads: List[Dict[str, Any]] = []
    real_load = torch.load
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: loads.append(kwargs) or real_load(*args, **kwargs))
    path = artifacts.save_checkpoint(2, model, T.build_optimizer(model, cfg), cfg)
    monkeypatch.setattr(torch, "load", real_load)
    assert loads == []                                                 # best_model.pt is not read back
    checkpoint = torch.load(path, weights_only=True)
    embedded = checkpoint["best_model_bundle"]
    for name, tensor in checkpoint["model_state"].items():
        assert embedded["model_state"][name].data_ptr() == tensor.data_ptr()   # one copy of the weights
    on_disk = torch.load(artifacts.best_path, weights_only=True)
    assert {k: v for k, v in embedded.items() if k != "model_state"} == {
        k: v for k, v in on_disk.items() if k != "model_state"}
    assert all(torch.equal(embedded["model_state"][k], v) for k, v in on_disk["model_state"].items())
    # The resumed selection is the selected weights, bit for bit.
    fresh = VFEModel(cfg)
    resumed = _selection_artifacts(tmp_path / "B", cfg, fresh)
    ra.load_checkpoint(path, fresh, artifacts=resumed)
    assert resumed.best_step == 2 and resumed.best_val_ppl == 9.0
    published = torch.load(resumed.best_path, weights_only=True)["model_state"]
    assert all(torch.equal(published[name], tensor) for name, tensor in model.state_dict().items())


def test_training_checkpoint_at_an_improving_eval_holds_one_copy_of_the_weights(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_git(monkeypatch)
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    cfg = tiny_config(**_TIED, max_steps=2, warmup_steps=0, checkpoint_interval=2, log_interval=0, eval_interval=2)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model)
    train_loader, val_loader, _ = _loaders(cfg)
    T.train(model, train_loader, cfg, n_steps=2, grad_clip=cfg.grad_clip, log_interval=0, eval_interval=2,
            val_loader=val_loader, device=CPU, logger=LOG, artifacts=artifacts, generate_samples=False)
    assert artifacts.best_step == 2 and artifacts.last_checkpoint_step == 2   # the eval at step 2 selected
    path = artifacts.ckpt_dir / "step_2.pt"
    members = optimizer_state_bytes(path)["other_member_bytes"]
    assert members["model_state"] == _unique_state_bytes(model)
    assert "best_model_bundle" not in members                          # its weights are model_state's storages
    checkpoint = torch.load(path, weights_only=True)
    embedded = checkpoint["best_model_bundle"]["model_state"]
    assert all(embedded[k].data_ptr() == v.data_ptr() for k, v in checkpoint["model_state"].items())


def test_checkpoint_after_the_selected_weights_moved_reads_best_model(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_git(monkeypatch)
    cfg = tiny_config(**_TIED)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    artifacts = _selection_artifacts(tmp_path / "A", cfg, model)
    assert artifacts.maybe_save_best(2, model, 9.0, background=True)
    selected = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with torch.no_grad():
        model.prior_bank.mu_embed.add_(0.5)                            # moved after selection, at the same step
    loads: List[Dict[str, Any]] = []
    real_load = torch.load
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: loads.append(kwargs) or real_load(*args, **kwargs))
    path = artifacts.save_checkpoint(2, model, T.build_optimizer(model, cfg), cfg)
    monkeypatch.setattr(torch, "load", real_load)
    assert [kwargs.get("mmap") for kwargs in loads] == [True]          # read memory-mapped, once
    embedded = torch.load(path, weights_only=True)["best_model_bundle"]["model_state"]
    assert all(torch.equal(embedded[name], tensor) for name, tensor in selected.items())
    assert not torch.equal(embedded["prior_bank.mu_embed"], model.prior_bank.mu_embed.detach())
    aliases = tied_block_mlp_aliases(embedded, cfg)
    assert aliases and all(embedded[a].data_ptr() == embedded[f].data_ptr() for a, f in aliases.items())
    # The memory map is released with the checkpoint: the next improving evaluation replaces best_model.pt (on
    # Windows a file mapped by this process cannot be replaced).
    assert artifacts.maybe_save_best(3, model, 8.0) is True and artifacts.best_step == 3
    assert "best_model_save" not in artifacts.diagnostics_failures


def test_host_state_copies_keep_the_tied_aliases(tmp_path: Path) -> None:
    cfg = tiny_config(**_TIED, max_steps=1, warmup_steps=0)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    copy = ra._host_state_copy(model.state_dict())
    aliases = tied_block_mlp_aliases(copy, cfg)
    assert aliases and all(copy[a].data_ptr() == copy[f].data_ptr() for a, f in aliases.items())
    assert all(copy[name].data_ptr() != tensor.data_ptr() and torch.equal(copy[name], tensor)
               for name, tensor in model.state_dict().items())
    # The terminal state train() hands to a terminal callback is the same alias-preserving host copy.
    captured: Dict[str, Any] = {}
    train_loader = _loaders(cfg)[0]
    T.train(model, train_loader, cfg, n_steps=1, grad_clip=cfg.grad_clip, log_interval=0, eval_interval=0,
            device=CPU, logger=LOG, generate_samples=False,
            terminal_callback=lambda state, losses: captured.update(state=state.raw_model_state))
    raw = captured["state"]
    assert all(raw[a].data_ptr() == raw[f].data_ptr() for a, f in tied_block_mlp_aliases(raw, cfg).items())
    assert all(torch.equal(raw[name], tensor) for name, tensor in model.state_dict().items())


# ------------------------------------------------------------------------------------------------ A264

def test_a_checkpoint_without_scaler_state_starts_an_enabled_scaler_at_its_initial_scale(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_git(monkeypatch)
    cfg = tiny_config()
    torch.manual_seed(0)
    source = VFEModel(cfg)
    path = ra.RunArtifacts(tmp_path / "source", cfg, source).save_checkpoint(
        0, source, T.build_optimizer(source, cfg), cfg, scaler=None)          # a bf16 or fp32 run
    assert torch.load(path, weights_only=True)["scaler_state"] is None
    target = VFEModel(cfg)
    scaler = torch.amp.GradScaler(device="cpu", enabled=True, init_scale=16.0)
    before = scaler.state_dict()
    with pytest.warns(UserWarning, match="initial scale"):
        assert ra.load_checkpoint(path, target, T.build_optimizer(target, cfg), cfg=cfg, scaler=scaler) == 0
    assert scaler.state_dict() == before
    assert all(torch.equal(a, b) for a, b in zip(source.state_dict().values(), target.state_dict().values()))
    # A present but malformed scaler state is still refused.
    bundle = torch.load(path, weights_only=True)
    bundle["scaler_state"] = {"scale": 2.0}
    torch.save(bundle, path)
    with pytest.raises(RuntimeError, match="scaler_state has an invalid schema"):
        ra.load_checkpoint(path, VFEModel(cfg), cfg=cfg, scaler=scaler)


# ------------------------------------------------------------------------------------------------ A278

def test_module_docstring_names_the_weights_finalize_scores() -> None:
    doc = ra.__doc__
    assert "reloaded best checkpoint" not in doc and "reloads the best-val" not in doc
    assert "final weights" in doc and "test_best_" in doc


# ------------------------------------------------------------------------------------------------ A300

def test_parameters_read_per_token_count_unique_parameters_and_report_layer_reuse() -> None:
    for tied in (True, False):
        cfg = tiny_config(**{**_TIED, "block_mlp_tie_layers": tied})
        torch.manual_seed(0)
        model = VFEModel(cfg)
        out = parameter_breakdown(dict(model.named_parameters()), cfg)
        norm = sum(p.numel() for p in model.block_norm.parameters())
        mlp = sum(p.numel() for p in model.block_mlps[0].parameters()) if tied else 0
        assert norm > 0 and (mlp > 0) is tied
        assert out["read_per_token_components"]["layer_shared_parameters"] == norm + mlp
        assert out["parameter_reads_per_token_with_layer_reuse"] == (
            out["parameters_read_per_token"] + (cfg.n_layers - 1) * (norm + mlp))
        assert "unique" in out["definition"]
        state = model.state_dict()
        from_state = parameter_breakdown(state, cfg, frozen=frozen_parameter_names(cfg, state.keys()))
        for key in ("parameters_read_per_token", "parameter_reads_per_token_with_layer_reuse"):
            assert from_state[key] == out[key]


def _reduced_view(cfg: VFE3Config, **fields: Any) -> SimpleNamespace:
    r"""The fields a reduced record view carries (vfe3.paper.runs._parameter_cfg_like before n_layers)."""
    return SimpleNamespace(model_family="vfe", vocab_size=cfg.vocab_size, decode_mode=cfg.decode_mode,
                           gauge_transport=cfg.gauge_transport, use_block_mlp=cfg.use_block_mlp,
                           block_mlp_tie_layers=cfg.block_mlp_tie_layers, **fields)


def test_layer_reuse_count_is_unknown_without_a_layer_count() -> None:
    cfg = tiny_config(**_TIED)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    named = dict(model.named_parameters())
    full = parameter_breakdown(named, cfg)
    assert full["parameter_reads_per_token_with_layer_reuse"] > full["parameters_read_per_token"]
    # A view without n_layers cannot count reads per layer: the reuse count is None, not the unique count.
    reduced = parameter_breakdown(named, _reduced_view(cfg))
    assert reduced["parameters_read_per_token"] == full["parameters_read_per_token"]
    assert reduced["parameter_reads_per_token_with_layer_reuse"] is None
    assert (parameter_breakdown(named, _reduced_view(cfg, n_layers=cfg.n_layers))
            ["parameter_reads_per_token_with_layer_reuse"] == full["parameter_reads_per_token_with_layer_reuse"])
    # Without a module every layer applies the two counts agree whatever the view records.
    bare = tiny_config(n_layers=2, use_block_mlp=False, norm_type_block="none")
    bare_named = dict(VFEModel(bare).named_parameters())
    out = parameter_breakdown(bare_named, _reduced_view(bare))
    assert out["read_per_token_components"]["layer_shared_parameters"] == 0
    assert out["parameter_reads_per_token_with_layer_reuse"] == out["parameters_read_per_token"]


def test_paper_parameter_view_carries_the_layer_count(tmp_path: Path) -> None:
    from vfe3.paper.runs import RunRecord, _parameter_cfg_like

    cfg = tiny_config(**_TIED)
    run = RunRecord(run_dir=tmp_path, config_raw=serialize_config(cfg))
    view = _parameter_cfg_like(run)
    assert view.n_layers == cfg.n_layers
    torch.manual_seed(0)
    named = dict(VFEModel(cfg).named_parameters())
    assert (parameter_breakdown(named, view)["parameter_reads_per_token_with_layer_reuse"]
            == parameter_breakdown(named, cfg)["parameter_reads_per_token_with_layer_reuse"])


def test_resources_tables_label_the_unique_count_and_add_layer_reuse() -> None:
    from vfe3.paper.tables import RESOURCE_ROWS, _resource_cells
    from vfe3.posthoc.report import resources_table

    row = RESOURCE_ROWS.index("Parameters read per token (unique)")
    cfg = tiny_config(**_TIED)
    torch.manual_seed(0)
    breakdown = parameter_breakdown(dict(VFEModel(cfg).named_parameters()), cfg)
    unique = breakdown["parameters_read_per_token"]
    reuse = breakdown["parameter_reads_per_token_with_layer_reuse"]
    assert reuse > unique

    def cell(params: Dict[str, Any], latex: bool = False) -> str:
        record = {"parameters": params, "optimizer_state_bytes": None, "peak_memory_mb": None,
                  "tokens_per_s_median": None, "device_hours": None, "latency": None}
        return _resource_cells(record, latex)[row]

    assert cell(breakdown) == f"{unique:,} ({reuse:,} with layer reuse)"
    assert cell(breakdown, latex=True) == ("$" + f"{unique:,}".replace(",", "{,}") + "$ ($"
                                           + f"{reuse:,}".replace(",", "{,}") + "$ with layer reuse)")
    # A record without the reuse count (written before it existed) or one where both agree shows the unique count.
    for value in (None, unique):
        assert cell({**breakdown, "parameter_reads_per_token_with_layer_reuse": value}) == f"{unique:,}"
    markdown = resources_table({"parameter_breakdown": breakdown})
    assert f"| Parameters read per token (unique) | {unique:,} ({reuse:,} with layer reuse) |" in markdown


# ------------------------------------------------------------------------------------------------ A309

def test_theory_pure_conjunction_is_documented_as_unreachable_on_current_routes() -> None:
    doc = " ".join(ra._pure_path_report.__doc__.split())
    assert "no current route satisfies" in doc.lower() and "canonical_attention_entropy" in doc
    for overrides in ({}, dict(use_block_mlp=False, decode_mode="diagonal_chunked", untie_decode_bank=True),
                      dict(**_DIAGONAL_ROUTE, relational_routing="noisy_key")):
        report = ra._pure_path_report(tiny_config(**overrides), [])
        assert report["theory_flags"]["canonical_attention_entropy"] is False
        assert report["on_theory_pure_path"] is False


# ------------------------------------------------------------------------------------------------ A310

def test_fd_gradient_check_records_its_device_and_kernels(_restore_diag_moments: Any) -> None:
    cfg = tiny_config(**_DIAGONAL_ROUTE, decode_mode="diagonal_chunked", untie_decode_bank=True)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    rs._DIAG_MOMENTS_TRITON_CALLS, rs._DIAG_MOMENTS_TORCH_CALLS = 6, 0
    record: Dict[str, Any] = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        worst = ra._fd_gradient_check(model, [_batch(cfg)], CPU, record=record)
    assert not [w for w in caught if "requires_grad=True to a scalar" in str(w.message)]
    assert math.isfinite(worst) and record["fd_gradient_probe"] == "prior_bank.decode_mu_embed"
    assert record["fd_gradient_device"] == "cpu"
    kernels = record["fd_gradient_kernels"]
    assert kernels["decode_ce"] == "torch" and kernels["diag_moments"] == "torch"
    assert (rs._DIAG_MOMENTS_TRITON_CALLS, rs._DIAG_MOMENTS_TORCH_CALLS) == (6, 0)   # the run's record is kept


def test_research_record_carries_the_fd_scope(finalized: SimpleNamespace) -> None:
    research = finalized.research
    assert research["fd_gradient_device"] == "cpu" and "decode_ce" in research["fd_gradient_kernels"]


# ------------------------------------------------------------------------------------------------ A327

def test_a_blocked_best_model_publish_is_counted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_git(monkeypatch)
    cfg = tiny_config()
    model = VFEModel(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model)

    def blocked(final: Path, tmp: Path, **kwargs: Any) -> None:
        raise PermissionError("best_model.pt held open by a reader")

    monkeypatch.setattr(ra, "_atomic_replace", blocked)
    assert artifacts.maybe_save_best(2, model, 9.0) is False
    assert artifacts.diagnostics_failures == {"best_model_save": 1}
    assert artifacts.best_step is None and artifacts.best_val_ppl == float("inf")


def test_failed_checkpoint_saves_are_counted_and_the_last_written_step_is_kept(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_git(monkeypatch)
    cfg = tiny_config(max_steps=4, warmup_steps=0, checkpoint_interval=2, checkpoint_at_end=True, log_interval=0,
                      eval_interval=0)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model)
    real_save = ra.RunArtifacts.save_checkpoint

    def flaky(self: ra.RunArtifacts, step: int, *args: Any, **kwargs: Any) -> Path:
        if step == 4:
            raise OSError(28, "No space left on device")
        return real_save(self, step, *args, **kwargs)

    monkeypatch.setattr(ra.RunArtifacts, "save_checkpoint", flaky)
    T.train(model, _loaders(cfg)[0], cfg, n_steps=4, grad_clip=cfg.grad_clip, log_interval=0, eval_interval=0,
            device=CPU, logger=LOG, artifacts=artifacts, generate_samples=False)
    assert artifacts.diagnostics_failures["checkpoint_save"] == 2       # the periodic save and the end-of-run retry
    assert artifacts.last_checkpoint_step == 2


def test_summary_publishes_the_last_written_checkpoint(finalized: SimpleNamespace) -> None:
    assert finalized.summary["last_checkpoint_step"] == 3
    assert "checkpoint_save" not in finalized.summary["diagnostics_failures"]
    assert sorted(p.name for p in finalized.artifacts.ckpt_dir.glob("step_*.pt")) == ["step_2.pt", "step_3.pt"]


# ------------------------------------------------------------------------------------------------ A328

def test_launcher_resume_records_equal_drift_fingerprints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_git(monkeypatch)
    cfg = tiny_config()
    torch.manual_seed(0)
    source = VFEModel(cfg)
    path = ra.RunArtifacts(tmp_path / "source", cfg, source).save_checkpoint(
        0, source, T.build_optimizer(source, cfg), cfg)
    resumed_cfg = tiny_config(resume_from=str(path))                   # the launcher's resume route
    target = VFEModel(resumed_cfg)
    artifacts = ra.RunArtifacts(tmp_path / "restored", resumed_cfg, target)
    ra.load_checkpoint(path, target, T.build_optimizer(target, resumed_cfg), cfg=resumed_cfg, artifacts=artifacts)
    record = json.loads((artifacts.run_dir / "resume_provenance.json").read_text(encoding="utf-8"))
    assert record["config_drift_fields"] == []
    assert record["config_drift_excluded_fields"] == ["resume_from", "trust_resume_checkpoint"]
    assert record["saved_config_fingerprint"] != record["current_config_fingerprint"]   # resume_from differs
    assert record["saved_config_drift_fingerprint"] == record["current_config_drift_fingerprint"]
    changed = tiny_config(resume_from=str(path), m_p_mu_lr=0.5)
    assert ra._resume_drift_fingerprint(serialize_config(changed)) != record["current_config_drift_fingerprint"]


# ------------------------------------------------------------------------------------------------ A331

def test_validation_only_finalizer_evaluates_the_weight_ema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_git(monkeypatch)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    cfg = tiny_config(max_steps=3, warmup_steps=0, weight_ema_decay=0.5, weight_ema_start_frac=0.0,
                      checkpoint_interval=0, log_interval=0, eval_interval=0)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model)
    train_loader, val_loader, _ = _loaders(cfg)
    merged: Dict[str, Any] = {}

    def finalize(state: Any, losses: List[float]) -> None:
        merged.update(ra.finalize_validation_run(model, artifacts, cfg, val_loader, losses=losses,
                                                 terminal_state=state, device=CPU, logger=LOG))

    T.train(model, train_loader, cfg, n_steps=3, grad_clip=cfg.grad_clip, log_interval=0, eval_interval=0,
            val_loader=val_loader, device=CPU, logger=LOG, artifacts=artifacts, generate_samples=False,
            terminal_callback=finalize)
    summary = json.loads((artifacts.run_dir / "summary.json").read_text(encoding="utf-8"))
    validation = json.loads((artifacts.run_dir / "validation_results.json").read_text(encoding="utf-8"))
    assert summary["weight_ema_status"] == "evaluated" and math.isfinite(summary["val_ema_ppl"])
    assert validation["val_ema_ppl"] == summary["val_ema_ppl"] and validation["weight_ema_updates"] >= 1
    assert not [key for key in summary if key.startswith("test_")]    # the validation finalizer opens no test split
    assert merged["primary_val_ppl"] == summary["primary_val_ppl"]
