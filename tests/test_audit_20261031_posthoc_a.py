r"""Remediation package posthoc-a of the 2026-10-31 audit: post-hoc evaluation, KL cap, calibration and report.

Tiny CPU models only (vocabulary at most 24, K at most 8, two heads, two layers, windows of 8 tokens). Covers:

* A048 and A214: every post-hoc evaluation forward runs on the replay ladder of :func:`vfe3.train.evaluate` inside
  :func:`vfe3.contracts.deferred_host_checks`, and the replays are recorded per pass;
* A094: degenerate positions are decided on the covariance the decoder reads;
* A210: only the scored rows are decoded, in tiles bounded by a byte budget;
* A113: an evenly spaced ``max_windows`` subset never contains window 0;
* A121 and A301: the KL-cap row minimum on the scale the cap tests, and :math:`\mathrm{TV} \le m_i`;
* A122: the calibration headline is never filled from another checkpoint;
* A197: the minimal-agent gap replays at most its token budget;
* A340: the weight-EMA shadow of the latest step checkpoint is evaluated with its own reference;
* A369: the M5 table names its split and restoration verdict, and an optional map or figure failure is recorded
  instead of discarding ``posthoc.json``.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch

from tests.profiles import live_config, test_config, tiny_config
from vfe3.config import VFE3Config, serialize_config
from vfe3.model.model import VFEModel
from vfe3.posthoc import evaluate as ev
from vfe3.posthoc import report as rp
from vfe3.posthoc import streams as st

V = 24
DEVICE = torch.device("cpu")


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


@pytest.fixture
def matmul_policy() -> Any:
    r"""Restore the process fp32 matmul policy after a test that changes it."""
    before = torch.get_float32_matmul_precision()
    yield
    torch.set_float32_matmul_precision(before)


def _cfg(**over: Any) -> VFE3Config:
    values: Dict[str, Any] = dict(
        vocab_size=V, embed_dim=4, n_heads=2, max_seq_len=8, eval_stride=4, n_layers=2, batch_size=2,
        attention_mode="projected_predictive", block_mlp_expansion=2, norm_type_block="none",
        layernorm_affine=False, skip_belief_sigma_update=False, beta_attention_prior="causal_alibi")
    values.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


def _model(cfg: VFE3Config, seed: int = 0) -> VFEModel:
    r"""A tiny model moved off its initialization."""
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg).eval()
    with torch.no_grad():
        model.prior_bank.sigma_log_embed.normal_(0.0, 0.5)
        model.prior_bank.phi_embed.normal_(0.0, 0.3)
        if getattr(model, "pos_phi_free", None) is not None:
            model.pos_phi_free.normal_(0.0, 0.3)
        for op in model.relational_operators_by_layer:
            for p in op.parameters():
                p.add_(0.3 * torch.randn_like(p))
    return model


def _stream(n: int, seed: int, vocab: int = V) -> torch.Tensor:
    return torch.randint(0, vocab, (n,), generator=torch.Generator().manual_seed(seed))


def _policy_dependent_failure(model: torch.nn.Module,
                              fail: Callable[[int], bool] = lambda k: True) -> Dict[str, int]:
    r"""Make ``model.forward_beliefs`` raise the frame-certificate error under a reduced fp32 matmul policy.

    CPU matmuls do not round at TF32, so the real certificate cannot trip here; this injects the fault the replay
    ladder exists for (the audit's probe). ``fail(k)`` decides whether the ``k``-th reduced-policy call raises.
    """
    original = model.forward_beliefs
    calls = {"reduced": 0, "highest": 0}

    def forward_beliefs(*args: Any, **kwargs: Any) -> Any:
        if torch.get_float32_matmul_precision() != "highest":
            calls["reduced"] += 1
            if fail(calls["reduced"]):
                raise ValueError("canonical forward and inverse factors must be mutual inverses within a "
                                 "dtype-scaled residual (injected under a reduced fp32_matmul_precision)")
        else:
            calls["highest"] += 1
        return original(*args, **kwargs)

    model.forward_beliefs = forward_beliefs
    return calls


# ----------------------------------------------------------------------------------------------
# A048 and A214: the replay ladder and the deferred certificates
# ----------------------------------------------------------------------------------------------

def test_nll_pass_replays_a_reduced_precision_failure_and_records_it(matmul_policy: Any) -> None:
    model = _model(_cfg())
    dataset = st.eval_windows(_stream(61, 2), 8, 4)
    indices = np.arange(len(dataset))
    torch.set_float32_matmul_precision("highest")
    clean = ev.nll_pass(model, dataset, indices, batch_size=2, device=DEVICE)
    n_batches = math.ceil(len(dataset) / 2)
    assert clean.precision_replays == {} and clean.n_batches == n_batches
    calls = _policy_dependent_failure(model)
    torch.set_float32_matmul_precision("high")
    result = ev.nll_pass(model, dataset, indices, batch_size=2, device=DEVICE)
    assert np.array_equal(result.target_pos, clean.target_pos) and np.array_equal(result.nll, clean.nll)
    assert result.precision_replays == {"highest": n_batches} and result.n_batches == n_batches
    assert calls == {"reduced": n_batches, "highest": n_batches}
    assert torch.get_float32_matmul_precision() == "high"                 # the ladder restores the run's policy


def test_latency_forwards_replay_and_exclude_replayed_batches(matmul_policy: Any) -> None:
    model = _model(_cfg())
    dataset = st.eval_windows(_stream(61, 2), 8, 4)
    _policy_dependent_failure(model, fail=lambda k: k % 2 == 1)          # every other reduced-policy forward fails
    torch.set_float32_matmul_precision("high")
    latency = rp.measure_latency(model, dataset, 2, DEVICE, 4, n_warmup=1)
    # Batches 0 (warmup), 2 and 4 replay; 1 and 3 are timed.
    assert latency["status"] == "ok" and latency["n_batches"] == 2
    assert latency["replayed_batches"] == 2 and latency["replay_policies"] == {"highest": 2}


def test_a_replayed_pass_is_not_matched_arithmetic() -> None:
    result = ev.TargetNLL(nll=np.ones(4), target_pos=np.arange(1, 5), token=np.zeros(4, dtype=np.int64),
                          window=np.zeros(4, dtype=np.int64), context=np.ones(4, dtype=np.int32), expected=4,
                          excluded=0, n_batches=2)
    assert rp._pass_arithmetic_matched(True, result) and not rp._pass_arithmetic_matched(False, result)
    result.precision_replays = {"highest": 1}
    assert not rp._pass_arithmetic_matched(True, result)
    record = rp._replay_record(result)
    assert record == {"n_batches": 2, "replayed_batches": 1, "completed_at": {"highest": 1}}


def test_nll_pass_defers_every_raise_only_certificate(monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.contracts as contracts
    import vfe3.geometry.transport as transport
    import vfe3.gradients.full_link as full_link
    import vfe3.model.covariance_ceiling as covariance_ceiling
    import vfe3.model.prototype_agents as prototype_agents
    import vfe3.relational_scoring as relational_scoring

    counts = {"immediate": 0, "deferred": 0}
    original = contracts.host_check

    def counting(flags: torch.Tensor, verdict: Any, detail: Any = None) -> None:
        counts["immediate" if contracts._DEFERRED_CHECKS is None else "deferred"] += 1
        return original(flags, verdict, detail)

    for module in (contracts, transport, full_link, covariance_ceiling, prototype_agents, relational_scoring):
        if hasattr(module, "host_check"):
            monkeypatch.setattr(module, "host_check", counting)
    model = _model(_cfg())
    dataset = st.eval_windows(_stream(61, 2), 8, 4)
    ev.nll_pass(model, dataset, np.arange(2), batch_size=2, device=DEVICE)          # warm the cached verdicts
    counts.update(immediate=0, deferred=0)
    ev.nll_pass(model, dataset, np.arange(2), batch_size=2, device=DEVICE)          # one batch
    assert counts["deferred"] > 0 and counts["immediate"] == 0


def test_restoration_pass_failure_is_recorded_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.posthoc import interventions as iv

    model = _model(_cfg(attention_residual=True))
    dataset = st.eval_windows(_stream(61, 3), 8, 4)
    indices = np.arange(len(dataset))
    base = ev.nll_pass(model, dataset, indices, batch_size=2, device=DEVICE)
    original = ev.nll_pass

    def failing(model: Any, dataset: Any, idx: Any, **kwargs: Any) -> ev.TargetNLL:
        if len(idx) == rp.RESTORATION_CHECK_WINDOWS:
            raise ValueError("restoration pass failed (injected)")
        return original(model, dataset, idx, **kwargs)

    monkeypatch.setattr(ev, "nll_pass", failing)
    rows = rp._run_interventions(iv, model, dataset, indices, base, base.window.copy(), batch_size=2,
                                 device=DEVICE, keys=["token_frames_identity"], n_resamples=0,
                                 log=lambda message: None, save_to=None)
    restoration = [r for r in rows if r["key"] == "restoration_check"][0]
    assert restoration["status"] == "failed" and "injected" in restoration["error"]
    assert [r["status"] for r in rows if r["key"] == "token_frames_identity"] == ["ok"]


# ----------------------------------------------------------------------------------------------
# A094: degenerate positions on the decode covariance
# ----------------------------------------------------------------------------------------------

def _full_family_model(decode_covariance: str) -> VFEModel:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(tiny_config(family="gaussian_full", decode_mode="full_chunked", vocab_size=V,
                                     decode_covariance=decode_covariance, n_layers=2, max_seq_len=8,
                                     eval_stride=4, sigma_init=2.0)).eval()
    with torch.no_grad():
        model.prior_bank.sigma_log_embed.uniform_(-0.5, 1.0, generator=torch.Generator().manual_seed(1))
    return model


def _non_positive_definite_belief(model: VFEModel, position: Tuple[int, int] = (0, 5)) -> None:
    r"""Make the stack output's covariance at ``position`` of every batch not positive definite."""
    original = model.forward_beliefs

    def forward_beliefs(*args: Any, **kwargs: Any) -> Any:
        belief, logits = original(*args, **kwargs)
        sigma = belief.sigma.clone()
        sigma[position] = -torch.eye(sigma.shape[-1], dtype=sigma.dtype)              # (K, K), not PD
        return belief._replace(sigma=sigma), logits

    model.forward_beliefs = forward_beliefs


@pytest.mark.parametrize("decode_covariance, n_scored", [("constant", 40), ("encoded", 40), ("belief", 35)])
def test_degenerate_positions_are_decided_on_the_decode_covariance(decode_covariance: str, n_scored: int) -> None:
    model = _full_family_model(decode_covariance)
    dataset = st.eval_windows(_stream(41, 9), 8, 4)                     # 40 transitions in 9 windows, 5 batches
    indices = np.arange(len(dataset))
    _non_positive_definite_belief(model)
    result = ev.nll_pass(model, dataset, indices, batch_size=2, device=DEVICE)
    reference = ev.protocol_evaluate(model, dataset, indices, batch_size=2, device=DEVICE)
    assert result.n == int(reference["scored_targets"]) == n_scored
    assert result.excluded == result.expected - n_scored
    assert abs(result.ce - float(reference["ce"])) < ev.CROSS_CHECK_TOL


# ----------------------------------------------------------------------------------------------
# A210: only the scored rows are decoded
# ----------------------------------------------------------------------------------------------

def _dense_reference(model: VFEModel, dataset: Any, indices: np.ndarray) -> np.ndarray:
    r"""Per-target NLL from the dense ``(B, N, V)`` readout of every position (the former ``nll_pass``)."""
    import torch.nn.functional as F

    out: List[np.ndarray] = []
    for _, inputs, targets in ev.iter_batches(dataset, indices, 2):
        with torch.no_grad():
            _, logits = model.forward_beliefs(inputs, return_logits=True)                 # (B, N, V)
        valid = targets != -100
        out.append(F.cross_entropy(logits[valid].double(), targets[valid], reduction="none").numpy())
    return np.concatenate(out)


@pytest.mark.parametrize("over", [dict(decode_mode="diagonal_chunked"), dict(decode_mode="linear")],
                         ids=["diagonal_chunked", "linear"])
def test_nll_pass_decodes_only_the_scored_rows(over: Dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model(_cfg(**over))
    dataset = st.eval_windows(_stream(61, 2), 8, 4)
    indices = np.arange(len(dataset))
    dense = _dense_reference(model, dataset, indices)
    decoded: List[int] = []
    original = model.prior_bank.decode

    def decode(mu_q: torch.Tensor, sigma_q: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        decoded.append(int(mu_q.shape[:-1].numel()))
        return original(mu_q, sigma_q, **kwargs)

    monkeypatch.setattr(model.prior_bank, "decode", decode)
    result = ev.nll_pass(model, dataset, indices, batch_size=2, device=DEVICE)
    assert result.n == 60 and sum(decoded) == 60 and len(decoded) == math.ceil(len(dataset) / 2)
    np.testing.assert_allclose(result.nll, dense, rtol=0, atol=2e-5)
    decoded.clear()
    monkeypatch.setattr(ev, "NLL_DECODE_TILE_BYTES", 1)                  # one row per tile
    tiled = ev.nll_pass(model, dataset, indices, batch_size=2, device=DEVICE)
    assert sum(decoded) == len(decoded) == 60
    np.testing.assert_allclose(tiled.nll, result.nll, rtol=0, atol=1e-6)
    assert np.array_equal(tiled.target_pos, result.target_pos)


@pytest.mark.parametrize("over", [
    dict(decode_mode="diagonal_chunked"),
    dict(decode_mode="linear", decode_coordinates="reference", attention_residual=True,
         norm_type_block="layernorm", layernorm_affine=True),
    dict(family="gaussian_full", decode_mode="full_chunked", decode_coordinates="reference",
         attention_residual=True, norm_type_block="layernorm", layernorm_affine=True),
], ids=["diagonal", "linear_reference", "full_reference"])
def test_decode_rows_equal_the_selected_rows_of_the_full_readout(over: Dict[str, Any]) -> None:
    model = _model(_cfg(**over))
    tokens = _stream(16, 4).reshape(2, 8)
    frame_out: Dict[str, Any] = {}
    encoded_out: Dict[str, Any] = {}
    rows = torch.zeros(2, 8, dtype=torch.bool)
    rows[0, 3:] = True
    rows[1, ::3] = True
    with torch.no_grad():
        belief, logits = model.forward_beliefs(tokens, return_logits=True, _decode_frame_out=frame_out,
                                               _encoded_sigma_out=encoded_out)
        packed = model._decode_belief_with_context(belief.mu, belief.sigma, canonical_frame=frame_out.get("frame"),
                                                   encoded_sigma=encoded_out.get("sigma"), decode_rows=rows)
    assert packed.shape == (1, int(rows.sum()), V)                                       # (1, S, V)
    torch.testing.assert_close(packed[0], logits[rows], rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="decode_rows"):
        model._decode_belief_with_context(belief.mu, belief.sigma, canonical_frame=frame_out.get("frame"),
                                          encoded_sigma=encoded_out.get("sigma"), decode_rows=rows,
                                          decode_last=True)


def test_default_decode_seam_is_the_pre_change_body_on_the_live_profile() -> None:
    r"""Without ``decode_rows`` the seam computes exactly what it did: the selected covariance, then the bank decode.

    The training loss of the live route never calls the seam (its fused chunked CE decodes in the prior bank), so
    the ``decode_rows`` keyword leaves every live-route loss and gradient untouched; this pins the inference readout.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = live_config(vocab_size=9, max_seq_len=8, relational_routing="noisy_key",
                          routing_kl_max_reference="query_shifted", weight_ema_decay=None)
        torch.manual_seed(0)
        model = VFEModel(cfg).eval()
    tokens = _stream(16, 6, vocab=9).reshape(2, 8)
    frame_out: Dict[str, Any] = {}
    encoded_out: Dict[str, Any] = {}
    with torch.no_grad():
        belief, logits = model.forward_beliefs(tokens, return_logits=True, _decode_frame_out=frame_out,
                                               _encoded_sigma_out=encoded_out)
        frame = frame_out.get("frame")
        sigma = (belief.sigma if cfg.decode_mode == "linear" else
                 model._decode_covariance(belief.mu, belief.sigma, encoded_out.get("sigma"), frame))
        expected = model.prior_bank.decode(belief.mu, sigma, canonical_frame=frame)       # (B, N, V)
    assert torch.equal(logits, expected)


# ----------------------------------------------------------------------------------------------
# A113: max_windows subsets exclude window 0
# ----------------------------------------------------------------------------------------------

def test_evenly_spaced_subsets_skip_window_zero_and_its_short_contexts() -> None:
    N, s = 8, 2
    dataset = st.eval_windows(_stream(2001, 1), N, s)
    n = len(dataset)

    def short_context_share(indices: np.ndarray) -> float:
        short = total = 0
        for index in indices:
            _, targets = dataset[int(index)]
            context = torch.nonzero(targets != -100).squeeze(-1) + 1                       # tokens of context
            total += int(context.numel())
            short += int((context <= N - s).sum())
        return short / total

    full = short_context_share(np.arange(n))
    assert full > 0.0
    for m in (2, 5, 20, n - 1):
        chosen = st.evenly_spaced(n, m)
        assert chosen.size == m and 0 not in chosen.tolist() and np.all(np.diff(chosen) > 0)
        assert short_context_share(chosen) <= full
    assert np.array_equal(st.evenly_spaced(n, None), np.arange(n))
    assert np.array_equal(st.evenly_spaced(n, n), np.arange(n))
    assert np.array_equal(st.evenly_spaced(n, n - 1), np.arange(1, n))
    assert st.evenly_spaced(n, 1).tolist() == [1]
    assert st.evenly_spaced(1, 1).tolist() == [0]                         # a one-window split keeps its window


# ----------------------------------------------------------------------------------------------
# A121 and A301: the KL-cap observer
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("reference", ["query_shifted", "row_min", "absolute"])
def test_klcap_row_minimum_is_on_the_scale_the_cap_tests(reference: str) -> None:
    from vfe3.posthoc.klcap import KLCapObserver, causal_support

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = live_config(vocab_size=9, max_seq_len=8, eval_stride=4, relational_routing="noisy_key",
                          routing_kl_max_reference=reference, weight_ema_decay=None, kl_max=1.0e6)
        torch.manual_seed(0)
        model = VFEModel(cfg).eval()
    dataset = st.eval_windows(_stream(41, 5, vocab=9), 8, 4)
    observer = KLCapObserver(float(cfg.kl_max), cfg.routing_kl_max_reference)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev.nll_pass(model, dataset, np.arange(2), batch_size=2, device=DEVICE, observers=[observer])
        capture: Dict[str, Any] = {"diagnostic": {}}
        with torch.no_grad():
            model.forward_beliefs(torch.stack([dataset[i][0] for i in range(2)]), capture=capture)
    diagnostic = capture["diagnostic"]
    valid = torch.stack([dataset[i][1] for i in range(2)]) != -100                        # (B, N) scored rows
    rows = [r for r in observer.rows() if r["population"] == "scored"]
    assert rows
    for row in rows:
        cap = diagnostic["layer_conditional_captures"][row["layer"]]
        support = causal_support(cap["beta_pre"], diagnostic["log_prior"])                 # (B, H, N, N)
        kept = ((support.sum(-1) >= 2) & valid[:, None, :])[:, row["head"]]               # (B, N)
        assert not bool((cap["score_energy_pre"][:, row["head"]] >= cfg.kl_max).any())    # nothing is capped
        capped_scale_min = (cap["score_energy_pre"].double().masked_fill(~support, math.inf)
                            .amin(-1))[:, row["head"]][kept]
        assert row["raw_row_min_mean"] == pytest.approx(float(capped_scale_min.mean()), rel=1e-5, abs=1e-5)
        assert row["raw_row_min_max"] == pytest.approx(float(capped_scale_min.max()), rel=1e-5, abs=1e-5)
        gap = (cap["score_raw_row_max_pre"] - cap["score_raw_row_min_pre"])[:, row["head"]][kept]
        assert row["raw_row_gap_mean"] == pytest.approx(float(gap.mean()), rel=1e-9, abs=1e-12)


def test_klcap_total_variation_is_bounded_by_the_capped_mass() -> None:
    from vfe3.posthoc import klcap

    g = torch.Generator().manual_seed(0)
    kappa, rows, n = 4.0, 5000, 12
    scale = torch.tensor([2.0, 8.0, 30.0], dtype=torch.float64)[torch.randint(0, 3, (rows, 1), generator=g)]
    D = torch.rand(rows, n, generator=g, dtype=torch.float64) * scale
    log_prior = 0.3 * torch.randn(rows, n, generator=g, dtype=torch.float64)
    beta = torch.softmax(-D.clamp(max=kappa) + log_prior, dim=-1)
    beta_u = torch.softmax(-D + log_prior, dim=-1)
    m = (beta * (D >= kappa)).sum(-1)                                        # capped routing mass
    tv = 0.5 * (beta - beta_u).abs().sum(-1)
    assert float((tv - m).max()) <= 1e-12
    assert float((tv / m.clamp_min(1e-300))[m > 0.05].max()) > 0.95          # the bound is nearly attained
    doc = " ".join((klcap.__doc__ or "").split())
    assert "m_i/(1 - m_i)" not in doc and r"\le m_i" in doc


# ----------------------------------------------------------------------------------------------
# A122: the calibration headline
# ----------------------------------------------------------------------------------------------

def test_calibration_headline_is_never_another_checkpoints_value() -> None:
    from vfe3.posthoc.calibration import WordStream, calibration_for_run, calibration_table

    text = b" = A b = \nxy"
    tokens = np.frombuffer(text, dtype=np.uint8).astype(np.int64)
    words = WordStream(rows=[" = A b = \n", "xy"], source="fixture", split="test")
    T = tokens.size

    def run(passes: Dict[str, Any]) -> Dict[str, Any]:
        return calibration_for_run("wikitext-103", tokens, passes, seq_len=4, stride=4,
                                   headline_checkpoint="final_model.pt",
                                   decode_bytes=lambda ids: bytes(int(i) for i in ids), word_stream=words)

    passes = {"best_model.pt": (np.full(T - 1, 0.5), np.arange(1, T))}
    result = run(passes)
    assert result["status"] == "skipped" and "final_model.pt" in result["reason"]
    assert result["checkpoints"]["best_model.pt"]["status"] == "ok"
    best_ppl = result["checkpoints"]["best_model.pt"]["conventions"]["words_plus_eos_per_line"]["ppl"]
    table = calibration_table(result).splitlines()
    headline = [line for line in table if line.startswith("| Evaluated realization")]
    assert headline and f"{best_ppl:.2f}" not in headline[0] and "final_model.pt" in headline[0]
    assert any(line.startswith("| best_model.pt") for line in table)                     # per-checkpoint rows kept
    passes["final_model.pt"] = (np.full(T - 1, 0.25), np.arange(1, T))
    result = run(passes)
    assert result["status"] == "ok"
    final_ppl = result["checkpoints"]["final_model.pt"]["conventions"]["words_plus_eos_per_line"]["ppl"]
    headline = [line for line in calibration_table(result).splitlines() if line.startswith("| Evaluated")]
    assert f"{final_ppl:.2f}" in headline[0]


# ----------------------------------------------------------------------------------------------
# A197: the minimal-agent gap budget
# ----------------------------------------------------------------------------------------------

def _minimal_agent_model() -> VFEModel:
    cfg = test_config(attention_mode="projected_predictive", predictive_readout="minimal_agent",
                      minimal_agent_inference="mean_field", n_layers=2, max_seq_len=6, attention_residual=False,
                      predictive_covariance="moment_matched", block_mlp_expansion=2)
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFEModel(cfg)


def test_minimal_agent_gap_replays_at_most_its_token_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.model.relational_operators import RelationalOperators
    from vfe3.posthoc import feature_probes

    model = _minimal_agent_model()
    ids = torch.randint(0, model.cfg.vocab_size, (3, 6), generator=torch.Generator().manual_seed(5))
    exact_tokens: List[int] = []
    original = RelationalOperators.minimal_agent_update

    def counting(self: Any, mu: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if self.minimal_agent_inference == "exact":
            exact_tokens.append(int(mu.shape[:-1].numel()))
        return original(self, mu, *args, **kwargs)

    monkeypatch.setattr(RelationalOperators, "minimal_agent_update", counting)
    gap = feature_probes.minimal_agent_gap(model, [ids], DEVICE, gap_tokens=6)
    assert [row["tokens"] for row in gap["layers"]] == [6] * model.cfg.n_layers
    assert sum(exact_tokens) == 6 * model.cfg.n_layers                   # one sequence per layer, not all 18 tokens
    assert all(ops.minimal_agent_inference == "mean_field" for ops in model.relational_operators_by_layer)
    exact_tokens.clear()
    single = feature_probes.minimal_agent_gap(model, [ids[:1]], DEVICE, gap_tokens=None)
    for budgeted, alone in zip(gap["layers"], single["layers"]):
        np.testing.assert_allclose(budgeted["label_tv"], alone["label_tv"], rtol=1e-4, atol=1e-6)
        assert budgeted["whitened_mean_shift"] == pytest.approx(alone["whitened_mean_shift"], rel=1e-4, abs=1e-6)
        assert budgeted["logdet_ratio_per_dim"] == pytest.approx(alone["logdet_ratio_per_dim"], rel=1e-4, abs=1e-6)


def test_feature_probe_figures_pass_the_gap_budget_to_the_minimal_agent(monkeypatch: pytest.MonkeyPatch,
                                                                        tmp_path: Path) -> None:
    from vfe3.posthoc import feature_probes
    from vfe3.viz import report as viz_report

    seen: Dict[str, Any] = {}

    def gap(model: Any, batches: Any, device: Any, **kwargs: Any) -> None:
        seen.update(kwargs)
        return None

    monkeypatch.setattr(feature_probes, "minimal_agent_gap", gap)
    monkeypatch.setattr(feature_probes, "prototype_probe", lambda *args, **kwargs: None)
    viz_report.generate_feature_probe_figures(_minimal_agent_model(), [torch.zeros(1, 6, dtype=torch.long)],
                                              tmp_path, tmp_path, DEVICE)
    assert seen.get("gap_tokens") == viz_report.FEATURE_GAP_TOKENS


# ----------------------------------------------------------------------------------------------
# A340: the weight-EMA shadow
# ----------------------------------------------------------------------------------------------

def _write_cache(cache_dir: Path, dataset: str, split: str, tokens: torch.Tensor) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tokens.to(torch.int64).contiguous(), cache_dir / f"{dataset}_{split}_tiktoken_tokens.pt")


def _ema_run(tmp_path: Path, *, stale: bool = False) -> Tuple[Path, Path, VFEModel, Any]:
    r"""A fake run directory whose latest step checkpoint carries a started weight-EMA shadow."""
    from torch.utils.data import DataLoader

    from vfe3.data.datasets import cache_source_identity
    from vfe3.train import evaluate
    from vfe3.weight_ema import WeightEMA

    dataset, run_dir, cache_dir = "synthetic-period3", tmp_path / "run", tmp_path / "cache"
    run_dir.mkdir()
    streams = {"validation": _stream(61, 11), "test": _stream(53, 12)}
    for split, tokens in streams.items():
        _write_cache(cache_dir, dataset, split, tokens)
    cfg = _cfg(weight_ema_decay=0.5, weight_ema_start_frac=0.0, max_steps=4, warmup_steps=1)
    model = _model(cfg)
    ema = WeightEMA.from_config(model, cfg)
    ema.observe(1, True)                                             # shadow = theta_1
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and p.is_floating_point():
                p.add_(0.05 * torch.randn_like(p))
    ema.observe(2, True)                                             # shadow = 0.5 theta_1 + 0.5 theta_2
    payload = serialize_config(cfg)
    identity = {"dataset": dataset, "split": "validation",
                "source": {"sha256": cache_source_identity(dataset, "validation", cache_dir=cache_dir)["sha256"]}}
    torch.save({"model_state": model.state_dict(), "config": payload, "selection_data_identity": identity},
               run_dir / "final_model.pt")
    (run_dir / "checkpoints").mkdir()
    torch.save({"step": 2, "model_state": model.state_dict(), "config": payload, "weight_ema": ema.state_dict()},
               run_dir / "checkpoints" / "step_2.pt")

    def ce(tokens: torch.Tensor) -> float:
        loader = DataLoader(st.eval_windows(tokens, 8, 4), batch_size=2, shuffle=False)
        return float(evaluate(model, loader, device=DEVICE)["ce"])

    with ema.applied():
        val_ema, test_ema = ce(streams["validation"]), ce(streams["test"])
    summary = {"n_steps": 2, "evaluation_weights": "final", "final_val_ce": ce(streams["validation"]),
               "final_val_scored_targets": 60, "test_ce": ce(streams["test"]), "test_scored_targets": 52,
               "val_ema_ce": val_ema, "test_ema_ce": test_ema, **ema.summary()}
    if stale:
        summary["weight_ema_updates"] = ema.updates + 3              # finalize evaluated a later shadow
    provenance = {"device_name": "cpu", "selection_data_identity": identity}
    for split, prefix in (("validation", "val"), ("test", "test")):
        provenance[f"{prefix}_data_sha256"] = st.token_content_sha256(streams[split])
        provenance[f"{prefix}_data_n_tokens"] = int(streams[split].numel())
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    return run_dir, cache_dir, model, ema


def test_load_model_reads_the_ema_shadow_of_the_latest_step_checkpoint(tmp_path: Path) -> None:
    run_dir, _, model, ema = _ema_run(tmp_path)
    loaded = ev.load_model(run_dir, ev.EMA_CHECKPOINT, device=DEVICE, precision="fp32")
    assert loaded.weights == "ema" and loaded.source.endswith("step_2.pt")
    assert loaded.ema["updates"] == ema.updates == 2 and loaded.ema["step"] == 2
    shadow = ema.shadow_state()
    for name, p in loaded.model.named_parameters():
        expected = shadow.get(name, dict(model.named_parameters())[name])
        assert torch.equal(p.detach(), expected), name
    tokens = _stream(16, 3).reshape(2, 8)
    with torch.no_grad(), ema.applied():
        reference = model.forward_beliefs(tokens, return_logits=True)[1]
    with torch.no_grad():
        assert torch.equal(loaded.model.forward_beliefs(tokens, return_logits=True)[1], reference)


def test_load_model_refuses_a_run_without_an_ema_shadow(tmp_path: Path) -> None:
    cfg = _cfg()
    (tmp_path / "checkpoints").mkdir()
    torch.save({"step": 3, "model_state": _model(cfg).state_dict(), "config": serialize_config(cfg),
                "weight_ema": None}, tmp_path / "checkpoints" / "step_3.pt")
    with pytest.raises(ValueError, match="weight-EMA shadow"):
        ev.load_model(tmp_path, ev.EMA_CHECKPOINT, device=DEVICE, precision="fp32")
    with pytest.raises(FileNotFoundError):
        ev.load_model(tmp_path / "missing", ev.EMA_CHECKPOINT, device=DEVICE, precision="fp32")


def _light_options(cache_dir: Path, **over: Any) -> Any:
    values: Dict[str, Any] = dict(
        device="cpu", cache_dir=str(cache_dir), splits=("validation", "test"), checkpoints=("final_model.pt",),
        interventions=(), maps=False, resources=False, figures=False, save_nll=False, latency_batches=0,
        bootstrap_resamples=0, klcap_splits=(), attention_split=None, residual_split=None, covariance=False,
        calibration=False, routing=False)
    values.update(over)
    return rp.PosthocOptions(**values)


@pytest.mark.parametrize("stale", [False, True], ids=["current_shadow", "stale_shadow"])
def test_run_posthoc_evaluates_the_ema_weights_against_their_own_reference(tmp_path: Path, stale: bool) -> None:
    run_dir, cache_dir, _, _ = _ema_run(tmp_path, stale=stale)
    report = rp.run_posthoc(run_dir, options=_light_options(cache_dir, ema_weights=True), log=lambda message: None)
    passes = {(p["checkpoint"], p["split"]): p for p in report["passes"]}
    final, ema_val, ema_test = (passes[("final_model.pt", "validation")], passes[(ev.EMA_CHECKPOINT, "validation")],
                                passes[(ev.EMA_CHECKPOINT, "test")])
    assert final["reproduction"]["status"] == "ok" and final["precision_replays"]["replayed_batches"] == 0
    assert ema_val["status"] == "ok" and ema_val["weights"] == "ema" and ema_val["step"] == 2
    assert ema_val["cross_check"]["ok"] and abs(ema_val["ce"] - final["ce"]) > 1e-6
    if stale:
        assert ema_val["reproduction"]["status"] == "no_reference"
        assert "weight_ema_updates" in ema_val["reproduction"]["reason"]
    else:
        assert ema_val["reproduction"]["status"] == "ok" and ema_test["reproduction"]["status"] == "ok"
        assert ema_val["reproduction"]["reference"]["source"] == "summary.json val_ema_ce"
    saved = json.loads((run_dir / "posthoc" / "posthoc.json").read_text(encoding="utf-8"))
    assert saved["ema_weights"]["source"].endswith("step_2.pt")
    table = (run_dir / "posthoc" / "tables" / "relational_run.md").read_text(encoding="utf-8")
    assert "EMA weights" in table


# ----------------------------------------------------------------------------------------------
# A369: report assembly
# ----------------------------------------------------------------------------------------------

def _m5_rows() -> List[Dict[str, Any]]:
    return [
        {"row": 1, "key": "baseline", "label": "None (final weights)", "status": "ok", "ce": 3.0,
         "delta_ce": 0.0, "lo": 0.0, "hi": 0.0, "ce_lo": 2.9, "ce_hi": 3.1, "n_targets": 100},
        {"row": 7, "key": "token_frames_identity", "label": "Token frames", "status": "ok",
         "ce": 3.2, "delta_ce": 0.2, "lo": 0.1, "hi": 0.3, "n_targets": 100, "n_targets_dropped": 0},
    ]


def test_interventions_table_names_its_split_and_the_restoration_verdict() -> None:
    restoration = {"row": 0, "key": "restoration_check", "label": "baseline re-run after all arms",
                   "status": "failed", "max_abs_nll_change": 0.25, "n_targets": 12}
    text = rp.interventions_table(_m5_rows(), n_units=8, unit="article", n_targets=100, subset=True,
                                  split="test", restoration=restoration)
    assert "every test target" in text and "SUBSET of the test split" in text and "validation" not in text
    assert "Restoration check" in text and "failed" in text and "0.25" in text
    plain = rp.interventions_table(_m5_rows(), n_units=8, unit="article", n_targets=100, subset=False)
    assert "every validation target" in plain and "Restoration check" not in plain


def test_map_invariant_curves_survive_a_zero_gain(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from vfe3.posthoc import figures as fg

    saved: List[Any] = []
    monkeypatch.setattr(fg, "_save", lambda fig, out_dir, name: saved.append(fig) or [])
    rows = [{"step": step, "layer": 0, "head": head, "scale_rms": 1.0, "gain_min": 0.0 if head == 0 else 0.5,
             "gain_max": 2.0, "angle_median_deg": 10.0, "M_sigma_max": 1.5, "link_spectral_radius": 1.1}
            for step in (5, 10) for head in (0, 1)]
    fg.plot_map_invariant_curves(rows, tmp_path)
    assert len(saved) == 1


def test_link_spectrum_draws_every_layer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from vfe3.posthoc import figures as fg

    titles: List[str] = []
    monkeypatch.setattr(fg, "_save", lambda fig, out_dir, name: titles.extend(
        ax.get_title(loc="left") for ax in fig.axes if ax.get_visible() and ax.get_title(loc="left")) or [])
    rows = [{"layer": layer, "head": 0, "k": 0, "re": 0.5, "im": 0.1, "abs": 0.51} for layer in range(17)]
    fg.plot_link_spectrum(rows, tmp_path)
    assert len(titles) == 17 and any("layer 16" in title for title in titles)


def test_a_failing_map_analysis_still_writes_posthoc_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.posthoc import maps as mp

    run_dir, cache_dir, _, _ = _ema_run(tmp_path)

    def failing(*args: Any, **kwargs: Any) -> Any:
        raise ZeroDivisionError("float division by zero (injected)")

    monkeypatch.setattr(mp, "map_invariants", failing)
    report = rp.run_posthoc(run_dir, options=_light_options(cache_dir, splits=("validation",), maps=True,
                                                            resources=True),
                            log=lambda message: None)
    saved = json.loads((run_dir / "posthoc" / "posthoc.json").read_text(encoding="utf-8"))
    assert saved["maps"]["status"] == "failed" and "injected" in saved["maps"]["error"]
    assert report["passes"] and saved["passes"][0]["status"] == "ok"
