r"""Type-level frame dropout (``phi_type_dropout``) outside the training forward (audit 2026-10-31, frame-dropout).

The dropout is a training-forward regularizer: every occurrence of a dropped type in a sequence reads the identity
frame. Pinned here on tiny CPU models of the frozen reference route (no launcher read):

* diagnostics, held-out probes, attention maps and generation run on the evaluation-mode model, draw no random
  numbers and restore every module's mode (evaluation-metrics-1, model-assembly-3, train-loop-lifecycle-2,
  model-assembly-9), so the logging and evaluation cadence leaves the training trajectory bitwise unchanged under
  the dropout; the log-step diagnostics and the evaluation block hand both random streams back
  (train-loop-lifecycle-8; the device stream through a CPU stand-in for the CUDA generators);
* the training forward withholds the content-row token ids while the dropout acts, so every position realizes its
  own sequence's draw (transport-geometry-1, model-assembly-1);
* ``token_type_frames`` returns the deterministic type frames :math:`U_v = \exp(\phi_v\cdot G)` from one validated
  context and reads only the frame table (model-assembly-2, model-assembly-10), and the frame-diagonal table Fisher
  step refuses the dropout (tables-families-1);
* the dropout is a gauge flag that both route switches clear (gauge-covariance-e2e-2).
"""
from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Tuple

import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.run_artifacts as ra
import vfe3.train as T
from tests.profiles import config_values, reference_k8_config
from vfe3 import contracts
from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.exact_route import principled_exact_route_failures, principled_exact_route_overrides
from vfe3.gauge_covariant import fully_gauge_covariant_overrides, gauge_covariance_failures
from vfe3.geometry import transport
from vfe3.geometry.lie_ops import embed_phi
from vfe3.model import prior_bank as prior_bank_module
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import PriorBank, phi_type_keep_mask
from vfe3.run_artifacts import _pure_path_report

CPU = torch.device("cpu")
F64 = torch.float64
B, N = 2, 4
# Types 1 and 2 occur in both sequences, so a per-(sequence, type) draw can keep a type in one and drop it in the other.
TOKENS = torch.tensor([[1, 2, 3, 1], [4, 1, 2, 5]])
DROP = dict(phi_type_dropout=0.5, phi_type_dropout_freq_scale=None)


@pytest.fixture(autouse=True)
def _quiet_two_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
    torch.set_num_threads(previous)


def _model(cfg: VFE3Config, *, frame_scale: float = 10.0, seed: int = 0) -> VFEModel:
    r"""A model whose frames are visibly away from the identity, so a dropped frame changes every reading."""
    torch.manual_seed(seed)
    model = VFEModel(cfg)
    with torch.no_grad():
        model.prior_bank.phi_embed.mul_(frame_scale)
    return model


def _targets(cfg: VFE3Config) -> torch.Tensor:
    return torch.randint(0, cfg.vocab_size, TOKENS.shape, generator=torch.Generator().manual_seed(3))


def _same(a: Any, b: Any) -> bool:
    r"""Bitwise equality of nested diagnostic values (NaN equals NaN)."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return a == b


def _snapshot_reading(model: VFEModel, tokens: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    snapshot = model.build_diagnostic_snapshot(tokens)
    return snapshot.encoded_belief.phi, snapshot.logits, snapshot.beta_maps


_READINGS: Dict[str, Callable[[VFEModel, torch.Tensor], Any]] = {
    "snapshot":              _snapshot_reading,
    "diagnostics":           lambda model, tokens: model.diagnostics(tokens),
    "diagnostics_per_layer": lambda model, tokens: model.diagnostics_per_layer(tokens),
    "attention_maps":        lambda model, tokens: model.attention_maps(tokens, score_at="entry"),
}


@pytest.mark.parametrize("reading", sorted(_READINGS))
def test_training_mode_diagnostics_read_the_evaluation_model_and_draw_nothing(reading: str) -> None:
    model = _model(reference_k8_config(**DROP))
    call = _READINGS[reading]
    model.eval()
    reference = call(model, TOKENS)
    model.train()
    model.block_mlps.eval()                                            # mixed per-module modes are restored exactly
    modes = [module.training for module in model.modules()]
    before = torch.get_rng_state().clone()
    first, second = call(model, TOKENS), call(model, TOKENS)
    assert torch.equal(torch.get_rng_state(), before)                  # no frame-drop mask was drawn
    assert _same(first, reference) and _same(second, reference)
    assert [module.training for module in model.modules()] == modes
    assert model.prior_bank.training


def test_held_out_probes_are_measured_on_the_evaluation_model() -> None:
    model = _model(reference_k8_config(**DROP))
    batch = (TOKENS, _targets(model.cfg))
    model.eval()
    reference = T._val_diagnostics_one(model, batch, CPU).metrics
    model.train()
    before = torch.get_rng_state().clone()
    for _ in range(2):
        assert _same(T._val_diagnostics_one(model, batch, CPU).metrics, reference)
    assert torch.equal(torch.get_rng_state(), before)
    assert all(module.training for module in model.modules())


def test_log_step_diagnostics_hand_back_both_random_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Whatever a log-step diagnostic draws, the training stream after it is unchanged, on the first attempt and
    after a replay of the precision ladder."""
    model = _model(reference_k8_config(**DROP))
    attempts: List[int] = []

    def drawing(token_ids: torch.Tensor) -> Dict[str, float]:
        attempts.append(int(torch.randint(0, 1 << 30, (1,))))         # a draw from the global stream
        if len(attempts) == 1 and torch.get_float32_matmul_precision() != "highest":
            raise ValueError("a certificate tripped at reduced precision")
        return {"value": 1.0}

    monkeypatch.setattr(model, "diagnostics", drawing)
    before = torch.get_rng_state().clone()
    assert T._diagnostics_with_precision_retry(model, TOKENS) == {"value": 1.0}
    assert torch.equal(torch.get_rng_state(), before)
    previous = torch.get_float32_matmul_precision()
    attempts.clear()
    torch.set_float32_matmul_precision("medium")
    try:
        assert T._diagnostics_with_precision_retry(model, TOKENS) == {"value": 1.0}
    finally:
        torch.set_float32_matmul_precision(previous)
    assert len(attempts) == 2 and attempts[0] == attempts[1]           # the replay drew what the first attempt drew
    assert torch.equal(torch.get_rng_state(), before)


def _cadence_loaders(cfg: VFE3Config) -> Tuple[DataLoader, DataLoader]:
    train_ds = TokenWindows(torch.arange(45) % cfg.vocab_size, cfg.max_seq_len)      # 11 windows, 5 full batches
    val_ds = TokenWindows((torch.arange(23) * 3) % cfg.vocab_size, cfg.max_seq_len, pad_final=True)
    return (DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True),            # global-RNG permutations
            DataLoader(val_ds, batch_size=2, shuffle=False))


def _cadence_run(cfg: VFE3Config, tmp: Path, *, log_interval: int, eval_interval: int,
                 artifacts: bool) -> Tuple[List[float], Dict[str, torch.Tensor], Dict[str, int]]:
    model = _model(cfg, seed=5)
    loader, val = _cadence_loaders(cfg)
    run = ra.RunArtifacts(tmp / "run", cfg, model, save_model_artifacts=False) if artifacts else None
    torch.manual_seed(11)
    losses = T.train(model, loader, cfg, n_steps=cfg.max_steps, grad_clip=1.0, log_interval=log_interval,
                     eval_interval=eval_interval, val_loader=val, device=CPU,
                     logger=logging.getLogger("frame-dropout-cadence"), artifacts=run, generate_samples=False)
    failures = dict(run.diagnostics_failures) if run is not None else {}
    return losses, {k: v.detach().clone() for k, v in model.state_dict().items()}, failures


@pytest.mark.parametrize("dropout", [0.0, 0.5])
def test_the_logging_and_evaluation_cadence_leaves_training_bitwise_unchanged(
        dropout: float, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Log-step diagnostics, the held-out probes and the attention-map replay of every eval run beside training;
    under the dropout they used to draw frame-drop masks from the training stream."""
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)

    def maps_only(token_ids: torch.Tensor, model: VFEModel, artifacts: Any, logger: Any, *, step: int) -> None:
        model.attention_maps(token_ids, score_at="entry")             # the replay, without the figure child

    monkeypatch.setattr(T, "_save_eval_attention_maps", maps_only)
    cfg = reference_k8_config(phi_type_dropout=dropout, phi_type_dropout_freq_scale=None, max_steps=8)
    silent, silent_state, _ = _cadence_run(cfg, tmp_path / "silent", log_interval=0, eval_interval=0,
                                           artifacts=False)
    loud, loud_state, failures = _cadence_run(cfg, tmp_path / "loud", log_interval=1, eval_interval=4,
                                              artifacts=True)
    assert loud == silent
    assert set(loud_state) == set(silent_state)
    for name in silent_state:
        assert torch.equal(loud_state[name], silent_state[name]), name
    assert failures == {}


def test_the_evaluation_block_hands_back_the_device_random_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""A device draw inside the evaluation block does not reach the next training step (CPU stand-in for the CUDA
    generators: the block captures and restores them through the two module helpers)."""

    class DeviceStream:
        def __init__(self) -> None:
            self.offset = 0

        def states(self) -> List[torch.Tensor]:
            return [torch.tensor([self.offset])]

        def restore(self, states: List[torch.Tensor]) -> None:
            self.offset = int(states[0])

    stream = DeviceStream()
    monkeypatch.setattr(T, "_cuda_rng_states", stream.states)
    monkeypatch.setattr(T, "_set_cuda_rng_states", stream.restore)
    real_evaluate, real_step = T.evaluate, T._train_step_with_precision_retry

    def evaluate_with_a_device_draw(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        stream.offset += 1
        return real_evaluate(*args, **kwargs)

    seen: List[int] = []

    def recording_step(*args: Any, **kwargs: Any) -> Any:
        seen.append(stream.offset)
        return real_step(*args, **kwargs)

    monkeypatch.setattr(T, "evaluate", evaluate_with_a_device_draw)
    monkeypatch.setattr(T, "_train_step_with_precision_retry", recording_step)
    cfg = reference_k8_config(max_steps=6)
    loader, val = _cadence_loaders(cfg)
    T.train(_model(cfg), loader, cfg, n_steps=6, grad_clip=1.0, log_interval=0, eval_interval=2, val_loader=val,
            device=CPU, logger=logging.getLogger("frame-dropout-device-stream"), generate_samples=False)
    assert seen == [0] * 6


class _PerOccurrenceModel(VFEModel):
    r"""Fails the content gate's ``type(self) is VFEModel`` check, so it always takes the per-occurrence frame route."""


def _spy_content_ids(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[bool, torch.Tensor]]:
    r"""Record, per compact frame exponential, whether token ids were passed and the occurrence frames it returned."""
    seen: List[Tuple[bool, torch.Tensor]] = []
    original = transport._stable_compact_glk_exp_pair

    def recorded(blocks: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        pair = original(blocks, **kwargs)
        seen.append((kwargs.get("_content_token_ids") is not None, pair[0].detach().clone()))
        return pair

    monkeypatch.setattr(transport, "_stable_compact_glk_exp_pair", recorded)
    return seen


@pytest.mark.parametrize("dropout, training, deduplicated", [(0.5, True, False), (0.5, False, True),
                                                             (0.0, True, True)])
def test_the_content_row_gate_withholds_token_ids_while_frames_drop(
        dropout: float, training: bool, deduplicated: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model(reference_k8_config(phi_type_dropout=dropout, phi_type_dropout_freq_scale=None))
    model.train(training)
    seen = _spy_content_ids(monkeypatch)
    torch.manual_seed(7)
    model(TOKENS, _targets(model.cfg))[1].backward()
    assert seen and all(passed == deduplicated for passed, _ in seen)


def test_every_position_realizes_its_own_sequences_frame_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = reference_k8_config(**DROP)
    model = _model(cfg)
    torch.manual_seed(0)
    per_occurrence = _PerOccurrenceModel(cfg)
    per_occurrence.load_state_dict(model.state_dict())
    targets = _targets(cfg)

    def draw(seed: int) -> torch.Tensor:                               # (B, N) the keep mask the forward draws
        torch.manual_seed(seed)
        return phi_type_keep_mask(model.prior_bank, TOKENS)

    def splits_a_shared_type(keep: torch.Tensor) -> bool:
        return any(bool(keep[0][TOKENS[0] == v].all()) != bool(keep[1][TOKENS[1] == v].all()) for v in (1, 2))

    # The first seed whose draw keeps a shared type in one sequence and drops it in the other.
    seed = next(s for s in range(100) if splits_a_shared_type(draw(s)))
    keep = draw(seed)
    seen = _spy_content_ids(monkeypatch)
    results = []
    for m in (model, per_occurrence):
        m.train()
        torch.manual_seed(seed)
        _, loss, _ = m(TOKENS, targets)
        loss.backward()
        results.append((loss.detach(), {n: p.grad.detach().clone() for n, p in m.named_parameters()
                                         if p.grad is not None}))
    assert torch.equal(results[0][0], results[1][0])
    assert set(results[0][1]) == set(results[1][1])
    for name, grad in results[0][1].items():
        assert torch.equal(grad, results[1][1][name]), name
    frames = seen[0][1]                                                # (B, N, H, d, d) occurrence frames of the loss
    identity = torch.eye(frames.shape[-1], dtype=frames.dtype).expand_as(frames)
    is_identity = (frames == identity).flatten(2).all(-1)              # (B, N)
    assert torch.equal(is_identity, ~keep)


def _frame_diagonal(**over: Any) -> VFE3Config:
    r"""The frame-diagonal family on the frozen reference route at K = 8, block_glk per-head frames of width 4."""
    values = dict(family="gaussian_frame_diagonal", predictive_readout="kernel", block_mlp_mode="coordinate",
                  block_mlp_covariance="passthrough", decode_mode="diagonal_chunked", norm_type_block="layernorm",
                  untie_decode_bank=True, gauge_group="block_glk", gauge_tying="per_head", pos_phi="none",
                  rope_mode="none", vocab_size=9, embed_dim=8, n_heads=2, weight_decay=0.0, mu_weight_decay=0.0,
                  sigma_weight_decay=0.0)
    return reference_k8_config(**{**values, **over})


def _float64_frame_model(cfg: VFE3Config) -> VFEModel:
    torch.manual_seed(0)
    model = VFEModel(cfg).double()
    gen = torch.Generator().manual_seed(1)
    pb = model.prior_bank
    with torch.no_grad():
        pb.phi_embed.copy_(0.5 * torch.randn(pb.phi_embed.shape, generator=gen, dtype=F64))
        pb.mu_embed.copy_(torch.randn(pb.mu_embed.shape, generator=gen, dtype=F64))
        pb.sigma_log_embed.copy_(0.4 * torch.randn(pb.sigma_log_embed.shape, generator=gen, dtype=F64))
    return model


def _definition_blocks(model: VFEModel, rows: torch.Tensor) -> torch.Tensor:   # (R, H, d, d) U_v head blocks
    G = model.group.generators.to(F64)                                 # (n_gen, K, K)
    U = torch.linalg.matrix_exp(embed_phi(model.prior_bank.phi_embed.detach().to(F64), G))[rows]   # (R, K, K)
    d = model.cfg.embed_dim // model.cfg.n_heads
    return torch.stack([U[:, h * d:(h + 1) * d, h * d:(h + 1) * d] for h in range(model.cfg.n_heads)], dim=1)


def test_type_frames_ignore_the_training_mode_dropout_and_draw_nothing() -> None:
    model = _float64_frame_model(_frame_diagonal(m_table_update_mode="adamw", **DROP))
    rows = torch.arange(model.cfg.vocab_size)
    model.eval()
    reference = model.token_type_frames(rows)
    model.train()
    before = torch.get_rng_state().clone()
    first, second = model.token_type_frames(rows), model.token_type_frames(rows)
    assert torch.equal(torch.get_rng_state(), before)
    for frame in (first, second):
        assert frame.block_diagonal and torch.equal(frame.forward, reference.forward)
        assert torch.equal(frame.inverse, reference.inverse)
    torch.testing.assert_close(first.forward, _definition_blocks(model, rows), rtol=0.0, atol=1e-12)
    assert model.prior_bank.training


def test_type_frames_validate_one_context_and_read_only_the_frame_table(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _float64_frame_model(_frame_diagonal(m_table_update_mode="fisher"))
    rows = torch.tensor([1, 2, 3, 5, 7])
    validated: List[Tuple[int, ...]] = []
    original_post_init = contracts.CanonicalFrameContext.__post_init__

    def counted(self: contracts.CanonicalFrameContext) -> None:
        validated.append(tuple(self.forward.shape))
        original_post_init(self)

    mean_reads: List[bool] = []
    original_mean = PriorBank._prior_mu_table

    def mean_table(self: PriorBank) -> torch.Tensor:
        mean_reads.append(True)
        return original_mean(self)

    monkeypatch.setattr(contracts.CanonicalFrameContext, "__post_init__", counted)
    monkeypatch.setattr(PriorBank, "_prior_mu_table", mean_table)
    frame = model.token_type_frames(rows)
    assert validated == [(rows.numel(), 2, 4, 4)]                     # one float64 check, on the R rows
    assert mean_reads == []                                            # no mean (or variance) rows are gathered
    torch.testing.assert_close(frame.forward, _definition_blocks(model, rows), rtol=0.0, atol=1e-12)


def test_type_frames_of_another_encoder_run_it_in_evaluation_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""An encoder other than the built-in table lookup is called with the prior bank in evaluation mode, whose
    training mode is restored afterwards."""
    model = _float64_frame_model(_frame_diagonal(m_table_update_mode="adamw", **DROP))
    rows = torch.arange(model.cfg.vocab_size)
    model.eval()
    reference = model.token_type_frames(rows)
    builtin = prior_bank_module._ENCODERS["per_token"]
    modes: List[bool] = []

    def wrapped(pb: PriorBank, token_ids: torch.Tensor) -> Any:
        modes.append(pb.training)
        return builtin(pb, token_ids)

    monkeypatch.setitem(prior_bank_module._ENCODERS, "per_token", wrapped)
    model.train()
    before = torch.get_rng_state().clone()
    frame = model.token_type_frames(rows)
    assert modes == [False] and model.prior_bank.training
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(frame.forward, reference.forward) and torch.equal(frame.inverse, reference.inverse)


def test_the_table_fisher_step_captures_deterministic_frames() -> None:
    r"""Even with the dropout switched on behind the validator's back, the optimizer's frame capture reads the
    deterministic type frames and draws nothing: the step equals the evaluation-mode step bitwise."""
    cfg = _frame_diagonal(m_table_update_mode="fisher", m_table_fisher_lr=0.3, m_table_fisher_trust_radius=1e9)
    steps = []
    for training in (False, True):
        model = _float64_frame_model(cfg)
        model.prior_bank.phi_type_dropout = 0.5
        model.train(training)
        g = torch.Generator().manual_seed(0)
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.randn(p.shape, generator=g, dtype=p.dtype)
        optimizer = T.build_optimizer(model, cfg)
        assert optimizer.table_statistics_provider.reads_frames
        pb = model.prior_bank
        before_mu = pb.mu_embed.detach().clone()
        optimizer.table_statistics_provider.set_batch(torch.tensor([[1, 2, 2, 5], [1, 1, 3, 7]]), torch.tensor(8))
        rng = torch.get_rng_state().clone()
        optimizer.step()
        assert torch.equal(torch.get_rng_state(), rng)
        steps.append(pb.mu_embed.detach() - before_mu)
    assert torch.equal(steps[0], steps[1])
    assert float(steps[0].abs().max()) > 0.0


def test_the_frame_diagonal_table_fisher_step_refuses_type_dropout() -> None:
    with pytest.raises(ValueError, match="m_table_update_mode='fisher' with phi_type_dropout=0.1"):
        _frame_diagonal(m_table_update_mode="fisher", phi_type_dropout=0.1)
    # Identity frames, the AdamW table step and no dropout are admitted.
    assert _frame_diagonal(m_table_update_mode="fisher", phi_type_dropout=0.1, gauge_transport="off").phi_type_dropout
    assert _frame_diagonal(m_table_update_mode="adamw", phi_type_dropout=0.1).phi_type_dropout == 0.1
    assert _frame_diagonal(m_table_update_mode="fisher").phi_type_dropout == 0.0


def test_generation_samples_the_evaluation_model_and_restores_every_mode() -> None:
    cfg = reference_k8_config(**DROP, block_mlp_dropout=0.5)
    model = _model(cfg)
    prompt = TOKENS[:, :2]
    model.eval()
    greedy = model.generate(prompt, 4, greedy=True)
    torch.manual_seed(3)
    sampled = model.generate(prompt, 4, greedy=False, top_k=3)
    model.train()
    model.block_mlps.eval()
    modes = [module.training for module in model.modules()]
    before = torch.get_rng_state().clone()
    assert torch.equal(model.generate(prompt, 4, greedy=True), greedy)
    assert torch.equal(torch.get_rng_state(), before)                  # greedy decoding drew nothing
    torch.manual_seed(3)
    assert torch.equal(model.generate(prompt, 4, greedy=False, top_k=3), sampled)
    assert [module.training for module in model.modules()] == modes


def _covariant_raw(**over: Any) -> Dict[str, Any]:
    r"""Projected-predictive kernel readout on the frame-diagonal family with the frame-dependent defaults of the test
    profile, so the gauge switch has work to do."""
    return config_values(attention_mode="projected_predictive", predictive_readout="kernel",
                         predictive_covariance="moment_matched", lambda_alpha_mode="state_dependent_per_coord",
                         family="gaussian_frame_diagonal", decode_mode="diagonal_chunked", pos_phi="none",
                         vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=N, n_layers=2, batch_size=B,
                         block_mlp_expansion=2, **over)


def test_frame_dropout_is_a_gauge_flag_that_both_switches_clear() -> None:
    raw = _covariant_raw(**DROP)
    overrides = fully_gauge_covariant_overrides(raw)
    assert overrides["phi_type_dropout"] == 0.0
    assert gauge_covariance_failures(VFE3Config(**{**raw, **overrides})) == []
    reverted = VFE3Config(**{**raw, **overrides, "phi_type_dropout": 0.5})
    assert gauge_covariance_failures(reverted) == ["no_frame_dropout"]
    report = _pure_path_report(reverted, [])
    assert report["gauge_flags"]["no_frame_dropout"] is False and report["on_gauge_pure_path"] is False
    # Inert on identity frames, and off by default.
    off = VFE3Config(**config_values(**DROP, gauge_transport="off"))
    assert _pure_path_report(off, [])["gauge_flags"]["no_frame_dropout"] is True
    assert _pure_path_report(VFE3Config(**config_values()), [])["gauge_flags"]["no_frame_dropout"] is True
    assert "phi_type_dropout" not in fully_gauge_covariant_overrides(_covariant_raw())
    # The exact route inherits the override and names the flag when the dropout is put back.
    values = config_values(**DROP)
    exact = principled_exact_route_overrides(values)
    assert exact["phi_type_dropout"] == 0.0
    assert principled_exact_route_failures(VFE3Config(**{**values, **exact})) == []
    back = VFE3Config(**{**values, **exact, "phi_type_dropout": 0.5})
    assert "gauge flag no_frame_dropout" in principled_exact_route_failures(back)


def _gauge_model(cfg: VFE3Config) -> VFEModel:
    r"""A float64 model with every parameter moved off its initialization."""
    torch.manual_seed(0)
    model = VFEModel(cfg).double()
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=gen, dtype=p.dtype))
        model.prior_bank.phi_embed.mul_(5.0)
    return model


def _table_frame_change(model: VFEModel, s: float = 0.3) -> None:
    r"""The passive action of one :math:`g_v` per type on the tables: :math:`\phi_v \to \phi_v'`,
    :math:`g_v = e^{\phi_v'\cdot G}e^{-\phi_v\cdot G}`, :math:`\mu_v \to g_v\mu_v`, with the reference variances
    :math:`c_v` and the untied reference decode bank fixed."""
    pb = model.prior_bank
    G = model.group.generators.to(F64)
    gen = torch.Generator().manual_seed(7)
    with torch.no_grad():
        U = torch.linalg.matrix_exp(embed_phi(pb.phi_embed, G))                        # (V, K, K)
        phi_new = pb.phi_embed + s * torch.randn(pb.phi_embed.shape, generator=gen, dtype=F64)
        g = torch.linalg.matrix_exp(embed_phi(phi_new, G)) @ torch.linalg.inv(U)      # (V, K, K)
        pb.phi_embed.copy_(phi_new)
        pb.mu_embed.copy_((g @ pb.mu_embed.unsqueeze(-1)).squeeze(-1))


def _loss(model: VFEModel, targets: torch.Tensor, *, training: bool, seed: int) -> float:
    model.train(training)
    torch.manual_seed(seed)                                            # the same keep masks before and after
    with torch.no_grad():
        return float(model(TOKENS, targets)[1])


@pytest.mark.parametrize("dropout, training, invariant", [(0.0, True, True), (0.5, False, True),
                                                          (0.5, True, False)])
def test_a_table_level_frame_change_moves_only_a_frame_dropped_training_forward(
        dropout: float, training: bool, invariant: bool) -> None:
    raw = _covariant_raw(**DROP)
    cfg = VFE3Config(**{**raw, **fully_gauge_covariant_overrides(raw), "phi_type_dropout": dropout})
    model = _gauge_model(cfg)
    targets = _targets(cfg)
    seed = 7
    if dropout > 0.0:
        torch.manual_seed(seed)
        assert not bool(phi_type_keep_mask(model.prior_bank, TOKENS).all())   # the seed drops a type
    before = _loss(model, targets, training=training, seed=seed)
    _table_frame_change(model)
    after = _loss(model, targets, training=training, seed=seed)
    change = abs(after - before) / abs(before)
    assert (change < 1e-9) if invariant else (change > 1e-6), change
