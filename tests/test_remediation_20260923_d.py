r"""Deep-audit 2026-09-23 remediation, group D: model, decode, training, gradients, data tools.

Each test pins one punch-list item on a tiny CPU model (single-digit dims, one or two layers).
"""

import csv
import inspect
import json
import math
import warnings
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import build_wikitext103_word_cache as builder
import generate_text
import vfe3.data.datasets as datasets_mod
import vfe3.train as train_module
from tests.test_relational_filtering import _fixture
from tests.test_relational_integration import _cfg as _relational_cfg
from tests.test_word_level_20260918 import fake_corpus  # noqa: F401  (pytest fixture)
from vfe3.config import VFE3Config
from vfe3.contracts import CanonicalFrameContext
from vfe3.data.datasets import TokenWindows, make_dataloader
from vfe3.data.word_vocab import WordVocabulary, vocabulary_path
from vfe3.families.base import get_family
from vfe3.free_energy import self_divergence_for_alpha
from vfe3.geometry.transport import RopeTransport, transport_mean
from vfe3.gradients.qkv import qkv_filtering_gradients
from vfe3.gradients.relational import relational_score
from vfe3.model import prior_bank as prior_bank_module
from vfe3.model.model import VFEModel, _derived_seed
from vfe3.model.relational_operators import RelationalOperators
from vfe3.run_artifacts import RunArtifacts
from vfe3.train import _eval_protocol_train_ce, build_optimizer, train


def _tiny(**overrides: object) -> VFE3Config:
    base = dict(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, e_q_mu_lr=0.1, e_phi_lr=0.0, m_phi_lr=0.0,
                warmup_steps=1, max_steps=4)
    base.update(overrides)
    return VFE3Config(**base)


def _sequential_loader(seq_len: int = 8, batch_size: int = 4, n: int = 105,
                       stride: int | None = None, evaluation: bool = False) -> DataLoader:
    tokens = torch.randint(0, 6, (n,), generator=torch.Generator().manual_seed(123))
    windows = TokenWindows(tokens.to(torch.long), seq_len, stride=stride, pad_final=evaluation)
    return DataLoader(windows, batch_size=batch_size, shuffle=False, drop_last=not evaluation)


# ---- #2 decode temperature ------------------------------------------------------------------

def test_decode_temperature_learned_false_freezes_scale_and_keeps_its_name() -> None:
    torch.manual_seed(0)
    cfg = _tiny(decode_mode="diagonal_chunked", decode_bias=False, decode_tau=0.5,
                decode_temperature_learned=False)
    model = VFEModel(cfg)
    scale = model.prior_bank.decode_log_scale
    assert "prior_bank.decode_log_scale" in model.state_dict()
    assert not scale.requires_grad
    optimizer = build_optimizer(model, cfg)
    assert all(parameter is not scale for group in optimizer.param_groups
               for parameter in group["params"])
    tokens = torch.randint(0, 6, (2, 8))
    _, loss, _ = model(tokens, torch.roll(tokens, -1, dims=1))
    loss.backward()
    assert scale.grad is None
    assert float(model.prior_bank._tau_eff()) == pytest.approx(0.5, abs=0.0)


def test_decode_temperature_learned_default_keeps_the_learned_scale() -> None:
    cfg = _tiny(decode_mode="diagonal_chunked", decode_bias=False)
    assert cfg.decode_temperature_learned is True
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    sigma_groups = [group for group in optimizer.param_groups if group.get("role") == "sigma"]
    assert any(parameter is model.prior_bank.decode_log_scale
               for group in sigma_groups for parameter in group["params"])


def test_frozen_decode_temperature_is_inert_under_linear_decode() -> None:
    cfg = _tiny(decode_mode="linear", decode_temperature_learned=False)
    assert "decode_temperature_learned" in cfg.inactive_fields()


# ---- #4 decode GEMM precision ---------------------------------------------------------------

def _decode_av_operands() -> tuple:
    generator = torch.Generator().manual_seed(4)
    sq = torch.rand(5, 4, generator=generator) + 0.5
    mc_q = torch.randn(5, 4, generator=generator)
    mc_v = torch.randn(7, 4, generator=generator)
    inv_v = torch.rand(7, 4, generator=generator) + 0.5
    lsum = torch.randn(7, generator=generator)
    return sq, mc_q, mc_v, inv_v, lsum


def test_decode_av_gemm_runs_ieee_fp32_and_restores_the_global_policy() -> None:
    from torch.overrides import TorchFunctionMode

    sq, mc_q, mc_v, inv_v, lsum = _decode_av_operands()
    seen: list = []

    class _RecordMatmulPrecision(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            if getattr(func, "__name__", "") in ("matmul", "__matmul__"):
                seen.append(torch.get_float32_matmul_precision())
            return func(*args, **(kwargs or {}))

    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        reference = prior_bank_module._decode_av(
            sq, mc_q, mc_v, inv_v, lsum, return_error_bound=True, precision="fp32")
        torch.set_float32_matmul_precision("high")
        with _RecordMatmulPrecision():
            actual = prior_bank_module._decode_av(
                sq, mc_q, mc_v, inv_v, lsum, return_error_bound=True, precision="fp32")
        assert seen and set(seen) == {"highest"}
        assert torch.get_float32_matmul_precision() == "high"
        assert torch.equal(actual[0], reference[0]) and torch.equal(actual[1], reference[1])
        with pytest.raises(RuntimeError):                       # (5, 8) @ (6, 7): the GEMM raises
            prior_bank_module._decode_av(sq, mc_q, mc_v[:, :3], inv_v[:, :3], lsum,
                                         precision="fp32")
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(previous)


# ---- #16 relational output init ------------------------------------------------------------

@pytest.mark.parametrize("scope", ["head", "full"])
def test_relational_output_init_identity_and_depth_scaled(scope: str) -> None:
    common = dict(attention_mode="projected_predictive", relational_output_scope=scope,
                  attention_residual=True, n_layers=2)
    identity = VFEModel(_relational_cfg(**common))
    scaled = VFEModel(_relational_cfg(**common, relational_output_init="depth_scaled"))
    assert identity.cfg.relational_output_init == "identity"
    factor = 1.0 / math.sqrt(2.0 * 2)
    for plain, depth in zip(identity.relational_operators_by_layer,
                            scaled.relational_operators_by_layer):
        eye_blocks = torch.eye(2).expand(2, 2, 2)
        if scope == "head":
            assert torch.equal(plain.O, eye_blocks)
        else:
            for h in range(2):
                assert torch.equal(plain.O[h, 2 * h:2 * h + 2], eye_blocks[h])
        torch.testing.assert_close(depth.O, plain.O * factor, rtol=0, atol=1e-7)
        assert torch.equal(depth.T, plain.T)
        assert torch.equal(depth.raw_score_q, plain.raw_score_q)


def test_depth_scaled_output_init_is_rejected_without_learned_output_maps() -> None:
    with pytest.raises(ValueError, match="relational_output_init"):
        _tiny(relational_output_init="depth_scaled")
    with pytest.raises(ValueError, match="relational_output_init"):
        _tiny(relational_output_init="bogus")


# ---- #17 generalization gap ----------------------------------------------------------------

def test_eval_protocol_train_ce_scores_only_the_sliding_window_tail() -> None:
    torch.manual_seed(1)
    model = VFEModel(_tiny(decode_mode="diagonal_chunked", decode_bias=False))
    model.train()
    tokens = torch.randint(0, 6, (2, 8))
    targets = torch.roll(tokens, -1, dims=1)
    rng = torch.get_rng_state()
    actual = _eval_protocol_train_ce(model, tokens, targets, eval_stride=3)
    assert model.training
    assert torch.equal(torch.get_rng_state(), rng)
    model.eval()
    with torch.no_grad():
        logits = model(tokens)                                   # (B, N, V) inference path
    expected = F.cross_entropy(logits[:, -3:].reshape(-1, 6), targets[:, -3:].reshape(-1))
    assert actual == pytest.approx(float(expected), rel=1e-4, abs=1e-5)


def test_generalization_gap_uses_the_validation_protocol_under_a_sliding_window(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list = []
    real = train_module._eval_protocol_train_ce

    def _record(model, tokens, targets, eval_stride):
        value = real(model, tokens, targets, eval_stride)
        recorded.append((eval_stride, value))
        return value

    monkeypatch.setattr(train_module, "_eval_protocol_train_ce", _record)
    cfg = _tiny(decode_mode="diagonal_chunked", decode_bias=False, eval_stride=4, max_steps=2)
    torch.manual_seed(2)
    model = VFEModel(cfg)
    artifacts = RunArtifacts(tmp_path / "run", cfg, model)
    train(model, _sequential_loader(), cfg, n_steps=2, eval_interval=1,
          val_loader=_sequential_loader(stride=4, evaluation=True), artifacts=artifacts,
          generate_samples=False)
    with open(tmp_path / "run" / "metrics.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [stride for stride, _ in recorded] == [4, 4]
    for row, (_, train_side) in zip(rows, recorded):
        assert float(row["generalization_gap"]) == pytest.approx(
            float(row["val_ce"]) - train_side, rel=1e-12, abs=1e-12)
        # The prior-bank KL decode is active, so the realized temperature is logged every row.
        assert float(row["decode_tau_eff"]) > 0.0


def test_generalization_gap_is_unchanged_with_disjoint_validation(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _forbidden(*args, **kwargs):
        raise AssertionError("disjoint validation must keep the per-step train CE")

    monkeypatch.setattr(train_module, "_eval_protocol_train_ce", _forbidden)
    cfg = _tiny(max_steps=1)
    torch.manual_seed(3)
    model = VFEModel(cfg)
    artifacts = RunArtifacts(tmp_path / "run", cfg, model)
    train(model, _sequential_loader(), cfg, n_steps=1, eval_interval=1,
          val_loader=_sequential_loader(evaluation=True), artifacts=artifacts,
          generate_samples=False)
    with open(tmp_path / "run" / "metrics.csv", newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["generalization_gap"]) == pytest.approx(
        float(row["val_ce"]) - float(row["train_ce"]), rel=1e-12, abs=1e-12)
    assert "decode_tau_eff" not in row                      # linear decode reads no temperature


# ---- #20 transported F diagnostics ---------------------------------------------------------

def _self_divergence(model: VFEModel, q, prior) -> float:
    cfg = model.cfg
    fam = get_family(cfg.family)
    mu_p, sigma_p = prior
    return float(self_divergence_for_alpha(
        model._family_instance(fam, q.mu, q.sigma),
        model._family_instance(fam, mu_p, sigma_p),
        alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
        divergence_family=cfg.divergence_family, lambda_alpha_mode=cfg.lambda_alpha_mode,
    ).sum())


@pytest.mark.parametrize("overrides", [
    dict(n_layers=2, n_e_steps=2, prior_handoff_rho=1.0, norm_type_block="none",
         use_block_mlp=False),
    dict(n_layers=1, n_e_steps=2, norm_type_block="layernorm", use_block_mlp=True),
])
def test_transported_diagnostics_score_the_e_step_state_against_its_live_prior(
        overrides: dict) -> None:
    torch.manual_seed(5)
    cfg = _tiny(vocab_size=12, max_seq_len=4, e_q_mu_lr=1.0, mu_init_std=0.5, pos_phi="none",
                pos_rotation="none", **overrides)
    model = VFEModel(cfg).eval()
    tokens = torch.tensor([[0, 5, 2, 9]])
    snapshot = model.build_diagnostic_snapshot(tokens)
    q_star = snapshot.layer_converged[-1]
    prior = snapshot.layer_priors[-1]
    q0 = q_star._replace(mu=q_star.mu[0], sigma=q_star.sigma[0])
    expected = _self_divergence(model, q0, (prior[0][0], prior[1][0]))
    # A real E-step displacement. The retired handoff reconstruction set mu_p = out.mu at rho=1,
    # L=2, which deleted exactly this Mahalanobis term (it reported 0 here).
    assert expected > 1e-2
    replayed = model.diagnostics(tokens)
    from_snapshot = model.diagnostics(tokens, snapshot=snapshot)
    assert replayed["self_divergence"] == pytest.approx(expected, rel=1e-5, abs=1e-7)
    assert from_snapshot["self_divergence"] == pytest.approx(expected, rel=1e-5, abs=1e-7)
    if overrides["use_block_mlp"]:
        # The post-transform state keeps its own, separately named column.
        assert replayed["post_block_mlp_self_divergence"] != pytest.approx(expected, rel=1e-3)


# ---- #22 projected_conditional value gauge ----------------------------------------------------

def _rope(n: int, dtype: torch.dtype) -> torch.Tensor:
    rotation = torch.zeros(n, 4, 4, dtype=dtype)
    for position in range(n):
        for block, frequency in enumerate((0.7, 0.3)):
            angle = torch.tensor(frequency * (position + 1), dtype=dtype)
            c, s = torch.cos(angle), torch.sin(angle)
            rotation[position, 2 * block:2 * block + 2, 2 * block:2 * block + 2] = torch.stack(
                (torch.stack((c, -s)), torch.stack((s, c))))
    return rotation                                              # (N, K, K) block-diagonal


@pytest.mark.parametrize("on_value", [False, True])
def test_conditional_values_use_the_value_gauge(on_value: bool) -> None:
    mu, sigma, omega, frame, ops, prior = _fixture()
    rope = RopeTransport(base=omega, rope=_rope(3, mu.dtype), on_value=on_value,
                         same_frame_flat_cocycle=True)
    capture: dict = {}
    qkv_filtering_gradients(mu, sigma, mu.detach() + 0.2, sigma, rope, frame,
                            value_frame=frame, log_prior=prior, tau=0.9, capture=capture,
                            relational_operators=ops)
    values = capture["conditional_context"].value_templates      # (1, N, N, K)
    content = ops.apply_content(mu.detach(), frame)
    unrotated = transport_mean(omega, content)                   # U_i O T U_j^{-1} mu_j
    rotated = transport_mean(rope, content)                      # U_i R_i^T R_j O T U_j^{-1} mu_j
    assert not torch.allclose(unrotated, rotated)
    torch.testing.assert_close(values, unrotated if not on_value else rotated,
                               rtol=1e-12, atol=1e-12)


# ---- #25 word cache builder and loader --------------------------------------------------------

def test_refused_word_cache_rerun_leaves_every_artifact_untouched(
        tmp_path: Path, fake_corpus) -> None:  # noqa: F811
    builder.build(min_count=1, cache_dir=tmp_path)
    sidecar = vocabulary_path(builder.DATASET, tag="word", cache_dir=tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(SystemExit, match="already exists"):
        builder.build(min_count=2, cache_dir=tmp_path)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
    assert WordVocabulary.load(sidecar).min_count == 1
    assert not any(name.endswith(".tmp") for name in before)


def test_word_loader_fails_closed_when_the_sidecar_disagrees_with_the_cache(
        tmp_path: Path, fake_corpus) -> None:  # noqa: F811
    builder.build(min_count=1, cache_dir=tmp_path)
    loader = make_dataloader(builder.DATASET, "train", 2, 1, shuffle=False, drop_last=False,
                             cache_dir=tmp_path)
    assert loader.dataset.data_identity["tokenizer_vocab_size"] == 3
    sidecar = vocabulary_path(builder.DATASET, tag="word", cache_dir=tmp_path)
    WordVocabulary(["<unk>", "the", "cat", "dog", "sat"], min_count=0).save(sidecar)
    with pytest.raises(ValueError, match="vocabulary sidecar"):
        make_dataloader(builder.DATASET, "train", 2, 1, shuffle=False, drop_last=False,
                        cache_dir=tmp_path)
    sidecar.unlink()
    with pytest.raises(FileNotFoundError, match="vocabulary sidecar"):
        make_dataloader(builder.DATASET, "train", 2, 1, shuffle=False, drop_last=False,
                        cache_dir=tmp_path)


# ---- Low: click-to-run entry points ---------------------------------------------------------

def test_entry_points_are_click_to_run() -> None:
    for module in (generate_text, builder):
        source = inspect.getsource(module)
        assert "argparse" not in source
        assert list(inspect.signature(module.main).parameters) == []
    assert builder.MIN_COUNT == builder.DEFAULT_MIN_COUNT and builder.FORCE is False
    assert generate_text.RUN_DIR is None


def test_generate_text_defaults_to_the_newest_launcher_run(tmp_path: Path) -> None:
    import os
    import time

    def _run(name: str, stamp: int) -> Path:
        run = tmp_path / name
        run.mkdir(parents=True)
        for artifact in ("config.json", "best_model.pt"):
            (run / artifact).write_text("x", encoding="utf-8")
            os.utime(run / artifact, ns=(stamp, stamp))
        return run

    now = time.time_ns()
    _run("109.33_wikitext-103_K20_block_glk_mlp_s6", now - 10**9)
    newest = _run("20260923-104116_wikitext-103_K20_block_glk_mlp_s7", now)
    (tmp_path / "stale_without_weights").mkdir()
    (tmp_path / "stale_without_weights" / "config.json").write_text("x", encoding="utf-8")
    assert generate_text.newest_run_dir(tmp_path) == newest
    with pytest.raises(FileNotFoundError, match="RUN_DIR"):
        generate_text.newest_run_dir(tmp_path / "empty")


# ---- Low: stream_ce residue -----------------------------------------------------------------

class _SurpriseAtThree(torch.nn.Module):
    """Predicts token 0 with logit 30 everywhere, so only a target 3 costs about 30 nats."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(*inputs.shape, 5)
        logits[..., 0] = 30.0
        return logits


@pytest.mark.parametrize("n_prefix,n_tail,seq_len", [
    (5, 9, 4), (5, 10, 4), (128, 129, 128), (128, 257, 128), (1, 5, 4)])
def test_stream_ce_scores_every_tail_token_exactly_once(
        n_prefix: int, n_tail: int, seq_len: int) -> None:
    prefix = torch.zeros(n_prefix, dtype=torch.long)
    tail = torch.zeros(n_tail, dtype=torch.long)
    tail[0] = 3                                                   # only the first tail token surprises
    surprise = math.log(math.exp(30.0) + 4.0)                     # CE of target 3
    background = math.log(1.0 + 4.0 * math.exp(-30.0))            # CE of target 0
    expected = (surprise + (n_tail - 1) * background) / n_tail
    actual = generate_text.stream_ce(_SurpriseAtThree(), prefix, tail, seq_len)
    assert actual == pytest.approx(expected, rel=1e-6)


# ---- Low: relational_score frame consistency ------------------------------------------------

def test_relational_score_rejects_head_coupling_dense_frames_and_accepts_block_ones() -> None:
    mu, sigma, _, frame, ops, _ = _fixture()
    blocks = torch.stack([frame.inverse[..., 0:2, 0:2], frame.inverse[..., 2:4, 2:4]], dim=-3)
    compact = CanonicalFrameContext(
        torch.stack([frame.forward[..., 0:2, 0:2], frame.forward[..., 2:4, 2:4]], dim=-3),
        blocks, block_diagonal=True)
    torch.testing.assert_close(relational_score(mu, sigma, mu, sigma, frame, ops),
                               relational_score(mu, sigma, mu, sigma, compact, ops),
                               rtol=0, atol=0)
    coupled = frame.forward.detach().clone()
    coupled[..., 0, 3] = 0.25                                     # an off-diagonal head block
    dense = CanonicalFrameContext(coupled, torch.linalg.inv(coupled))
    with pytest.raises(ValueError, match="head-block-diagonal"):
        relational_score(mu, sigma, mu, sigma, dense, ops)


# ---- Low: decode_last slices before decoding ------------------------------------------------

def test_decode_last_decodes_only_the_final_position(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(7)
    model = VFEModel(_tiny(decode_mode="diagonal_chunked", decode_bias=False)).eval()
    tokens = torch.randint(0, 6, (2, 5))
    with torch.no_grad():
        _, full = model.forward_beliefs(tokens, return_logits=True)
    shapes: list = []
    real_decode = model.prior_bank.decode

    def _recorded(mu_q, sigma_q, **kwargs):
        shapes.append(tuple(mu_q.shape))
        return real_decode(mu_q, sigma_q, **kwargs)

    monkeypatch.setattr(model.prior_bank, "decode", _recorded)
    with torch.no_grad():
        _, last = model.forward_beliefs(tokens, return_logits=True, decode_last=True)
    assert shapes == [(2, 1, 4)]
    torch.testing.assert_close(last, full[:, -1:], rtol=1e-5, atol=1e-5)


# ---- Low: distinct derived init streams -----------------------------------------------------

def test_module_init_streams_are_distinct_and_deterministic() -> None:
    seeds = {_derived_seed(6, "pos_phi_free"), _derived_seed(6, "relational_operators", 0),
             _derived_seed(6, "relational_operators", 1), _derived_seed(7, "relational_operators", 0)}
    assert len(seeds) == 4 and all(0 <= seed < 2 ** 63 for seed in seeds)
    assert _derived_seed(6, "pos_phi_free") == _derived_seed(6, "pos_phi_free")

    def build(seed: int) -> VFEModel:
        return VFEModel(_relational_cfg(attention_mode="projected_predictive", n_layers=2,
                                        seed=seed))

    six, six_again, seven = build(6), build(6), build(7)
    assert torch.equal(six.pos_phi_free, six_again.pos_phi_free)
    assert torch.equal(six.relational_operators.raw_score_q,
                       six_again.relational_operators.raw_score_q)
    # Layer 1 of seed 6 no longer replays layer 0 of seed 7.
    assert not torch.equal(six.relational_operators_extra[0].raw_score_q,
                           seven.relational_operators.raw_score_q)
    # pos_phi_free no longer replays the layer-0 comparison draw.
    ops = six.relational_operators
    raw = torch.randn(2, ops.input_dim, 2, generator=torch.Generator().manual_seed(
        _derived_seed(6, "relational_operators", 0)))
    replay = six.pos_phi_free.detach().flatten()[:raw.numel()] / six.cfg.pos_phi_scale
    assert not torch.allclose(replay, raw.flatten())


# ---- Low: chart-bound early warning on the AdamW route --------------------------------------

def test_phi_clamp_warning_trips_at_the_chart_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    model = VFEModel(_tiny(transport_chart_max_norm=0.5))
    with torch.no_grad():
        model.prior_bank.phi_embed.fill_(0.5)                   # embedded ||M||_F ~ 1.4 < 20
    monkeypatch.setattr(train_module, "_PHI_CLAMP_WARNED", False)
    with pytest.warns(RuntimeWarning, match="transport_chart_max_norm=0.5"):
        train_module._warn_phi_transport_clamp(model)

    unbounded = VFEModel(_tiny())
    with torch.no_grad():
        unbounded.prior_bank.phi_embed.fill_(0.5)
    monkeypatch.setattr(train_module, "_PHI_CLAMP_WARNED", False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        train_module._warn_phi_transport_clamp(unbounded)      # below the clamp: silent


# ---- Low: cumulative realized budget after a resume -----------------------------------------

def test_resumed_realized_budget_is_cumulative(tmp_path: Path) -> None:
    cfg = _tiny(checkpoint_interval=2, max_steps=4)
    torch.manual_seed(0)
    partial = VFEModel(cfg)
    train(partial, _sequential_loader(), cfg, n_steps=2,
          artifacts=RunArtifacts(tmp_path / "partial", cfg, partial))
    checkpoint = tmp_path / "partial" / "checkpoints" / "step_2.pt"
    resumed = VFEModel(cfg)
    artifacts = RunArtifacts(tmp_path / "resumed", cfg, resumed)
    train(resumed, _sequential_loader(), cfg, n_steps=4, resume_from=checkpoint,
          artifacts=artifacts)
    realized = artifacts.realized_updates
    assert realized["attempted_steps"] == 4
    assert realized["accepted_updates"] == 4
    assert realized["skipped_steps"] == 0
    assert realized["accepted_update_frac"] == 1.0
    assert json.loads(json.dumps(realized))                      # summary-serializable
