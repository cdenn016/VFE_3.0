r"""Audit 2026-10-31, package paper-tables-baseline: the conventional baseline and its launcher.

Tiny CPU models only (vocabulary 11, K = 8, two heads of width 4, two or three layers, 6-token windows, one
thread); configuration from ``tests/profiles.py``. Covered: ``BaselineConfig.matched_to`` refuses a tied block
MLP and every block-MLP mode whose parameter count is not the arm's, and ``paper_arms`` records a measured
role-count difference (A006, baseline-generation-1 and paper-arms-harness-3); under ``grad_clip_scope='group'``
the hidden-matrix group shares the clip budget of the decay group, while the VFE's groups keep one bound each
(A075, baseline-generation-2); one cached prior and rotation, and a last-position decode in ``generate``
(A172, baseline-generation-10); the opt-in fused attention kernel and the booked rotation and exponential cost
(A174, baseline-generation-9); and the run labels of ``train_baseline.py`` (A280, baseline-generation-7).
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch

import vfe3.train as T
from tests.profiles import live_config, test_config
from vfe3.baseline import BaselineAttention, BaselineConfig, BaselineTransformer, build_optimizer
from vfe3.config import VFE3Config, serialize_config
from vfe3.model.model import VFEModel
from vfe3.run_resources import parameter_breakdown

ROOT = Path(__file__).resolve().parents[1]
V, K, H, L, E, N, B = 11, 8, 2, 2, 2, 6, 2
D = K // H


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _vcfg(**over: Any) -> VFE3Config:
    values: Dict[str, Any] = dict(
        vocab_size=V, embed_dim=K, n_heads=H, batch_size=B, max_seq_len=N, eval_stride=3,
        n_layers=L, n_e_steps=1, warmup_steps=0, max_steps=6, log_interval=0, eval_interval=0,
        checkpoint_interval=0, generate_embedding_figures=False, pos_phi="learned", use_block_mlp=True,
        block_mlp_expansion=E, beta_attention_prior="causal_alibi", family="gaussian_diagonal",
        gauge_group="block_glk", attention_mode="projected_predictive", attention_residual=True,
        decode_mode="linear", norm_type_block="layernorm", layernorm_affine=True, skip_belief_sigma_update=True)
    values.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


def _bcfg(**over: Any) -> BaselineConfig:
    values: Dict[str, Any] = dict(
        vocab_size=V, embed_dim=K, n_heads=H, n_layers=L, mlp_expansion=E, max_seq_len=N,
        batch_size=B, eval_stride=3, max_steps=8, warmup_steps=1, lr=3e-3, log_interval=0,
        eval_interval=0, checkpoint_interval=0, seed=5)
    values.update(over)
    return BaselineConfig(**values)


def _tokens(seed: int, shape: Tuple[int, int] = (B, N)) -> torch.Tensor:
    return torch.randint(0, V, shape, generator=torch.Generator().manual_seed(seed))


# ---- A006 (baseline-generation-1, paper-arms-harness-3): the matched arm's MLP count -----------

def test_matched_to_refuses_a_tied_block_mlp() -> None:
    tied = _vcfg(block_mlp_tie_layers=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vfe = VFEModel(tied)
    mlp = parameter_breakdown(dict(vfe.named_parameters()), tied)["context_processing"]["block_mlp"]
    assert mlp == 2 * E * K * K + E * K + K                              # one MLP, counted once
    with pytest.raises(ValueError, match="block_mlp_tie_layers"):
        BaselineConfig.matched_to(tied)
    with pytest.raises(ValueError, match="block_mlp_tie_layers"):
        BaselineConfig.matched_to({"config": serialize_config(tied)})
    # One layer has one block MLP either way, so the tie is inert there and the base is accepted.
    single = BaselineConfig.matched_to(_vcfg(block_mlp_tie_layers=True, n_layers=1))
    assert single.n_layers == 1


@pytest.mark.parametrize("mode", ["gauge_gate", "gaussian_template", "prototype_agents", "cue_result_memory",
                                  "a_future_mode"])
def test_matched_to_refuses_block_mlp_modes_of_another_count(mode: str) -> None:
    record = {"config": {**serialize_config(_vcfg()), "block_mlp_mode": mode}}
    with pytest.raises(ValueError, match="block_mlp_mode"):
        BaselineConfig.matched_to(record)


def test_matched_to_keeps_the_dense_equivalent_modes() -> None:
    for mode in ("coordinate", "canonical_frame"):
        record = {"config": {**serialize_config(_vcfg()), "block_mlp_mode": mode}}
        assert BaselineConfig.matched_to(record).mlp_expansion == E


def test_paper_arms_refuses_baselines_on_a_tied_base_and_records_a_role_difference(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import paper_arms as pa
    import train_baseline  # noqa: F401
    from tests.test_paper_arms_20260928 import _tiny_cfg, _write_base

    label = "Matched context-processing parameters"
    tied = pa.load_base("tied", _write_base(tmp_path / "tied", _tiny_cfg(block_mlp_tie_layers=True)))
    plan = pa.build_plan(tied, budget="base_protocol", seeds=[(1, 1)], labels=[label], counts=False)[0]
    assert plan.status == "invalid" and "block_mlp_tie_layers" in plan.reason

    untied = pa.load_base("untied", _write_base(tmp_path / "untied", _tiny_cfg()))
    plan = pa.build_plan(untied, budget="base_protocol", seeds=[(1, 1)], labels=[label])[0]
    assert plan.status == "ready" and plan.deviations == []
    assert plan.params["block_mlp"] > 0 and plan.params["relational_content"] > 0
    assert any("context-processing parameters" in note for note in plan.notes)
    # A role the arm does not match (here forced) is a recorded deviation, not a silent label.
    real = pa._vfe_counts
    monkeypatch.setattr(pa, "_vfe_counts", lambda cfg: {**real(cfg), "block_mlp": real(cfg)["block_mlp"] // 2})
    plan = pa.build_plan(untied, budget="base_protocol", seeds=[(1, 1)], labels=[label])[0]
    assert any("block_mlp parameters" in deviation for deviation in plan.deviations)
    exp = pa.experiment_dir(tmp_path / "out", "untied", "base_protocol")
    manifest = pa.build_manifest(exp, [plan], base=untied, budget="base_protocol", seeds=[(1, 1)])
    assert any("block_mlp" in d for d in manifest["paper_arms"]["rows"][label]["declared_deviations"])


# ---- A075 (baseline-generation-2): the hidden group's clip budget under group scope ------------

def _consumed_gradients(model: torch.nn.Module, cfg: Any, optimizer: torch.optim.Optimizer, *,
                        clip: Any, scope: str) -> Tuple[Dict[str, torch.Tensor], float]:
    r"""``(gradients that reach optimizer.step after unscale and clip, the step's loss)`` of one ``train_step``."""
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    names = {id(p): n for n, p in model.named_parameters()}
    seen: Dict[str, torch.Tensor] = {}
    optimizer.register_step_pre_hook(lambda *_: seen.update(
        {names[id(p)]: p.grad.clone() for g in optimizer.param_groups for p in g["params"] if p.grad is not None}))
    generator = torch.Generator().manual_seed(1)
    tokens, targets = (torch.randint(0, cfg.vocab_size, (B, cfg.max_seq_len), generator=generator)
                       for _ in range(2))
    loss = T.train_step(model, optimizer, scheduler, tokens, targets, grad_clip=clip, grad_clip_scope=scope)
    return seen, float(loss)


def test_hidden_lr_scale_keeps_the_clip_partition_under_group_scope() -> None:
    clip = 1e-3
    grads: Dict[float, Dict[str, torch.Tensor]] = {}
    for scale in (1.0, 2.0):
        cfg = _bcfg(grad_clip=clip, grad_clip_scope="group", hidden_lr_scale=scale)
        torch.manual_seed(0)
        model = BaselineTransformer(cfg)
        optimizer = build_optimizer(model, cfg)
        assert len(optimizer.param_groups) == (2 if scale == 1.0 else 3)
        grads[scale], _ = _consumed_gradients(model, cfg, optimizer, clip=clip, scope="group")
    assert grads[1.0].keys() == grads[2.0].keys()
    for name in grads[1.0]:
        torch.testing.assert_close(grads[2.0][name], grads[1.0][name], rtol=1e-6, atol=0.0, msg=name)
    # The partition is the joint one: the matrices share one L2 bound at both scales.
    cfg = _bcfg(grad_clip=clip, grad_clip_scope="group", hidden_lr_scale=2.0)
    groups = build_optimizer(BaselineTransformer(cfg), cfg).param_groups
    assert [g.get("clip_set") for g in groups] == ["matrices", None, "matrices"]
    assert [g["lr"] for g in groups] == [cfg.lr, cfg.lr, cfg.lr * 2.0]


def test_a_clip_set_spanning_two_roles_is_refused() -> None:
    cfg = _bcfg(grad_clip=1e-3, grad_clip_scope="group", hidden_lr_scale=2.0)
    model = BaselineTransformer(cfg)
    optimizer = build_optimizer(model, cfg)
    optimizer.param_groups[1]["clip_set"] = "matrices"                    # no-decay group: also role mu
    optimizer.param_groups[1]["role"] = "phi"
    with pytest.raises(ValueError, match="clip_set"):
        _consumed_gradients(model, cfg, optimizer, clip=1e-3, scope="group")


def test_vfe_group_scope_clip_is_bitwise_one_bound_per_group() -> None:
    r"""The live profile's groups carry no clip set, so each keeps its own bound (bitwise, as before)."""
    cfg = live_config(grad_clip=1e-3, grad_clip_scope="group")
    clip = 1e-3

    def build() -> Tuple[VFEModel, torch.optim.Optimizer]:
        torch.manual_seed(3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VFEModel(cfg)
        return model, T.build_optimizer(model, cfg)

    model, optimizer = build()
    assert all("clip_set" not in group for group in optimizer.param_groups)
    clipped, clipped_loss = _consumed_gradients(model, cfg, optimizer, clip=clip, scope="group")
    model, optimizer = build()
    raw, raw_loss = _consumed_gradients(model, cfg, optimizer, clip=None, scope="group")
    names = {id(p): n for n, p in model.named_parameters()}
    assert clipped_loss == raw_loss and math.isfinite(raw_loss)          # the loss precedes the clip
    assert clipped.keys() == raw.keys()
    for group in optimizer.param_groups:
        members = [names[id(p)] for p in group["params"] if names[id(p)] in raw]
        if not members:
            continue
        if group.get("table_fisher"):
            expected = [raw[n] for n in members]
        else:
            expected = [g.clone() for g in (raw[n] for n in members)]
            norm = torch.nn.utils.get_total_norm(expected)
            torch.nn.utils.clip_grads_with_norm_(_as_parameters(expected), clip, norm)
        for name, value in zip(members, expected):
            assert torch.equal(clipped[name], value), name


def _as_parameters(grads: List[torch.Tensor]) -> List[torch.nn.Parameter]:
    r"""Parameters whose ``.grad`` are ``grads`` (in place), as ``clip_grads_with_norm_`` reads them."""
    out = []
    for grad in grads:
        parameter = torch.nn.Parameter(torch.zeros_like(grad))
        parameter.grad = grad
        out.append(parameter)
    return out


# ---- A172 (baseline-generation-10): one cache entry and a last-position decode -----------------

def test_prior_and_rotation_caches_hold_one_entry() -> None:
    model = BaselineTransformer(_bcfg(pos_rotation="rope", max_seq_len=16)).eval()
    model.generate(_tokens(3, (1, 2)), 20, greedy=True)
    assert len(model._log_prior_cache) == 1 and len(model._rope_cache) == 1
    (prior_key,), (rope_key,) = model._log_prior_cache, model._rope_cache
    assert 16 in prior_key and 16 in rope_key


def test_generate_decodes_the_last_position_only(monkeypatch: pytest.MonkeyPatch) -> None:
    model = BaselineTransformer(_bcfg(max_seq_len=N)).eval()
    model.set_unigram_log_prior(torch.arange(1, V + 1, dtype=torch.float32))
    prompt = _tokens(4, (2, 3))
    rows: List[int] = []
    original = model.decoder.logits

    def logits(h: torch.Tensor) -> torch.Tensor:
        rows.append(h.shape[1])
        return original(h)

    monkeypatch.setattr(model.decoder, "logits", logits)
    out = model.generate(prompt, 5, greedy=True)
    assert rows == [1] * 5 and out.shape == (2, 8)
    monkeypatch.setattr(model.decoder, "logits", original)
    seq = prompt
    for _ in range(5):                                   # the full-sequence decode picks the same tokens
        nxt = model(seq[:, -N:])[:, -1, :].argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, nxt], dim=1)
    assert torch.equal(out, seq)


# ---- A174 (baseline-generation-9): the opt-in fused kernel and the booked cost -----------------

def test_fused_kernel_is_opt_in_and_validated() -> None:
    assert BaselineConfig().attention_kernel == "explicit"
    with pytest.raises(ValueError, match="attention_kernel"):
        _bcfg(attention_kernel="sdpa")
    with pytest.raises(ValueError, match="deterministic"):
        _bcfg(attention_kernel="fused", deterministic=True)
    cfg = _bcfg(attention_kernel="fused", deterministic=False)
    assert BaselineConfig.from_serialized(cfg.serialize()) == cfg
    legacy = {key: value for key, value in _bcfg().serialize().items() if key != "attention_kernel"}
    assert BaselineConfig.from_serialized(legacy).attention_kernel == "explicit"     # older records


@pytest.mark.parametrize("over", [dict(), dict(pos_rotation="rope", rope_fraction=0.5),
                                  dict(pos_rotation="rope", per_type_table="block_rw",
                                       per_type_table_init="phi_like", per_type_table_init_scale=0.3)])
def test_fused_kernel_matches_the_explicit_forward_and_gradient(over: Dict[str, Any]) -> None:
    explicit = _bcfg(deterministic=False, **over)
    fused = dataclasses.replace(explicit, attention_kernel="fused")
    torch.manual_seed(0)
    a = BaselineTransformer(explicit)
    torch.manual_seed(0)
    b = BaselineTransformer(fused)
    for name, tensor in a.state_dict().items():
        assert torch.equal(tensor, b.state_dict()[name]), name           # identical parameters
    tokens, targets = _tokens(5), _tokens(6)
    torch.testing.assert_close(b(tokens), a(tokens), rtol=1e-5, atol=1e-5)
    _, loss_a, _ = a(tokens, targets)
    _, loss_b, _ = b(tokens, targets)
    loss_a.backward()
    loss_b.backward()
    torch.testing.assert_close(loss_b, loss_a, rtol=1e-6, atol=1e-6)
    grads_b = dict(b.named_parameters())
    for name, parameter in a.named_parameters():
        torch.testing.assert_close(grads_b[name].grad, parameter.grad, rtol=1e-4, atol=1e-6, msg=name)
    # The diagnostics read the explicit weights under either kernel.
    torch.testing.assert_close(b.attention_maps(tokens), a.attention_maps(tokens), rtol=1e-5, atol=1e-6)


def test_fused_attention_gradient_matches_float64_finite_differences() -> None:
    model = BaselineTransformer(_bcfg(pos_rotation="rope", rope_fraction=0.5, deterministic=False,
                                      attention_kernel="fused"))
    cpu = torch.device("cpu")
    cos, sin, partner = model._rope_factors(N, cpu)
    factors = (cos.double(), sin.double(), partner)
    prior = model._attention_log_prior(N, cpu).double()                  # (H, N, N), -inf above the diagonal
    torch.manual_seed(1)
    attention = BaselineAttention(K, H, 0.3, 0.3, kernel="fused").double()
    u = torch.randn(B, N, K, dtype=torch.float64, requires_grad=True)
    w_q = attention.w_q.detach().clone().requires_grad_(True)

    def fused(u: torch.Tensor, w_q: torch.Tensor) -> torch.Tensor:
        return torch.func.functional_call(attention, {"w_q": w_q}, (u, prior, None), {"rope_factors": factors})

    assert torch.autograd.gradcheck(fused, (u, w_q), eps=1e-6, atol=1e-7, rtol=1e-5)


def test_fused_rotation_is_the_dense_rotation() -> None:
    model = BaselineTransformer(_bcfg(pos_rotation="rope", rope_fraction=0.5, embed_dim=10, n_heads=2))
    R = model._rope_rotation(N, torch.device("cpu"))                   # (N, d, d) with d = 5 (odd head)
    cos, sin, partner = model._rope_factors(N, torch.device("cpu"))
    q = torch.randn(3, 2, N, 5, dtype=torch.float32, generator=torch.Generator().manual_seed(7))
    dense = torch.einsum("nst,bhnt->bhns", R, q)
    elementwise = q * cos + q[..., partner] * sin
    torch.testing.assert_close(elementwise, dense, rtol=1e-6, atol=1e-7)


def test_cost_model_books_the_rotation_and_the_block_rw_exponentials() -> None:
    def fields(**over: Any) -> Dict[str, Any]:
        return BaselineTransformer(_bcfg(**over)).cost_model_fields(1000, 6000)

    plain, rope, fused = fields(), fields(pos_rotation="rope"), fields(pos_rotation="rope",
                                                                       attention_kernel="fused",
                                                                       deterministic=False)
    assert plain["flops_per_token_rope"] == 0.0
    assert rope["flops_per_token_rope"] == pytest.approx(4.0 * L * K * D)
    assert fused["flops_per_token_rope"] == pytest.approx(6.0 * L * K)
    assert rope["flops_per_token_forward"] - plain["flops_per_token_forward"] == pytest.approx(4.0 * L * K * D)
    read, rw = fields(per_type_table="block_read"), fields(per_type_table="block_rw")
    assert read["flops_per_token_type_table_exp"] == 0.0
    exp_forward = 4.0 * 5 * K * D * D                                 # two exponentials, five d x d products each
    assert rw["flops_per_token_type_table_exp"] == pytest.approx(exp_forward)
    assert rw["flops_per_token_train"] == pytest.approx(
        3.0 * (rw["flops_per_token_forward"] - exp_forward) + 9.0 * exp_forward)
    assert rw["est_flops_analytic"] == pytest.approx(rw["flops_per_token_train"] * 6000)


# ---- A280 (baseline-generation-7): train_baseline.py labels and the grid comment ----------------

def test_default_label_names_every_arm_type() -> None:
    import train_baseline as tb

    assert tb._default_label(_bcfg(), "wikitext-103") == "wikitext-103_K8_baseline_s5"
    assert tb._default_label(_bcfg(per_type_table="block_read"), "wikitext-103") == \
        "wikitext-103_K8_baseline_block_read_s5"
    assert tb._default_label(_bcfg(per_type_table="block_rw"), "wikitext-103") == \
        "wikitext-103_K8_baseline_block_rw_s5"
    assert tb._default_label(_bcfg(per_type_table="block_read", norm_placement="post_shared"),
                             "wikitext-103") == "wikitext-103_K8_baseline_block_read_post_shared_s5"


def test_the_lr_grid_comment_names_the_grid_top() -> None:
    source = (ROOT / "train_baseline.py").read_text(encoding="utf-8")
    start = source.index("# The baseline's peak lr")
    comment = source[start:source.index("LR_SWEEP:", start)]
    assert "(3e-3)" not in comment and "top of the grid" in comment
