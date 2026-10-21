"""Confound controls for the per-type read-table comparison (GL(K) v2 review, October 7, 2026).

Covers the covariance-blind routing cost ``relational_routing='mean_distance'``, the baseline's
read-and-write per-type table (``per_type_table='block_rw'``) and shared post-norm layout
(``norm_placement='post_shared'``), and the confound arms of ``paper_arms`` (TIER_A, TIER_B).
Tiny models only; CPU, one thread.
"""
import warnings
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

import paper_arms as pa
from tests.profiles import test_config
from tests.test_paper_arms_20260928 import _tiny_cfg, _write_base
from vfe3.baseline import BaselineConfig, BaselineTransformer, baseline_parameter_role
from vfe3.model.model import VFEModel
from vfe3.relational_scoring import (
    ROUTING_COSTS, projected_gaussian_kl, projected_mean_distance_cost, routing_cost,
)

V, K, H, L, E, N, B = 11, 8, 2, 2, 2, 6, 2


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _moments(seed: int, n: int = 4, heads: int = 2, r: int = 3) -> tuple:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(1, n, heads, r, generator=g, dtype=torch.float64)                 # (1, N, H, r)
    root = torch.randn(1, n, heads, r, r, generator=g, dtype=torch.float64)
    G = root @ root.transpose(-1, -2) + .3 * torch.eye(r, dtype=torch.float64)        # (1, N, H, r, r)
    return a, .5 * (G + G.transpose(-1, -2))


# ---- mean-distance routing ---------------------------------------------------------------------

def test_mean_distance_cost_is_the_identity_covariance_kl_and_ignores_covariances() -> None:
    a, G = _moments(1)
    b, G_k = _moments(2)
    D = projected_mean_distance_cost(a, G, b, G_k)                                      # (1, H, Nq, Nk)
    eye = torch.eye(3, dtype=torch.float64).expand_as(G)
    torch.testing.assert_close(D, projected_gaussian_kl(a, eye, b, eye), rtol=1e-12, atol=1e-12)
    delta = a.movedim(-2, -3).unsqueeze(-2) - b.movedim(-2, -3).unsqueeze(-3)
    torch.testing.assert_close(D, .5 * delta.pow(2).sum(-1), rtol=0., atol=0.)
    torch.testing.assert_close(projected_mean_distance_cost(a, 3 * G, b, 7 * G_k), D, rtol=0., atol=0.)
    assert float(D.min()) >= 0.


def test_routing_cost_dispatch() -> None:
    assert set(ROUTING_COSTS) == {"gaussian_kl", "noisy_query", "noisy_key", "mean_distance"}
    assert routing_cost("mean_distance") is projected_mean_distance_cost
    with pytest.raises(ValueError, match="relational routing"):
        routing_cost("dot_product")


def _vfe(**over: Any):
    values: Dict[str, Any] = dict(
        vocab_size=V, embed_dim=K, n_heads=H, batch_size=B, max_seq_len=N, n_layers=L, n_e_steps=1,
        use_block_mlp=True, block_mlp_expansion=E, family="gaussian_diagonal", gauge_group="block_glk",
        attention_mode="projected_predictive", attention_residual=True, decode_mode="linear",
        generate_embedding_figures=False, pos_phi="learned", norm_type_block="layernorm",
        layernorm_affine=True, skip_belief_sigma_update=True)
    values.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


def test_mean_distance_routing_trains_in_a_tiny_model_and_is_predictive_only() -> None:
    cfg = _vfe(relational_routing="mean_distance")
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg)
    tokens = torch.randint(0, V, (B, N), generator=torch.Generator().manual_seed(3))
    _, loss, _ = model(tokens, tokens.roll(-1, dims=1))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with pytest.raises(ValueError, match="mean_distance"):
        _vfe(relational_routing="mean_distance", attention_mode="projected_gradient")


# ---- baseline: read-and-write table and shared post-norm ---------------------------------------

def _bcfg(**over: Any) -> BaselineConfig:
    values: Dict[str, Any] = dict(
        vocab_size=V, embed_dim=K, n_heads=H, n_layers=L, mlp_expansion=E, max_seq_len=N,
        batch_size=B, max_steps=4, warmup_steps=1, lr=3e-3, log_interval=0, eval_interval=0,
        checkpoint_interval=0, seed=5)
    values.update(over)
    return BaselineConfig(**values)


def _bmodel(cfg: BaselineConfig) -> BaselineTransformer:
    torch.manual_seed(0)
    return BaselineTransformer(cfg)


def _tokens() -> torch.Tensor:
    return torch.randint(0, V, (B, N), generator=torch.Generator().manual_seed(4))


def test_block_rw_starts_as_the_plain_baseline_and_has_the_frame_table_size() -> None:
    plain, rw = _bmodel(_bcfg()), _bmodel(_bcfg(per_type_table="block_rw"))
    with torch.no_grad():
        torch.testing.assert_close(rw(_tokens()), plain(_tokens()), rtol=1e-6, atol=1e-6)
    read = _bcfg(per_type_table="block_read").per_type_parameters()["type_table"]
    assert _bcfg(per_type_table="block_rw").per_type_parameters()["type_table"] == read == V * K * (K // H)


def test_block_rw_reads_through_the_inverse_of_what_it_writes_and_trains() -> None:
    cfg = _bcfg(per_type_table="block_rw", per_type_table_init="phi_like", per_type_table_init_scale=0.3)
    model = _bmodel(cfg)
    X = model.type_table[_tokens()].detach().double()                                  # (B, N, H, d, d)
    product = torch.linalg.matrix_exp(-X) @ torch.linalg.matrix_exp(X)
    torch.testing.assert_close(product, torch.eye(K // H, dtype=torch.float64).expand_as(product),
                               rtol=1e-12, atol=1e-12)
    tokens = _tokens()
    _, loss, _ = model(tokens, tokens.roll(-1, dims=1))
    loss.backward()
    assert torch.isfinite(loss) and float(model.type_table.grad.abs().sum()) > 0.
    # A nonzero table changes the forward pass relative to the plain baseline.
    with torch.no_grad():
        assert not torch.allclose(model(tokens), _bmodel(_bcfg())(tokens))


def test_post_shared_norm_has_the_vfe_context_count_and_one_shared_layernorm() -> None:
    cfg = _bcfg(norm_placement="post_shared")
    model = _bmodel(cfg)
    assert model.final_norm is None and model.shared_norm is not None
    assert not any(".ln1." in n or ".ln2." in n for n, _ in model.named_parameters())
    context = sum(p.numel() for n, p in model.named_parameters()
                  if baseline_parameter_role(n)[0] == "context_processing")
    vfe_count = L * (4 * K * K + 2 * E * K * K + E * K + K) + 2 * K
    assert context == cfg.context_processing_parameters() == vfe_count
    assert _bcfg().context_processing_parameters() - vfe_count == 4 * K * L
    tokens = _tokens()
    _, loss, _ = model(tokens, tokens.roll(-1, dims=1))
    loss.backward()
    assert torch.isfinite(loss) and float(model.shared_norm.weight.grad.abs().sum()) > 0.
    with pytest.raises(ValueError, match="norm_placement"):
        _bcfg(norm_placement="sandwich")


# ---- paper_arms confound arms ------------------------------------------------------------------

def test_tiers_name_existing_arms_and_base_protocol_keeps_the_base_budget() -> None:
    assert set(pa.TIER_A) <= set(pa.ARMS) and set(pa.TIER_B) <= set(pa.ARMS)
    # The run selection in CONFIG is the user's (2026-10-21): it only has to name registered bases and a budget.
    assert set(pa.CONFIG["bases"]) <= set(pa.BASES) and pa.CONFIG["budget"] in pa.BUDGETS
    cfg = _tiny_cfg(max_steps=9, eval_interval=3)
    assert pa.base_protocol(cfg, "unused") == {"max_steps": 9, "eval_interval": 3, "checkpoint_interval": 9}


def test_confound_arms_resolve_with_declared_changes_only(tmp_path: Path) -> None:
    import train_baseline  # noqa: F401
    base = pa.load_base("tiny", _write_base(tmp_path, _tiny_cfg(beta_attention_prior="causal")))
    plans = {p.label: p for p in pa.build_plan(
        base, budget="base_protocol", seeds=[(6, 3)], labels=[*pa.TIER_A, *pa.TIER_B, "Matched measured compute"],
        baseline_overrides={"lr": 3e-3})}
    as_run = plans["Per-type read table, ALiBi (as run)"]
    assert as_run.status == "ready" and as_run.cfg.per_type_table == "block_read"
    assert as_run.cfg.beta_attention_prior == "causal_alibi" and as_run.cfg.lr == 3e-3
    assert as_run.diff == {"beta_attention_prior": ("causal", "causal_alibi")}
    causal = plans["Per-type table of the frame table's size"]
    assert causal.status == "ready" and causal.cfg.beta_attention_prior == "causal" and causal.diff == {}
    opt = plans["Per-type read table, ALiBi, realization optimizer"].cfg
    assert (opt.adam_beta2, opt.weight_decay, opt.per_type_table_weight_decay) == (0.999, 0.0, 0.1)
    assert plans["Per-type read-and-write table, ALiBi"].cfg.per_type_table == "block_rw"
    assert plans["Per-type read table, ALiBi, shared post-norm"].cfg.norm_placement == "post_shared"
    assert plans["Matched measured compute"].status == "deferred"
    alibi = plans["Learned frames, ALiBi prior"]
    assert alibi.status == "ready" and set(alibi.diff) == {"beta_attention_prior"}
    mean = plans["Covariance-blind routing (mean distance)"]
    assert mean.status == "ready" and set(mean.diff) == {"relational_routing"}
    # Every as-run variant keeps the base's protocol apart from its declared fields.
    for label in pa.TIER_A[2:5]:
        assert set(plans[label].diff) == {"beta_attention_prior"}


def test_the_base_run_serves_as_the_evaluated_replicate_at_its_own_seed_pair(tmp_path: Path) -> None:
    import json

    run = _write_base(tmp_path, _tiny_cfg(seed=6))
    (run / "provenance.json").write_text(json.dumps({"data_seed": 3}))
    (run / "summary.json").write_text(json.dumps({"n_steps": 100, "wall_time_s": 20.0, "test_ce": 3.5}))
    base = pa.load_base("tiny", run)
    assert base.data_seed == 3
    exp = tmp_path / "exp"
    assert pa.completed_run(exp, pa.EVALUATED, 6, 3, base=base, budget="base_protocol") == base.run_dir
    assert pa.completed_run(exp, pa.EVALUATED, 23, 23, base=base, budget="base_protocol") is None
    assert pa.completed_run(exp, pa.EVALUATED, 6, 3, base=base, budget="one_epoch") is None
    assert pa.completed_run(exp, "Learned frames, ALiBi prior", 6, 3, base=base, budget="base_protocol") is None
    plans = {p.label: p for p in pa.build_plan(base, budget="base_protocol", seeds=[(6, 3), (23, 23)],
                                               labels=[pa.EVALUATED], counts=False)}
    assert plans[pa.EVALUATED].reused == [(6, 3)]
    manifest = pa.build_manifest(exp, list(plans.values()), base=base, budget="base_protocol",
                                 seeds=[(6, 3), (23, 23)])
    assert manifest["arms"][pa.EVALUATED]["runs"] == [str(base.run_dir)]
