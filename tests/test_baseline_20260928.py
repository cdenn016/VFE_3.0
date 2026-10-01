"""Conventional-transformer baseline (vfe3.baseline, train_baseline.py) and the shared-loop fixes.

Tiny CPU models only (vocabulary 11, K=8, two heads of width 4, two layers, 6-token windows, one
thread). Covers the baseline's forward/backward, causality, the attention prior, an independent
float64 reference forward, the context-processing count against a tiny VFEModel, the per-type read
map's size, config and checkpoint serialization, the end-to-end train + finalize path, exact resume,
the post-hoc preset, the compute-matching helpers, and the shared-loop fixes: frozen-table reporting
and projection skip under gauge_transport='off', the per_type_hidden_bias optimizer group, and the
host-side terminal snapshot.
"""

from tests.profiles import test_config
import csv
import dataclasses
import json
import logging
import math
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import train_baseline as tb
import vfe3.run_artifacts as ra
import vfe3.train as T
from vfe3.attention_prior import prior_causal_alibi
from vfe3.baseline import (
    MODEL_FAMILY,
    BaselineConfig,
    BaselineTransformer,
    baseline_config_from_serialized,
    calibrated_tokens_per_s,
    measured_train_device_hours,
    steps_for_device_hours,
    width_ladder_rule,
)
from vfe3.config import VFE3Config, serialize_config
from vfe3.contracts import is_vfe_config, is_vfe_model
from vfe3.data.datasets import TokenWindows
from vfe3.gauge_optim import project_phi_parameter_rows_
from vfe3.model.model import VFEModel
from vfe3.run_resources import exclude_frozen_from_breakdown, frozen_parameter_names, parameter_breakdown

V, K, H, L, E, N, B = 11, 8, 2, 2, 2, 6, 2
D = K // H
CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _one_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.set_num_threads(1)
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)


def _bcfg(**over: Any) -> BaselineConfig:
    values: Dict[str, Any] = dict(
        vocab_size=V, embed_dim=K, n_heads=H, n_layers=L, mlp_expansion=E, max_seq_len=N,
        batch_size=B, eval_stride=3, max_steps=8, warmup_steps=1, lr=3e-3, log_interval=0,
        eval_interval=0, checkpoint_interval=0, seed=5)
    values.update(over)
    return BaselineConfig(**values)


def _vcfg(**over: Any) -> VFE3Config:
    values: Dict[str, Any] = dict(
        vocab_size=V, embed_dim=K, n_heads=H, batch_size=B, max_seq_len=N, eval_stride=3,
        n_layers=L, n_e_steps=1, warmup_steps=0, max_steps=6, log_interval=0, eval_interval=0,
        checkpoint_interval=0, e_phi_lr=0.0, oracle_unroll_grad=True,
        generate_embedding_figures=False, pos_phi="learned", use_block_mlp=True,
        block_mlp_expansion=E, beta_attention_prior="causal_alibi", family="gaussian_diagonal",
        gauge_group="block_glk", attention_mode="projected_predictive", attention_residual=True,
        decode_mode="linear",
        # pinned: independent of the config defaults the WIP flips
        norm_type_block="layernorm", layernorm_affine=True, skip_belief_sigma_update=True)
    values.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


def _vmodel(cfg: VFE3Config, seed: int = 0) -> VFEModel:
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFEModel(cfg)


def _bmodel(cfg: BaselineConfig, seed: int = 0) -> BaselineTransformer:
    torch.manual_seed(seed)
    return BaselineTransformer(cfg)


def _tokens(seed: int, shape: Tuple[int, int] = (B, N)) -> torch.Tensor:
    return torch.randint(0, V, shape, generator=torch.Generator().manual_seed(seed))


def _rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _num(value: str) -> float:
    return float("nan") if value == "" else float(value)


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------

def test_forward_backward_and_fused_ce_matches_dense() -> None:
    cfg = _bcfg(per_type_table="block_read", per_type_table_init="phi_like", per_type_table_init_scale=0.3)
    model = _bmodel(cfg)
    counts = torch.arange(1, V + 1, dtype=torch.float32)
    model.set_unigram_log_prior(counts)
    tokens, targets = _tokens(1), _tokens(2)
    targets[0, 0] = -100
    logits = model(tokens)
    assert logits.shape == (B, N, V) and logits.dtype == torch.float32
    _, loss, stats = model(tokens, targets, return_decode_stats=True)
    dense = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100)
    torch.testing.assert_close(loss, dense, rtol=1e-6, atol=1e-6)
    assert int(stats.scored_tokens) == B * N - 1 and int(stats.excluded_tokens) == 1
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    grad_out: Dict[str, float] = {}
    model(tokens, targets, estep_grad_out=grad_out)
    assert set(grad_out) == {"mu", "sigma", "phi"} and all(math.isnan(v) for v in grad_out.values())


def test_causality() -> None:
    model = _bmodel(_bcfg(per_type_table="block_read", per_type_table_init="phi_like",
                          per_type_table_init_scale=0.3)).eval()
    tokens = _tokens(3)
    base = model(tokens)
    for t in range(N - 1):
        perturbed = tokens.clone()
        perturbed[:, t + 1:] = (perturbed[:, t + 1:] + 1) % V
        changed = model(perturbed)
        assert torch.equal(changed[:, :t + 1], base[:, :t + 1]), t
        assert not torch.equal(changed[:, t + 1:], base[:, t + 1:])


def test_attention_prior_is_the_vfe_causal_alibi() -> None:
    cfg = _bcfg(alibi_slope=0.7)
    model = _bmodel(cfg)
    expected = prior_causal_alibi(N, N, n_heads=H, alibi_slope=0.7)                   # (H, N, N)
    assert torch.equal(model._attention_log_prior(N, CPU), expected)
    maps = model.attention_maps(_tokens(4))                                            # (L, H, N, N)
    assert maps.shape == (L, H, N, N)
    assert torch.equal(torch.triu(maps, diagonal=1), torch.zeros_like(maps))           # causal
    torch.testing.assert_close(maps.sum(-1), torch.ones(L, H, N))
    vfe = _vmodel(_vcfg(alibi_slope=0.7))
    torch.testing.assert_close(vfe._attention_log_prior(N, CPU).float(), expected)
    from vfe3.posthoc.evaluate import alibi_slopes
    assert alibi_slopes(model, N) == pytest.approx([0.7 * 2.0 ** -4, 0.7 * 2.0 ** -8])


def _reference_logits(model: BaselineTransformer, tokens: torch.Tensor) -> torch.Tensor:
    r"""Independent float64 pre-LN forward from the module docstring equations."""
    cfg = model.cfg
    dbl = {name: p.detach().double() for name, p in model.named_parameters()}
    b, n = tokens.shape
    prior = prior_causal_alibi(n, n, n_heads=H, alibi_slope=cfg.alibi_slope, dtype=torch.float64)

    def ln(x: torch.Tensor, prefix: str) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) ** 2).mean(-1, keepdim=True)
        return (x - mean) / torch.sqrt(var + cfg.layernorm_eps) * dbl[f"{prefix}.weight"] + dbl[f"{prefix}.bias"]

    h = dbl["tok_embed"][tokens]                                                       # (B, N, K)
    X = dbl["type_table"][tokens] if "type_table" in dbl else None                    # (B, N, H, d, d)
    for layer in range(cfg.n_layers):
        p = f"blocks.{layer}"
        u = ln(h, f"{p}.ln1")
        if X is not None:
            u_h = u.reshape(b, n, H, D)
            u = (u_h + (X @ u_h.unsqueeze(-1)).squeeze(-1)).reshape(b, n, K)
        q, k, v = (u @ dbl[f"{p}.attn.w_{r}"].T for r in "qkv")
        z = torch.zeros_like(u)
        for head in range(H):
            sl = slice(head * D, (head + 1) * D)
            weights = torch.softmax(q[..., sl] @ k[..., sl].transpose(-1, -2) / math.sqrt(D) + prior[head], -1)
            z[..., sl] = weights @ v[..., sl]
        h = h + z @ dbl[f"{p}.attn.w_o"].T
        m = ln(h, f"{p}.ln2") @ dbl[f"{p}.mlp.fc1.weight"].T + dbl[f"{p}.mlp.fc1.bias"]
        m = 0.5 * m * (1.0 + torch.erf(m / math.sqrt(2.0)))                             # exact GELU
        h = h + m @ dbl[f"{p}.mlp.fc2.weight"].T + dbl[f"{p}.mlp.fc2.bias"]
    logits = ln(h, "final_norm") @ dbl["decoder.output_proj_weight"].T + dbl["decoder.output_proj_bias"]
    return logits + cfg.unigram_kappa * model.decoder.unigram_log_prior.double()


def test_forward_matches_an_independent_float64_reference() -> None:
    cfg = _bcfg(per_type_table="block_read", per_type_table_init="phi_like", per_type_table_init_scale=0.3,
                unigram_kappa=0.8)
    model = _bmodel(cfg)
    with torch.no_grad():
        model.decoder.output_proj_bias.normal_(0.0, 0.5)
        for block in model.blocks:
            block.ln1.weight.normal_(1.0, 0.2)
            block.mlp.fc1.bias.normal_(0.0, 0.3)
    counts = torch.arange(V, dtype=torch.float32) ** 2
    model.set_unigram_log_prior(counts)
    expected_prior = torch.log((counts + 1.0) / (counts.sum() + V))
    torch.testing.assert_close(model.decoder.unigram_log_prior, expected_prior)
    tokens = _tokens(5)
    torch.testing.assert_close(model(tokens).double(), _reference_logits(model, tokens), rtol=0, atol=2e-5)


def test_zero_read_map_is_the_plain_baseline_and_phi_like_is_not() -> None:
    plain = _bmodel(_bcfg(), seed=7)
    zero = _bmodel(_bcfg(per_type_table="block_read"), seed=7)
    phi_like = _bmodel(_bcfg(per_type_table="block_read", per_type_table_init="phi_like",
                             per_type_table_init_scale=0.3), seed=7)
    tokens = _tokens(6)
    assert torch.equal(zero.type_table, torch.zeros(V, H, D, D))
    assert torch.equal(plain(tokens), zero(tokens))                    # A = I: step 0 is arm 1
    assert not torch.equal(plain(tokens), phi_like(tokens))
    for name, tensor in plain.state_dict().items():                     # the table draws no global RNG
        assert torch.equal(tensor, phi_like.state_dict()[name]), name
    assert 0.2 < float(phi_like.type_table.detach().std()) < 0.4


def test_context_processing_count_against_a_tiny_vfe_model() -> None:
    vcfg = _vcfg()
    vfe = _vmodel(vcfg)
    vfe_counts = parameter_breakdown(dict(vfe.named_parameters()), vcfg)
    assert vfe_counts["context_processing"]["total"] == L * (4 * K * K + 2 * E * K * K + E * K + K) + 2 * K
    bcfg = BaselineConfig.matched_to(vcfg)
    model = BaselineTransformer(bcfg)
    counts = parameter_breakdown(dict(model.named_parameters()), bcfg)
    context = counts["context_processing"]
    assert context["total"] == vfe_counts["context_processing"]["total"] + 4 * K * L
    assert context["total"] == bcfg.context_processing_parameters() == model.context_processing_parameter_count()
    for role in ("relational_score", "relational_content", "block_mlp"):
        assert context[role] == vfe_counts["context_processing"][role], role
    assert counts["other"]["total"] == 0 and counts["positional_frames"]["total"] == 0
    assert counts["total"] == bcfg.total_parameters() == sum(p.numel() for p in model.parameters())
    assert counts["per_type_tables"]["token_embed"] == V * K
    assert counts["per_type_tables"]["decode"] == V * K + V
    # One token row plus the whole decoder plus every context-processing parameter is read per token.
    assert counts["parameters_read_per_token"] == K + V * K + V + context["total"]
    for name in ("vocab_size", "embed_dim", "n_heads", "n_layers", "max_seq_len", "batch_size",
                 "eval_stride", "max_steps", "alibi_slope", "seed", "decode_unigram_prior"):
        assert getattr(bcfg, name) == getattr(vcfg, name), name
    assert bcfg.mlp_expansion == E and bcfg.model_family == MODEL_FAMILY
    assert BaselineConfig.matched_to({"config": serialize_config(vcfg)}) == bcfg


def test_block_read_table_has_the_frame_table_size() -> None:
    vcfg = _vcfg()
    vfe = _vmodel(vcfg)
    bcfg = BaselineConfig.matched_to(vcfg, per_type_table="block_read")
    model = BaselineTransformer(bcfg)
    assert model.type_table.shape == (V, H, D, D)
    assert model.type_table.numel() == vfe.prior_bank.phi_embed.numel() == V * K * D
    counts = parameter_breakdown(dict(model.named_parameters()), bcfg)
    assert counts["per_type_tables"]["type_table"] == vfe.prior_bank.phi_embed.numel()
    assert bcfg.per_type_parameters()["type_table"] == vfe.prior_bank.phi_embed.numel()
    assert counts["context_processing"]["total"] == bcfg.context_processing_parameters()   # no context cost
    assert bcfg.per_type_table_init_scale == vcfg.phi_scale
    optimizer = T.build_optimizer(model, bcfg)                          # dispatches to the baseline
    table_groups = [g for g in optimizer.param_groups if any(p is model.type_table for p in g["params"])]
    assert len(table_groups) == 1 and table_groups[0]["role"] == "phi"
    assert table_groups[0]["lr"] == bcfg.lr and table_groups[0]["weight_decay"] == bcfg.weight_decay
    assert T._parameter_group("type_table") == "per_type_table"
    assert {T._parameter_group(n) for n, _ in model.named_parameters()} <= set(T.PARAMETER_GROUPS) - {"other"}


def test_config_and_checkpoint_serialization_round_trip(tmp_path: Path) -> None:
    cfg = _bcfg(per_type_table="block_read", amp_dtype="bf16", eval_stride=None, grad_clip=None)
    payload = cfg.serialize()
    assert payload["model_family"] == MODEL_FAMILY and not is_vfe_config(payload)
    assert BaselineConfig.from_serialized(json.loads(json.dumps(payload))) == cfg
    torch.save({"config": payload}, tmp_path / "bundle.pt")
    loaded = torch.load(tmp_path / "bundle.pt", weights_only=True)["config"]
    assert BaselineConfig.from_serialized(loaded) == cfg
    assert ra._config_from_run_serialized(payload, source="test") == cfg
    assert ra._serialize_run_config(cfg) == payload
    assert ra.semantic_config_fingerprint(ra._selection_semantic_config(payload)) == \
        ra.semantic_config_fingerprint(ra._selection_semantic_config(cfg))
    with pytest.raises(ValueError, match="undeclared"):
        BaselineConfig.from_serialized({**payload, "not_a_field": 1})
    with pytest.raises(ValueError, match="model_family"):
        BaselineConfig.from_serialized({**payload, "model_family": "vfe"})
    lenient, dropped = baseline_config_from_serialized({**payload, "retired": 2}, source="t",
                                                       drop_unknown_config_fields=True)
    assert lenient == cfg and dropped == {"retired": 2}
    from vfe3.paper.runs import load_config
    from vfe3.posthoc.streams import config_from_bundle
    assert config_from_bundle(payload, source="t") == (cfg, {})
    assert load_config(payload, source="t") == (cfg, "strict", {})
    # The VFE path is untouched: an untagged mapping is a VFE3Config.
    vcfg = _vcfg()
    assert ra._config_from_run_serialized(serialize_config(vcfg), source="t") == vcfg
    model, other = _bmodel(cfg, seed=1), _bmodel(cfg, seed=2)
    other.load_state_dict(model.state_dict(), strict=True)
    tokens = _tokens(8)
    assert torch.equal(model(tokens), other(tokens))
    assert "decoder.unigram_log_prior" in model.state_dict()


# ----------------------------------------------------------------------------------------------
# Shared loop: end to end, resume, post hoc
# ----------------------------------------------------------------------------------------------

_STREAMS = {"train": (torch.arange(97) * 7 + 3) % V, "validation": (torch.arange(29) * 3) % V,
            "test": (torch.arange(23) * 5 + 1) % V}


def _fake_select_loader(dataset: str, cfg: BaselineConfig, *, split: str, data_seed: int,
                        max_tokens: Any = None) -> DataLoader:
    if split == "train":
        return DataLoader(TokenWindows(_STREAMS["train"], cfg.max_seq_len), batch_size=cfg.batch_size,
                          shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(data_seed))
    return DataLoader(TokenWindows(_STREAMS[split], cfg.max_seq_len, pad_final=True, stride=cfg.eval_stride),
                      batch_size=2, shuffle=False)


def test_end_to_end_train_and_finalize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb, "_select_loader", _fake_select_loader)
    monkeypatch.setattr(tb, "_tokens_per_char", lambda *args, **kwargs: None)
    monkeypatch.setattr(tb, "_tokenizer_tag", lambda dataset: None)
    cfg = _bcfg(per_type_table="block_read", per_type_table_init="phi_like", per_type_table_init_scale=0.2,
                max_steps=8, log_interval=2, eval_interval=4, checkpoint_interval=4)
    run_dir = tb.run_baseline(cfg, dataset="tiny-synthetic", out_root=tmp_path, seed=3, data_seed=None,
                              label="tiny", device="cpu", generate_samples=False,
                              logger=logging.getLogger("baseline-test"))
    assert run_dir.parent == tmp_path and run_dir.name.endswith("_tiny") and run_dir.is_dir()
    for name in ("config.json", "metrics.csv", "summary.json", "resources.json", "provenance.json",
                 "test_results.json", "best_model.pt", "final_model.pt", "checkpoints/step_4.pt",
                 "checkpoints/step_8.pt"):
        assert (run_dir / name).is_file(), name
    config = json.loads((run_dir / "config.json").read_text())
    assert config["config"]["model_family"] == MODEL_FAMILY and config["config"]["seed"] == 3
    summary = json.loads((run_dir / "summary.json").read_text())
    assert math.isfinite(summary["test_ce"]) and math.isfinite(summary["test_best_ce"])
    assert summary["test_best_status"] == "evaluated" or summary["test_best_status"].startswith("best weights are")
    assert math.isfinite(summary["train_sample_ce_final"]) and math.isfinite(summary["train_sample_ce_best"])
    assert summary["model_family"] == MODEL_FAMILY and "zero_e_steps_counterfactual" in summary["skipped_analyses"]
    assert summary["n_params_trainable"] == summary["n_params"]
    assert summary["alibi_slopes"] == pytest.approx([2.0 ** -4, 2.0 ** -8])
    assert summary["diagnostics_failures"] == {} and summary["accepted_updates"] == 8
    assert summary["scaling_point"]["model_family"] == MODEL_FAMILY
    resources = json.loads((run_dir / "resources.json").read_text())
    breakdown = resources["parameter_breakdown"]
    assert breakdown["model_family"] == MODEL_FAMILY
    assert breakdown["context_processing"]["total"] == cfg.context_processing_parameters() > 0
    assert breakdown["per_type_tables"]["type_table"] == V * K * D
    assert breakdown["frozen"]["total"] == 0 and breakdown["total"] == summary["n_params"]
    assert resources["device_hours"] > 0 and resources["optimizer_state_bytes"]["total"] > 0
    rows = _rows(run_dir / "metrics.csv")
    assert [int(r["step"]) for r in rows] == [2, 4, 6, 8]
    for column in ("val_ce", "train_sample_ce", "tokens_seen", "wall_clock_s", "lr_mu", "lr_phi",
                   "grad_norm_group_per_type_table", "grad_norm_group_mu_embed", "attn_entropy"):
        assert column in rows[0], column
    for row in rows:
        assert int(row["tokens_seen"]) == int(row["step"]) * B * N
        assert math.isfinite(_num(row["wall_clock_s"])) and math.isfinite(_num(row["attn_entropy"]))
        assert math.isfinite(_num(row["grad_norm_group_per_type_table"]))
        assert math.isnan(_num(row["lr_sigma"]))
        if int(row["step"]) % 4 == 0:
            assert math.isfinite(_num(row["val_ce"])) and math.isfinite(_num(row["train_sample_ce"]))
        else:
            assert math.isnan(_num(row["val_ce"]))
    provenance = json.loads((run_dir / "provenance.json").read_text())
    assert provenance["data_seed"] == 3 and provenance["seed"] == 3
    # The paper assembler builds the baseline config and reads the context/total cells.
    from vfe3.paper.runs import load_run, parameter_counts, select_checkpoint
    record = load_run(run_dir)
    assert record.config_load == "strict" and isinstance(record.cfg, BaselineConfig)
    counts = parameter_counts(record)
    assert counts["context_processing"]["total"] == cfg.context_processing_parameters()
    assert select_checkpoint(record)["status"] == "ok"
    # The post-hoc loader rebuilds the model from the bundle and replays the recorded test CE.
    from vfe3.posthoc import evaluate as ev
    loaded = ev.load_model(run_dir, "final_model.pt", device=CPU)
    assert isinstance(loaded.model, BaselineTransformer)
    test = DataLoader(TokenWindows(_STREAMS["test"], N, pad_final=True, stride=3), batch_size=2)
    assert T.evaluate(loaded.model, test)["ce"] == pytest.approx(summary["test_ce"], abs=1e-6)


def test_exact_resume_from_a_baseline_checkpoint(tmp_path: Path) -> None:
    cfg = _bcfg(per_type_table="block_read", per_type_table_init="phi_like", max_steps=6,
                checkpoint_interval=3)

    def run(directory: str, resume: Any = None) -> Tuple[List[float], Dict[str, torch.Tensor]]:
        model = _bmodel(cfg, seed=0 if resume is None else 9)
        artifacts = ra.RunArtifacts(tmp_path / directory, cfg, model)
        torch.manual_seed(1)
        losses = T.train(model, _fake_select_loader("x", cfg, split="train", data_seed=4), cfg,
                         n_steps=cfg.max_steps, grad_clip=cfg.grad_clip, device=CPU, artifacts=artifacts,
                         resume_from=resume, generate_samples=False)
        return losses, model.state_dict()

    full_losses, full_state = run("a")
    resumed_losses, resumed_state = run("b", tmp_path / "a" / "checkpoints" / "step_3.pt")
    assert resumed_losses == full_losses[3:]
    for name, tensor in full_state.items():
        assert torch.equal(tensor, resumed_state[name]), name


def _write_cache(cache_dir: Path, dataset: str, split: str, tokens: torch.Tensor) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tokens.to(torch.int64).contiguous(), cache_dir / f"{dataset}_{split}_tiktoken_tokens.pt")


def test_posthoc_baseline_preset_end_to_end(tmp_path: Path) -> None:
    from vfe3.data.datasets import cache_source_identity
    from vfe3.posthoc import streams as st
    from vfe3.posthoc.report import PosthocOptions, run_posthoc

    dataset, run_dir, cache_dir = "synthetic-period3", tmp_path / "run", tmp_path / "cache"
    run_dir.mkdir()
    cfg = _bcfg(max_seq_len=8, eval_stride=4)
    streams = {"validation": _tokens(11, (1, 61))[0], "test": _tokens(12, (1, 53))[0],
               "train": _tokens(13, (1, 400))[0]}
    for split, tokens in streams.items():
        _write_cache(cache_dir, dataset, split, tokens)
    model = _bmodel(cfg, seed=0)
    model.set_unigram_log_prior(torch.bincount(streams["train"], minlength=V).float())
    identity = {"dataset": dataset, "split": "validation",
                "source": {"sha256": cache_source_identity(dataset, "validation", cache_dir=cache_dir)["sha256"]}}
    torch.save({"model_state": model.state_dict(), "config": cfg.serialize(), "selection_data_identity": identity},
               run_dir / "final_model.pt")

    def ce(tokens: torch.Tensor) -> float:
        return float(T.evaluate(model, DataLoader(st.eval_windows(tokens, 8, 4), batch_size=2))["ce"])

    (run_dir / "summary.json").write_text(json.dumps({
        "n_steps": 20, "evaluation_weights": "final", "final_val_ce": ce(streams["validation"]),
        "test_ce": ce(streams["test"])}), encoding="utf-8")
    (run_dir / "provenance.json").write_text(json.dumps({"device_name": "cpu"}), encoding="utf-8")
    options = PosthocOptions(device="cpu", cache_dir=str(cache_dir), checkpoints=("final_model.pt",),
                             bootstrap_resamples=50, latency_batches=1, figures=False)
    report = run_posthoc(run_dir, options=options, log=lambda message: None)
    assert report["model_family"] == MODEL_FAMILY
    assert set(report["skipped_analyses"]) == {"interventions", "klcap", "attention", "residual",
                                               "covariance", "routing", "maps"}
    assert report["interventions"] is None and report["klcap"] == {} and report["attention"] is None
    passes = {(p["checkpoint"], p["split"]): p for p in report["passes"]}
    assert all(p["status"] == "ok" and p["cross_check"]["ok"] for p in passes.values())
    assert report["reproduction"]["final_weights_reproduced"] is True
    assert report["resources"]["parameter_breakdown"]["context_processing"]["total"] == \
        cfg.context_processing_parameters()
    assert report["alibi_slopes"] == pytest.approx([2.0 ** -4, 2.0 ** -8])


# ----------------------------------------------------------------------------------------------
# Compute matching
# ----------------------------------------------------------------------------------------------

def test_steps_for_device_hours_and_the_width_ladder_rule(tmp_path: Path) -> None:
    assert steps_for_device_hours(1000.0, 2.0, 512) == math.floor(3600 * 2 * 1000 / 512)
    with pytest.raises(ValueError):
        steps_for_device_hours(0.0, 1.0, 512)
    base = _bcfg(embed_dim=8, n_heads=2, batch_size=4, max_seq_len=8)       # head width 4
    tokens_per_s = {8: 4000.0, 16: 2000.0, 32: 500.0}
    rule = width_ladder_rule(base, hours=0.01, tokens_per_s=tokens_per_s, unique_train_tokens=1000)
    by_width = {row["width"]: row for row in rule["candidates"]}
    for width, row in by_width.items():
        cfg = base.with_width(width)
        assert row["context_parameters"] == cfg.context_processing_parameters() and cfg.d_head == 4
        assert row["required_tokens"] == min(20 * row["context_parameters"], 16 * 1000)
        assert row["max_steps"] == steps_for_device_hours(tokens_per_s[width], 0.01, 32)
        assert row["fits"] == (row["affordable_tokens"] >= row["required_tokens"])
    fitting = [w for w, row in by_width.items() if row["fits"]]
    assert rule["chosen_width"] == max(fitting)
    chosen = rule["config"]
    assert chosen.embed_dim == rule["chosen_width"] and chosen.n_heads == chosen.embed_dim // 4
    assert chosen.max_steps == by_width[chosen.embed_dim]["max_steps"]
    # Calibration and budget read metrics.csv through run_resources.throughput_summary.
    run = tmp_path / "calibration"
    run.mkdir()
    (run / "metrics.csv").write_text(
        "step,tokens_per_s,train_step_tokens_per_s,pipeline_tokens_per_s,train_step_ms_mean,peak_mem_mb\n"
        "100,900,1000,900,50,1\n200,950,1100,950,40,1\n300,960,1200,960,30,1\n", encoding="utf-8")
    assert calibrated_tokens_per_s(run)["tokens_per_s"] == pytest.approx(1100.0)
    assert measured_train_device_hours(run)["train_hours"] == pytest.approx(
        (100 * 0.050 + 100 * 0.040 + 100 * 0.030) / 3600.0)


# ----------------------------------------------------------------------------------------------
# Shared-loop fixes on the VFE path
# ----------------------------------------------------------------------------------------------

def test_frozen_tables_are_reported_apart_under_identity_frames(tmp_path: Path) -> None:
    cfg = _vcfg(gauge_transport="off", decode_temperature_learned=True)
    model = _vmodel(cfg)
    phi = model.prior_bank.phi_embed
    assert not phi.requires_grad and getattr(model, "pos_phi_free", None) is None
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    everything = sum(p.numel() for p in model.parameters())
    out = parameter_breakdown(dict(model.named_parameters()), cfg)
    assert out["frozen"] == {"prior_bank.phi_embed": phi.numel(), "total": phi.numel()}
    assert out["total"] == trainable and out["total_including_frozen"] == everything
    assert out["per_type_tables"]["phi_embed"] == 0
    assert out["roles_by_name"]["prior_bank.phi_embed"]["frozen"] is True
    state = model.state_dict()
    assert frozen_parameter_names(cfg, state) == ["prior_bank.phi_embed"]
    from_state = parameter_breakdown(state, cfg, frozen=frozen_parameter_names(cfg, state))
    assert from_state["total"] == trainable and from_state["frozen"] == out["frozen"]
    assert from_state["parameters_read_per_token"] == out["parameters_read_per_token"]
    # A breakdown recorded before the frozen section existed is corrected by the paper assembler.
    old = parameter_breakdown(state, cfg)
    old = {k: v for k, v in old.items() if k not in ("frozen", "total_including_frozen")}
    assert old["total"] == everything
    assert exclude_frozen_from_breakdown(old, cfg)["total"] == trainable
    from vfe3.paper.runs import RunRecord, parameter_counts
    record = RunRecord(run_dir=tmp_path, config_raw=serialize_config(cfg), cfg=cfg,
                       resources={"parameter_breakdown": old})
    assert parameter_counts(record)["total"] == trainable
    # The live run: summary and resources report trainable counts, frozen listed apart.
    loaders = (_fake_select_loader("x", SimpleNamespace(max_seq_len=N, batch_size=B, eval_stride=3),
                                   split=s, data_seed=2) for s in ("train", "validation", "test"))
    train_loader, val_loader, test_loader = loaders
    run_dir = tmp_path / "run"
    artifacts = ra.RunArtifacts(run_dir, cfg, model, dataset="tiny")
    torch.manual_seed(0)
    losses = T.train(model, train_loader, cfg, n_steps=2, grad_clip=1.0, eval_interval=2,
                     val_loader=val_loader, device=CPU, artifacts=artifacts, generate_samples=False)
    ra.finalize_run(model, artifacts, cfg, final_validation=T.evaluate(model, val_loader),
                    train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                    losses=losses, device=CPU, wall_time=1.0, logger=logging.getLogger("baseline-test"))
    summary = json.loads((run_dir / "summary.json").read_text())
    resources = json.loads((run_dir / "resources.json").read_text())
    assert summary["n_params"] == everything and summary["n_params_trainable"] == trainable
    assert resources["parameter_breakdown"]["total"] == trainable
    assert resources["parameter_breakdown"]["frozen"]["prior_bank.phi_embed"] == phi.numel()
    assert "model_family" not in summary and "skipped_analyses" not in summary     # VFE summary as before


def test_frame_projection_skips_frozen_tables() -> None:
    radius = 5.0
    frozen = _vmodel(_vcfg(gauge_transport="off", phi_mstep_max_matrix_norm=radius))
    counter = torch.zeros(2, dtype=torch.long)
    stats = project_phi_parameter_rows_(frozen, radius, count_out=counter)
    assert stats["phi_chart_total_rows"] == 0.0 and counter.tolist() == [0, 0]
    live = _vmodel(_vcfg(phi_mstep_max_matrix_norm=radius))
    counter = torch.zeros(2, dtype=torch.long)
    stats = project_phi_parameter_rows_(live, radius, count_out=counter)
    assert stats["phi_chart_total_rows"] == float(V + N) and counter.tolist()[1] == V + N


def test_per_type_hidden_bias_optimizer_group_and_roles() -> None:
    from vfe3.run_resources import parameter_role

    assert parameter_role("per_type_hidden_bias") == ("per_type_tables", "per_type_hidden_bias")
    assert T._parameter_group("per_type_hidden_bias") == "per_type_table"
    assert T._parameter_group("_orig_mod.per_type_hidden_bias") == "per_type_table"
    fields = {f.name for f in dataclasses.fields(VFE3Config)}
    cfg = _vcfg(gauge_transport="off", m_phi_lr=0.02, phi_weight_decay=0.07,
                **({"per_type_hidden_bias": True} if "per_type_hidden_bias" in fields else {}))
    model = _vmodel(cfg)
    if getattr(model, "per_type_hidden_bias", None) is None:            # before the model_core option
        n_gen = model.prior_bank.phi_embed.shape[1]
        model.per_type_hidden_bias = torch.nn.Parameter(torch.zeros(V, n_gen))
    table = model.per_type_hidden_bias
    optimizer = T.build_optimizer(model, cfg)
    groups = [g for g in optimizer.param_groups if any(p is table for p in g["params"])]
    assert len(groups) == 1 and groups[0]["params"] == [table]
    assert cfg.m_phi_lr == 0.0                                           # coerced by gauge_transport='off'
    expected_lr = getattr(cfg, "m_per_type_hidden_bias_lr", None)
    assert groups[0]["lr"] == (0.02 if expected_lr is not None else cfg.m_phi_lr)
    assert groups[0]["weight_decay"] == 0.07 and groups[0]["role"] == "phi"
    assert "pullback_group" not in groups[0] and "lr_report_role" not in groups[0]
    counts = parameter_breakdown(dict(model.named_parameters()), cfg)
    assert counts["per_type_tables"]["per_type_hidden_bias"] == table.numel()


def test_vfe_path_is_unchanged_by_the_family_gates(tmp_path: Path) -> None:
    cfg = _vcfg()
    model = _vmodel(cfg)
    assert is_vfe_model(model) and is_vfe_config(cfg) and is_vfe_config(serialize_config(cfg))
    optimizer = T.build_optimizer(model, cfg)
    assert {g["lr_report_role"] for g in optimizer.param_groups if "lr_report_role" in g} == {"mu", "sigma", "phi"}
    lrs = T._learning_rates_by_role(optimizer.param_groups, [g["lr"] for g in optimizer.param_groups])
    assert set(lrs) == {"mu", "sigma", "phi"} and all(math.isfinite(v) for v in lrs.values())
    broken = [dict(g) for g in optimizer.param_groups if g.get("lr_report_role") != "phi"]
    with pytest.raises(RuntimeError, match="report roles"):
        T._learning_rates_by_role(broken, [g["lr"] for g in broken])
    assert ra._serialize_run_config(cfg) == serialize_config(cfg)
    assert T._numerics_counter_skip_reasons(model)["cap_activation"] is None
    # The terminal snapshot is a host copy, bitwise the final iterate; the trajectory is unchanged.
    grabbed: Dict[str, Any] = {}

    def callback(state: T.TrainingTerminalState, losses: List[float]) -> None:
        grabbed["state"] = state.raw_model_state

    loader = _fake_select_loader("x", SimpleNamespace(max_seq_len=N, batch_size=B, eval_stride=3),
                                 split="train", data_seed=2)
    reference = _vmodel(cfg)
    torch.manual_seed(0)
    expected = T.train(reference, _fake_select_loader("x", SimpleNamespace(max_seq_len=N, batch_size=B,
                                                                          eval_stride=3),
                                                      split="train", data_seed=2),
                       cfg, n_steps=3, device=CPU, generate_samples=False)
    torch.manual_seed(0)
    losses = T.train(model, loader, cfg, n_steps=3, device=CPU, terminal_callback=callback,
                     generate_samples=False)
    assert losses == expected
    live = model.state_dict()
    assert set(grabbed["state"]) == set(live)
    for name, tensor in grabbed["state"].items():
        assert tensor.device.type == "cpu" and torch.equal(tensor, live[name]), name
        assert tensor.data_ptr() != live[name].data_ptr(), name
