"""Tiny depth regressions for projected predictive-content attention."""
from pathlib import Path
import pytest
import torch
from tests.test_relational_integration import _cfg, _tokens
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer
from vfe3.run_artifacts import RunArtifacts, load_checkpoint


def _depth_cfg(layers: int, **kwargs) -> VFE3Config:
    return _cfg(n_layers=layers, attention_mode="projected_predictive",
                relational_input_scope="full", relational_output_scope="full",
                attention_residual=True,
                prior_handoff_rho=1.0, prior_handoff_sigma=0.1, **kwargs)


@pytest.mark.parametrize("layers", [2, 3])
def test_active_launcher_accepts_depth(layers: int) -> None:
    # Source of truth is the shipped click-to-run launcher dict (the docs/changes copy of this
    # config was deleted with the docs tree on 2026-09-18).
    import train_vfe3

    settings = dict(train_vfe3.config)
    settings["n_layers"] = layers
    cfg = VFE3Config(**settings)
    assert cfg.n_layers == layers and cfg.attention_residual


@pytest.mark.parametrize("layers", [2, 3])
@pytest.mark.parametrize("amp", [False, True])
def test_depth_forward_backward_and_snapshot(layers: int, amp: bool) -> None:
    torch.manual_seed(17)
    cfg = _depth_cfg(layers)
    model = VFEModel(cfg)
    ids, targets = _tokens()
    capture = {"diagnostic": {}}
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        belief, logits = model.forward_beliefs(ids, return_logits=True, capture=capture)
    torch.nn.functional.cross_entropy(logits.float().reshape(-1, 7), targets.reshape(-1)).backward()
    assert torch.isfinite(belief.mu).all()
    for module in [*model.relational_operators_by_layer, *model.block_mlps]:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert sum(p.grad.abs().sum() for p in module.parameters()) > 0
    diagnostic = capture["diagnostic"]
    assert len(diagnostic["layer_outputs"]) == layers
    for i in range(1, layers):
        torch.testing.assert_close(diagnostic["layer_priors"][i][0], diagnostic["layer_outputs"][i-1].mu.float())
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        snapshot = model.build_diagnostic_snapshot(ids)
    assert snapshot.beta_maps.shape == (layers, 2, 4, 4)
    assert len(snapshot.layer_outputs) == layers
    per_layer = model.diagnostics_per_layer(ids, snapshot=snapshot)
    assert len(per_layer["attn_entropy"]) == layers
    for i in range(layers):
        torch.testing.assert_close(snapshot.beta_maps[i], diagnostic["layer_conditional_captures"][i]["beta_pre"][0].float())


def test_depth_checkpoint_and_optimizer(tmp_path: Path) -> None:
    cfg = _depth_cfg(2)
    source = VFEModel(cfg)
    optimizer = build_optimizer(source, cfg)
    ids, targets = _tokens()
    _, loss, _ = source(ids, targets)
    loss.backward()
    optimizer.step()
    path = RunArtifacts(tmp_path / "run", cfg, source).save_checkpoint(1, source, optimizer, cfg)
    target = VFEModel(cfg)
    restored = build_optimizer(target, cfg)
    assert load_checkpoint(path, target, restored, cfg=cfg, restore_rng=False) == 1
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, target.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(source(ids), target(ids), rtol=0, atol=0)
    for source_layer, target_layer in zip(source.relational_operators_by_layer, target.relational_operators_by_layer):
        for left, right in zip(source_layer.parameters(), target_layer.parameters()):
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(optimizer.state[left][key], restored.state[right][key], rtol=0, atol=0)
    assert len({id(p) for g in optimizer.param_groups for p in g["params"]}) == sum(len(g["params"]) for g in optimizer.param_groups)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_scaled_active_config_training_step(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    import train_vfe3

    settings = dict(train_vfe3.config)
    settings.update(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=4,
                    attention_window=4, eval_stride=1, batch_size=1,
                    block_mlp_expansion=2, n_layers=3, max_steps=2, warmup_steps=0)
    cfg = VFE3Config(**settings)
    torch.manual_seed(23)
    model = VFEModel(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    ids, targets = (x.to(device) for x in _tokens())
    _, loss, _ = model(ids, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for mlp in model.block_mlps for p in mlp.parameters())
    optimizer.step()
    snapshot = model.build_diagnostic_snapshot(ids)
    assert snapshot.beta_maps.shape[0] == 3
    assert len(model.diagnostics_per_layer(ids, snapshot=snapshot)["attn_entropy"]) == 3
    assert torch.isfinite(snapshot.logits).all()


def test_depth_gradient_matches_finite_difference() -> None:
    torch.manual_seed(5)
    model = VFEModel(_depth_cfg(2)).double()
    ids, _ = _tokens()
    param = model.relational_operators_by_layer[1].raw_score_q
    objective = model.forward_beliefs(ids, return_logits=False)[0].mu.square().sum()
    derivative = torch.autograd.grad(objective, param)[0].flatten()[0]
    step = 1e-5
    with torch.no_grad():
        initial = param.flatten()[0].item()
        param.flatten()[0] = initial + step
        plus = model.forward_beliefs(ids, return_logits=False)[0].mu.square().sum()
        param.flatten()[0] = initial - step
        minus = model.forward_beliefs(ids, return_logits=False)[0].mu.square().sum()
        param.flatten()[0] = initial
    torch.testing.assert_close(derivative, (plus-minus)/(2*step), rtol=1e-3, atol=1e-6)



def test_projection_parameters_are_independent_and_optimized() -> None:
    from vfe3.run_artifacts import _relational_artifact_fields
    cfg = _depth_cfg(3, m_relational_score_lr=0.003, m_relational_content_lr=0.007)
    model = VFEModel(cfg)
    ops = model.relational_operators_by_layer
    assert len(ops) == 3 and ops[0] is model.relational_operators
    for name in ("raw_score_q", "raw_score_k", "T", "O"):
        assert len({getattr(layer, name).data_ptr() for layer in ops}) == 3
    assert not torch.equal(ops[0].raw_score_q, ops[1].raw_score_q)
    optimizer = build_optimizer(model, cfg)
    for layer in ops:
        for name in ("raw_score_q", "raw_score_k", "T", "O"):
            param = getattr(layer, name)
            owners = [g for g in optimizer.param_groups if any(p is param for p in g["params"])]
            assert len(owners) == 1
            assert owners[0]["lr"] == (0.007 if name in ("T", "O") else 0.003)
    metadata = _relational_artifact_fields(cfg)["relational_operators"]
    assert metadata["layer_parameter_sharing"] == "independent"
    assert metadata["total_parameters"] == sum(p.numel() for layer in ops for p in layer.parameters())


def test_middle_projection_changes_only_middle_and_later_layers() -> None:
    torch.manual_seed(31)
    model = VFEModel(_depth_cfg(3))
    ids, _ = _tokens()
    before = model.build_diagnostic_snapshot(ids)
    with torch.no_grad():
        model.relational_operators_by_layer[1].O.mul_(0.3)
        model.relational_operators_by_layer[1].raw_score_q.mul_(1.5)
    after = model.build_diagnostic_snapshot(ids)
    torch.testing.assert_close(before.layer_outputs[0].mu, after.layer_outputs[0].mu, rtol=0, atol=0)
    for index in (1, 2):
        assert not torch.equal(before.layer_outputs[index].mu, after.layer_outputs[index].mu)
    metrics = model.diagnostics_per_layer(ids, snapshot=after)
    assert metrics["relational_score_q_raw_min_singular_value"][1] == pytest.approx(1.5, abs=1e-5)
    assert metrics["relational_score_q_raw_min_singular_value"][0] == pytest.approx(1.0, abs=1e-5)


def test_single_layer_projection_state_keys_remain_compatible() -> None:
    model = VFEModel(_depth_cfg(1))
    keys = {key for key in model.state_dict() if key.startswith("relational_")}
    assert keys == {"relational_operators." + name for name in ("raw_score_q", "raw_score_k", "T", "O")}
    assert len(model.relational_operators_by_layer) == 1
