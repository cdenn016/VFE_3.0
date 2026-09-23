"""Small end-to-end reachability, optimizer and snapshot tests on CPU."""
from pathlib import Path
from typing import Any

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer, _learning_rates_by_aux_role, _val_diagnostics_one
from vfe3.run_artifacts import RunArtifacts, load_checkpoint, _pure_path_report
from vfe3.viz.extract import e_step_belief_trace, numerical_health


def _cfg(**overrides: Any) -> VFE3Config:
    values = dict(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=4,
                  n_layers=1, n_e_steps=1, family="gaussian_diagonal",
                  gauge_group="block_glk", skip_belief_sigma_update=True,
                  e_phi_lr=0.0, oracle_unroll_grad=True, warmup_steps=0,
                  max_steps=2, log_interval=0, eval_interval=0,
                  pos_phi="learned", use_head_mixer=True, use_block_mlp=True,
                  block_mlp_expansion=1, attention_mode="projected_conditional")
    values.update(overrides)
    return VFE3Config(**values)


def _tokens() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor([[0, 1, 2, 3]]), torch.tensor([[1, 2, 3, 4]])


def test_real_forward_uses_router_content_and_live_outer_gradients() -> None:
    torch.manual_seed(17)
    model = VFEModel(_cfg())
    ids, targets = _tokens()
    cap = {}
    belief, logits = model.forward_beliefs(ids, return_logits=True, capture=cap)
    actual = cap['relational_capture']
    assert actual['beta_pre'].shape == (1, 2, 4, 4)
    assert actual['projected_covariance_pre'].shape == (1, 4, 2, 2, 2)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1))
    derivatives = torch.autograd.grad(loss, tuple(model.relational_operators.parameters()))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 1e-10 for g in derivatives)
    with torch.no_grad():
        model.relational_operators.O.zero_()
        changed, changed_logits = model.forward_beliefs(ids, return_logits=True)
    assert not torch.equal(changed_logits, logits)
    assert torch.equal(changed.sigma, belief.sigma)


def test_disabled_mode_has_no_parameters_or_rng_effect() -> None:
    cfg = _cfg(attention_mode="transported")
    torch.manual_seed(91)
    first = VFEModel(cfg)
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(91)
    second = VFEModel(cfg)
    assert torch.equal(expected_rng, torch.random.get_rng_state())
    assert first.relational_operators is None
    assert not any(k.startswith("relational_operators.") for k in first.state_dict())
    ids, _ = _tokens()
    torch.testing.assert_close(first.forward_beliefs(ids, return_logits=False)[0].mu, second.forward_beliefs(ids, return_logits=False)[0].mu)


@pytest.mark.parametrize("identity_content", [False, True])
def test_optimizer_owns_each_parameter_once(identity_content: bool) -> None:
    cfg = _cfg(relational_identity_content=identity_content,
               m_relational_score_lr=0.0,
               m_relational_content_lr=None if identity_content else 0.002)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    assert len(parameters) == len({id(p) for p in parameters})
    assert {id(p) for p in parameters} == {id(p) for p in model.parameters() if p.requires_grad}
    roles = {g.get('lr_aux_role'): g for g in optimizer.param_groups}
    assert 'relational_score' in roles
    assert ('relational_content' in roles) == (not identity_content)
    rates = _learning_rates_by_aux_role(optimizer.param_groups, [g['lr'] for g in optimizer.param_groups])
    assert rates['relational_score'] == 0.0
    if not identity_content:
        assert rates['relational_content'] == 0.002


def test_checkpoint_restores_operator_and_optimizer_state(tmp_path: Path) -> None:
    cfg = _cfg()
    source = VFEModel(cfg)
    optimizer = build_optimizer(source, cfg)
    for p in source.relational_operators.parameters():
        p.grad = torch.ones_like(p)
    optimizer.step()
    checkpoint = RunArtifacts(tmp_path / "run", cfg, source).save_checkpoint(1, source, optimizer, cfg)
    target = VFEModel(cfg)
    restored = build_optimizer(target, cfg)
    assert load_checkpoint(checkpoint, target, restored, cfg=cfg, restore_rng=False) == 1
    for key, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[key])
    for a, b in zip(source.relational_operators.parameters(), target.relational_operators.parameters()):
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(optimizer.state[a][key], restored.state[b][key])
    ids, _ = _tokens()
    torch.testing.assert_close(source.forward_beliefs(ids, return_logits=False)[0].mu, target.forward_beliefs(ids, return_logits=False)[0].mu)


@pytest.mark.parametrize("autocast", [False, True])
def test_snapshot_reports_actual_conditional_semantics(autocast: bool) -> None:
    model = VFEModel(_cfg())
    ids, targets = _tokens()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        snapshot = model.build_diagnostic_snapshot(ids)
    assert snapshot.relational_capture is not None
    assert snapshot.trace_free_energy.numel() == 0
    diagnostics = model.diagnostics(ids, snapshot=snapshot)
    assert diagnostics["relational_semantics"] == "relational_projected_conditional_filtering"
    assert diagnostics["relational_legacy_global_f_available"] == 0.0
    assert diagnostics["relational_covariance_repair_count"] == 0.0
    trajectory = e_step_belief_trace(model, ids, snapshot=snapshot)
    assert trajectory['free_energy_available'] is False
    assert 'relational_conditional_cost_before' in trajectory
    health = numerical_health(model, ids, snapshot=snapshot)
    assert health['nan_energy'] == health['nan_beta'] == 0
    validation = _val_diagnostics_one(model, (ids, targets), torch.device("cpu"))
    assert validation.metrics['val_relational_snapshot_ce'] >= 0
    assert _pure_path_report(model.cfg, [])['pure_flags']['canonical_attention_entropy'] is False
