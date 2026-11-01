r"""Whole-model frame covariance at the table level and layer by layer (audit 2026-10-31 gauge-covariance-e2e-5).

``tests/test_audit_20261016_a1_passive_gauge_model.py`` patches ``prior_bank.encode`` and moves the encoded beliefs and
frames, checks logits and the loss only, and runs its forwards without reseeding. A patched encoder also closes the
per-content frame route that every training loss takes. Two checks are added here.

The table-level action changes the frame of every agent of type :math:`c` by one :math:`g_c`, applied to the token
tables, and runs the unpatched model:

.. math::
    \phi_c \mapsto \mathrm{Log}\big(g_c\,e^{\phi_c\cdot G}\big), \qquad \mu_c \mapsto g_c\mu_c,

with the covariance table as the family requires (``gaussian_frame_diagonal`` stores reference variances, which the
action leaves fixed, so any :math:`g_c` of the group is representable; the diagonal table of ``gaussian_full`` and
``gaussian_block`` represents a diagonal :math:`g_c`, :math:`\log s_c \mapsto \log s_c + 2\log\operatorname{diag} g_c`).
The loss forward (the deduplicated per-content frame route) and the captured forward (per occurrence) must not move, in
evaluation and in training mode, with the generator reseeded before every forward; moving the means without their
frames must move the loss.

The agent-level action (one :math:`g_i` per row and position, as in the existing test) is checked layer by layer: the
captured state at every stage must transform as :math:`(\mu, \Sigma) \mapsto (g\mu, g\Sigma g^\top)` (:math:`c` fixed on
the frame-diagonal family), the routing weights, logits and loss must not move, and neither may the gradient of any
parameter that is not a token table. The cases include the block MLPs, links and energy terms the existing test does
not run. Float64, tiny shapes (K = 8, two heads, two layers, N = 4); the live launcher route supplies everything the
overrides do not set.
"""
import math
import warnings
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import pytest
import torch

from tests.profiles import live_values
from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.exact_route import principled_exact_route_overrides
from vfe3.gauge_covariant import fully_gauge_covariant_overrides, gauge_covariance_failures
from vfe3.geometry.lie_ops import embed_phi, extract_phi
from vfe3.geometry.matrix_log import logm_principal
from vfe3.model.model import VFEModel

F64 = torch.float64
B, N = 2, 4
_REL = 1e-9                     # observed residuals are below 1e-13 (2.8e-11 on one delta-full logit case)
# Initialization scales at which the beliefs-only controls discriminate (audit 2026-10-31 config-launcher-9); the
# frame condition bound is off, so the gauge flags of every switched case are all true.
_INIT = dict(mu_init_std=0.5, sigma_init=1.0, phi_scale=0.2, phi_frame_condition_max=None)
_FULL_REFERENCE = dict(family="gaussian_full", block_coordinates="reference", decode_coordinates="reference",
                       decode_mode="full_chunked", m_table_update_mode="adamw")
_SO3 = dict(gauge_group="so_n", group_n=3, irrep_spec=[("l1", 1), ("l2", 1)], embed_dim=8, n_heads=2,
            gauge_tying="shared")
_TOKEN_TABLES = ("prior_bank.mu_embed", "prior_bank.sigma_log_embed", "prior_bank.phi_embed",
                 "prior_bank.sigma_chol_embed")


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _switched(**overrides: Any) -> dict:
    raw = live_values(**{**_INIT, **overrides})
    return {**raw, **fully_gauge_covariant_overrides(raw)}


def _exact(**overrides: Any) -> dict:
    raw = live_values(**{**_INIT, **overrides})
    return {**raw, **principled_exact_route_overrides(raw), "phi_frame_condition_max": None}


def _config(raw: dict) -> VFE3Config:
    cfg = VFE3Config(**raw)
    assert gauge_covariance_failures(cfg) == []
    return cfg


def _build(cfg: VFE3Config) -> VFEModel:
    r"""A float64 model with every parameter moved off its initialization."""
    torch.manual_seed(0)
    model = VFEModel(cfg).double()
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=gen, dtype=p.dtype))
    return model


def _relative(x: torch.Tensor, reference: torch.Tensor) -> float:
    scale = float(reference.abs().max())
    return float((x - reference).abs().max()) / (scale if scale > 0 else 1.0)


# --------------------------------------------------------------------------------------------- table-level action


def _table_action(model: VFEModel, *, diagonal: bool, means_only: bool = False, s: float = 0.3) -> None:
    r"""Apply one :math:`g_c` per token type to the tables in place (``means_only`` moves :math:`\mu_c` alone)."""
    pb = model.prior_bank
    G = model.group.generators.to(F64)                                     # (n_gen, K, K)
    V, K = pb.mu_embed.shape
    gen = torch.Generator().manual_seed(7)
    with torch.no_grad():
        U = torch.linalg.matrix_exp(embed_phi(pb.phi_embed, G))            # (V, K, K)
        if diagonal:
            t = s * torch.randn(V, K, generator=gen, dtype=F64)
            g = torch.diag_embed(t.exp())                                  # (V, K, K)
            log_gU, failed, _ = logm_principal(g @ U)
            assert not bool(failed.any())
            phi = extract_phi(log_gU, G)
            assert float((embed_phi(phi, G) - log_gU).abs().max()) < 1e-9  # g_c U_c stays in the chart
            if not means_only and pb.sigma_log_embed is not None and not pb.frame_diagonal:
                pb.sigma_log_embed.add_(2.0 * t)
        else:
            assert pb.frame_diagonal                                       # only c_v is invariant under any g_c
            phi = pb.phi_embed + s * torch.randn(pb.phi_embed.shape, generator=gen, dtype=F64)
            g = torch.linalg.matrix_exp(embed_phi(phi, G)) @ torch.linalg.inv(U)
        if not means_only:
            pb.phi_embed.copy_(phi)
        pb.mu_embed.copy_((g @ pb.mu_embed.unsqueeze(-1)).squeeze(-1))


def _measure(model: VFEModel, ids: torch.Tensor, targets: torch.Tensor) -> Dict[Tuple[bool, str], torch.Tensor]:
    r"""The loss (deduplicated frame route) and the captured logits (per occurrence), in evaluation and training
    mode, each forward from the same generator state."""
    out: Dict[Tuple[bool, str], torch.Tensor] = {}
    for train in (False, True):
        model.train(train)
        with torch.no_grad():
            torch.manual_seed(11)
            out[(train, "loss")] = model(ids, targets)[1]
            torch.manual_seed(11)
            out[(train, "captured_logits")] = model.forward_beliefs(ids, return_logits=True, capture={})[1]
    return out


_TABLE_CASES: Dict[str, Tuple[Callable[[], dict], bool]] = {
    # name: (configuration, diagonal g_c)
    "frame_diagonal_reference": (lambda: _switched(family="gaussian_frame_diagonal"), False),
    "exact_route": (lambda: _exact(), False),
    "full_untied_bank": (lambda: _switched(**_FULL_REFERENCE), True),
    "block_untied_bank": (lambda: _switched(family="gaussian_block", block_coordinates="reference",
                                            decode_coordinates="reference", decode_mode="diagonal_chunked",
                                            m_table_update_mode="adamw"), True),
    "full_linear_decode": (lambda: _switched(**{**_FULL_REFERENCE, "decode_mode": "linear"}), True),
}


@pytest.mark.parametrize("case", sorted(_TABLE_CASES))
def test_a_per_type_frame_change_of_the_tables_leaves_both_frame_routes_invariant(case: str) -> None:
    make, diagonal = _TABLE_CASES[case]
    cfg = _config(make())
    model = _build(cfg)
    assert model.cfg.untie_decode_bank or model.cfg.decode_mode == "linear"
    ids = torch.tensor([[1, 2, 3, 1], [4, 1, 2, 5]])                      # repeated types within and across rows
    targets = torch.randint(0, cfg.vocab_size, ids.shape, generator=torch.Generator().manual_seed(3))
    state = {name: p.detach().clone() for name, p in model.state_dict().items()}
    before = _measure(model, ids, targets)
    _table_action(model, diagonal=diagonal)
    after = _measure(model, ids, targets)
    for key, value in before.items():
        assert _relative(after[key], value) < _REL, key
    model.load_state_dict(state)
    _table_action(model, diagonal=diagonal, means_only=True)              # control: the means without the frames
    moved = _measure(model, ids, targets)
    assert abs(float(moved[(True, "loss")] - before[(True, "loss")])) > 1e-3 * abs(float(before[(True, "loss")]))


def test_the_table_level_action_moves_a_route_that_reads_local_coordinates() -> None:
    r"""Control of the table-level check: the unswitched live route (local block norm, MLP and decoder) is not
    covariant, and the same per-type action moves its loss and logits."""
    cfg = VFE3Config(**live_values(**_INIT))
    assert gauge_covariance_failures(cfg)
    model = _build(cfg)
    ids = torch.tensor([[1, 2, 3, 1], [4, 1, 2, 5]])
    targets = torch.randint(0, cfg.vocab_size, ids.shape, generator=torch.Generator().manual_seed(3))
    before = _measure(model, ids, targets)
    _table_action(model, diagonal=True)
    after = _measure(model, ids, targets)
    assert _relative(after[(True, "captured_logits")], before[(True, "captured_logits")]) > 1e-3
    assert abs(float(after[(False, "loss")] - before[(False, "loss")])) > 1e-3 * abs(float(before[(False, "loss")]))


# --------------------------------------------------------------------------------------------- agent level, by layer


_STAGES = ("layer_converged", "layer_block_mlp_inputs", "layer_block_mlp_outputs", "layer_outputs")


def _gauge_draw(model: VFEModel, s: float = 0.3) -> torch.Tensor:                 # (B, N, K, K)
    G = model.group.generators.to(F64)
    a = torch.randn(B, N, G.shape[0], generator=torch.Generator().manual_seed(2), dtype=F64)
    return torch.linalg.matrix_exp(s * embed_phi(a, G))


def _frame_change(model: VFEModel, g: torch.Tensor, *, frames: bool) -> Callable[[torch.Tensor], BeliefState]:
    encode = model.prior_bank.encode
    G = model.group.generators.to(F64)

    def changed(token_ids: torch.Tensor) -> BeliefState:
        prior = encode(token_ids)
        mu = (g @ prior.mu.unsqueeze(-1)).squeeze(-1)                     # (B, N, K)
        sigma = prior.sigma if model.prior_bank.frame_diagonal else g @ prior.sigma @ g.transpose(-1, -2)
        if not frames:
            return prior._replace(mu=mu, sigma=sigma)
        log_gU, failed, _ = logm_principal(g @ torch.linalg.matrix_exp(embed_phi(prior.phi, G)))
        assert not bool(failed.any())
        return prior._replace(mu=mu, sigma=sigma, phi=extract_phi(log_gU, G))
    return changed


def _agent_run(model: VFEModel, encode: Callable[[torch.Tensor], BeliefState], *, grads: bool = True) -> dict:
    gen = torch.Generator().manual_seed(3)
    ids = torch.randint(0, model.cfg.vocab_size, (B, N), generator=gen)
    targets = torch.randint(0, model.cfg.vocab_size, (B, N), generator=gen)
    model.prior_bank.encode = encode
    try:
        torch.manual_seed(123)
        with torch.no_grad():
            capture: Dict[str, Any] = {"diagnostic": {}}
            _, logits = model.forward_beliefs(ids, return_logits=True, capture=capture)
        torch.manual_seed(123)
        model.zero_grad(set_to_none=True)
        _, loss, _ = model(ids, targets)
        gradients: Dict[str, torch.Tensor] = {}
        if grads:
            loss.backward()
            gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()
                         if p.grad is not None and name not in _TOKEN_TABLES}
    finally:
        del model.prior_bank.encode
    return {"diagnostic": capture["diagnostic"], "logits": logits, "loss": loss.detach(), "grads": gradients}


def _layer_deviation(model: VFEModel, base: dict, moved: dict, g: torch.Tensor) -> Dict[str, float]:
    r"""Worst relative deviation of the moved state from the transformed base state, per stage, and of the routing
    weights."""
    worst: Dict[str, float] = {}
    for stage in _STAGES:
        layers = list(zip(base["diagnostic"][stage], moved["diagnostic"][stage]))
        assert layers, stage
        for b0, b1 in layers:
            deviation = _relative(b1.mu, (g @ b0.mu.unsqueeze(-1)).squeeze(-1))
            sigma = b0.sigma if model.prior_bank.frame_diagonal else g @ b0.sigma @ g.transpose(-1, -2)
            worst[stage] = max(worst.get(stage, 0.0), deviation, _relative(b1.sigma, sigma))
    weights = 0
    for c0, c1 in zip(base["diagnostic"].get("layer_conditional_captures", []),
                      moved["diagnostic"].get("layer_conditional_captures", [])):
        for key, value in c0.items():
            other = c1.get(key)
            if key.startswith("beta") and torch.is_tensor(value) and torch.is_tensor(other):
                worst["routing_weights"] = max(worst.get("routing_weights", 0.0), _relative(other, value))
                weights += 1
    assert weights, "no routing weights were captured"
    return worst


_LAYER_CASES: Dict[str, Callable[[], dict]] = {
    "switched_live_route": lambda: _switched(),
    "exact_route": lambda: _exact(),
    "gaussian_template_mlp_full_reference": lambda: _switched(**_FULL_REFERENCE, block_mlp_mode="gaussian_template"),
    "cue_result_memory_full_reference": lambda: _switched(
        **_FULL_REFERENCE, block_mlp_mode="cue_result_memory", block_mlp_covariance="posterior",
        block_mlp_inference="exact", block_memory_triton=False, block_memory_regression_rank=2,
        block_mlp_template_init="absolute"),
    "prototype_em_term": lambda: _switched(
        **_FULL_REFERENCE, block_mlp_mode="prototype_agents", block_mlp_covariance="posterior",
        block_mlp_inference="mean_field", block_mlp_em_weight=0.3, block_mlp_template_init="absolute"),
    "canonical_frame_mlp_mahalanobis_norm": lambda: _switched(block_mlp_mode="canonical_frame",
                                                             block_mlp_tie_layers=False,
                                                             norm_type_block="mahalanobis"),
    "latent_links_loop_prior": lambda: _switched(**_FULL_REFERENCE, content_link_edges="lowrank",
                                                 content_link_latent=True, content_link_loop_weight=0.5),
    "per_type_hidden_bias": lambda: _switched(per_type_hidden_bias=True),
    "so3_cg_energy": lambda: _switched(**_SO3, use_cg_coupling=True, cg_energy_weight=0.3, use_block_mlp=False,
                                       norm_type_block="none", rope_mode="none", beta_attention_prior="causal"),
}


@pytest.mark.parametrize("case", sorted(_LAYER_CASES))
def test_every_layer_routing_weight_and_gradient_is_covariant_under_a_local_change_of_frame(case: str) -> None:
    cfg = _config(_LAYER_CASES[case]())
    model = _build(cfg)
    model.train()
    g = _gauge_draw(model)
    plain = model.prior_bank.encode
    base = _agent_run(model, lambda token_ids: plain(token_ids))
    moved = _agent_run(model, _frame_change(model, g, frames=True))
    control = _agent_run(model, _frame_change(model, g, frames=False), grads=False)
    worst = _layer_deviation(model, base, moved, g)
    assert max(worst.values()) < _REL, worst
    assert _relative(moved["logits"], base["logits"]) < _REL
    assert abs(float(moved["loss"] - base["loss"])) < _REL * abs(float(base["loss"]))
    assert base["grads"] and set(moved["grads"]) == set(base["grads"])
    for name, grad in base["grads"].items():
        assert _relative(moved["grads"][name], grad) < _REL, name
    assert _relative(control["logits"], base["logits"]) > 1e-3            # the frames carry the invariance
