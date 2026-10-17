r"""The ``fully_gauge_covariant`` switch yields a whole model invariant under a local change of frame (audit 2026-10-16,
F127).

The switch's own tests stop at the configuration (``gauge_covariance_failures(cfg) == []``). Here the switched model
is built in float64, every parameter is moved off its initialization, and the model is run. A change of local frame
:math:`g_i \in G`, drawn independently for every batch row and position, acts on the encode-time prior as

.. math::
    (\mu_i, \Sigma_i, U_i) \mapsto (g_i\mu_i,\ g_i\Sigma_ig_i^\top,\ g_iU_i), \qquad
    \phi_i' = \mathrm{Log}\big(g_i\,e^{\phi_i\cdot G}\big),

with the principal logarithm (:func:`vfe3.geometry.matrix_log.logm_principal`, which also handles the
non-diagonalizable unipotent frames). It is applied by patching ``prior_bank.encode``; a positional right factor
composes after it, :math:`g_iU_ie^{Y_i}`. The logits and the loss must not move. Two controls show the check
discriminates: moving the beliefs without their frames changes the logits, and so does reverting one override the
switch made (``decode_coordinates`` or ``block_coordinates``), which ``gauge_covariance_failures`` then reports.

The reference run patches ``encode`` with the identity, because any patch leaves the per-content factor fast path.
"""
from __future__ import annotations

import warnings
from typing import Callable, Dict, Tuple

import pytest
import torch

from tests.profiles import _CONFIG_VALUES, _LIVE_OMITTED, _LIVE_TINY, config_values, launcher_values
from vfe3.belief import BeliefState
from vfe3.config import _RETIRED_GAUGE_GROUPS, VFE3Config
from vfe3.gauge_covariant import fully_gauge_covariant_overrides, gauge_covariance_failures
from vfe3.geometry.groups import _GROUPS
from vfe3.geometry.lie_ops import embed_phi, extract_phi
from vfe3.geometry.matrix_log import logm_principal
from vfe3.model.model import VFEModel

B, N = 2, 4
_REL = 1e-9          # observed: below 1e-14 on these cases, about 2e-12 on the launcher route of 2026-10-16


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# Projected-predictive kernel readout on the frame-dependent defaults of the test profile (diagonal family, local
# block and decoder, sigma_max, per-coordinate self term), so the switch has work to do in every case.
_BASE = dict(attention_mode="projected_predictive", predictive_readout="kernel",
             predictive_covariance="moment_matched", lambda_alpha_mode="state_dependent_per_coord",
             vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=N, n_layers=2, batch_size=B, block_mlp_expansion=2)
_SO3 = dict(gauge_group="so_n", group_n=3, irrep_spec=[("l1", 1), ("l2", 1)], embed_dim=8)          # 3 + 5
_SU3 = dict(gauge_group="su_n", group_n=3, irrep_spec=[("pq1_0", 1), ("pq1_1", 1)], embed_dim=14)   # 6 + 8

CASES: Dict[str, dict] = {
    # every registered structure group, under each tying it admits
    "block_glk_per_head": {},
    "block_glk_shared":   {"gauge_tying": "shared"},
    "block_so":           {"gauge_group": "block_so"},
    "block_co":           {"gauge_group": "block_co"},
    "block_sl":           {"gauge_group": "block_sl"},
    "block_borel":        {"gauge_group": "block_borel"},
    "block_unipotent":    {"gauge_group": "block_unipotent"},
    "block_glmc":         {"gauge_group": "block_glmc"},
    "so_n_shared":        _SO3,
    "so_n_per_head":      {**_SO3, "gauge_tying": "per_head"},
    "so_n_cg_coupling":   {**_SO3, "use_cg_coupling": True, "use_block_mlp": False, "norm_type_block": "none"},
    "su_n_shared":        _SU3,
    "su_n_per_head":      {**_SU3, "gauge_tying": "per_head"},
    # routes, on block_glk
    "learned_pos_phi":    {"pos_phi": "learned"},
    "frame_rope":         {"rope_mode": "frame"},                        # the switch moves it to comparison space
    "minimal_agent":      {"predictive_readout": "minimal_agent"},
    "prototype_agents":   {"block_mlp_mode": "prototype_agents", "block_mlp_covariance": "posterior"},
    "gauge_gate":         {"block_mlp_mode": "gauge_gate", "norm_type_block": "none"},
    "no_block_mlp":       {"use_block_mlp": False, "norm_type_block": "none"},
    "kl_decode":          {"decode_mode": "diagonal_chunked"},           # the switch moves it to full_chunked
    "gaussian_block":     {"family": "gaussian_block", "decode_mode": "full_chunked"},
    "lowrank_edges":      {"content_link_chart": "exponential", "content_link_edges": "lowrank"},
}


def _raw(case: str) -> dict:
    return config_values(**{**_BASE, **CASES[case]})


def _switched(raw: dict) -> VFE3Config:
    r"""The configuration the launcher runs with ``fully_gauge_covariant=True``, checked as the launcher checks it."""
    cfg = VFE3Config(**{**raw, **fully_gauge_covariant_overrides(raw)})
    assert gauge_covariance_failures(cfg) == []
    return cfg


def _model(cfg: VFE3Config, state: Dict[str, torch.Tensor] | None = None) -> VFEModel:
    r"""A float64 model with every parameter moved off its initialization (or loaded from ``state``)."""
    torch.manual_seed(0)
    model = VFEModel(cfg).double()
    if state is not None:
        model.load_state_dict(state, strict=True)
        return model
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=gen, dtype=p.dtype))
    return model


def _gauge_draw(model: VFEModel, s: float = 0.3, seed: int = 2) -> torch.Tensor:   # (B, N, K, K)
    r"""Independent group elements :math:`g_i = \exp(s\,a_i\cdot G)`, :math:`a_i \sim \mathcal N(0, I)`, per row and
    position."""
    G = model.group.generators.to(torch.float64)                       # (n_gen, K, K)
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(B, N, G.shape[0], generator=gen, dtype=torch.float64)
    return torch.linalg.matrix_exp(s * embed_phi(a, G))


def _frame_change(model: VFEModel, g: torch.Tensor, *, frames: bool) -> Callable[[torch.Tensor], BeliefState]:
    r"""``prior_bank.encode`` followed by the change of frame ``g`` (``frames=False`` moves the beliefs only)."""
    encode = model.prior_bank.encode
    G = model.group.generators.to(torch.float64)                       # (n_gen, K, K)

    def changed(token_ids: torch.Tensor) -> BeliefState:
        prior = encode(token_ids)
        mu = (g @ prior.mu.unsqueeze(-1)).squeeze(-1)                  # (B, N, K)
        sigma = g @ prior.sigma @ g.transpose(-1, -2)                  # (B, N, K, K)
        if not frames:
            return prior._replace(mu=mu, sigma=sigma)
        log_gU, failed, _ = logm_principal(g @ torch.linalg.matrix_exp(embed_phi(prior.phi, G)))   # (B, N, K, K)
        assert not bool(failed.any())
        phi = extract_phi(log_gU, G)                                   # (B, N, n_gen)
        assert (embed_phi(phi, G) - log_gU).abs().max() < 1e-9         # Log(g_i U_i) lies in the algebra's span
        return prior._replace(mu=mu, sigma=sigma, phi=phi)
    return changed


def _run(model: VFEModel, encode: Callable[[torch.Tensor], BeliefState]) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Logits (B, N, V) and loss with ``prior_bank.encode`` replaced by ``encode``."""
    gen = torch.Generator().manual_seed(3)
    ids = torch.randint(0, model.cfg.vocab_size, (B, N), generator=gen)
    targets = torch.randint(0, model.cfg.vocab_size, (B, N), generator=gen)
    model.prior_bank.encode = encode
    try:
        with torch.no_grad():
            logits = model(ids)
            _, loss, _ = model(ids, targets)
    finally:
        del model.prior_bank.encode                                    # the class method again
    return logits, loss


def _relative(x: torch.Tensor, reference: torch.Tensor) -> float:
    return float((x - reference).abs().max() / reference.abs().max())


def _invariance(model: VFEModel) -> Tuple[float, float, float]:
    r"""Relative logit change under the frame change, relative loss change, and the beliefs-only logit change."""
    original = model.prior_bank.encode
    g = _gauge_draw(model)
    logits, loss = _run(model, lambda token_ids: original(token_ids))
    gauged_logits, gauged_loss = _run(model, _frame_change(model, g, frames=True))
    moved_logits, _ = _run(model, _frame_change(model, g, frames=False))
    return (_relative(gauged_logits, logits), float((gauged_loss - loss).abs() / loss.abs()),
            _relative(moved_logits, logits))


def test_the_cases_cover_every_registered_group() -> None:
    covered = {_raw(case)["gauge_group"] for case in CASES}
    assert covered == set(_GROUPS) - _RETIRED_GAUGE_GROUPS


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_switched_model_is_invariant_under_a_local_change_of_frame(case: str) -> None:
    raw = _raw(case)
    assert gauge_covariance_failures(VFE3Config(**raw))                # unswitched, the route is not covariant
    cfg = _switched(raw)
    assert cfg.use_cg_coupling == raw["use_cg_coupling"]               # the CG case keeps its coupling
    logits_change, loss_change, beliefs_only_change = _invariance(_model(cfg))
    assert logits_change < _REL and loss_change < _REL
    assert beliefs_only_change > 1e-3                                  # the frames carry the invariance


@pytest.mark.parametrize("key, flag", [("decode_coordinates", "decoder_gauge_compatible"),
                                       ("block_coordinates", "block_norm_gauge_equivariant")])
def test_reverting_one_override_breaks_the_invariance(key: str, flag: str) -> None:
    raw = _raw("block_glk_per_head")                                   # LayerNorm + coordinate MLP, linear decode
    overrides = fully_gauge_covariant_overrides(raw)
    assert overrides[key] == "reference"
    model = _model(_switched(raw))
    assert _invariance(model)[0] < _REL
    reverted = VFE3Config(**{**raw, **overrides, key: "local"})
    assert flag in gauge_covariance_failures(reverted)
    assert _invariance(_model(reverted, state=model.state_dict()))[0] > 1e-3


def test_the_launcher_route_under_the_switch_is_invariant() -> None:
    r"""The route ``train_vfe3.py`` trains, at tiny scale, with the switch applied as the launcher applies it."""
    launcher = launcher_values()
    try:
        overrides = fully_gauge_covariant_overrides(launcher)
    except ValueError as exc:                                          # a refusal is the switch's contract
        pytest.skip(f"the launcher route has no covariant reading: {exc}")
    values = {**_CONFIG_VALUES, **_LIVE_OMITTED, **launcher, **_LIVE_TINY, **overrides,
              "max_seq_len": N, "batch_size": B}
    logits_change, loss_change, beliefs_only_change = _invariance(_model(_switched(values)))
    assert logits_change < _REL and loss_change < _REL
    assert beliefs_only_change > 1e-3
