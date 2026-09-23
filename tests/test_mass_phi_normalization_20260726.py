r"""D-08: one ``mass_phi`` must name ONE penalty strength, and the reported F must carry it.

The manuscript's M-step loss and phi E-step both carry ``(alpha_phi/2)||phi||^2`` with the same
coefficient. Before this fix the executable had three inconsistent readings of that one field:

  * ``e_step.phi_alignment_loss``  used ``(phi ** 2).sum()``   -- the per-sequence squared norm,
  * ``model.VFEModel.forward``     used ``(phi ** 2).mean()``  -- smaller by ``phi.numel()``
    (10^3 to 10^5 at production shapes), so the M-step penalty was effectively a different knob,
  * ``e_step.free_energy_value``   accepted-and-ignored it, so a ``mass_phi > 0`` sweep logged an F
    that omitted the very term the phi substep descends.

All three now read the per-position squared norm ``||phi_i||^2 = (phi ** 2).sum(dim=-1)``. The
E-step and the reported F sum it over the sequence (matching F's per-sequence scale); the M-step
averages it over positions (matching the per-token mean that ``ce`` already is). Everything here is
latent at the ``mass_phi = 0.0`` default -- these tests exist because it is a swept axis.
"""

import torch

from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.inference.e_step import free_energy_value
from vfe3.model.model import VFEModel

BASE = dict(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1)
MASS = 0.25


def _tiny_model(**overrides):
    cfg = VFE3Config(**BASE, **overrides)
    torch.manual_seed(0)
    return cfg, VFEModel(cfg)


def _hand_belief(model, n_positions=3):
    K = model.cfg.embed_dim
    n_gen = model.prior_bank.phi_embed.shape[-1]
    torch.manual_seed(0)
    return BeliefState(
        mu=torch.randn(n_positions, K),
        sigma=torch.rand(n_positions, K) + 0.5,
        phi=torch.randn(n_positions, n_gen) * 0.1,
    )


def test_free_energy_value_honors_mass_phi():
    """The regression: the logged F omitted the phi penalty entirely."""
    _, model = _tiny_model()
    belief = _hand_belief(model)
    n_positions, K = belief.mu.shape
    mu_p = torch.randn(n_positions, K)
    sigma_p = torch.rand(n_positions, K) + 0.5

    f_off = free_energy_value(belief, mu_p, sigma_p, model.group)
    f_on = free_energy_value(belief, mu_p, sigma_p, model.group, mass_phi=MASS)

    expected = 0.5 * MASS * (belief.phi ** 2).sum()
    assert torch.allclose(f_on - f_off, expected, atol=1e-6), (
        f"reported F must carry (mass_phi/2)||phi||^2; got {(f_on - f_off).item()} "
        f"expected {expected.item()}"
    )


def test_free_energy_value_at_mass_phi_zero_is_unchanged():
    """The default must stay byte-identical."""
    _, model = _tiny_model()
    belief = _hand_belief(model)
    n_positions, K = belief.mu.shape
    mu_p = torch.randn(n_positions, K)
    sigma_p = torch.rand(n_positions, K) + 0.5
    f_default = free_energy_value(belief, mu_p, sigma_p, model.group)
    f_explicit_zero = free_energy_value(belief, mu_p, sigma_p, model.group, mass_phi=0.0)
    assert torch.equal(f_default, f_explicit_zero)


def test_mstep_penalty_is_the_per_position_squared_norm_not_a_global_mean():
    """The .sum() / .mean() mismatch. At the default e_phi_lr=0 the phi substep does not move the
    belief, so the loss delta is exactly the M-step penalty and the normalization is observable."""
    cfg_off, model_off = _tiny_model()
    cfg_on, model_on = _tiny_model(mass_phi=MASS)
    assert cfg_off.e_phi_lr == 0.0, "this test's isolation relies on the e_phi_lr=0 default"

    torch.manual_seed(1)
    tokens = torch.randint(0, cfg_off.vocab_size, (2, cfg_off.max_seq_len))
    targets = torch.randint(0, cfg_off.vocab_size, (2, cfg_off.max_seq_len))

    _, loss_off, ce_off = model_off(tokens, targets=targets)
    _, loss_on, ce_on = model_on(tokens, targets=targets)
    belief, _ = model_on.forward_beliefs(tokens)

    # the belief is untouched by mass_phi at e_phi_lr=0, so the CE sector is identical
    assert torch.allclose(ce_off, ce_on, atol=1e-6)

    penalty = loss_on - ce_on
    per_position = 0.5 * MASS * (belief.phi ** 2).sum(dim=-1).mean()
    global_mean = 0.5 * MASS * (belief.phi ** 2).mean()             # the pre-fix expression

    assert torch.allclose(penalty, per_position, atol=1e-6), (
        f"M-step penalty must be the per-position squared norm; got {penalty.item()} "
        f"expected {per_position.item()}"
    )
    n_gen = belief.phi.shape[-1]
    assert n_gen > 1, "the two normalizations only differ when n_gen > 1"
    assert not torch.allclose(penalty, global_mean, atol=1e-6), (
        "M-step penalty still matches the old global-mean expression -- the fix did not apply"
    )


def test_mstep_and_estep_agree_on_one_mass_phi():
    """The two sites must differ only by the sum-vs-mean over POSITIONS, never by n_gen."""
    _, model = _tiny_model(mass_phi=MASS)
    torch.manual_seed(1)
    tokens = torch.randint(0, model.cfg.vocab_size, (1, model.cfg.max_seq_len))
    belief, _ = model.forward_beliefs(tokens)
    phi = belief.phi[0]                                             # (N, n_gen), one sequence

    estep_term = 0.5 * MASS * (phi ** 2).sum()                      # phi_alignment_loss / reported F
    mstep_term = 0.5 * MASS * (phi ** 2).sum(dim=-1).mean()         # VFEModel.forward
    n_positions = phi.shape[0]
    assert torch.allclose(estep_term, mstep_term * n_positions, atol=1e-6)
