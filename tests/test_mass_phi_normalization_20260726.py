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

from tests.profiles import test_config

import torch

from vfe3.model.model import VFEModel

BASE = dict(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1)
MASS = 0.25


def _tiny_model(**overrides):
    cfg = test_config(**BASE, **overrides)
    torch.manual_seed(0)
    return cfg, VFEModel(cfg)


def test_mstep_penalty_is_the_per_position_squared_norm_not_a_global_mean():
    """The .sum() / .mean() mismatch. The projected routes run no phi substep, so the loss delta
    is exactly the M-step penalty and the normalization is observable."""
    cfg_off, model_off = _tiny_model()
    cfg_on, model_on = _tiny_model(mass_phi=MASS)

    torch.manual_seed(1)
    tokens = torch.randint(0, cfg_off.vocab_size, (2, cfg_off.max_seq_len))
    targets = torch.randint(0, cfg_off.vocab_size, (2, cfg_off.max_seq_len))

    _, loss_off, ce_off = model_off(tokens, targets=targets)
    _, loss_on, ce_on = model_on(tokens, targets=targets)
    belief, _ = model_on.forward_beliefs(tokens)

    # the belief is untouched by mass_phi at so the CE sector is identical
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


