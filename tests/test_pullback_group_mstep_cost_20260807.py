r"""``pullback_group`` M-step: cheaper staging without moving the certificates (audit 2026-08-07).

Four changes, all on the ``m_phi_update_mode='pullback_group'`` route, none of which relaxes a
certificate:

**(1) The series-order floor was a cost, not a tolerance.** ``_adaptive_phi_differentials`` bounds
its own truncation error with a geometric majorant and returns the first candidate order that
majorant accepts -- but the candidate grid started at 40. Measured against the live ``block_glk``
structure constants at ``phi_scale=0.06``, the majorant certifies by order 17, so the loop ran ~23
redundant matmuls of the whole ``(n_blocks, active_rows, n_gen, n_gen)`` workspace: 78.5% of the
staging call at 256 active rows. The grid now starts at 8 in steps of 4. The certificate is
untouched; only how early it is TESTED moved.

That is safe only because lowering the floor does not increase the mixed-order fallback rate. It
could have: a partial certification raises ``_MixedPhiDifferentialOrder``, which the caller catches
to abandon the batched fast path for the per-block loop -- so a lower floor trading matmuls for
fallbacks would have been a net pessimization. Measured over 108 trials at K=20/H=2 (scale
0.06/0.0632/0.2 x 64/256/1024 active rows): ZERO fallbacks under both the old and new grids, because
both leading blocks draw phi from one init and certify together. The fallback remains, and remains
bit-exact, for the lopsided case that actually needs it.

**(2) ``exp(phi)`` was recomputed once per backtracking attempt.** ``phi`` is loop-invariant while
the trust factor is halved; only ``delta`` and the candidate move. Hoisted.

**(3) Host syncs.** Every ``bool(tensor)`` on CUDA is a ``cudaStreamSynchronize``; 33 were measured
per staging call. The predicates are unchanged and raise the same errors -- they are now reduced on
device and read once.

**(4) ``fused`` is forwarded to ``GaugeManifoldAdamW``.** Leaving it unset is not the same as off:
AdamW defaults ``fused=None``, and with ``foreach`` also unresolved the multi-tensor path runs at 9
kernel launches per group per step against 1. Not exercised here -- ``fused`` is CUDA-only.

Pins: (a) the certified order actually drops below the old floor at the live scale; (b) the
majorant still holds at the returned order; (c) hoisting ``exp(phi)`` is value-identical; (d) a
nonfinite group-product residual still raises rather than being committed; (e) nonfinite inputs and
certificates still raise; (f) the lopsided mixed-order fallback still fires.
"""

import pytest
import torch

import vfe3.gauge_optim as gauge_optim
import vfe3.geometry.phi_preconditioner as pp
from vfe3.geometry.generators import generate_glk_multihead
from vfe3.geometry.groups import get_group

K, H = 8, 2
LIVE_PHI_SCALE = 0.06


def _group():
    return get_group("block_glk")(K, H, dtype=torch.float64)


def _staging_inputs(rows=6, scale=LIVE_PHI_SCALE, seed=0):
    g = torch.Generator().manual_seed(seed)
    n_gen = _group().generators.shape[0]
    phi = scale * torch.randn(rows, n_gen, generator=g, dtype=torch.float64)
    grad = 0.01 * torch.randn(rows, n_gen, generator=g, dtype=torch.float64)
    return grad, phi


def _stage(grad, phi, group, **kw):
    params = dict(learning_rate=2.5e-3, trust_radius=0.1, chart_max_norm=12.0,
                  bch_residual_max=1e-6, phi_precond_mode="pullback_per_block")
    params.update(kw)
    return gauge_optim.stage_pullback_group_candidate(grad, phi, group, **params)


# -- (a)/(b) the floor no longer pads the series, and the majorant still holds ------------------

def test_certified_series_order_drops_below_the_old_floor_at_the_live_phi_scale():
    grad, phi = _staging_inputs()
    candidate = _stage(grad, phi, _group())

    assert candidate.direction.series_order < 40, (
        "the certificate converges well before 40 at phi_scale=0.06; a floor of 40 was pure "
        f"padding, got {candidate.direction.series_order}"
    )
    assert candidate.direction.series_order in range(
        pp._PHI_GROUP_MIN_SERIES_ORDER, pp._PHI_GROUP_MAX_SERIES_ORDER + 1,
        pp._PHI_GROUP_SERIES_ORDER_STEP,
    )


def test_returned_order_still_satisfies_the_geometric_majorant():
    r"""The floor moved; the error bound did not. Re-derive it independently of the loop."""
    basis = generate_glk_multihead(K // H, 1).double()
    prep = pp._build_strict_basis_preparation(basis)
    g = torch.Generator().manual_seed(3)
    phi = LIVE_PHI_SCALE * torch.randn(H, 5, basis.shape[0], generator=g, dtype=torch.float64)
    ad = torch.einsum("...a,abc->...cb", phi, prep.structure)

    psi_right, _, order = pp._adaptive_phi_differentials(ad, require_uniform_leading_batch=True)

    # bound_k = alpha^order/(order+1)!, tail = first_omitted / (1 - alpha/(order+2))
    alpha = torch.minimum(
        torch.linalg.matrix_norm(ad, ord=1, dim=(-2, -1)),
        torch.linalg.matrix_norm(ad, ord=float("inf"), dim=(-2, -1)),
    )
    bound = torch.ones_like(alpha)
    for k in range(1, order + 1):
        bound = bound * alpha / float(k + 1)
    tail = (bound * alpha / float(order + 1)) / (1.0 - alpha / float(order + 2))
    scale = torch.maximum(
        torch.ones_like(tail),
        torch.linalg.matrix_norm(psi_right, ord=1, dim=(-2, -1)),
    )
    assert bool((tail <= pp._PHI_GROUP_TAIL_TOL * scale).all()), "majorant violated at the returned order"


# -- (c) hoisting exp(phi) out of the backtracking loop is value-identical ----------------------

def test_hoisted_current_element_matches_a_recomputed_one():
    group = _group()
    _, phi = _staging_inputs()
    g = torch.Generator().manual_seed(7)
    delta = 1e-3 * torch.randn_like(phi)
    candidate_phi = phi + delta

    hoisted = gauge_optim._pullback_group_current_element(phi, group)
    residual_hoisted = gauge_optim._pullback_group_product_residual(
        candidate_phi, hoisted, delta, group)
    # Recompute the element from scratch, as the pre-fix code did on every attempt.
    residual_fresh = gauge_optim._pullback_group_product_residual(
        candidate_phi, gauge_optim._pullback_group_current_element(phi, group), delta, group)

    assert torch.equal(residual_hoisted, residual_fresh)
    assert torch.isfinite(residual_hoisted).all()


# -- (d) THE regression: a nonfinite residual must raise, never be accepted ---------------------

def test_nonfinite_group_product_residual_raises_instead_of_being_committed(monkeypatch):
    r"""``nan > limit`` is False, so a bound test alone reads NaN as SUCCESS.

    Folding finiteness into the failure mask is what keeps a nonfinite residual from being
    committed as an accepted candidate. Without it the staging call returns a poisoned chart and
    the M-step writes it into phi.
    """
    group = _group()
    grad, phi = _staging_inputs()

    def _nan_residual(candidate_phi, current_element, delta, grp):
        return torch.full((candidate_phi.shape[0],), float("nan"), dtype=torch.float64)

    monkeypatch.setattr(gauge_optim, "_pullback_group_product_residual", _nan_residual)

    with pytest.raises(FloatingPointError, match="nonfinite group-product residual"):
        _stage(grad, phi, group)


# -- (e) the batched sync reductions raise the same errors as the separate ones ------------------

@pytest.mark.parametrize("poison", ["grad", "phi"])
def test_nonfinite_inputs_still_raise(poison):
    group = _group()
    grad, phi = _staging_inputs()
    if poison == "grad":
        grad = grad.clone(); grad[0, 0] = float("nan")
    else:
        phi = phi.clone(); phi[0, 0] = float("inf")

    with pytest.raises(FloatingPointError, match="nonfinite grad or chart"):
        _stage(grad, phi, group)


def test_chart_norm_bound_still_raises_with_its_measured_maximum():
    group = _group()
    grad, phi = _staging_inputs()

    with pytest.raises(FloatingPointError, match="exceeds bound"):
        _stage(grad, phi, group, chart_max_norm=1e-6)


# -- (f) the lopsided mixed-order case still falls back, bit-exactly ----------------------------

def test_mixed_order_blocks_still_fall_back_to_the_per_block_route():
    r"""The fallback is what makes the batched result bit-identical to the per-block one.

    Lowering the floor must not have removed it -- only made it rarer at the live distribution.
    """
    basis = generate_glk_multihead(K // H, 1).double()
    prep = pp._build_strict_basis_preparation(basis)
    n_gen = basis.shape[0]
    # One block near zero, one large: they cannot certify at the same order.
    quiet = torch.zeros(1, n_gen, dtype=torch.float64)
    loud = torch.zeros(1, n_gen, dtype=torch.float64)
    loud[0, 1], loud[0, 2] = -4.5, 4.5
    ad = torch.einsum("...a,abc->...cb", torch.stack((quiet, loud), dim=0), prep.structure)

    with pytest.raises(pp._MixedPhiDifferentialOrder):
        pp._adaptive_phi_differentials(ad, require_uniform_leading_batch=True)
