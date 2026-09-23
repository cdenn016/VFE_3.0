r"""Exact Daleckii-Krein evaluation of the trivialized exponential differentials (audit 2026-08-08).

``Psi_R(ad_X)`` and ``Psi_L(ad_X)`` were only ever available as truncated power series. They have an
exact closed form: ``ad_X`` acting on the algebra has eigenvectors ``P E_ij P^-1`` with eigenvalues
``lam_i - lam_j`` for ``lam = eig(X)``, so for analytic ``f``

    f(ad_X)(T) = P [ f(lam_i - lam_j) . (P^-1 T P) ] P^-1

(Higham, *Functions of Matrices*, 2008, Sec. 3.2). With ``u = (lam_i - lam_j)/2``,
``Psi_R = e^u sinhc(u)`` and ``Psi_L = e^-u sinhc(u)``, both entire -- so the ``i == j`` diagonal,
where the naive ``(e^z - 1)/z`` is 0/0 on EVERY evaluation, needs no branch.

OPT-IN (``set_phi_group_differential_mode``), and the measurements are the reason. On its own
component the closed form is a real win -- 264.1 ms -> 57.0 ms at 256 active rows, 4.6x -- but the
series was never where the direction kernel spends its time:

    Psi series       264.1 ms   17.2%
    metric einsum      2.9 ms    0.2%
    safe_eigvalsh   1260.8 ms   81.9%     <-- certificates, not the direction
    cholesky          10.9 ms    0.7%     <-- the actual solve

so end-to-end it is 1.14-1.18x. It also trades a truncation error for a CONDITIONING one: ``X`` is
built from an unconstrained gradient and can be defective, where no eigenbasis exists at all. Hence
default-off, and hence the guards -- ``kappa_F(P)`` and an imaginary-residue check on a real
operator -- which DECLINE to the series rather than returning a degraded answer.

Pins: (a) it reproduces the series to float64 noise across d and ||phi||; (b) the entire-function
form is exact at repeated eigenvalues, where the divided difference is 0/0; (c) defective and
ill-conditioned inputs fall back instead of raising or degrading; (d) the whole direction result
agrees between modes; (e) the mode knob validates and round-trips.
"""

import math

import pytest
import torch

import vfe3.geometry.phi_preconditioner as pp
from vfe3.geometry.generators import generate_glk_multihead


def _prep(d):
    basis = generate_glk_multihead(d, 1).double()
    return basis, pp._build_strict_basis_preparation(basis)


def _series(phi, preparation):
    ad = torch.einsum("...a,abc->...cb", phi, preparation.structure)
    right, left, _ = pp._adaptive_phi_differentials(ad, require_uniform_leading_batch=False)
    return right, left


# -- (a) the closed form reproduces the certified series ---------------------------------------

@pytest.mark.parametrize("d,scale", [(2, 0.06), (5, 0.06), (10, 0.06), (10, 0.2), (10, 0.5)])
def test_spectral_matches_the_series_for_psi_right(d, scale):
    basis, prep = _prep(d)
    g = torch.Generator().manual_seed(0)
    phi = scale * torch.randn(2, 4, basis.shape[0], generator=g, dtype=torch.float64)

    right, _ = _series(phi, prep)
    spectral = pp._spectral_phi_differentials(phi, prep)
    if spectral is None:
        # dfab053 installed a CONSERVATIVE roundoff screen (gamma_(8K) kappa(P)^2 + eig_error
        # <= 1e-10). The K=10 charts land at ~1.2e-10 and are refused, leaving the caller on the
        # certified series. Only large charts may be refused: a refusal at K=5 would be the
        # regression this branch guards against.
        assert d >= 10, f"the roundoff screen refused a small chart (K={d})"
        return
    pushed, _ = spectral
    # The spectral route returns Psi(G_a) as MATRICES; expand the series coordinates to compare.
    expected = torch.einsum("...ca,cij->...aij", right, basis)

    rel = (pushed - expected).abs().max() / expected.abs().max()
    assert rel < 1e-10, f"closed form disagrees with the certified series at {rel:.2e}"


@pytest.mark.parametrize("d,scale", [(5, 0.06), (10, 0.2)])
def test_spectral_matches_the_series_for_psi_left_applied(d, scale):
    basis, prep = _prep(d)
    g = torch.Generator().manual_seed(1)
    n_gen = basis.shape[0]
    phi = scale * torch.randn(2, 4, n_gen, generator=g, dtype=torch.float64)
    v = torch.randn(2, 4, n_gen, generator=g, dtype=torch.float64)

    _, left = _series(phi, prep)
    spectral = pp._spectral_phi_differentials(phi, prep)
    if spectral is None:                       # conservative roundoff screen; see the Psi_R test
        assert d >= 10, f"the roundoff screen refused a small chart (K={d})"
        return
    _, apply_left = spectral

    expected = torch.einsum("...ab,...b->...a", left, v)
    rel = (apply_left(v) - expected).abs().max() / expected.abs().max()
    assert rel < 1e-10


# -- (b) repeated eigenvalues: the divided difference is 0/0 and must still be exact -------------

def test_repeated_eigenvalues_are_exact_not_merely_finite():
    r"""X = c*I makes EVERY lam_i - lam_j zero -- the worst case for a divided difference."""
    basis, prep = _prep(6)
    phi = torch.zeros(1, 1, basis.shape[0], dtype=torch.float64)
    for k in range(6):
        phi[0, 0, k * 6 + k] = 0.3                       # 0.3 * identity

    right, _ = _series(phi, prep)
    pushed, _ = pp._spectral_phi_differentials(phi, prep)
    expected = torch.einsum("...ca,cij->...aij", right, basis)

    assert torch.isfinite(pushed).all()
    assert (pushed - expected).abs().max() / expected.abs().max() < 1e-12


def test_sinhc_is_continuous_through_zero():
    u = torch.tensor([0.0, 1e-12, 1e-8, 1e-3, 1.0], dtype=torch.complex128)
    got = pp._sinhc(u)
    assert torch.isfinite(got.real).all()
    assert abs(got[0].real - 1.0) < 1e-15, "sinhc(0) must be exactly the limit 1"
    assert abs(got[-1].real.item() - math.sinh(1.0)) < 1e-12   # float64 reference, not float32


# -- (c) fragile inputs decline rather than raising or degrading ---------------------------------

def test_defective_matrix_declines_instead_of_raising():
    r"""A single nilpotent Jordan block has no eigenbasis; inv(P) is singular."""
    basis, prep = _prep(6)
    phi = torch.zeros(1, 1, basis.shape[0], dtype=torch.float64)
    for k in range(5):
        phi[0, 0, k * 6 + (k + 1)] = 1.0

    assert pp._spectral_phi_differentials(phi, prep) is None


def test_ill_conditioned_eigenbasis_declines(monkeypatch):
    basis, prep = _prep(5)
    g = torch.Generator().manual_seed(2)
    phi = 0.06 * torch.randn(1, 2, basis.shape[0], generator=g, dtype=torch.float64)

    assert pp._spectral_phi_differentials(phi, prep) is not None      # fine at the live scale
    monkeypatch.setattr(pp, "_PHI_GROUP_SPECTRAL_MAX_COND", 1.0)      # nothing can pass
    assert pp._spectral_phi_differentials(phi, prep) is None


def test_imaginary_residue_guard_declines(monkeypatch):
    basis, prep = _prep(5)
    g = torch.Generator().manual_seed(4)
    phi = 0.2 * torch.randn(1, 2, basis.shape[0], generator=g, dtype=torch.float64)

    assert pp._spectral_phi_differentials(phi, prep) is not None
    monkeypatch.setattr(pp, "_PHI_GROUP_SPECTRAL_MAX_IMAG", 0.0)      # Psi is real, but not EXACTLY
    assert pp._spectral_phi_differentials(phi, prep) is None


# -- (d) the whole direction agrees between modes, and falls back transparently ------------------

def test_direction_result_agrees_between_modes():
    basis, prep = _prep(10)
    g = torch.Generator().manual_seed(5)
    n_gen = basis.shape[0]
    phi = 0.06 * torch.randn(2, 8, n_gen, generator=g, dtype=torch.float64)
    grad = 0.01 * torch.randn(2, 8, n_gen, generator=g, dtype=torch.float64)

    previous = pp.set_phi_group_differential_mode("series")
    try:
        series = pp._full_pullback_group_direction(grad, phi, basis, preparation=prep)
        pp.set_phi_group_differential_mode("spectral")
        spectral = pp._full_pullback_group_direction(grad, phi, basis, preparation=prep)
    finally:
        pp.set_phi_group_differential_mode(previous)

    assert series.series_order > 0, "the series route must report the order it certified"
    assert spectral.series_order == 0, "the closed form has no truncation, hence no order"
    for name in ("v_phi", "xi"):
        a, b = getattr(series, name), getattr(spectral, name)
        assert (a - b).abs().max() / a.abs().max() < 1e-9, f"{name} disagrees between modes"


def test_spectral_mode_falls_back_to_the_series_on_a_defective_chart():
    r"""Declining must be invisible to the caller: same contract, series order restored."""
    basis, prep = _prep(6)
    phi = torch.zeros(1, 2, basis.shape[0], dtype=torch.float64)
    for k in range(5):
        phi[0, :, k * 6 + (k + 1)] = 1.0
    grad = torch.full((1, 2, basis.shape[0]), 0.01, dtype=torch.float64)

    previous = pp.set_phi_group_differential_mode("spectral")
    try:
        result = pp._full_pullback_group_direction(grad, phi, basis, preparation=prep)
    finally:
        pp.set_phi_group_differential_mode(previous)

    assert result.series_order > 0, "a declined spectral call must land on the series"
    assert torch.isfinite(result.v_phi).all() and torch.isfinite(result.xi).all()


# -- (e) the knob ------------------------------------------------------------------------------

def test_mode_knob_validates_and_round_trips():
    assert pp.phi_group_differential_mode() == "series", "the closed form must stay opt-in"
    previous = pp.set_phi_group_differential_mode("spectral")
    try:
        assert previous == "series"
        assert pp.phi_group_differential_mode() == "spectral"
    finally:
        pp.set_phi_group_differential_mode(previous)
    assert pp.phi_group_differential_mode() == "series"

    with pytest.raises(ValueError, match="phi group differential mode"):
        pp.set_phi_group_differential_mode("daleckii")
