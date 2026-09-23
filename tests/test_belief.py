r"""Tests for the extensible BeliefState (roadmap M3).

The 3-field default (mu, sigma, phi) must stay byte-identical in behavior; the new
optional channel fields (s, r) must round-trip and default to None.
"""

import torch

from vfe3.belief import BeliefState


def _fields():
    mu    = torch.randn(3, 2)
    sigma = torch.rand(3, 2) + 0.5
    phi   = 0.1 * torch.randn(3, 4)
    return mu, sigma, phi


# --- 3-field default unchanged ---------------------------------------------

def test_three_field_construction_and_attribute_access():
    mu, sigma, phi = _fields()
    b = BeliefState(mu=mu, sigma=sigma, phi=phi)
    assert b.mu is mu and b.sigma is sigma and b.phi is phi


def test_positional_construction_still_works():
    mu, sigma, phi = _fields()
    b = BeliefState(mu, sigma, phi)
    assert b.mu is mu and b.sigma is sigma and b.phi is phi


def test_replace_preserves_namedtuple_semantics():
    # _replace is a NamedTuple behavior; confirm it survives the added fields.
    mu, sigma, phi = _fields()
    b = BeliefState(mu=mu, sigma=sigma, phi=phi)
    phi2 = phi + 0.5
    b2 = b._replace(phi=phi2)
    assert b2.mu is mu and b2.sigma is sigma and b2.phi is phi2


# --- optional trailing channels ---------------------------------------------

def test_optional_channels_default_to_none():
    mu, sigma, phi = _fields()
    b = BeliefState(mu=mu, sigma=sigma, phi=phi)
    assert b.right_phi is None
