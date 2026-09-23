from collections.abc import MutableMapping

import pytest
import torch

from vfe3.alpha_i import alpha_regularizer, register_alpha, self_coupling_alpha


_MISSING = object()


def _restore_registry_entry(
    registry: MutableMapping[str, object],
    name:     str,
    previous: object,
) -> None:
    if previous is _MISSING:
        registry.pop(name, None)
    else:
        registry[name] = previous


def test_constant_alpha_is_value_zero_reg():
    kl = torch.rand(3, 5)
    a, r = self_coupling_alpha(kl, mode="constant", value=1.0)
    assert torch.allclose(a, torch.ones(3, 5))
    assert torch.allclose(r, torch.zeros(3, 5))


def test_state_dependent_alpha_formula_and_minimizes_objective():
    # alpha* = c0/(b0 + KL) is the stationary point of  alpha*KL + b0*alpha - c0*log(alpha).
    kl = torch.tensor([0.0, 1.0, 4.0])
    b0, c0 = 0.5, 2.0
    a, r = self_coupling_alpha(kl, mode="state_dependent", b0=b0, c0=c0)
    assert torch.allclose(a, c0 / (b0 + kl), atol=1e-6)
    # d/d alpha [alpha*KL + b0*alpha - c0*log alpha] = KL + b0 - c0/alpha == 0 at alpha*
    grad = kl + b0 - c0 / a
    assert torch.allclose(grad, torch.zeros_like(grad), atol=1e-5)


def test_per_coord_alpha_uses_per_dimension_kl():
    kl = torch.rand(2, 4, 3) + 0.1                       # (..., N, K) per-coordinate KL
    b0 = torch.full((3,), 0.5)
    c0 = torch.full((3,), 2.0)
    a, r = self_coupling_alpha(kl, mode="state_dependent_per_coord", b0=b0, c0=c0)
    assert torch.allclose(a, c0 / (b0 + kl), atol=1e-6)


@pytest.mark.registry_mutation
def test_new_form_with_novel_kwarg_reachable_without_editing_dispatcher():
    # Modularity: a registered form's OWN param must flow through the dispatcher's
    # **kwargs (not a hard-coded value/b0/c0 union), so a new form selects-with-config
    # without editing the call site.
    from vfe3.alpha_i import _ALPHAS, _ALPHA_PER_COORD
    name = "_test_scaled"
    previous_alpha = _ALPHAS.get(name, _MISSING)
    previous_flag = _ALPHA_PER_COORD.get(name, _MISSING)
    try:
        @register_alpha(name, override=previous_alpha is not _MISSING)
        def _scaled(kl, *, scale=2.0, **kwargs):
            return scale * torch.ones_like(kl), torch.zeros_like(kl)

        kl = torch.zeros(3)
        a, r = self_coupling_alpha(kl, mode=name, scale=5.0)
        assert torch.allclose(a, torch.full((3,), 5.0))
    finally:
        _restore_registry_entry(_ALPHAS, name, previous_alpha)
        _restore_registry_entry(_ALPHA_PER_COORD, name, previous_flag)


def test_alpha_is_per_coord_declares_reduction_need():
    # Modularity: each alpha form DECLARES whether it consumes a per-coordinate (unsummed)
    # self-divergence, so the routing seam reads that flag rather than hard-coding a mode
    # name at the call sites. A future per-coordinate form slots in by registering with
    # per_coord=True -- no consumer is edited.
    from vfe3.alpha_i import alpha_is_per_coord
    assert alpha_is_per_coord("state_dependent_per_coord") is True
    assert alpha_is_per_coord("state_dependent") is False
    assert alpha_is_per_coord("constant") is False


@pytest.mark.registry_mutation
def test_register_alpha_per_coord_flag_is_modular():
    from vfe3.alpha_i import _ALPHAS, _ALPHA_PER_COORD, alpha_is_per_coord, register_alpha
    name = "_test_pc"
    previous_alpha = _ALPHAS.get(name, _MISSING)
    previous_flag = _ALPHA_PER_COORD.get(name, _MISSING)
    try:
        @register_alpha(name, per_coord=True, override=previous_alpha is not _MISSING)
        def _pc(kl, **kwargs):
            return kl, torch.zeros_like(kl)

        assert alpha_is_per_coord(name) is True
    finally:
        _restore_registry_entry(_ALPHAS, name, previous_alpha)
        _restore_registry_entry(_ALPHA_PER_COORD, name, previous_flag)


from vfe3.alpha_i import alpha_gradient_coefficient


def test_alpha_grad_coefficient_constant_is_value():
    kl = torch.rand(3, 5)
    assert torch.allclose(alpha_gradient_coefficient(kl, mode="constant", value=2.0),
                          torch.full((3, 5), 2.0))


def test_alpha_grad_coefficient_state_dependent_is_alpha_star():
    # By the alpha-envelope, the effective coefficient is alpha* itself (the
    # alpha'*D and R' paths cancel at the stationary alpha* = c0/(b0+KL)).
    kl = torch.tensor([0.0, 1.0, 4.0])
    b0, c0 = 0.5, 2.0
    coef = alpha_gradient_coefficient(kl, mode="state_dependent", b0=b0, c0=c0)
    assert torch.allclose(coef, c0 / (b0 + kl), atol=1e-6)


# --- audit 2026-07-01 F12-registry: duplicate keys fail closed across the registry decorators ---
# A second @register under an existing name used to silently shadow the first, so a
# config-selected seam could dispatch to an unintended implementation. Each decorator now
# raises KeyError on a duplicate key unless override=True is passed (the deliberate
# replacement escape hatch). register_alpha lives here; register_prior / register_compose
# share the guard and are mirrored below.


@pytest.mark.registry_mutation
def test_register_alpha_duplicate_raises_and_override_replaces():
    from vfe3.alpha_i import _ALPHAS, _ALPHA_PER_COORD
    name = "_test_dup_alpha"
    previous_alpha = _ALPHAS.get(name, _MISSING)
    previous_flag = _ALPHA_PER_COORD.get(name, _MISSING)
    try:
        @register_alpha(name, override=previous_alpha is not _MISSING)
        def _first(kl, **kwargs):
            return torch.ones_like(kl), torch.zeros_like(kl)

        with pytest.raises(KeyError, match="already registered"):
            @register_alpha(name)
            def _second(kl, **kwargs):
                return kl, torch.zeros_like(kl)

        assert _ALPHAS[name] is _first                    # the first registration survives

        @register_alpha(name, override=True)
        def _third(kl, **kwargs):
            return 2.0 * torch.ones_like(kl), torch.zeros_like(kl)

        assert _ALPHAS[name] is _third                    # explicit override replaces
    finally:
        _restore_registry_entry(_ALPHAS, name, previous_alpha)
        _restore_registry_entry(_ALPHA_PER_COORD, name, previous_flag)


@pytest.mark.registry_mutation
def test_register_prior_duplicate_raises_and_override_replaces():
    from vfe3.attention_prior import _PRIORS, register_prior
    name = "_test_dup_prior"
    previous = _PRIORS.get(name, _MISSING)
    try:
        @register_prior(name, override=previous is not _MISSING)
        def _first(n_query, n_key, **kwargs):
            return torch.zeros(n_query, n_key)

        with pytest.raises(KeyError, match="already registered"):
            register_prior(name)(_first)

        @register_prior(name, override=True)
        def _second(n_query, n_key, **kwargs):
            return torch.zeros(n_query, n_key)

        assert _PRIORS[name] is _second
    finally:
        _restore_registry_entry(_PRIORS, name, previous)


@pytest.mark.registry_mutation
def test_register_compose_duplicate_raises_and_override_replaces():
    from vfe3.geometry.lie_ops import _COMPOSE, register_compose
    name = "_test_dup_compose"
    previous = _COMPOSE.get(name, _MISSING)
    try:
        @register_compose(name, override=previous is not _MISSING)
        def _first(phi1, phi2, generators, **kwargs):
            return phi1 + phi2

        with pytest.raises(KeyError, match="already registered"):
            register_compose(name)(_first)

        @register_compose(name, override=True)
        def _second(phi1, phi2, generators, **kwargs):
            return phi1 + phi2

        assert _COMPOSE[name] is _second
    finally:
        _restore_registry_entry(_COMPOSE, name, previous)
