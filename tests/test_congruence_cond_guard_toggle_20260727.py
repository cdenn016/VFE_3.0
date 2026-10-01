r"""The diagonal-congruence CONDITIONING guard is opt-in (cfg.congruence_cond_escalation, default OFF).

``_escalate_if_negative`` recomputes a diagonal congruence in float64 when the working-dtype result
goes negative. The 2026-07-27 punch list added a second trigger: an accuracy proxy built from
``cond(exp_phi)``, which runs ``torch.linalg.svdvals`` on the vertex blocks and then a ``bool(...)``
on the result. Both run on the HEALTHY path (``amin >= 0``, the common case), so every covariance
transport paid an SVD launch plus a GPU->CPU sync -- measured as a flat step-time regression on the
K=20 wikitext run (85 ms/step -> 126 ms/step), with no change to the returned values in the
well-conditioned case.

The sign-only guard remains the default and is unchanged. The conditioning proxy is now gated, so
the accuracy escalation is available where it matters (large K, ill-conditioned frames) without
charging every step for it.

Everything is tiny and CPU-bound: K = 4, H = 2, d = 2, V = 16, N = 4.
"""

from tests.profiles import test_config
import pytest
import torch

from vfe3.geometry.transport import build_factored_transport, transport_covariance
from vfe3.model.model import VFEModel

BASE = dict(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
            e_step_update="gradient", decode_mode="linear")


def _model(**overrides) -> VFEModel:
    torch.manual_seed(0)
    return VFEModel(test_config(**BASE, **overrides))


def _count_svdvals(monkeypatch) -> list:
    r"""Record every ``torch.linalg.svdvals`` call, still delegating to the real kernel."""
    calls: list = []
    real = torch.linalg.svdvals

    def counting(matrix, *args, **kwargs):
        calls.append(tuple(matrix.shape))
        return real(matrix, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svdvals", counting)
    return calls


def _factored(cond_escalation: bool, *, batch: int = 2, n_tok: int = 4):
    r"""A compact factored transport plus a matching diagonal sigma, both tiny."""
    model = _model()
    torch.manual_seed(0)
    n_gen = model.group.generators.shape[0]
    phi = 0.05 * torch.randn(batch, n_tok, n_gen)
    omega = build_factored_transport(phi, model.group, compact_blocks=True,
                                     cond_escalation=cond_escalation)
    sigma = torch.rand(batch, n_tok, model.group.generators.shape[-1]) + 0.5
    return omega, sigma


# --------------------------------------------------------------------------- the config seam

def test_conditioning_escalation_is_off_by_default():
    r"""The expensive accuracy trigger must be opt-in."""
    assert test_config(**BASE).congruence_cond_escalation is False


# --------------------------------------------------------------------------- the transport seam

def test_transport_covariance_runs_no_svd_when_the_guard_is_off(monkeypatch):
    r"""Default path: sign-only guard, so no svdvals launch and no bool() sync."""
    omega, sigma = _factored(cond_escalation=False)
    calls = _count_svdvals(monkeypatch)
    transport_covariance(omega, sigma)
    assert calls == []


def test_transport_covariance_runs_the_svd_when_the_guard_is_on(monkeypatch):
    r"""Opt-in path: the conditioning proxy runs on the vertex blocks."""
    omega, sigma = _factored(cond_escalation=True)
    calls = _count_svdvals(monkeypatch)
    transport_covariance(omega, sigma)
    assert calls, "conditioning guard was enabled but svdvals never ran"


def test_the_toggle_does_not_change_a_well_conditioned_result():
    r"""Gating is a COST decision, not a numerics change, where the frames are well conditioned."""
    omega_off, sigma = _factored(cond_escalation=False)
    omega_on, _ = _factored(cond_escalation=True)
    torch.testing.assert_close(transport_covariance(omega_off, sigma),
                               transport_covariance(omega_on, sigma))


# --------------------------------------------------------------------------- the model wiring
