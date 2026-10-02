r"""``rope_insertion``: where the positional rotation sits relative to the learned gauge frame.

The shipped composition was ``Omega_ij = R_i U_i U_j^-1 R_j^T`` -- the coboundary of ``W_i = R_i U_i``,
with the rotations OUTSIDE the content operator so they never meet. Writing
``R_i Omega R_j^T = (R_i Omega R_i^T) R(theta_i - theta_j)``, the relative factor is there but the
transport is conjugated by the query's ABSOLUTE angle. Consequences measured on the 2026-07-26 K=20
rope checkpoint: 69% of the pair energy's spread is absolute-position contamination, and gauge
covariance fails with residual 5.6.

``rope_insertion='right'`` folds the rotation into the vertex frame instead, ``V_i = U_i R_i^T``, so
``Omega_ij = U_i R(theta_j - theta_i) U_j^-1`` -- an ordinary flat coboundary with the relative
rotation BETWEEN the query and key content factors (Su et al. 2021 Eq 16; GL(K)_attention.tex).

Every test here is TINY (K < 6, single-digit everything) and CPU-bound per CLAUDE.md. The two
properties are pinned against a NON-IDENTITY gauge frame on purpose: at ``Omega = I`` the two
insertions coincide exactly, which is why `tests/test_rope.py` never caught this.
"""
from tests.profiles import test_config

import pytest
import torch

from vfe3.geometry.groups import get_group
from vfe3.geometry.rope import build_rope_rotation
from vfe3.geometry.transport import (
    ROPE_INSERTIONS,
    FactoredTransport,
    RopeTransport,
)

K, H, N, PERIOD = 4, 2, 12, 4


def _setup(seed: int = 0):
    r"""A periodic-content flat coboundary plus its rope rotation."""
    torch.manual_seed(seed)
    group = get_group("block_glk")(K=K, n_heads=H)
    table = 0.3 * torch.randn(PERIOD, group.generators.shape[0])
    phi = table[torch.arange(N) % PERIOD]                       # period-PERIOD content
    frame = torch.matrix_exp(torch.einsum("na,aij->nij", phi, group.generators))
    base = FactoredTransport(exp_phi=frame, exp_neg_phi=torch.linalg.inv(frame),
                             irrep_dims=list(group.irrep_dims))
    rope = build_rope_rotation(torch.arange(N), group.irrep_dims, base=100.0)
    return group, frame, base, rope


def _dense(omega: RopeTransport) -> torch.Tensor:
    r"""The effective (N, N, K, K) score operator of either insertion."""
    folded = omega.score_operator()                             # RIGHT: the folded vertex frame
    if folded is not None:
        return folded.to_dense_omega()
    base = omega.base.to_dense_omega() if hasattr(omega.base, "to_dense_omega") else omega.base
    return torch.einsum("ikl,ijlm,jnm->ijkn", omega.rope, base, omega.rope)   # R_i Omega_ij R_j^T


def test_left_and_right_agree_at_an_identity_frame() -> None:
    r"""The blind spot itself: at U = I both insertions reduce to R(theta_i - theta_j).

    Pinned so the reason the shipped tests missed this is explicit and stays explicit.
    """
    group, _frame, _base, rope = _setup()
    identity = torch.eye(K).unsqueeze(0).expand(N, K, K).contiguous()
    trivial = FactoredTransport(exp_phi=identity.clone(), exp_neg_phi=identity.clone(),
                                irrep_dims=list(group.irrep_dims))

    left = _dense(RopeTransport(base=trivial, rope=rope, insertion="left"))
    right = _dense(RopeTransport(base=trivial, rope=rope, insertion="right"))

    assert torch.allclose(left, right.transpose(-1, -2), atol=1e-5)   # same rotation, opposite sign


def test_right_insertion_is_exactly_relative_and_left_is_not() -> None:
    r"""Periodic content: a relative scheme satisfies Omega[i+P, j+P] == Omega[i, j] exactly."""
    _group, _frame, base, rope = _setup()

    residual = {}
    for insertion in ROPE_INSERTIONS:
        dense = _dense(RopeTransport(base=base, rope=rope, insertion=insertion))
        shifted = dense[PERIOD:, PERIOD:]
        residual[insertion] = float((dense[:N - PERIOD, :N - PERIOD] - shifted).abs().max())

    assert residual["right"] < 1e-5
    assert residual["left"] > 1e-2                              # the defect, still reproducible


def test_right_insertion_is_gauge_covariant_and_left_is_not() -> None:
    r"""Under a global gauge U_i -> g U_i the operator must transform as Omega -> g Omega g^-1.

    This is the property the config warning claimed gauge-RoPE could not have for ANY group. It can;
    it just needs the rotation inside the frame rather than outside it.
    """
    group, frame, _base, rope = _setup()
    torch.manual_seed(3)
    g = torch.matrix_exp(torch.einsum("a,aij->ij", 0.4 * torch.randn(group.generators.shape[0]),
                                      group.generators))
    g_inv = torch.linalg.inv(g)

    def build(frames: torch.Tensor, insertion: str) -> torch.Tensor:
        base = FactoredTransport(exp_phi=frames, exp_neg_phi=torch.linalg.inv(frames),
                                 irrep_dims=list(group.irrep_dims))
        return _dense(RopeTransport(base=base, rope=rope, insertion=insertion))

    gauged_frame = torch.einsum("ij,njk->nik", g, frame)
    residual = {}
    for insertion in ROPE_INSERTIONS:
        plain = build(frame, insertion)
        gauged = build(gauged_frame, insertion)
        want = torch.einsum("ij,abjk,kl->abil", g, plain, g_inv)
        residual[insertion] = float((gauged - want).abs().max())

    assert residual["right"] < 1e-4
    assert residual["left"] > 1e-1                              # the defect, still reproducible


def test_right_insertion_keeps_the_cocycle_and_the_identity_self_link() -> None:
    r"""Folding must not cost the flat-cocycle structure: Omega_ii = I and Omega_ij Omega_jk = Omega_ik."""
    _group, _frame, base, rope = _setup()

    dense = _dense(RopeTransport(base=base, rope=rope, insertion="right"))

    identity = torch.eye(K).expand(N, K, K)
    assert torch.allclose(dense[range(N), range(N)], identity, atol=1e-5)
    # One intermediate vertex per triple: Omega_ij @ Omega_jk == Omega_ik.
    for i, j, k in ((0, 3, 5), (2, 6, 9), (7, 4, 1)):
        assert torch.allclose(dense[i, j] @ dense[j, k], dense[i, k], atol=1e-4), (i, j, k)


def test_right_insertion_rejects_a_base_it_cannot_fold_into() -> None:
    r"""A dense base exposes only the composed Omega_ij, so the per-vertex frame is unrecoverable.

    Failing closed matters: silently falling back to the left insertion would reintroduce the exact
    defect this toggle exists to remove.
    """
    _group, _frame, _base, rope = _setup()
    dense_base = torch.eye(K).expand(N, N, K, K).contiguous()

    with pytest.raises(ValueError, match="per-vertex frame"):
        RopeTransport(base=dense_base, rope=rope, insertion="right").score_operator()


def test_insertion_is_validated() -> None:
    _group, _frame, base, rope = _setup()

    with pytest.raises(ValueError, match="insertion"):
        RopeTransport(base=base, rope=rope, insertion="middle")


def test_config_rope_insertion_is_fixed_to_right() -> None:

    cfg = test_config(embed_dim=4, n_heads=2, vocab_size=7, max_seq_len=3)
    assert cfg.rope_insertion == "right"
    with pytest.raises(TypeError, match="rope_insertion"):
        test_config(rope_insertion="left")
    with pytest.raises(AttributeError):
        cfg.rope_insertion = "left"



def test_every_rope_insertion_default_agrees(monkeypatch) -> None:
    r"""E-07 was three contradicting defaults for rope_on_value across config/block/e_step.

    rope_insertion threads the same path, so pin every declared default to one value rather than
    discovering the disagreement from a run that silently used the wrong composition.
    """
    import inspect

    from vfe3.geometry import transport as transport_module
    from vfe3.inference import e_step as e_step_module

    declared = [test_config().rope_insertion,
                inspect.signature(transport_module.RopeTransport).parameters["insertion"].default]
    for name, function in inspect.getmembers(e_step_module, inspect.isfunction):
        parameter = inspect.signature(function).parameters.get("rope_insertion")
        if parameter is not None and parameter.default is not inspect.Parameter.empty:
            declared.append(parameter.default)

    assert len(declared) >= 3, "rope_insertion is not threaded through e_step"
    assert set(declared) == {"right"}, declared
