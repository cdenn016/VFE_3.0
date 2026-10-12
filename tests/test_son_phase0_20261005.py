r"""Phase 0 of the SO(N) mixed-tower plan (docs/implementation/so-n-mixed-towers-and-frame-tying-plan-2026-10-05.md).

Pins, on CPU at tiny size:

* equal-head goldens captured at 389413b before any Phase 0 edit (rewritten October 9, 2026 for the
  centered pair-KL GEMM, a float32 rounding-level change, under the CUDA interpreter's torch build; the four
  predictive routes rewritten October 12, 2026, same build, for the prior-anchored covariance floor f Sigma_p of
  audit 2026-10-12 T5, which changes only their moment-matched sigma and the full route's loss and gradients) (block_glk K=4 H=2 and so_n
  [("l1", 2)] K=6 H=2 on four projected routes): initial state, forward beliefs, loss and every
  gradient stay bitwise identical, and so_n reproduces them in BOTH irrep bases because l1 is the
  defining representation in each;
* the canonical irrep basis: closed-form SO(3) spins of any order, deterministic, exactly related
  to the eigh basis by an orthogonal intertwiner, and a deterministic tensor basis for N != 3;
* the irrep_basis config field, its construction check and the loader default 'eigh' for so_n
  configs saved before the field existed;
* the generator fingerprint refusing a checkpoint from another basis, and the CG intertwiners
  re-verified after a buffer load;
* the so_n Frobenius Gram certificate and the skew-group exemption from the transport radius.
"""
from pathlib import Path
import math
import warnings

import pytest
import torch

from tests.profiles import test_config, tiny_config
from vfe3.config import config_from_serialized, serialize_config
from vfe3.geometry.generators import generate_son
from vfe3.geometry.groups import get_group
from vfe3.geometry.irreps import irrep_generators
from vfe3.model.model import VFEModel

_GOLDENS = Path(__file__).parent / "data" / "equal_head_goldens_20261005.pt"
_CASES = {
    "block_glk": dict(gauge_group="block_glk", embed_dim=4, n_heads=2),
    "so_n": dict(gauge_group="so_n", group_n=3, irrep_spec=[("l1", 2)], embed_dim=6, n_heads=2),
}
_ROUTES = {
    "predictive_diagonal": dict(),
    "predictive_full": dict(family="gaussian_full", decode_mode="full_chunked"),
    "gradient_diagonal": dict(attention_mode="projected_gradient", predictive_covariance="passthrough",
                              attention_residual=False),
    "gradient_full_closed_form": dict(attention_mode="projected_gradient", predictive_covariance="passthrough",
                                      attention_residual=False, family="gaussian_full",
                                      decode_mode="full_chunked", relational_receiver_update="closed_form"),
}
_TOKENS, _TARGETS = torch.tensor([[1, 4, 2, 5]]), torch.tensor([[4, 2, 5, 3]])


@pytest.fixture(autouse=True)
def _one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _golden_run(case: str, route: str, **extra: object) -> dict:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(tiny_config(**{**_CASES[case], **_ROUTES[route], **extra}))
    with torch.no_grad():
        model.prior_bank.phi_embed.normal_(mean=0.0, std=0.3, generator=torch.Generator().manual_seed(1))
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    belief, _ = model.forward_beliefs(_TOKENS)
    _, loss, _ = model(_TOKENS, _TARGETS)
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return dict(state=state, mu=belief.mu, sigma=belief.sigma, phi=belief.phi, loss=loss, grads=grads)


@pytest.mark.parametrize("route", sorted(_ROUTES))
@pytest.mark.parametrize("case, extra", [
    ("block_glk", {}),
    ("so_n", {"irrep_basis": "eigh"}),
    ("so_n", {"irrep_basis": "canonical"}),
])
def test_equal_head_goldens_are_bitwise_unchanged(case: str, extra: dict, route: str) -> None:
    goldens = torch.load(_GOLDENS)
    golden = goldens[f"{case}/{route}"]
    # The goldens were written by one torch build (``torch_version``). Another build rounds a few float32
    # backward entries differently (about 12 ulp measured for torch 2.11.0+cpu against 2.10.0.dev+cu128),
    # so gradients are bitwise only on the capturing build; everything else stays bitwise on any build.
    same_build = goldens.get("torch_version") == torch.__version__
    out = _golden_run(case, route, **extra)
    extra_keys = set(out["state"]) - set(golden["state"])
    assert extra_keys <= {"gauge_basis_fingerprint"}, extra_keys
    for key, value in golden["state"].items():
        assert torch.equal(out["state"][key], value), key
    for name in ("mu", "sigma", "phi", "loss"):
        assert torch.equal(out[name].detach(), golden[name]), name
    assert out["grads"].keys() == golden["grads"].keys()
    for name, value in golden["grads"].items():
        if same_build:
            assert torch.equal(out["grads"][name], value), name
        else:
            torch.testing.assert_close(out["grads"][name], value, rtol=1e-5, atol=1e-10, msg=name)


# ---------------------------------------------------------------- canonical irrep basis

_G3 = generate_son(3, dtype=torch.float64)


@pytest.mark.parametrize("l", [1, 2, 3, 6, 8, 12])
def test_canonical_so3_spin_blocks(l: int) -> None:
    rho = irrep_generators(_G3, algebra="so", label=f"l{l}", basis="canonical")
    d = 2 * l + 1
    assert rho.shape == (3, d, d)
    assert torch.equal(rho, -rho.transpose(-1, -2))                             # exactly skew
    casimir = -(rho @ rho).sum(0)
    torch.testing.assert_close(casimir, l * (l + 1) * torch.eye(d, dtype=torch.float64), atol=1e-12, rtol=0)
    for a in range(3):                                                         # ||rho_l(L)||_F^2
        assert abs(float((rho[a] ** 2).sum()) - l * (l + 1) * (2 * l + 1) / 3) < 1e-10
    assert torch.equal(rho, irrep_generators(_G3, algebra="so", label=f"l{l}", basis="canonical"))


def test_canonical_l1_is_the_defining_representation() -> None:
    assert torch.equal(irrep_generators(_G3, algebra="so", label="l1", basis="canonical"), _G3)
    assert torch.equal(irrep_generators(_G3, algebra="so", label="l1", basis="eigh"), _G3)


@pytest.mark.parametrize("label", ["l2", "l3", "l5"])
def test_canonical_and_eigh_bases_differ_by_an_orthogonal_intertwiner(label: str) -> None:
    can = irrep_generators(_G3, algebra="so", label=label, basis="canonical")
    old = irrep_generators(_G3, algebra="so", label=label, basis="eigh")
    d = can.shape[-1]
    eye = torch.eye(d, dtype=torch.float64)
    # X with can_a X = X old_a for every generator: null space of sum_a ||(I (x) can_a - old_a^T (x) I) vec X||^2.
    op = torch.cat([torch.kron(eye, can[a]) - torch.kron(old[a].T.contiguous(), eye) for a in range(3)])
    _, sv, Vh = torch.linalg.svd(op)
    assert float(sv[-1]) < 1e-10 and float(sv[-2]) > 1e-3                        # one-dimensional (Schur)
    X = Vh[-1].clone().reshape(d, d).T.contiguous()                             # vec is column-major
    X = X / X.norm() * math.sqrt(d)
    torch.testing.assert_close(X.T @ X, eye, atol=1e-10, rtol=0)                # orthogonal
    for a in range(3):
        torch.testing.assert_close(can[a] @ X, X @ old[a], atol=1e-10, rtol=0)


def test_eigh_basis_keeps_its_rank_cap_and_canonical_lifts_it_for_so3() -> None:
    with pytest.raises(ValueError, match="rank-7"):
        irrep_generators(_G3, algebra="so", label="l7", basis="eigh")
    assert irrep_generators(_G3, algebra="so", label="l7", basis="canonical").shape == (3, 15, 15)


@pytest.mark.parametrize("N, label", [(2, "l3"), (4, "l2"), (4, "l3"), (5, "l2")])
def test_canonical_tensor_basis_for_other_n_is_deterministic_and_a_representation(N: int, label: str) -> None:
    G = generate_son(N, dtype=torch.float64)
    first = irrep_generators(G, algebra="so", label=label, basis="canonical")    # homomorphism asserted inside
    assert torch.equal(first, irrep_generators(G, algebra="so", label=label, basis="canonical"))
    assert float((first + first.transpose(-1, -2)).abs().max()) < 1e-12


# ---------------------------------------------------------------- config field and loader

def test_irrep_basis_config_field_and_construction_check() -> None:
    cfg = test_config(gauge_group="so_n", group_n=3, irrep_spec=[("l8", 2)], embed_dim=34, n_heads=2)
    assert cfg.irrep_basis == "canonical"
    with pytest.raises(ValueError, match="rank-8"):
        test_config(gauge_group="so_n", group_n=3, irrep_spec=[("l8", 2)], embed_dim=34, n_heads=2,
                    irrep_basis="eigh")
    with pytest.raises(ValueError, match="irrep_basis"):
        test_config(gauge_group="so_n", group_n=3, irrep_spec=[("l1", 2)], embed_dim=6, n_heads=2,
                    irrep_basis="spherical")
    assert "irrep_basis" in test_config().inactive_fields()                     # block_glk never reads it
    assert "irrep_basis" not in test_config(gauge_group="so_n", group_n=3, irrep_spec=[("l1", 2)],
                                            embed_dim=6, n_heads=2).inactive_fields()


def test_saved_so_n_config_without_the_field_loads_in_the_eigh_basis() -> None:
    so_n = serialize_config(test_config(gauge_group="so_n", group_n=3, irrep_spec=[("l2", 2)],
                                        embed_dim=10, n_heads=2))
    legacy = {k: v for k, v in so_n.items() if k != "irrep_basis"}
    assert config_from_serialized(legacy, source="test").irrep_basis == "eigh"
    assert config_from_serialized(so_n, source="test").irrep_basis == "canonical"
    glk = {k: v for k, v in serialize_config(test_config()).items() if k != "irrep_basis"}
    assert config_from_serialized(glk, source="test").irrep_basis == "canonical"


# ---------------------------------------------------------------- fingerprint and CG reload

def _so_n_l2_model(basis: str, **extra: object) -> VFEModel:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFEModel(tiny_config(gauge_group="so_n", group_n=3, irrep_spec=[("l2", 2)], embed_dim=10,
                                    n_heads=2, irrep_basis=basis, **extra))


def test_fingerprint_refuses_a_checkpoint_from_another_basis() -> None:
    canonical, eigh = _so_n_l2_model("canonical"), _so_n_l2_model("eigh")
    state = canonical.state_dict()
    assert "gauge_basis_fingerprint" in state
    assert not torch.equal(canonical.group.generators, eigh.group.generators)
    with pytest.raises(ValueError, match="different so_n irrep basis"):
        eigh.load_state_dict(state)
    _so_n_l2_model("canonical").load_state_dict(state)                          # same basis loads
    assert "gauge_basis_fingerprint" not in VFEModel(tiny_config()).state_dict()  # block_glk: deterministic


def test_checkpoint_without_fingerprint_loads_with_a_warning() -> None:
    state = {k: v for k, v in _so_n_l2_model("eigh").state_dict().items() if k != "gauge_basis_fingerprint"}
    target = _so_n_l2_model("eigh")
    with pytest.warns(UserWarning, match="predates the irrep-basis fingerprint"):
        target.load_state_dict(state)


def test_cg_intertwiners_are_reverified_after_load() -> None:
    cg = dict(use_cg_coupling=True, family="gaussian_full", decode_mode="full_chunked")
    saved = _so_n_l2_model("canonical", **cg)
    assert saved.cg_coupling is not None
    state = {k: v for k, v in saved.state_dict().items() if k != "gauge_basis_fingerprint"}
    with pytest.warns(UserWarning, match="predates"):
        _so_n_l2_model("canonical", **cg).load_state_dict(state)               # same basis: equivariant
    with pytest.warns(UserWarning, match="predates"):
        with pytest.raises(RuntimeError, match="not equivariant for this build's irrep basis"):
            _so_n_l2_model("eigh", **cg).load_state_dict(state)


# ---------------------------------------------------------------- Gram certificate and chart radius

@pytest.mark.parametrize("spec, K", [([("l1", 2)], 6), ([("l3", 3), ("l8", 1)], 38), ([("l0", 1), ("l2", 1)], 6)])
def test_so_n_gram_certificate_matches_the_dense_norm(spec: list, K: int) -> None:
    from vfe3.gauge_optim import embedded_phi_frobenius_norm
    group = get_group("so_n")(K, group_n=3, irrep_spec=spec, dtype=torch.float64)
    assert group.frobenius_gram_diagonal and group.phi_norm_route() == "diagonal_gram"
    expected = sum(int(lab[1:]) * (int(lab[1:]) + 1) * (2 * int(lab[1:]) + 1) / 3 * mult for lab, mult in spec)
    assert abs(group.frobenius_gram_uniform - expected) < 1e-9
    phi = torch.randn(5, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    dense = torch.linalg.matrix_norm(torch.einsum("na,aij->nij", phi, group.generators), ord="fro")
    torch.testing.assert_close(embedded_phi_frobenius_norm(phi, group), dense, atol=1e-10, rtol=1e-12)


def test_skew_groups_are_exempt_from_the_transport_radius_and_block_glk_is_not() -> None:
    from vfe3.geometry.transport import build_factored_transport
    tower = get_group("so_n")(38, group_n=3, irrep_spec=[("l3", 3), ("l8", 1)])
    phi = torch.zeros(1, 2, 3)
    phi[0, :, 1] = 3.0                                     # ||M||_F = 3 sqrt 492 ~ 66.5 > 20
    out = build_factored_transport(phi, tower, validity_max_norm=20.0)
    eye = torch.eye(38).expand(1, 2, 38, 38)
    torch.testing.assert_close(out.exp_phi @ out.exp_phi.transpose(-1, -2), eye, atol=1e-5, rtol=0)
    glk = get_group("block_glk")(4, 2)
    big = torch.zeros(1, 2, 8)
    big[0, :, 0] = 30.0
    with pytest.raises(ValueError, match="validity bound exceeded"):
        build_factored_transport(big, glk, validity_max_norm=20.0)
