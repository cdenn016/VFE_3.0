r"""Block covariance remediation, audit 2026-10-09 round 2 (items 2, 10, 11, 12, 13, 17, 20).

* #2: the ``delta_full`` pushforward through the block LayerNorm is formed as the Gram matrix
  :math:`(JL)(JL)^\top` of :math:`\Sigma = LL^\top`, with autocast off and IEEE fp32 GEMMs, so bf16 autocast and a
  large :math:`(\gamma/s)^2\operatorname{cond}\Sigma` no longer return indefinite covariances.
* #13: its floor is :math:`fD\Sigma D`, :math:`D = \operatorname{diag}(\gamma)/s`, in post-norm units, so the
  pushforward is exactly homogeneous of degree :math:`-2` in the pre-norm scale; without a norm it is :math:`f\Sigma`.
* #11: ``gaussian_block`` reads a block-MLP covariance through its head-block M-projection.
* #10: ``gaussian_block`` is declared, and executably verified, invariant for the block groups.
* #12: the gauge gate on ``gaussian_block`` is covariant under per-token block GL.
* #20: ``gaussian_block`` under passthrough is bitwise ``gaussian_diagonal`` and is marked inert.
* #17: the ``block_cholesky`` chart :math:`L = \operatorname{diag}(e^{\ell/2})(I + N)` steps correlations and
  log-variances independently of ``sigma_init``.

Tiny dims, CPU; float64 references.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch

from tests.profiles import tiny_config
from vfe3.belief import BeliefState
from vfe3.contracts import CanonicalFrameContext
from vfe3.families.gaussian import head_block_projection
from vfe3.geometry.groups import _GROUPS, check_admissible, declared_invariant_families, get_group
from vfe3.geometry.norms import AffineLayerNorm
from vfe3.model.block import vfe_block
from vfe3.model.block_mlp import BlockMLP, PrototypeAgentBlockMLP, block_norm_linearization, block_norm_radial_term
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import packed_cholesky_covariance

FLOOR = 1e-4
H, D = 2, 3
K = H * D


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _spd(n: int, k: int, cond: float, g: torch.Generator) -> torch.Tensor:   # (n, k, k) float64, cond(Sigma) = cond
    Q, _ = torch.linalg.qr(torch.randn(n, k, k, generator=g, dtype=torch.float64))
    ev = torch.logspace(0.0, -float(torch.log10(torch.tensor(cond))), k, dtype=torch.float64)
    S = (Q * ev.unsqueeze(-2)) @ Q.transpose(-1, -2)
    return 0.5 * (S + S.transpose(-1, -2))


def _norm_and_mlp(k: int, g: torch.Generator) -> 'tuple[AffineLayerNorm, BlockMLP]':
    torch.manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=g)))    # module init from the global stream
    norm = AffineLayerNorm(k, eps=1e-6)
    mlp = BlockMLP(k, 2, "gelu", 0.0, covariance_contract="delta_full", covariance_floor=FLOOR).eval()
    with torch.no_grad():
        norm.weight.copy_(1.0 + 0.3 * torch.randn(k, generator=g))
        for p in mlp.parameters():
            p.add_(0.2 * torch.randn(p.shape, generator=g))
    return norm, mlp


def _pushforward(norm, mlp, u: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    lin = block_norm_linearization(norm, u)
    return mlp.forward_moments(norm(u, S), S, input_jacobian=lin.jacobian, input_scale=lin.scale).sigma


def _reference(norm, mlp, u: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    r"""The same pushforward evaluated in float64 on float64 copies of the modules."""
    import copy
    norm64, mlp64 = copy.deepcopy(norm).double(), copy.deepcopy(mlp).double()
    return _pushforward(norm64, mlp64, u.double(), S.double())


def _min_eig(P: torch.Tensor) -> torch.Tensor:
    return torch.linalg.eigvalsh(P.double())[..., 0]


def _relative(P: torch.Tensor, ref: torch.Tensor) -> float:
    return float(((P.double() - ref).abs().amax((-1, -2)) / ref.abs().amax((-1, -2))).max())


# ----------------------------------------------------------------------------------------------
# #2 and #13: positive definiteness and the post-norm floor
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("spread,cond", [(1.0, 1e2), (0.01, 1e3)])
def test_bf16_autocast_pushforward_is_positive_definite(spread: float, cond: float) -> None:
    g = torch.Generator().manual_seed(0)
    k, n = 8, 1024
    norm, mlp = _norm_and_mlp(k, g)
    u = spread * torch.randn(n, k, generator=g)
    S = _spd(n, k, cond, g).float()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        P = _pushforward(norm, mlp, u, S)
        # Negative control: the pre-fix composition, every product in bf16 under the ambient autocast.
        lin = block_norm_linearization(norm, u)
        _, J_mlp = mlp._coordinate_update_with_jacobian(norm(u, S))
        J = J_mlp.bfloat16() @ lin.jacobian.bfloat16()
        old = (J @ S.bfloat16()) @ J.transpose(-1, -2) + FLOOR * S.bfloat16()
    ref = _reference(norm, mlp, u, S)
    assert P.dtype == torch.float32
    assert bool((_min_eig(P) > 0).all())
    assert _relative(P, ref) < 1e-2              # the mean path (fc1 pre-activation) stays bf16 under autocast
    print(f"spread {spread}, cond {cond}: bf16 relative error {_relative(P, ref):.2e}, "
          f"pre-fix indefinite fraction {float((_min_eig(old) <= 0).double().mean()):.3f}")
    assert float((_min_eig(old) <= 0).double().mean()) > 0.1


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("cond", [1e2, 1e6])
def test_fp32_pushforward_at_large_gamma_over_s_squared_cond_stays_positive_definite(cond: float, seed: int) -> None:
    r""":math:`(\gamma/s)^2\operatorname{cond}\Sigma` of :math:`10^6` and :math:`10^{10}` in fp32. The post-norm floor
    removes the :math:`(\gamma/s)^2` factor; past :math:`\operatorname{cond}\Sigma \approx f/(Ku)` the Gram's own rounding
    can exceed the floor, and the Cholesky certificate with its float64 recompute keeps the result definite."""
    g = torch.Generator().manual_seed(1 + seed)
    k, n = 8, 1024
    norm, mlp = _norm_and_mlp(k, g)
    u = 0.01 * torch.randn(n, k, generator=g)                               # s ~ 1e-2: (gamma/s)^2 ~ 1e4
    S = _spd(n, k, cond, g).float()
    with torch.no_grad():
        P = _pushforward(norm, mlp, u, S)
        ref = _reference(norm, mlp, u, S)
        # Negative control: the pre-norm floor f Sigma, a share of order f s^2 = 1e-8 of J Sigma J^T.
        lin = block_norm_linearization(norm, u)
        _, J_mlp = mlp._coordinate_update_with_jacobian(norm(u, S))
        J = J_mlp @ lin.jacobian
        old = (J @ S) @ J.transpose(-1, -2) + FLOOR * S
    assert bool((_min_eig(P) > 0).all()) and bool((_min_eig(ref) > 0).all())
    assert _relative(P, ref) < 1e-5
    assert float((_min_eig(old) <= 0).double().mean()) > 0.1


def test_floor_share_is_independent_of_the_pre_norm_scale_and_unchanged_without_a_norm() -> None:
    r"""LN is scale invariant, so :math:`J(cu) = J(u)/c` and :math:`D(cu) = D(u)/c` up to :math:`\epsilon`: the
    whole pushforward, floor included, scales as :math:`c^{-2}`. Without a norm the floor is :math:`f\Sigma`."""
    g = torch.Generator().manual_seed(2)
    norm, mlp = _norm_and_mlp(K, g)
    norm.eps = 0.0
    norm, mlp = norm.double(), mlp.double()
    u = torch.randn(5, K, generator=g, dtype=torch.float64)
    S = _spd(5, K, 10.0, g)
    base = _pushforward(norm, mlp, u, S)
    for c in (1e-2, 30.0):
        assert torch.allclose(_pushforward(norm, mlp, c * u, S) * c * c, base, rtol=1e-10, atol=0)
    # No norm: J_mlp Sigma J_mlp^T + f Sigma, as before the change (float64, to roundoff).
    out, J = mlp._coordinate_update_with_jacobian(u)
    plain = mlp.forward_moments(u, S).sigma
    assert torch.allclose(plain, J @ S @ J.transpose(-1, -2) + FLOOR * S, rtol=1e-13, atol=1e-15)


# ----------------------------------------------------------------------------------------------
# #11: gaussian_block reads the block-MLP covariance through its head-block projection
# ----------------------------------------------------------------------------------------------

def _cfg(**over: Any):
    values = dict(embed_dim=K, n_heads=H, vocab_size=9, max_seq_len=5, n_layers=2, family="gaussian_block",
                  decode_mode="full_chunked", attention_mode="projected_predictive",
                  predictive_covariance="moment_matched", content_link_chart="exponential",
                  norm_type_block="layernorm", layernorm_affine=True, use_block_mlp=True,
                  block_mlp_mode="coordinate", block_mlp_expansion=2)
    values.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tiny_config(**values)


def _model(cfg, seed: int = 0) -> VFEModel:
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg).double().eval()
    g = torch.Generator().manual_seed(seed + 50)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad or name.endswith("phi_embed"):
                p.add_((0.3 if name.endswith("phi_embed") else 0.2) * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return model


def _off_block(S: torch.Tensor) -> torch.Tensor:
    return S - head_block_projection(S, [D] * H)


@pytest.mark.parametrize("over", [
    dict(block_mlp_covariance="delta_full", block_coordinates="local"),
    dict(block_mlp_covariance="delta_full", block_coordinates="local", norm_type_block="none"),
    dict(block_mlp_mode="prototype_agents", block_mlp_covariance="posterior", block_coordinates="local"),
    dict(block_mlp_mode="prototype_agents", block_mlp_covariance="posterior", block_coordinates="reference",
         decode_coordinates="reference"),
], ids=["delta_full_ln", "delta_full", "posterior_local", "posterior_reference"])
def test_block_family_reads_the_mlp_covariance_through_its_head_block_projection(over: dict) -> None:
    model = _model(_cfg(**over))
    tokens = torch.randint(0, 9, (2, 5), generator=torch.Generator().manual_seed(3))
    capture: dict = {}
    with torch.no_grad():
        model.forward_beliefs(tokens, capture=capture)
    pre, post = capture["block_mlp_input"], capture["block_mlp_output"]
    assert float(_off_block(post.sigma).abs().max()) == 0.0                   # stays in the family
    assert bool((_min_eig(post.sigma) > 0).all())
    assert float((post.sigma - pre.sigma).abs().max()) > 1e-6                 # the covariance moved
    if over.get("block_mlp_covariance") == "delta_full":
        # It is the head-block M-projection of the dense pushforward the full family would keep.
        mlp = model.block_mlps[-1]
        converged = capture["converged"]
        if model.block_norm is not None:
            lin = block_norm_linearization(model.block_norm, converged.mu)
            radial = block_norm_radial_term(model.block_norm, converged.mu, converged.sigma)   # round 2 (10-10), #11
            from vfe3.model.block_mlp import block_norm_trace_cap   # the sphere-bound cap (2026-10-12, item 1)
            cap = block_norm_trace_cap(model.block_norm, converged.mu, converged.sigma,
                                       model.cfg.block_mlp_covariance_floor)
            dense = mlp.forward_moments(pre.mu, converged.sigma, input_jacobian=lin.jacobian,
                                        input_scale=lin.scale, input_radial=radial, input_cap=cap).sigma
        else:
            dense = mlp.forward_moments(pre.mu, converged.sigma).sigma
        assert float(_off_block(dense).abs().max()) > 1e-6                     # the projection is not vacuous
        assert torch.allclose(post.sigma, head_block_projection(dense, [D] * H), rtol=1e-12, atol=1e-14)


def test_mean_field_posterior_of_a_block_covariance_is_exactly_block_diagonal() -> None:
    r""":math:`P = (C^{-1} + \operatorname{diag}D)^{-1}` of a block-diagonal :math:`C` has exactly zero off-head blocks."""
    g = torch.Generator().manual_seed(4)
    mlp = PrototypeAgentBlockMLP(K, 2, 0.0, covariance_contract="posterior", inference="mean_field").double()
    blocks = _spd(3 * H, D, 20.0, g).reshape(3, H, D, D)
    C = torch.zeros(3, K, K, dtype=torch.float64)
    for h in range(H):
        C[:, h * D:(h + 1) * D, h * D:(h + 1) * D] = blocks[:, h]
    out = mlp.forward_moments(torch.randn(3, K, generator=g, dtype=torch.float64), C).sigma
    assert float(_off_block(out).abs().max()) == 0.0 and float((out - C).abs().max()) > 1e-3


# ----------------------------------------------------------------------------------------------
# #10: invariance declaration and verifier
# ----------------------------------------------------------------------------------------------

def _group(name: str, tying: str):
    builder = get_group(name)
    if name == "so_n":
        return builder(4, group_n=3, irrep_spec=[("l1", 1), ("l0", 1)], tying=tying, dtype=torch.float64)
    return builder(4, 2, **({"tying": tying} if name == "block_glk" else {}), dtype=torch.float64)


@pytest.mark.parametrize("name,tying", [("block_glk", "per_head"), ("block_glk", "shared"), ("block_so", "per_head"),
                                        ("block_co", "per_head"), ("block_glmc", "per_head"), ("so_n", "shared"),
                                        ("so_n", "per_head")])
def test_block_family_declaration_matches_the_verifier(name: str, tying: str) -> None:
    group = _group(name, tying)
    assert group.invariant_for("gaussian_block") and "gaussian_block" in declared_invariant_families(name)
    assert check_admissible(group, "gaussian_block", n_samples=4)
    assert not check_admissible(group, "gaussian_diagonal", n_samples=4) or name == "block_co"


def test_every_registered_builtin_group_declares_the_block_family() -> None:
    for name in ("block_glk", "so_n", "block_so", "block_co", "block_glmc"):
        assert name in _GROUPS and "gaussian_block" in declared_invariant_families(name)


def test_dense_gl_breaks_the_block_family() -> None:
    r"""Negative control: a dense :math:`g` couples the heads, so the head-block readout is not invariant."""
    from vfe3.families.base import get_family, get_functional
    fam, kl = get_family("gaussian_block"), get_functional("renyi")
    g = torch.Generator().manual_seed(6)
    A = torch.linalg.matrix_exp(0.3 * torch.randn(4, 4, generator=g, dtype=torch.float64))
    mu_q, mu_p = torch.randn(2, 4, generator=g, dtype=torch.float64), torch.randn(2, 4, generator=g, dtype=torch.float64)
    S_q, S_p = _spd(2, 4, 5.0, g), _spd(2, 4, 5.0, g)
    read = lambda m, S: fam.from_covariance(m, S, head_dims=[2, 2])                         # noqa: E731
    base = kl(read(mu_q, S_q), read(mu_p, S_p), alpha=1.0, kl_max=1e12)
    moved = kl(read(mu_q @ A.T, A @ S_q @ A.T), read(mu_p @ A.T, A @ S_p @ A.T), alpha=1.0, kl_max=1e12)
    assert float((base - moved).abs().max()) > 1e-3


def test_run_artifacts_label_the_block_family_and_book_its_dense_layout() -> None:
    from vfe3.run_artifacts import _pure_path_report, _relational_artifact_fields

    def find(tree: Any, key: str) -> Any:
        if isinstance(tree, dict):
            if key in tree:
                return tree[key]
            for value in tree.values():
                found = find(value, key)
                if found is not None:
                    return found
        return None

    cfg = _cfg(use_block_mlp=True, block_mlp_covariance="delta_full", block_coordinates="local")
    model = _model(cfg)
    assert model.executable_build.block_mlp.flops_per_token == pytest.approx(
        cfg.n_layers * (4.0 * cfg.block_mlp_expansion * K * K + (4.0 + 2.0 / 3.0) * K ** 3))
    report = _pure_path_report(cfg, [], executable_build=model.executable_build)
    contract = find(report, "block_mlp_covariance_contract")
    assert contract == "head_block_projection_of_delta_full_through_layernorm_plus_post_norm_scaled_floor"
    assert find(report, "regime_ii_covariant_exactness") != "diagonal_projection_approximation"
    record = _relational_artifact_fields(cfg)["relational_operators"]
    assert record["content_covariance_contract"] == \
        "head_block_moment_projection_of_predictive_kernel_plus_covariant_floor"


# ----------------------------------------------------------------------------------------------
# #12: gauge gate on gaussian_block, covariant under per-token block GL
# ----------------------------------------------------------------------------------------------

def _gl_blocks(B: int, N: int, scale: float, seed: int) -> torch.Tensor:          # (B, N, H, d, d)
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.matrix_exp(scale * torch.randn(B, N, H, D, D, generator=g, dtype=torch.float64))


def _block_diag(blocks: torch.Tensor) -> torch.Tensor:                             # (..., H, d, d) -> (..., K, K)
    out = torch.zeros(*blocks.shape[:-3], K, K, dtype=blocks.dtype)
    for h in range(H):
        out[..., h * D:(h + 1) * D, h * D:(h + 1) * D] = blocks[..., h, :, :]
    return out


def _one_layer(model: VFEModel, mu, sigma, frame: CanonicalFrameContext):
    B, N = mu.shape[:2]
    n_gen = model.prior_bank.phi_embed.shape[-1]
    belief = BeliefState(mu=mu, sigma=sigma, phi=torch.zeros(B, N, n_gen, dtype=torch.float64))
    with torch.no_grad():
        out = vfe_block(belief, mu, sigma, model.group, model.cfg,
                        log_prior=model._attention_log_prior(N, mu.device),
                        block_norm=model.block_norm, block_mlp=model.block_mlps[0],
                        relational_operators=model.relational_operators_by_layer[0],
                        relational_frame=frame, relational_value_frame=frame)
        logits = model.prior_bank.decode(out.mu, out.sigma, canonical_frame=frame)
    return out, logits


@pytest.mark.parametrize("covariance", ["passthrough", "delta_full"])
def test_gauge_gate_on_the_block_family_is_covariant_under_per_token_block_gl(covariance: str) -> None:
    cfg = _cfg(n_layers=1, block_mlp_mode="gauge_gate", block_mlp_covariance=covariance,
               norm_type_block="mahalanobis" if covariance == "passthrough" else "none", layernorm_affine=False,
               decode_coordinates="reference", block_coordinates="local", pos_phi="none")
    model = _model(cfg)
    gen = torch.Generator().manual_seed(4)
    with torch.no_grad():
        for p in model.block_mlps.parameters():                            # the gate is the identity at init
            p.add_(0.5 * torch.randn(p.shape, generator=gen, dtype=p.dtype))
    B, N = 2, 5
    mu = torch.randn(B, N, K, generator=gen, dtype=torch.float64)
    sigma = _block_diag(_spd(B * N * H, D, 10.0, gen).reshape(B, N, H, D, D))
    U_blocks, g_blocks = _gl_blocks(B, N, 0.4, 5), _gl_blocks(B, N, 0.5, 6)
    U = CanonicalFrameContext(U_blocks, torch.linalg.inv(U_blocks), block_diagonal=True)
    gU = CanonicalFrameContext(g_blocks @ U_blocks, torch.linalg.inv(g_blocks @ U_blocks), block_diagonal=True)
    g = _block_diag(g_blocks)
    gmv = lambda x: (g @ x.unsqueeze(-1)).squeeze(-1)                        # noqa: E731
    out, logits = _one_layer(model, mu, sigma, U)
    out_g, logits_g = _one_layer(model, gmv(mu), g @ sigma @ g.transpose(-1, -2), gU)
    assert float((out_g.mu - gmv(out.mu)).abs().max()) < 1e-10
    assert float((out_g.sigma - g @ out.sigma @ g.transpose(-1, -2)).abs().max()) < 1e-9
    assert float((torch.log_softmax(logits_g, -1) - torch.log_softmax(logits, -1)).abs().max()) < 1e-10
    assert float((out.mu - mu).abs().max()) > 1e-2
    # Negative control: moving the beliefs without the frames changes the logits.
    _, unmoved = _one_layer(model, gmv(mu), g @ sigma @ g.transpose(-1, -2), U)
    assert float((torch.log_softmax(unmoved, -1) - torch.log_softmax(logits, -1)).abs().max()) > 1e-3


# ----------------------------------------------------------------------------------------------
# #20: gaussian_block under passthrough is gaussian_diagonal
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("use_block_mlp", [False, True])
def test_block_family_under_passthrough_matches_the_diagonal_family_and_marked_inert(use_block_mlp: bool) -> None:
    over = dict(predictive_covariance="passthrough", decode_mode="diagonal_chunked", use_block_mlp=use_block_mlp)
    block_cfg, diag_cfg = _cfg(**over), _cfg(family="gaussian_diagonal", **over)
    assert "family" in block_cfg.inactive_fields() and "family" not in diag_cfg.inactive_fields()
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        block, diag = VFEModel(block_cfg).eval(), VFEModel(diag_cfg).eval()
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for p in block.parameters():
            p.add_(0.2 * torch.randn(p.shape, generator=g))
    diag.load_state_dict(block.state_dict(), strict=True)
    tokens, targets = torch.randint(0, 9, (2, 5), generator=g), torch.randint(0, 9, (2, 5), generator=g)
    # The head-block projection reads the diagonal blocks of a dense Sigma (projected_moments head_dims), a
    # different contraction order from the diagonal family's, so the two agree to float32 rounding.
    with torch.no_grad():
        torch.testing.assert_close(block(tokens), diag(tokens), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(block(tokens, targets)[1], diag(tokens, targets)[1], rtol=1e-5, atol=1e-5)
    # Moment matching or a covariance-producing MLP makes the families differ, and the marker drops.
    assert "family" not in _cfg(**dict(over, predictive_covariance="moment_matched")).inactive_fields()
    assert "family" not in _cfg(**dict(over, block_mlp_covariance="delta_full", use_block_mlp=True)).inactive_fields()


# ----------------------------------------------------------------------------------------------
# #17: block_cholesky chart
# ----------------------------------------------------------------------------------------------

def _correlation(C: torch.Tensor) -> torch.Tensor:
    s = torch.diagonal(C, dim1=-2, dim2=-1).rsqrt()
    return s.unsqueeze(-1) * C * s.unsqueeze(-2)


def test_block_cholesky_chart_steps_do_not_depend_on_sigma_init() -> None:
    r"""One AdamW step on a scale-free objective (correlations and log-variances) moves the correlations and the
    log-variances by the same amounts at every ``sigma_init``; the log-variance moves at the optimizer's rate."""
    g = torch.Generator().manual_seed(7)
    rows, cols = torch.tril_indices(D, D)
    N = 0.3 * torch.randn(4, H, rows.numel(), generator=g, dtype=torch.float64)
    weights = torch.randn(4, H, D, D, generator=g, dtype=torch.float64)
    lr, eps = 1e-2, 1e-12
    moves = []
    for sigma_init in (1.0, 5.0, 15.0):
        packed = torch.where(rows == cols, torch.tensor(float(torch.log(torch.tensor(sigma_init)))), N)
        packed = torch.nn.Parameter(packed.clone())
        optimizer = torch.optim.AdamW([packed], lr=lr, weight_decay=0.0)
        C0 = packed_cholesky_covariance(packed, D, eps=eps).detach()
        C = packed_cholesky_covariance(packed, D, eps=eps)
        loss = (weights * _correlation(C)).sum() + torch.diagonal(C, dim1=-2, dim2=-1).log().sum()
        loss.backward()
        optimizer.step()
        C1 = packed_cholesky_covariance(packed, D, eps=eps).detach()
        moves.append(((_correlation(C1) - _correlation(C0)).abs().max(),
                      (torch.diagonal(C1, dim1=-2, dim2=-1).log() - torch.diagonal(C0, dim1=-2, dim2=-1).log())))
    for corr, logvar in moves[1:]:
        assert torch.allclose(corr, moves[0][0], rtol=1e-6, atol=0)
        assert torch.allclose(logvar, moves[0][1], rtol=1e-6, atol=1e-12)
    assert float(moves[0][0]) > 1e-3
    # The first diagonal slot has no N entries: its log-variance moves by exactly one optimizer step.
    assert torch.allclose(moves[0][1][..., 0].abs(), torch.full((4, H), lr, dtype=torch.float64), rtol=1e-6)
