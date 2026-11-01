r"""Audit 2026-10-31 remediation, package ``geometry-numerics``: regression tests.

One section per finding, keyed by the lane ID of ``docs/audits/ultradeep-2026-10-31``. Configurations come from
``tests.profiles``; every model is tiny (single-digit dimensions, two heads, one or two layers).
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import pytest
import torch

import vfe3.train as T
from tests.profiles import test_config, tiny_config
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel

U32 = 2.0 ** -24                                   # float32 unit roundoff


# ------------------------------------------------------------------------------------------------
# tables-families-3 (A097): the far branch of ratio_minus_one_minus_log
# ------------------------------------------------------------------------------------------------

def _ratio_reference(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    r"""float64 :math:`r - 1 - \log r` of the float32 inputs, :math:`r = \text{num}/\text{den}`."""
    ratio = num.double() / den.double()
    return ratio - 1.0 - torch.log(ratio)


@pytest.mark.parametrize("den", [15.0, 1e3, 1e6])
@pytest.mark.parametrize("ratio", [0.3, 1.6, 4.0])
def test_far_branch_error_does_not_grow_with_the_logarithm_of_the_operands(den: float, ratio: float) -> None:
    from vfe3.families.base import ratio_minus_one_minus_log
    steps = torch.arange(-20, 21, dtype=torch.float64) * 1e-3
    num = (ratio * den * (1.0 + steps)).float()                       # (41,) all in the far branch
    denominator = torch.full_like(num, den)
    value = ratio_minus_one_minus_log(num, denominator)
    reference = _ratio_reference(num, denominator)
    relative = float(((value.double() - reference).abs() / reference).max())
    assert relative <= 16 * U32, relative


def test_far_branch_keeps_the_two_log_form_where_the_ratio_is_not_a_normal_float() -> None:
    from vfe3.families.base import ratio_minus_one_minus_log
    num = torch.tensor([1e-30], dtype=torch.float32, requires_grad=True)
    den = torch.tensor([1e10], dtype=torch.float32, requires_grad=True)   # r = 1e-40 is subnormal in float32
    value = ratio_minus_one_minus_log(num, den)
    reference = _ratio_reference(num.detach(), den.detach())
    assert torch.isfinite(value).all()
    assert abs(float(value.detach()) - float(reference)) <= 64 * U32 * float(reference)
    value.sum().backward()
    assert torch.isfinite(num.grad).all() and torch.isfinite(den.grad).all()


def test_far_branch_gradient_matches_finite_differences() -> None:
    from vfe3.families.base import ratio_minus_one_minus_log
    num = torch.tensor([0.3, 4.0, 1.6e6, 2.0e-3, 0.9], dtype=torch.float64, requires_grad=True)
    den = torch.tensor([1.0, 1.0, 1.0e6, 1.0e-2, 1.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(ratio_minus_one_minus_log, (num, den))


# ------------------------------------------------------------------------------------------------
# transport-geometry-3 (A102): frames_identity_reason and the additive encoder
# ------------------------------------------------------------------------------------------------

def test_additive_encoder_frames_are_reported_as_the_identity() -> None:
    from vfe3.model.frame_actions import frames_identity_reason
    from vfe3.posthoc.interventions import frames_inactive_reason
    cfg = tiny_config(encode_mode="per_token_additive", pos_phi="none")
    n_gen = 8
    reason = frames_identity_reason(cfg, n_gen)
    assert reason is not None and "per_token_additive" in reason
    assert frames_identity_reason(cfg, n_gen, positional=False) is not None
    torch.manual_seed(0)
    model = VFEModel(cfg)
    beliefs = model.prior_bank.encode(torch.tensor([[0, 1, 2]]))
    assert torch.equal(beliefs.phi, torch.zeros_like(beliefs.phi))         # the realized token frames are exp(0) = I
    assert bool((model.prior_bank.phi_embed != 0).any())                    # while the table itself is not zero
    assert frames_inactive_reason(model) is not None
    # A positional factor keeps the composed frame active (the config refuses that pairing, so a namespace stands in).
    positional = SimpleNamespace(gauge_transport="on", phi_scale=0.01, m_phi_lr=0.001,
                                 encode_mode="per_token_additive", pos_phi="learned", rope_mode="none")
    assert frames_identity_reason(positional, n_gen) is None
    assert frames_identity_reason(positional, n_gen, positional=False) is not None
    assert frames_identity_reason(tiny_config(), n_gen) is None             # per_token frames stay active


# ------------------------------------------------------------------------------------------------
# type-contracts-2 (A104): rope_rotated_pairs floors the configured decimal
# ------------------------------------------------------------------------------------------------

def test_rope_rotated_pairs_at_the_three_widths_the_float_floor_missed() -> None:
    from vfe3.geometry.rope import rope_rotated_pairs
    assert rope_rotated_pairs(100, 0.58) == 29
    assert rope_rotated_pairs(180, 0.7) == 63
    assert rope_rotated_pairs(200, 0.29) == 29
    assert rope_rotated_pairs(200, 0.57) == 57
    assert rope_rotated_pairs(200, 0.58) == 58
    assert rope_rotated_pairs(32, 1) == 16 and rope_rotated_pairs(7, 1.0) == 3 and rope_rotated_pairs(2, 0.5) == 0


def test_rope_rotated_pairs_equals_the_exact_floor_for_every_width_and_thousandth() -> None:
    from vfe3.geometry.rope import rope_rotated_pairs
    mismatches = [(d, thousandths) for thousandths in range(1, 1001) for d in range(2, 257)
                  if rope_rotated_pairs(d, thousandths / 1000) != (thousandths * d) // 2000]
    assert mismatches == []


def test_config_counts_rotated_planes_through_the_shared_function(monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.config as config_module
    monkeypatch.setattr(config_module, "_DIAGONAL_COMPARISON_WARNED", set())
    with pytest.warns(UserWarning, match="of 29 planes for heads of size 100"):
        config_module._warn_diagonal_comparison_covariance([100], 0.58, 100.0, 8)


# ------------------------------------------------------------------------------------------------
# attention-scoring-7 (A111): reduced_free_energy and a length-one tau
# ------------------------------------------------------------------------------------------------

def test_reduced_free_energy_collapses_a_length_one_tau_on_a_headless_energy() -> None:
    from vfe3.free_energy import log_partition, reduced_free_energy
    energy = torch.rand(6, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    tau = torch.tensor([0.7], dtype=torch.float64)
    scalar = reduced_free_energy(energy, tau=0.7)
    vector = reduced_free_energy(energy, tau=tau)
    assert scalar.shape == vector.shape == log_partition(energy, tau=tau).shape == (6,)
    assert torch.equal(vector, scalar)
    batched = torch.rand(2, 6, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    assert reduced_free_energy(batched, tau=tau).shape == (2, 6)
    heads = torch.rand(3, 6, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    per_head = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    assert reduced_free_energy(heads, tau=per_head).shape == (3, 6)


# ------------------------------------------------------------------------------------------------
# gauge-structure-2 (A116): declared Frobenius-Gram weights and a basis changed before the first read
# ------------------------------------------------------------------------------------------------

_GRAM_GROUPS = [
    ("block_glk", dict(K=4, n_heads=2)),
    ("block_glk", dict(K=4, n_heads=2, tying="shared")),
    ("block_so", dict(K=8, n_heads=2)),
    ("block_co", dict(K=8, n_heads=2)),
    ("block_sl", dict(K=8, n_heads=2)),
    ("block_glmc", dict(K=8, n_heads=2)),
    ("block_borel", dict(K=8, n_heads=2)),
    ("block_unipotent", dict(K=8, n_heads=2)),
    ("so_n", dict(K=8, group_n=3, irrep_spec=[("l1", 1), ("l2", 1)])),
    ("so_n", dict(K=8, group_n=3, irrep_spec=[("l1", 1), ("l2", 1)], tying="per_head")),
    ("su_n", dict(K=8, group_n=3, irrep_spec=[("pq1_1", 1)])),
]


@pytest.mark.parametrize("name,kwargs", _GRAM_GROUPS, ids=lambda value: str(value))
@pytest.mark.parametrize("change", ["in_place", "replacement"])
def test_gram_weights_follow_a_basis_changed_before_the_first_read(name: str, kwargs: Dict[str, Any],
                                                                   change: str) -> None:
    from vfe3.gauge_optim import embedded_phi_frobenius_norm
    from vfe3.geometry.groups import get_group
    group = get_group(name)(dtype=torch.float64, **kwargs)
    if change == "in_place":
        group.generators.mul_(2.0)
    else:
        group.generators = group.generators * 2.0
    G = group.generators                                                    # (n_gen, K, K)
    diagonal = group.gram_diagonal()
    if diagonal is not None:
        torch.testing.assert_close(diagonal, torch.einsum("aij,aij->a", G, G), rtol=1e-12, atol=0.0)
    phi = torch.randn(5, G.shape[0], dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    dense = torch.linalg.matrix_norm(torch.einsum("...a,aij->...ij", phi, G))
    torch.testing.assert_close(embedded_phi_frobenius_norm(phi, group, warn_fallback=False), dense,
                               rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("name,kwargs", _GRAM_GROUPS, ids=lambda value: str(value))
def test_unchanged_bases_keep_their_declared_weights(name: str, kwargs: Dict[str, Any]) -> None:
    from vfe3.geometry.groups import get_group
    group = get_group(name)(dtype=torch.float64, **kwargs)
    route = group.phi_norm_route()
    uniform = group.frobenius_gram_uniform
    diagonal = group.gram_diagonal()
    assert group.phi_norm_route() == route
    if uniform is not None:
        assert diagonal is not None and torch.equal(diagonal, torch.full_like(diagonal, uniform))
    G = group.generators
    if diagonal is not None:
        torch.testing.assert_close(diagonal, torch.einsum("aij,aij->a", G, G), rtol=1e-12, atol=1e-12)
    group.apply_generators(lambda tensor: tensor.to(torch.float64))       # a move keeps the certificate
    if uniform is not None:
        assert group.gram_diagonal_uniform() == uniform


# ------------------------------------------------------------------------------------------------
# numerics-2 (A066): the eps floor of the log-variance tables
# ------------------------------------------------------------------------------------------------

def _floor_cfg(**overrides: Any) -> VFE3Config:
    values: Dict[str, Any] = dict(
        vocab_size=9, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=1, batch_size=2,
        decode_mode="diagonal_chunked", untie_decode_bank=True, m_table_update_mode="fisher",
        m_decode_update_mode="adamw", sigma_weight_decay=0.0, warmup_steps=0)
    values.update(overrides)
    return test_config(**values)


def _floor_batch(cfg: VFE3Config) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(4)
    shape = (cfg.batch_size, cfg.max_seq_len)
    return (torch.randint(0, cfg.vocab_size, shape, generator=g),
            torch.randint(0, cfg.vocab_size, shape, generator=g))


def test_the_projected_floor_value_keeps_a_nonzero_derivative() -> None:
    from vfe3.numerics import bounded_variance_from_log, log_variance_floor
    eps = 1e-6
    for dtype in (torch.float32, torch.float64):
        floor = torch.tensor([log_variance_floor(eps)], dtype=dtype, requires_grad=True)
        variance = bounded_variance_from_log(floor, eps=eps)
        variance.sum().backward()
        assert float(floor.grad) > 0.0 and float(variance) >= eps
        below = torch.tensor([math.log(eps) - 0.3], dtype=dtype, requires_grad=True)
        bounded_variance_from_log(below, eps=eps).sum().backward()
        assert float(below.grad) == 0.0                                    # the absorbing floor itself


def test_projection_raises_only_the_entries_below_the_floor() -> None:
    from vfe3.numerics import log_variance_floor, project_log_variance_floor_
    eps = 1e-6
    table = torch.tensor([[math.log(eps) - 2.0, 0.5], [-13.0, math.log(eps) - 1e-3]])
    before = table.clone()
    project_log_variance_floor_(table, eps=eps)
    floor = torch.tensor(log_variance_floor(eps), dtype=table.dtype)
    below = before < floor
    assert torch.equal(table[below], floor.expand(int(below.sum())))
    assert torch.equal(table[~below], before[~below])


@pytest.mark.parametrize("mode", ["absorbing", "projected"])
def test_a_train_step_absorbs_or_projects_entries_below_the_floor(mode: str) -> None:
    from vfe3.numerics import log_variance_floor
    cfg = _floor_cfg(sigma_log_floor_mode=mode)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    bank = model.prior_bank
    tokens, targets = _floor_batch(cfg)
    rows = torch.unique(tokens)
    below = math.log(cfg.eps) - 0.3
    with torch.no_grad():
        bank.sigma_log_embed[rows, 0] = below                              # encoder table (table Fisher step)
        bank.decode_sigma_log_embed[rows, 1] = below                       # decode table (AdamW)
    optimizer = T.build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    T.train_step(model, optimizer, scheduler, tokens, targets, grad_clip=1.0)
    encoder = bank.sigma_log_embed.detach()[rows, 0]
    decoder = bank.decode_sigma_log_embed.detach()[rows, 1]
    if mode == "absorbing":
        # The defect the opt-in repairs: zero derivative, so neither optimizer moves the entries.
        assert torch.equal(encoder, torch.full_like(encoder, below))
        assert torch.equal(decoder, torch.full_like(decoder, below))
        return
    floor = log_variance_floor(cfg.eps)
    assert bool((encoder >= floor - 1e-6).all()) and bool((decoder >= floor - 1e-6).all())
    model.zero_grad(set_to_none=True)
    model(tokens, targets)[1].backward()                                   # (logits, loss, ce)
    assert bool((bank.decode_sigma_log_embed.grad[rows, 1] != 0).any())    # the entries take gradient again
    assert bool((bank.sigma_log_embed.grad[rows, 0] != 0).any())


def test_projection_lets_a_loss_raise_an_entry_that_crossed_the_floor() -> None:
    from vfe3.numerics import bounded_variance_from_log, log_variance_floor, project_log_variance_floor_
    eps = 1e-6
    for projected in (False, True):
        table = torch.nn.Parameter(torch.full((3, 2), math.log(eps) - 0.3))
        optimizer = torch.optim.AdamW([table], lr=0.05, weight_decay=0.0)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            (-bounded_variance_from_log(table, eps=eps).sum() * 1e6).backward()    # prefers a larger variance
            optimizer.step()
            if projected:
                project_log_variance_floor_(table.data, eps=eps)
        if projected:
            assert bool((table.detach() > log_variance_floor(eps)).all())
        else:
            assert torch.equal(table.detach(), torch.full((3, 2), math.log(eps) - 0.3))


def test_default_floor_mode_runs_no_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _floor_cfg()
    assert cfg.sigma_log_floor_mode == "absorbing"

    def _refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the absorbing default must not project")
    monkeypatch.setattr(T, "project_log_variance_tables_", _refuse)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    optimizer = T.build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    T.train_step(model, optimizer, scheduler, *_floor_batch(cfg), grad_clip=1.0)


def test_floor_mode_validation_and_inert_rule() -> None:
    with pytest.raises(ValueError, match="sigma_log_floor_mode"):
        _floor_cfg(sigma_log_floor_mode="clamped")
    assert "sigma_log_floor_mode" not in _floor_cfg(sigma_log_floor_mode="projected").inactive_fields()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cholesky = dict(embed_dim=6, n_heads=2, vocab_size=9, max_seq_len=5, n_layers=1, family="gaussian_block",
                        pos_phi="none", gauge_transport="off", prior_covariance_table="block_cholesky",
                        sigma_log_floor_mode="projected")
        linear = tiny_config(decode_mode="linear", **cholesky)
        untied = tiny_config(decode_mode="full_chunked", untie_decode_bank=True, block_coordinates="reference",
                             decode_coordinates="reference", **cholesky)
    assert "sigma_log_floor_mode" in linear.inactive_fields()             # no log-variance table at all
    assert "sigma_log_floor_mode" not in untied.inactive_fields()         # the untied decode table remains


# ------------------------------------------------------------------------------------------------
# transport-geometry-4 (A161): squarings bounded by the observed block norm, bitwise
# ------------------------------------------------------------------------------------------------

def _frame_rows(seed: int, scale: float, n: int = 6, heads: int = 2, d: int = 3) -> torch.Tensor:
    return scale * torch.randn(n, heads, d, d, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("scale", [1e-3, 0.4, 2.0, 6.0])
def test_tightened_squaring_bound_is_bitwise_in_values_and_gradients(scale: float) -> None:
    from vfe3.geometry.transport import _FixedScheduleMatrixExp
    radius = 20.0
    blocks = _frame_rows(1, scale)
    d = blocks.shape[-1]
    observed = float(blocks.square().sum(dim=(-2, -1)).amax().sqrt()) * (1.0 + 2.0 ** -10)
    weight = torch.randn(blocks.shape, generator=torch.Generator().manual_seed(2))
    results = []
    for bound in (radius, observed):
        x = blocks.clone().requires_grad_(True)
        out = _FixedScheduleMatrixExp.apply(x, math.sqrt(d) * bound)
        (grad,) = torch.autograd.grad((out * weight).sum(), x, create_graph=True)
        (second,) = torch.autograd.grad((grad * weight).sum(), x)
        results.append((out.detach(), grad.detach(), second))
    for static, tight in zip(*results):
        assert torch.equal(static, tight)


def _cpu_fixed_schedule(blocks: torch.Tensor, norm_bound: float) -> bool:
    r"""The CUDA route predicate of ``_frame_matrix_exp`` with the device test removed, for CPU tensors."""
    return math.isfinite(norm_bound) and blocks.dtype in (torch.float32, torch.float64)


def test_deduplicated_build_drops_unneeded_squarings_and_keeps_its_bits(monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.geometry.transport as transport
    table = 0.05 * torch.randn(5, 2, 3, 3, generator=torch.Generator().manual_seed(3))
    ids = torch.tensor([[0, 1, 0, 2], [4, 4, 1, 0]])
    original_square = transport._t18_scale_square
    original_bound = transport._squaring_bound
    runs = []
    for tighten in (False, True):
        counts: list = []

        def recording(matrix, direction, threshold, norm_bound):
            counts.append(max(0, math.ceil(math.log2(norm_bound / threshold))) if norm_bound > 0.0 else 0)
            return original_square(matrix, direction, threshold, norm_bound)
        monkeypatch.setattr(transport, "_fixed_schedule_route", _cpu_fixed_schedule)
        monkeypatch.setattr(transport, "_t18_scale_square", recording)
        monkeypatch.setattr(transport, "_squaring_bound",
                            original_bound if tighten else (lambda rows, norm_bound, synced: None))
        leaf = table.clone().requires_grad_(True)
        forward, inverse = transport._stable_compact_glk_exp_pair(
            leaf[ids], validity_max_norm=50.0, _content_token_ids=ids)
        ((forward * 1.3).sum() + (inverse * 0.7).sum()).backward()
        runs.append((forward.detach(), inverse.detach(), leaf.grad, list(counts)))
    (f0, i0, g0, c0), (f1, i1, g1, c1) = runs
    assert torch.equal(f0, f1) and torch.equal(i0, i1) and torch.equal(g0, g1)
    assert c0[0] == math.ceil(math.log2(math.sqrt(3) * 50.0 / 3.010066362817634))     # the chart's static count
    assert c1[0] < c0[0] and sum(c1) < sum(c0)                                         # initialization scale


def test_squaring_bound_needs_the_unique_synchronization_and_the_cuda_schedule() -> None:
    from vfe3.geometry.transport import _squaring_bound
    rows = _frame_rows(4, 0.3)
    assert _squaring_bound(rows, 50.0, synced=True) is None                 # CPU keeps torch.linalg.matrix_exp
    assert _squaring_bound(rows, 50.0, synced=False) is None


def test_squaring_bound_covers_every_block_and_skips_a_nonfinite_maximum(monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.geometry.transport as transport
    monkeypatch.setattr(transport, "_fixed_schedule_route", _cpu_fixed_schedule)
    rows = _frame_rows(5, 0.7)
    bound = transport._squaring_bound(rows, 50.0, synced=True)
    assert bound is not None and bound >= float(rows.square().sum(dim=(-2, -1)).sqrt().max())
    assert transport._squaring_bound(rows, 50.0, synced=False) is None
    rows[0, 0, 0, 0] = float("nan")
    assert transport._squaring_bound(rows, 50.0, synced=True) is None


def test_the_tightened_bound_reaches_a_two_parameter_frame_exponential(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""The bound travels as the norm bound of ``_frame_matrix_exp(blocks, norm_bound)``, whose signature is the one
    the existing wrappers of tests/test_frame_reuse_20260906.py and tests/test_live_route_profile.py replace."""
    import vfe3.geometry.transport as transport
    monkeypatch.setattr(transport, "_fixed_schedule_route", _cpu_fixed_schedule)
    original = transport._frame_matrix_exp
    bounds: list = []

    def counted(matrix: torch.Tensor, norm_bound: float) -> torch.Tensor:
        bounds.append(norm_bound)
        return original(matrix, norm_bound)
    monkeypatch.setattr(transport, "_frame_matrix_exp", counted)
    table = 0.05 * torch.randn(5, 2, 3, 3, generator=torch.Generator().manual_seed(6))
    ids = torch.tensor([[0, 1, 0, 2], [4, 4, 1, 0]])
    transport._stable_compact_glk_exp_pair(table[ids], validity_max_norm=50.0, _content_token_ids=ids)
    assert len(bounds) == 1 and 0.0 < bounds[0] < 1.0
    bounds.clear()
    transport._stable_compact_glk_exp_pair(table[ids], validity_max_norm=50.0)       # no ids: no synchronization
    assert bounds == [50.0]


# ------------------------------------------------------------------------------------------------
# transport-geometry-5 and -6, model-assembly-5 (A162, A163): the NaN map on the deduplicated rows and the
# context finiteness read from the unique rows; the chart check still sees every occurrence
# ------------------------------------------------------------------------------------------------

def _occurrence_guard_pair(blocks: torch.Tensor, ids: torch.Tensor,
                           radius: float) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""The 2026-10-30 order of ``_stable_compact_glk_exp_pair``: the NaN map on the occurrences, then dedup."""
    import vfe3.geometry.transport as transport
    from vfe3.contracts import host_check
    host_check(*transport._compact_chart_check(blocks, radius))
    blocks = transport._finite_or_nan(blocks)
    rows, inverse_index, occurrence_shape = transport._content_rows(blocks, ids)
    exp_pos = transport._frame_matrix_exp(rows.contiguous(), radius)
    return transport._compact_pair_from_exp(exp_pos, rows, validity_max_norm=radius, inverse_index=inverse_index,
                                            occurrence_shape=occurrence_shape, right_pair=None, force_fp64=False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_row_guards_are_bitwise_the_occurrence_guards(dtype: torch.dtype) -> None:
    import vfe3.geometry.transport as transport
    table = 0.3 * torch.randn(5, 2, 3, 3, generator=torch.Generator().manual_seed(7), dtype=dtype)
    ids = torch.tensor([[0, 1, 0, 2, 2], [4, 4, 1, 0, 3]])
    weights = torch.randn((2, 5, 2, 3, 3), generator=torch.Generator().manual_seed(8), dtype=dtype)
    runs = []
    for build in (_occurrence_guard_pair,
                  lambda b, i, r: transport._stable_compact_glk_exp_pair(b, validity_max_norm=r,
                                                                         _content_token_ids=i)):
        leaf = table.clone().requires_grad_(True)
        forward, inverse = build(leaf[ids], ids, 20.0)
        ((forward * weights).sum() + (inverse * weights.flip(0)).sum()).backward()
        runs.append((forward.detach(), inverse.detach(), leaf.grad))
    for before, after in zip(*runs):
        assert torch.equal(before, after)


def test_guards_on_deduplicated_rows_keep_values_gradients_and_failures() -> None:
    import vfe3.geometry.transport as transport
    from vfe3.contracts import deferred_host_checks
    table = 0.3 * torch.randn(5, 2, 3, 3, generator=torch.Generator().manual_seed(5))
    ids = torch.tensor([[0, 1, 0, 2], [4, 4, 1, 0]])
    outputs = []
    for token_ids in (None, ids):                                            # occurrence route, then dedup route
        leaf = table.clone().requires_grad_(True)
        forward, inverse = transport._stable_compact_glk_exp_pair(
            leaf[ids], validity_max_norm=20.0, _content_token_ids=token_ids)
        ((forward * 1.1).sum() + (inverse * 0.9).sum()).backward()
        outputs.append((forward.detach(), inverse.detach(), leaf.grad))
    for occurrence, deduplicated in zip(*outputs):                          # the table gradient sums in another order
        torch.testing.assert_close(deduplicated, occurrence, rtol=1e-5, atol=1e-6)
    bad = table.clone()
    bad[2, 1, 0, 0] = float("inf")                                           # a row seen once by the dedup route
    with pytest.raises(ValueError, match="nonfinite matrix norm"):
        with deferred_host_checks():
            transport._stable_compact_glk_exp_pair(bad[ids], validity_max_norm=20.0, _content_token_ids=ids)
    far = table.clone()
    far[4] = 30.0 * far[4] / far[4].norm()
    with pytest.raises(ValueError, match="_F=30"):
        with deferred_host_checks():
            transport._stable_compact_glk_exp_pair(far[ids], validity_max_norm=20.0, _content_token_ids=ids)


def test_gathered_context_reads_finiteness_from_the_unique_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.geometry.transport as transport
    from vfe3.contracts import CanonicalFrameContext
    table = 0.3 * torch.randn(5, 2, 3, 3, generator=torch.Generator().manual_seed(6))
    ids = torch.tensor([[0, 1, 0, 2, 2, 2], [4, 4, 1, 0, 0, 0]])
    forward, inverse = transport._stable_compact_glk_exp_pair(table[ids], validity_max_norm=20.0,
                                                              _content_token_ids=ids)
    sizes: list = []
    real_isfinite = torch.isfinite

    def recording(tensor: torch.Tensor) -> torch.Tensor:
        sizes.append(tensor.numel())
        return real_isfinite(tensor)
    monkeypatch.setattr(torch, "isfinite", recording)
    CanonicalFrameContext(forward=forward, inverse=inverse, block_diagonal=True)
    monkeypatch.undo()
    assert forward.numel() not in sizes                                      # no occurrence-sized finiteness pass
    assert 4 * forward[0, 0].numel() in sizes                                # the 4 distinct tokens' rows
    broken = forward.clone()
    broken[0, 0, 0, 0, 0] = float("nan")                                     # an ungathered pair is read in full
    with pytest.raises(ValueError, match="finite"):
        CanonicalFrameContext(forward=broken, inverse=inverse.clone(), block_diagonal=True)


# ------------------------------------------------------------------------------------------------
# transport-geometry-6 and model-assembly-5 (A163): the opt-in IEEE fp32 adjoint of the stored frame inverse
# ------------------------------------------------------------------------------------------------

@pytest.fixture
def adjoint_precision() -> Any:
    r"""Set the process-global frame inverse adjoint precision for one test and restore it afterwards."""
    import vfe3.geometry.transport as transport
    previous = transport.frame_inverse_adjoint_precision()
    yield transport.set_frame_inverse_adjoint_precision
    transport.set_frame_inverse_adjoint_precision(previous)


def _frame_table(seed: int, scale: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return scale * torch.randn(5, 2, 3, 3, generator=torch.Generator().manual_seed(seed), dtype=dtype)


def _pair_and_gradient(table: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
                       create_graph: bool = False) -> Tuple[torch.Tensor, ...]:
    import vfe3.geometry.transport as transport
    leaf = table.clone().requires_grad_(True)
    forward, inverse = transport._stable_compact_glk_exp_pair(leaf[ids], validity_max_norm=20.0,
                                                              _content_token_ids=ids)
    objective = (forward * weights).sum() + (inverse * weights.flip(0)).sum()
    (grad,) = torch.autograd.grad(objective, leaf, create_graph=create_graph)
    return forward, inverse, grad, leaf


def test_fp32_inverse_adjoint_keeps_the_pair_and_matches_the_float64_adjoint(adjoint_precision: Any) -> None:
    table = _frame_table(11, 0.6)
    ids = torch.tensor([[0, 1, 0, 2, 2], [4, 4, 1, 0, 3]])
    weights = torch.randn((2, 5, 2, 3, 3), generator=torch.Generator().manual_seed(12))
    runs = {}
    for precision in ("fp64", "fp32"):
        adjoint_precision(precision)
        forward, inverse, grad, _ = _pair_and_gradient(table, ids, weights)
        runs[precision] = (forward.detach(), inverse.detach(), grad)
    assert torch.equal(runs["fp64"][0], runs["fp32"][0]) and torch.equal(runs["fp64"][1], runs["fp32"][1])
    assert not torch.equal(runs["fp64"][2], runs["fp32"][2])                 # the backward did change route
    # Both float32 gradients sit at float32 rounding distance from the float64 evaluation of the same table.
    adjoint_precision("fp64")
    _, _, reference, _ = _pair_and_gradient(table.double(), ids, weights.double())
    scale = float(reference.abs().max())
    for precision in ("fp64", "fp32"):
        error = float((runs[precision][2].double() - reference).abs().max()) / scale
        assert error < 64 * U32, (precision, error)


def test_fp32_inverse_adjoint_runs_no_float64_matmul_in_the_backward(adjoint_precision: Any) -> None:
    from torch.utils._python_dispatch import TorchDispatchMode

    class Float64Matmuls(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.seen: list = []

        def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
            name = str(func.overloadpacket)
            if any(op in name for op in ("mm", "matmul", "linalg_inv")) and any(
                    isinstance(a, torch.Tensor) and a.dtype == torch.float64 for a in args):
                self.seen.append(name)
            return func(*args, **(kwargs or {}))

    table = _frame_table(13, 0.5)
    ids = torch.tensor([[0, 1, 0, 2], [4, 4, 1, 0]])
    weights = torch.randn((2, 4, 2, 3, 3), generator=torch.Generator().manual_seed(14))
    counts = {}
    for precision in ("fp64", "fp32"):
        adjoint_precision(precision)
        import vfe3.geometry.transport as transport
        leaf = table.clone().requires_grad_(True)
        forward, inverse = transport._stable_compact_glk_exp_pair(leaf[ids], validity_max_norm=20.0,
                                                                  _content_token_ids=ids)
        objective = (forward * weights).sum() + (inverse * weights.flip(0)).sum()
        mode = Float64Matmuls()
        with mode:
            objective.backward()
        counts[precision] = mode.seen
    assert counts["fp64"], "the default route's float64 adjoint was not observed"
    assert counts["fp32"] == []


def test_fp32_inverse_adjoint_supports_a_double_backward(adjoint_precision: Any) -> None:
    table = _frame_table(15, 0.4)
    ids = torch.tensor([[0, 1, 0, 2], [4, 4, 1, 0]])
    weights = torch.randn((2, 4, 2, 3, 3), generator=torch.Generator().manual_seed(16))
    direction = torch.randn(table.shape, generator=torch.Generator().manual_seed(17))
    second = {}
    for precision in ("fp64", "fp32"):
        adjoint_precision(precision)
        _, _, grad, leaf = _pair_and_gradient(table, ids, weights, create_graph=True)
        (hessian_direction,) = torch.autograd.grad((grad * direction).sum(), leaf)
        second[precision] = hessian_direction
    assert torch.isfinite(second["fp32"]).all()
    torch.testing.assert_close(second["fp32"], second["fp64"], rtol=1e-4, atol=1e-5)


def test_frame_inverse_adjoint_precision_config_and_installation(adjoint_precision: Any) -> None:
    import vfe3.geometry.transport as transport
    with pytest.raises(ValueError, match="frame_inverse_adjoint_precision"):
        test_config(frame_inverse_adjoint_precision="bf16")
    with pytest.raises(ValueError, match="frame_inverse_adjoint_precision"):
        test_config(frame_inverse_adjoint_precision=None)
    assert test_config().frame_inverse_adjoint_precision == "fp64"
    assert "frame_inverse_adjoint_precision" not in test_config(frame_inverse_adjoint_precision="fp32").inactive_fields()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert "frame_inverse_adjoint_precision" in test_config(gauge_transport="frozen").inactive_fields()
    torch.manual_seed(0)
    VFEModel(test_config(frame_inverse_adjoint_precision="fp32"))
    assert transport.frame_inverse_adjoint_precision() == "fp32"
    torch.manual_seed(0)
    VFEModel(test_config())
    assert transport.frame_inverse_adjoint_precision() == "fp64"


def test_fp32_inverse_adjoint_leaves_the_model_loss_and_moves_gradients_by_rounding(adjoint_precision: Any) -> None:
    results = {}
    for precision in ("fp64", "fp32"):
        cfg = test_config(frame_inverse_adjoint_precision=precision, vocab_size=7, embed_dim=6, n_heads=2,
                          max_seq_len=5, n_layers=1)
        torch.manual_seed(0)
        model = VFEModel(cfg)
        g = torch.Generator().manual_seed(18)
        tokens = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len), generator=g)
        targets = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len), generator=g)
        _, loss, _ = model(tokens, targets)
        loss.backward()
        results[precision] = (loss.detach(), model.prior_bank.phi_embed.grad.detach().clone())
    assert torch.equal(results["fp64"][0], results["fp32"][0])                # the forward is the same bits
    torch.testing.assert_close(results["fp32"][1], results["fp64"][1], rtol=1e-4, atol=1e-6)


# ------------------------------------------------------------------------------------------------
# gauge-structure-4 (A190): matrix-free tensor-power action and a guard on the memory actually allocated
# ------------------------------------------------------------------------------------------------

F64 = torch.float64


def test_matrix_free_leibniz_action_equals_the_dense_action() -> None:
    from vfe3.geometry import irreps
    from vfe3.geometry.generators import generate_son, generate_sun_real
    for N, p in ((4, 3), (5, 3)):
        G = generate_son(N, dtype=F64)
        Q = irreps._canonical_invariant_basis(N, p, traceless=True)
        for a in range(G.shape[0]):
            torch.testing.assert_close(irreps._leibniz_apply(G[a], Q, p), irreps._leibniz_action(G[a], p) @ Q,
                                       rtol=0, atol=1e-13)
    X = irreps._su_complex_basis(generate_sun_real(3))
    for p, q in ((2, 1), (2, 2)):
        Q = irreps._mixed_trace_basis(3, p, q).to(torch.complex128)
        for a in range(X.shape[0]):
            torch.testing.assert_close(irreps._leibniz_apply(X[a], Q, p + q, conj_from=p),
                                       irreps._mixed_leibniz(X[a], p, q) @ Q, rtol=0, atol=1e-13)
    Q = irreps._mixed_trace_basis(3, 2, 2).to(torch.complex128)
    torch.testing.assert_close(irreps._slot_swap_apply(Q, 3, 2), irreps._slot_swap(3, 2).to(torch.complex128) @ Q,
                               rtol=0, atol=0)


def _refuse_dense(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("the dense N^r x N^r action was built")


@pytest.mark.parametrize("algebra,N,label", [("so", 4, "l3"), ("so", 5, "l3"), ("so", 4, "l2"),
                                             ("su", 3, "pq2_1"), ("su", 3, "pq2_2")])
def test_matrix_free_images_equal_the_dense_images(monkeypatch: pytest.MonkeyPatch, algebra: str, N: int,
                                                   label: str) -> None:
    from vfe3.geometry import irreps
    from vfe3.geometry.generators import generate_son, generate_sun_real
    G = generate_son(N, dtype=F64) if algebra == "so" else generate_sun_real(N)
    dense = irreps.irrep_generators(G, algebra=algebra, label=label)
    monkeypatch.setattr(irreps, "_DENSE_CONSTRUCTION_BUDGET_BYTES", 0)
    for name in ("_leibniz_action", "_mixed_leibniz", "_slot_swap"):
        monkeypatch.setattr(irreps, name, _refuse_dense)
    free = irreps.irrep_generators(G, algebra=algebra, label=label)
    torch.testing.assert_close(free, dense, rtol=0, atol=1e-12)


def test_labels_within_the_dense_budget_keep_their_bytes() -> None:
    from vfe3.geometry import irreps
    from vfe3.geometry.generators import generate_son
    G = generate_son(4, dtype=F64)
    with irreps._single_threaded():
        Q = irreps._canonical_invariant_basis(4, 3, traceless=True)
        inline = torch.stack([Q.T @ irreps._leibniz_action(G[a], 3) @ Q for a in range(G.shape[0])])
    assert torch.equal(irreps.irrep_generators(G, algebra="so", label="l3"), inline)


def test_construction_guard_reads_the_planned_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.geometry import irreps
    from vfe3.geometry.generators import generate_son
    dense, peak = irreps._tensor_construction_plan(6, 6)
    assert not dense and peak < 2 * 2 ** 30                               # so(6) l6: was a 16 GiB dense action
    irreps.check_irrep_constructible(6, algebra="so", label="l6")
    # A label whose dense construction fits the budget keeps the dense route and so its bytes and fingerprint:
    # so(7) l5 (a 2.1 GiB action, about 6.5 GiB in all) stays dense, so(8) l5 (8 GiB per action) does not.
    dense, peak = irreps._tensor_construction_plan(7, 5)
    assert dense and 6 * 2 ** 30 < peak <= irreps._TENSOR_POWER_BUDGET_BYTES
    assert not irreps._tensor_construction_plan(8, 5)[0]
    with pytest.raises(ValueError, match="construction budget"):         # su(15) (2, 2): an 11 GiB basis alone
        irreps.check_irrep_constructible(15, algebra="su", label="pq2_2")
    monkeypatch.setattr(irreps, "_TENSOR_POWER_BUDGET_BYTES", 2 ** 16)
    monkeypatch.setattr(irreps, "_canonical_invariant_basis", _refuse_dense)
    with pytest.raises(ValueError, match="construction budget"):         # refused before any allocation
        irreps._tensor_power_irrep(generate_son(5, dtype=F64), 3, traceless=True)


@pytest.mark.parametrize("K", [64, 128, 192, 256, 512])
def test_random_towers_are_unchanged_by_the_memory_guard(monkeypatch: pytest.MonkeyPatch, K: int) -> None:
    from vfe3.geometry import irreps
    draws = {}
    for budget in (irreps._TENSOR_POWER_BUDGET_BYTES, math.inf):
        monkeypatch.setattr(irreps, "_TENSOR_POWER_BUDGET_BYTES", budget)
        for N in range(3, 13):
            try:
                draws.setdefault(N, []).append(irreps.random_irrep_spec(N, K, 4, seed=1))
            except ValueError as exc:
                draws.setdefault(N, []).append(str(exc))
    assert all(first == second for first, second in draws.values())


# ------------------------------------------------------------------------------------------------
# Rule 2 of the remediation brief: the live route is bitwise unchanged on the CPU
# ------------------------------------------------------------------------------------------------

def _head_ratio_minus_one_minus_log(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    r"""``vfe3.families.base.ratio_minus_one_minus_log`` as of 5bb0a57 (the two-log far branch)."""
    from vfe3.families.base import x_minus_log1p
    x = (num - den) / den
    near = x.abs() < 0.5
    ones = torch.ones_like(x)
    small = x_minus_log1p(torch.where(near, x, torch.zeros_like(x)))
    far_num, far_den = torch.where(near, ones, num), torch.where(near, ones, den)
    far = far_num / far_den - 1.0 + torch.log(far_den) - torch.log(far_num)
    return torch.where(near, small, far)


def _head_stable_compact_glk_exp_pair(blocks: torch.Tensor, *, validity_max_norm: Any = None,
                                      _content_token_ids: Any = None, _right_pair: Any = None,
                                      _force_fp64: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""``vfe3.geometry.transport._stable_compact_glk_exp_pair`` as of 5bb0a57 (NaN map on the occurrences)."""
    import vfe3.geometry.transport as transport
    from vfe3.contracts import host_check
    validity_max_norm = transport._resolve_chart_max_norm(validity_max_norm)
    host_check(*transport._compact_chart_check(blocks, validity_max_norm))
    blocks = transport._finite_or_nan(blocks)
    store_dtype = blocks.dtype if blocks.dtype in (torch.float32, torch.float64) else torch.float32
    rows, inverse_index, occurrence_shape = transport._content_rows(blocks, _content_token_ids)
    with torch.amp.autocast(rows.device.type, enabled=False):
        exp_pos = transport._frame_matrix_exp(rows.to(store_dtype).contiguous(), validity_max_norm)
    return transport._compact_pair_from_exp(
        exp_pos, rows, validity_max_norm=validity_max_norm, inverse_index=inverse_index,
        occurrence_shape=occurrence_shape, right_pair=_right_pair, force_fp64=_force_fp64)


def _live_loss_and_gradients(cfg: VFE3Config) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    model = VFEModel(cfg)
    g = torch.Generator().manual_seed(9)
    shape = (cfg.batch_size, cfg.max_seq_len)
    tokens = torch.randint(0, cfg.vocab_size, shape, generator=g)
    tokens[:, -1] = tokens[:, 0]                                             # repeated ids: the dedup route runs
    targets = torch.randint(0, cfg.vocab_size, shape, generator=g)
    _, loss, _ = model(tokens, targets)
    loss.backward()
    return loss.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("dropout", [dict(phi_type_dropout=0.0), dict(phi_type_dropout=0.5,
                                                                         phi_type_dropout_freq_scale=None)])
def test_live_profile_loss_and_gradients_are_bitwise_unchanged(monkeypatch: pytest.MonkeyPatch,
                                                               dropout: Dict[str, Any]) -> None:
    r"""The tiny live route against the 5bb0a57 forms of the two functions this package rewrote on it.

    The live route calls ``ratio_minus_one_minus_log`` nowhere (its router and decoder use their own kernels), so
    that replacement only guards a later route change; the deduplicated frame build runs without frame dropout (a
    training forward under the dropout builds every occurrence) and is compared.
    The remediation report records a further comparison of the whole tree against 5bb0a57 on this profile.
    """
    import vfe3.families.gaussian as gaussian
    import vfe3.geometry.transport as transport
    from tests.profiles import live_config
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = live_config(**dropout)
    deduplicated = []
    real_rows = transport._content_rows

    def recording(blocks: torch.Tensor, ids: Any) -> Any:
        out = real_rows(blocks, ids)
        deduplicated.append(out[1] is not None)
        return out
    monkeypatch.setattr(transport, "_content_rows", recording)
    current = _live_loss_and_gradients(cfg)
    # The deduplicated frame build runs without frame dropout; a training forward under type-level frame dropout
    # withholds the token ids, so every occurrence is built (frame-dropout package, transport-geometry-1).
    assert any(deduplicated) is (cfg.phi_type_dropout == 0.0)
    monkeypatch.setattr(gaussian, "ratio_minus_one_minus_log", _head_ratio_minus_one_minus_log)
    monkeypatch.setattr(transport, "_stable_compact_glk_exp_pair", _head_stable_compact_glk_exp_pair)
    head = _live_loss_and_gradients(cfg)
    assert torch.equal(current[0], head[0])
    assert current[1].keys() == head[1].keys() and len(current[1]) > 10
    for name, grad in head[1].items():
        assert torch.equal(current[1][name], grad), name


# ------------------------------------------------------------------------------------------------
# dead-code-4 (A336): frame RoPE calls build_rope_rotation directly
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fraction", [1, 1.0, 0.5])
def test_frame_rope_is_the_builder_rotation(fraction: float) -> None:
    from vfe3.geometry.rope import build_rope_rotation
    cfg = test_config(vocab_size=6, embed_dim=8, n_heads=2, max_seq_len=8, n_layers=1, gauge_group="block_glk",
                      rope_mode="frame", rope_base=100.0, rope_fraction=fraction)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    device = model.prior_bank.mu_embed.device
    rotation = model._rope_rotation(5, device)
    expected = build_rope_rotation(torch.arange(5), model.group.irrep_dims, base=100.0, fraction=float(fraction),
                                   device=device, dtype=model.group.generator_dtype)
    assert torch.equal(rotation, expected)


# ------------------------------------------------------------------------------------------------
# transport-geometry-7 (A330): the condition bound alone leaves the chart radius fail-closed
# ------------------------------------------------------------------------------------------------

def test_condition_bound_without_an_interior_radius_warns_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import warnings
    import vfe3.config as config_module
    monkeypatch.setattr(config_module, "_UNBOUNDED_FRAME_CHART_WARNED", set())
    base = dict(gauge_group="block_glk", embed_dim=4, n_heads=2, phi_frame_condition_max=20.0)
    with pytest.warns(UserWarning, match="phi_mstep_max_matrix_norm below transport_chart_max_norm"):
        test_config(**base)
    for overrides in (dict(), dict(phi_mstep_max_matrix_norm=10.0), dict(phi_frame_condition_max=None)):
        with warnings.catch_warnings(record=True) as caught:                # once per message, and not when bounded
            warnings.simplefilter("always")
            test_config(**{**base, **overrides})
        assert not [w for w in caught if "ends the run" in str(w.message)], overrides


def test_chart_radius_still_ends_a_run_on_a_conditioned_frame_without_the_interior_radius() -> None:
    from vfe3.geometry.groups import get_group
    from vfe3.geometry.transport import build_factored_transport
    group = get_group("block_glk")(4, 2)
    phi = torch.zeros(1, 1, group.generator_shape[0])
    rotation = torch.tensor([[0.0, 1.0], [-1.0, 0.0]]) * 30.0 / math.sqrt(2.0)   # skew: cond 1, ||M||_F = 30
    phi[0, 0, :4] = rotation.reshape(-1)
    with pytest.raises(ValueError, match="transport chart validity bound exceeded"):
        build_factored_transport(phi, group, compact_blocks=True, validity_max_norm=20.0)


# ------------------------------------------------------------------------------------------------
# tables-families-7 (A321): whole-table record of the encoder log-variances against their clamps
# ------------------------------------------------------------------------------------------------

def test_sigma_log_statistics_record_the_extremes_and_the_clamped_count() -> None:
    cfg = _floor_cfg()
    torch.manual_seed(0)
    model = VFEModel(cfg)
    table = model.prior_bank.sigma_log_embed
    with torch.no_grad():
        table[0, 0] = math.log(cfg.eps) - 0.3                              # below the eps floor
        table[1, 1] = 81.0                                                 # above the exponent cap
    stats = T._sigma_log_statistics(model)
    assert set(stats) == set(T.SIGMA_LOG_COLUMNS)
    values = table.detach()
    assert stats["sigma_log_min"] == float(values.min()) and stats["sigma_log_max"] == float(values.max())
    assert stats["sigma_log_clamped"] == 2.0
    assert stats["sigma_log_mean"] == float(torch.std_mean(values, correction=0)[1])


# ------------------------------------------------------------------------------------------------
# model-assembly-8 (A092): the frame-pair escalation is one verdict per call, as now documented
# ------------------------------------------------------------------------------------------------

def test_frame_pair_escalation_moves_the_whole_call_to_float64(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""One ill-conditioned block in the last position of the second sequence moves every row of the call,
    the first sequence's prefix included, to a float64 pair equal to the float32 one only to rounding: the
    trade the ``frame_pair_escalation_tol`` comment and the transport docstrings now state."""
    import vfe3.geometry.transport as transport
    monkeypatch.setattr(transport, "_FRAME_PAIR_FP64_ESCALATION", True)
    monkeypatch.setattr(transport, "_FRAME_PAIR_CONDITION_TOL", 1e-6)
    monkeypatch.setattr(transport, "_FRAME_PAIR_FP64_ESCALATIONS", transport._FRAME_PAIR_FP64_ESCALATIONS)
    calm = 0.05 * torch.randn(2, 4, 2, 3, 3, generator=torch.Generator().manual_seed(19))   # (B, N, H, d, d)
    forward, inverse = transport._stable_compact_glk_exp_pair(calm, validity_max_norm=20.0)
    assert forward.dtype == inverse.dtype == torch.float32
    stormy = calm.clone()
    stormy[1, 3, 0] = torch.diag(torch.tensor([2.0, -2.0, 0.0]))         # kappa_F(U) near 55 > 1e-6 / eps_32
    escalated, _ = transport._stable_compact_glk_exp_pair(stormy, validity_max_norm=20.0)
    assert escalated.dtype == torch.float64
    prefix = escalated[0]                                                   # the untouched first sequence
    torch.testing.assert_close(prefix.float(), forward[0], rtol=1e-6, atol=1e-6)
    assert not torch.equal(prefix, forward[0].double())
