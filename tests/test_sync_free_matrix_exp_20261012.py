"""The sync-free fixed-schedule frame exponential (audit 2026-10-12 round 2, item 19 and M5)."""
import math
import warnings
from typing import Dict

import pytest
import torch

from vfe3.contracts import deferred_host_checks
from vfe3.geometry import transport
from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import _FixedScheduleMatrixExp

RADIUS = 20.0     # the test profile's transport_chart_max_norm


def _families(d: int, n: int, generator: torch.Generator) -> Dict[str, torch.Tensor]:
    r"""float64 generators of the structures the frames use, scaled over ||A||_F in (0, RADIUS]."""
    norms = torch.cat([torch.logspace(-6, 0, 8, dtype=torch.float64),
                       torch.linspace(1.0, RADIUS, n - 8, dtype=torch.float64)])

    def scaled(a: torch.Tensor) -> torch.Tensor:
        return a / torch.linalg.matrix_norm(a)[:, None, None] * norms[:, None, None]

    gaussian = torch.randn(n, d, d, generator=generator, dtype=torch.float64)
    skew = torch.randn(n, d, d, generator=generator, dtype=torch.float64)
    torus = torch.zeros(n, d, d, dtype=torch.float64)
    angles = torch.randn(n, d // 2, generator=generator, dtype=torch.float64)
    for k in range(d // 2):
        torus[:, 2 * k, 2 * k + 1], torus[:, 2 * k + 1, 2 * k] = angles[:, k], -angles[:, k]
    u = torch.randn(n, d, 1, generator=generator, dtype=torch.float64)
    v = torch.randn(n, 1, d, generator=generator, dtype=torch.float64)
    upper = torch.triu(torch.randn(n, d, d, generator=generator, dtype=torch.float64), diagonal=1)
    column = torch.zeros(n, d, d, dtype=torch.float64)
    column[:, :, 0] = torch.where(torch.rand(n, d, generator=generator) < 0.5, -1.0, 1.0)  # ||A||_1 = sqrt(d) ||A||_F
    return {
        "gl": scaled(gaussian),
        "so": scaled(skew - skew.mT),
        "rope_torus": scaled(torus),
        "near_singular": scaled(u @ v + 1e-6 * torch.randn(n, d, d, generator=generator, dtype=torch.float64)),
        "non_normal": scaled(upper + torch.diag_embed(0.1 * torch.randn(n, d, generator=generator, dtype=torch.float64))),
        "one_norm_extremal": scaled(column),
    }


def _relative(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_norm(x.double() - reference) / torch.linalg.matrix_norm(reference)


def _fixed(matrix: torch.Tensor) -> torch.Tensor:
    return _FixedScheduleMatrixExp.apply(matrix, math.sqrt(matrix.shape[-1]) * RADIUS)


@pytest.mark.parametrize("d", [2, 4])
def test_forward_matches_matrix_exp_over_the_chart(d: int) -> None:
    generator = torch.Generator().manual_seed(d)
    for name, a64 in _families(d, 24, generator).items():
        a32 = a64.float()
        reference = torch.linalg.matrix_exp(a32.double())
        torch_error = _relative(torch.linalg.matrix_exp(a32), reference)
        fixed_error = _relative(_fixed(a32), reference)
        assert fixed_error.max() <= 4.0 * torch_error.max() + 1e-6, name


@pytest.mark.parametrize("gradient_scale", [1e-6, 1.0, 1e6])
def test_backward_matches_the_frechet_adjoint(gradient_scale: float) -> None:
    generator = torch.Generator().manual_seed(7)
    d = 3
    for name, a64 in _families(d, 16, generator).items():
        g64 = gradient_scale * torch.randn(a64.shape, generator=generator, dtype=torch.float64)
        a32, g32 = a64.float(), g64.float()

        def vjp(function, a: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
            a = a.clone().requires_grad_(True)
            (function(a) * g).sum().backward()
            return a.grad

        reference = vjp(torch.linalg.matrix_exp, a32.double(), g32.double())
        fixed_error = _relative(vjp(_fixed, a32, g32), reference)
        assert fixed_error.max() < 1e-4, name


@pytest.mark.parametrize("d", [2, 3])
def test_float64_gradcheck_and_gradgradcheck(d: int) -> None:
    generator = torch.Generator().manual_seed(11 + d)
    a = (2.5 * torch.randn(2, d, d, generator=generator, dtype=torch.float64)).requires_grad_(True)

    def function(x: torch.Tensor) -> torch.Tensor:
        return _FixedScheduleMatrixExp.apply(x, math.sqrt(d) * 6.0)
    assert torch.autograd.gradcheck(function, (a,))
    assert torch.autograd.gradgradcheck(function, (a,))


def test_static_schedule_caps_and_propagates_nan() -> None:
    a = torch.zeros(3, 2, 2)
    a[0, 0, 1] = 1.0
    a[1] = float("nan")
    a[2, 0, 0] = 1e3                                   # far outside the bound: capped squarings, no hang
    out = _FixedScheduleMatrixExp.apply(a, math.sqrt(2) * 2.0)
    assert torch.equal(out[0], torch.tensor([[1.0, 1.0], [0.0, 1.0]]))
    assert torch.isnan(out[1]).all()
    assert out.shape == a.shape


def test_device_split_keeps_matrix_exp_off_cuda() -> None:
    a = 0.3 * torch.randn(4, 2, 3, 3, generator=torch.Generator().manual_seed(3))
    assert torch.equal(transport._frame_matrix_exp(a, RADIUS), torch.linalg.matrix_exp(a))
    assert torch.equal(transport._frame_matrix_exp(a, math.inf), torch.linalg.matrix_exp(a))


@pytest.mark.parametrize("with_ids", [False, True])
def test_stacked_pairs_equal_the_sequential_calls(with_ids: bool) -> None:
    generator = torch.Generator().manual_seed(5)
    right = 0.05 * torch.randn(4, 2, 2, 2, generator=generator)
    ids = torch.tensor([[0, 1, 0, 2], [2, 2, 1, 0]])
    table = 0.5 * torch.randn(3, 2, 2, 2, generator=generator)
    left = table[ids]
    token_ids = ids if with_ids else None
    stacked = transport._stacked_compact_exp_pairs(
        right, left, validity_max_norm=RADIUS, _content_token_ids=token_ids)
    right_pair = transport._stable_compact_glk_exp_pair(right, validity_max_norm=RADIUS)
    left_pair = transport._stable_compact_glk_exp_pair(
        left, validity_max_norm=RADIUS, _content_token_ids=token_ids, _right_pair=right_pair)
    for got, expected in zip((*stacked[0], *stacked[1]), (*right_pair, *left_pair)):
        torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-7)


def _count_syncs(function) -> int:
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            function()
        finally:
            torch.cuda.set_sync_debug_mode(0)
    return sum("called a synchronizing CUDA operation" in str(w.message) for w in caught)


def test_cuda_frame_exponential_has_no_host_sync() -> None:
    import os
    if os.environ.get("VFE3_TEST_DEVICE", "cpu") != "cuda" or not torch.cuda.is_available():
        pytest.skip("CUDA lane only")
    group = get_group("block_glk")(4, 2, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(0)
    phi = 0.3 * torch.randn(2, 5, 8, device="cuda", generator=generator)
    right_phi = 0.02 * torch.randn(5, 8, device="cuda", generator=generator)

    def step(exp_function) -> None:
        x = phi.clone().requires_grad_(True)
        y = right_phi.clone().requires_grad_(True)
        blocks = torch.cat((x.reshape(-1, 2, 2, 2), y.reshape(-1, 2, 2, 2)))
        exp_function(blocks).sum().backward()

    assert _count_syncs(lambda: step(lambda b: transport._frame_matrix_exp(b, RADIUS))) == 0
    assert _count_syncs(lambda: step(torch.linalg.matrix_exp)) > 0

    def build() -> None:
        x = phi.clone().requires_grad_(True)
        with deferred_host_checks():
            built = transport.build_factored_transport(
                x, group, compact_blocks=True, right_phi=right_phi, validity_max_norm=RADIUS)
            loss = built.exp_blocks.sum() + built.inv_blocks.sum()
        loss.backward()
    # One read closes the deferred scope; the exponentials themselves add none, forward or backward.
    assert _count_syncs(build) == 1
