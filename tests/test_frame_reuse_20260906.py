"""Tiny numerical and dispatch regressions for S22 diagnostics and S18 factor reuse."""
from collections.abc import Iterator
import importlib
import os
import warnings

import pytest
import torch

from vfe3 import metrics
from vfe3.config import VFE3Config
from vfe3.geometry import transport
from vfe3.geometry.groups import GaugeGroup, get_group
from vfe3.model.model import VFEModel


@pytest.fixture
def device() -> torch.device:
    return torch.device(os.environ.get('VFE3_TEST_DEVICE', 'cpu'))


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _dense_norm(phi: torch.Tensor, group: GaugeGroup) -> torch.Tensor:
    return torch.linalg.matrix_norm(group.embed_algebra(phi, dtype=phi.dtype))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_coordinate_norm_avoids_matrix(dtype: torch.dtype, device: torch.device,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    group = get_group('block_glk')(6, 3, dtype=dtype, device=device)
    torch.manual_seed(18)
    phi = torch.randn(4, 12, dtype=dtype, device=device) * 0.1
    expected = _dense_norm(phi, group)
    def forbidden(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError('diagnostic allocated a dense frame')
    monkeypatch.setattr(group, 'embed_algebra', forbidden)
    actual = metrics.diagnostic_frame_norm(phi, group, max_norm=20.0, eps=1e-8)
    torch.testing.assert_close(actual, expected, rtol=2e-7, atol=1e-15)


@pytest.mark.parametrize('kind', ['threshold', 'eps', 'nan', 'inf', 'overflow', 'basis', 'grad', 'autocast'])
def test_norm_fallback_retains_reduction(kind: str, device: torch.device,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    group = get_group('block_glk')(4, 2, device=device)
    phi = torch.full((3, 8), 0.02, device=device)
    if kind in ('threshold', 'eps'):
        phi[0].zero_(); phi[0, 0] = 20.0 if kind == 'threshold' else 1e-8
        phi[1] = torch.nextafter(phi[0], torch.full_like(phi[0], float('inf')))
    elif kind in ('nan', 'inf', 'overflow'):
        phi[0, 0] = {'nan': float('nan'), 'inf': float('inf'), 'overflow': 1e30}[kind]
    elif kind == 'basis':
        group.generators[0].add_(group.generators[1], alpha=0.2)
    elif kind == 'grad':
        phi.requires_grad_()
    calls = []
    original = group.embed_algebra
    def counted(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(group, 'embed_algebra', counted)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=kind == 'autocast'):
        actual = metrics.diagnostic_frame_norm(phi, group, max_norm=20.0, eps=1e-8)
        assert len(calls) == 1
        expected = _dense_norm(phi, group)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    if kind == 'grad':
        gradient, = torch.autograd.grad(actual.sum(), phi, retain_graph=True)
        reference, = torch.autograd.grad(expected.sum(), phi)
        torch.testing.assert_close(gradient, reference, rtol=0, atol=0)
    else:
        assert torch.equal(actual > 20, expected > 20)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('threshold', [20.0, 0.0])
def test_unique_factors_values_gradients_hvp(dtype: torch.dtype, threshold: float,
                                           device: torch.device) -> None:
    torch.manual_seed(20)
    ids = torch.tensor([[2, 0, 2, 1], [0, 2, 2, 0]], device=device)
    table = (torch.randn(3, 2, 2, 2, dtype=dtype, device=device) * 0.04).requires_grad_()
    group = get_group('block_glk')(4, 2, dtype=dtype, device=device)
    position = (torch.randn(4, 8, dtype=dtype, device=device) * 0.03).requires_grad_()
    weights = [torch.randn(2, 4, 2, 2, 2, dtype=dtype, device=device) for _ in range(2)]
    directions = [torch.randn_like(table), torch.randn_like(position)]
    results = []
    for reuse in (False, True):
        factors = transport.build_factored_transport(
            table[ids].flatten(-3), group, compact_blocks=True, right_phi=position,
            exp_fp64_norm_threshold=threshold,
            _content_token_ids=ids if reuse else None, validity_max_norm=18.0, clamp_monitor=True)
        loss = (factors.exp_blocks * weights[0]).sum() + (factors.inv_blocks * weights[1]).sum()
        grads = torch.autograd.grad(loss, (table, position), create_graph=True)
        hvp = torch.autograd.grad(sum((g * d).sum() for g, d in zip(grads, directions)), (table, position))
        results.append((factors.exp_blocks, factors.inv_blocks, *grads, *hvp))
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)


def test_unique_exp_work_and_occurrence_clamp_fraction(device: torch.device,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    ids = torch.tensor([[0, 0, 1, 0]], device=device)
    table = torch.zeros(2, 2, 2, 2, device=device)
    table[0, 0, 0, 0] = 21
    shapes = []
    original = torch.linalg.matrix_exp
    def counted(matrix: torch.Tensor) -> torch.Tensor:
        shapes.append(tuple(matrix.shape))
        return original(matrix)
    monkeypatch.setattr(torch.linalg, 'matrix_exp', counted)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        transport._stable_compact_glk_exp_pair(table[ids], clamp_monitor=True, _content_token_ids=ids)
    assert shapes == [(2, 2, 2, 2)]
    assert any('75.0%' in str(w.message) for w in recorded)


@pytest.mark.parametrize('kind', ['nonfinite', 'bound'])
def test_unique_errors_precede_deduplication(kind: str, device: torch.device,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    blocks = torch.zeros(1, 4, 2, 2, 2, device=device)
    if kind == 'nonfinite': blocks[0, 0, 0, 0, 0] = float('nan')
    if kind == 'bound': blocks[0, 0, 0, 0, 0] = 19.0
    def forbidden(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError('deduplication preceded the original validation error')
    monkeypatch.setattr(torch, 'unique', forbidden)
    with pytest.raises(ValueError):
        transport._stable_compact_glk_exp_pair(
            blocks, _content_token_ids=torch.zeros(1, 4, dtype=torch.long, device=device),
            validity_max_norm=18.0)


def test_unique_table_directional_finite_difference(device: torch.device) -> None:
    torch.manual_seed(32)
    ids = torch.tensor([[0, 1, 0, 1]], device=device)
    table = (torch.randn(2, 2, 2, 2, dtype=torch.float64, device=device) * 0.02).requires_grad_()
    direction = torch.randn_like(table)
    weights = torch.randn(1, 4, 2, 2, 2, dtype=table.dtype, device=device)
    def objective(x: torch.Tensor) -> torch.Tensor:
        factors = transport._stable_compact_glk_exp_pair(x[ids], exp_fp64_norm_threshold=0.0, _content_token_ids=ids)
        return (factors[0] * weights).sum()
    gradient, = torch.autograd.grad(objective(table), table)
    delta = 1e-5
    finite_difference = (objective(table.detach() + delta * direction) - objective(table.detach() - delta * direction))/(2 * delta)
    torch.testing.assert_close((gradient * direction).sum(), finite_difference, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize('fallback', ['none', 'encoder', 'basis', 'capture', 'position', 'int32', 'public', 'empty_capture'])
def test_builtin_model_gate(fallback: str, device: torch.device, monkeypatch: pytest.MonkeyPatch) -> None:
    prior_bank = importlib.import_module('vfe3.model.prior_bank')
    cfg = VFE3Config(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=4,
                    batch_size=1, n_layers=1, n_e_steps=1, decode_mode='linear',
                    pos_phi='learned', e_phi_lr=0.0, block_mlp_expansion=2, amp_dtype=None)
    model = VFEModel(cfg).to(device)
    tokens = torch.tensor([[0, 1, 0, 1]], device=device)
    original = transport._stable_compact_glk_exp_pair
    seen = []
    def recorded(blocks: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        seen.append(kwargs.get('_content_token_ids') is not None)
        return original(blocks, **kwargs)
    monkeypatch.setattr(transport, '_stable_compact_glk_exp_pair', recorded)
    if fallback == 'encoder':
        builtin = prior_bank._ENCODERS['per_token']
        monkeypatch.setitem(prior_bank._ENCODERS, 'per_token', lambda *args: builtin(*args))
    if fallback == 'int32': tokens = tokens.int()
    if fallback == 'basis': model.group.generators[0].mul_(1.1)
    if fallback == 'position': monkeypatch.setattr(model, '_apply_pos_phi', lambda phi: phi + 0.001)
    if fallback in ('capture', 'empty_capture', 'public'):
        cap = {'diagnostic': {}} if fallback == 'capture' else {} if fallback == 'empty_capture' else None
        belief, _ = model.forward_beliefs(tokens, capture=cap, _reuse_content_factors=fallback != 'public')
        belief.mu.sum().backward()
    else:
        model(tokens, torch.tensor([[1, 2, 1, 2]], device=device))[1].backward()
    assert any(seen) == (fallback == 'none')
    assert model.prior_bank.phi_embed.grad is not None


@pytest.mark.parametrize('same_token', [False, True])
def test_reuse_extreme_token_counts(same_token: bool, device: torch.device,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
    ids = torch.zeros(1, 4, dtype=torch.long, device=device) if same_token else torch.arange(4, device=device).unsqueeze(0)
    table = torch.zeros(4, 2, 2, 2, device=device)
    original = torch.linalg.matrix_exp
    counts = []
    def counted(matrix: torch.Tensor) -> torch.Tensor:
        counts.append(matrix.numel() // 4)
        return original(matrix)
    monkeypatch.setattr(torch.linalg, 'matrix_exp', counted)
    actual = transport._stable_compact_glk_exp_pair(table[ids], _content_token_ids=ids)
    assert counts == [2 if same_token else 8]
    assert actual[0].shape == (1, 4, 2, 2, 2)
    assert torch.equal(actual[0], actual[1])


def test_unique_factors_under_active_bf16_autocast(device: torch.device) -> None:
    with torch.autocast(device.type, dtype=torch.bfloat16):
        test_unique_factors_values_gradients_hvp(torch.float32, 'norm', 20.0, device)
