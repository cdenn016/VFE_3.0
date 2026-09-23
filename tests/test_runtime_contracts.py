"""Tiny runtime contracts for cache selection, precision ownership and update recovery."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.profiles import tiny_config
from vfe3.model.model import VFEModel




@pytest.mark.parametrize("full", [False, True])
def test_decoder_precision_belongs_to_model(full: bool) -> None:
    mode = "full_chunked" if full else "diagonal_chunked"
    family = "gaussian_full" if full else "gaussian_diagonal"
    a = VFEModel(tiny_config(decode_mode=mode, decode_av_precision='fp64', family=family)).eval()
    mu, sigma = torch.zeros(1, 2, 4), torch.ones(1, 2, 4)
    if full:
        sigma = torch.diag_embed(sigma)
    before = a.prior_bank.decode(mu, sigma)
    VFEModel(tiny_config(decode_mode=mode, decode_av_precision='fp32', family=family))
    after = a.prior_bank.decode(mu, sigma)
    assert before.dtype == after.dtype == torch.float64
    torch.testing.assert_close(before, after, rtol=0., atol=0.)


def test_captured_block_mlp_beliefs_are_declared() -> None:
    from vfe3.contracts import MStepCapture
    model = VFEModel(tiny_config(use_block_mlp=True, block_mlp_expansion=2)).eval()
    capture: MStepCapture = {}
    model.forward_beliefs(torch.tensor([[0, 1]]), capture=capture)
    assert {'block_mlp_input', 'block_mlp_output'} <= set(capture)
    assert set(capture) <= set(MStepCapture.__annotations__)


def test_word_cache_metadata_and_normalization_use_selected_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.data.datasets as data
    from vfe3.data.word_vocab import WordVocabulary, vocabulary_path
    default, selected = tmp_path/'default', tmp_path/'selected'
    default.mkdir(); selected.mkdir()
    name = 'wikitext-103-word'
    WordVocabulary(['<unk>', 'default', 'extra', 'other']).save(vocabulary_path(name, tag='word', cache_dir=default))
    WordVocabulary(['<unk>', 'a', 'b']).save(vocabulary_path(name, tag='word', cache_dir=selected))
    path = data.cache_path(name, 'validation', cache_dir=selected)
    torch.save(torch.tensor([1, 2, 1, 2, 1]), path)
    Path(str(path)+'.meta.json').write_text(json.dumps(dict(n_tokens=5, n_chars_source=10)), encoding='utf-8')
    monkeypatch.setattr(data, 'default_cache_dir', lambda: default)
    loader = data.make_dataloader(name, 'validation', 2, 1, shuffle=False, drop_last=False, cache_dir=selected, vocab_size=3)
    assert loader.dataset.data_identity['tokenizer_vocab_size'] == 3
    assert data.get_tiktoken_decoder(name, cache_dir=selected)([1, 2]) == 'a b'
    monkeypatch.setattr(data, 'default_cache_dir', lambda: tmp_path/'absent')
    assert data.tokens_per_char(name, 'validation', cache_dir=selected) == .5
    from vfe3.train import _default_sample_decoder
    assert _default_sample_decoder(tiny_config(), dataset=name, cache_dir=selected)([1, 2]) == 'a b'


class _OverflowModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.))
        self.cfg = SimpleNamespace(grad_clip_per_role=False, m_phi_update_mode='adamw', phi_mstep_max_matrix_norm=None)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor, **kwargs: object) -> tuple:
        loss = self.weight * 1e35 + 1.
        return None, loss, SimpleNamespace(ce=loss, scored_tokens=torch.tensor(1), excluded_tokens=torch.tensor(0))


@pytest.mark.parametrize('metrics', [False, True])
def test_overflow_backoff_is_independent_of_logging(metrics: bool) -> None:
    from vfe3.train import train_step
    model = _OverflowModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    scaler = torch.amp.GradScaler('cpu', init_scale=65536.)
    tokens = torch.zeros(1, 1, dtype=torch.long)
    for expected in (32768., 16384., 8192.):
        status = {}
        train_step(model, optimizer, scheduler, tokens, tokens, scaler=scaler,
                   grad_clip=1., metrics_out={} if metrics else None, status_out=status)
        assert scaler.get_scale() == expected
        assert not status['did_step']
    assert scheduler.last_epoch == 0
    assert model.weight.item() == 0.


def test_predictive_mean_step_sweep_is_rejected() -> None:
    import ablation
    with pytest.raises(ValueError, match='inert|e_q_mu_lr'):
        ablation.validate_sweeps(['e_q_mu_lr'], require_construction=True)


@pytest.mark.parametrize('compact', [False, True])
def test_dense_transport_preserves_factor_precision_under_autocast(compact: bool) -> None:
    from vfe3.geometry.transport import FactoredTransport, CompactFactoredTransport
    U = torch.tensor([[[[1., .4], [.1, 1.]], [[1., -.2], [.3, 1.]]]])
    inv = torch.linalg.inv(U)
    transport = (CompactFactoredTransport(U.unsqueeze(-3), inv.unsqueeze(-3), 2)
                 if compact else FactoredTransport(U, inv, [2]))
    expected = transport.to_dense_omega()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = transport.to_dense_omega()
    assert actual.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)


def test_repaired_cholesky_has_finite_correct_derivative() -> None:
    from vfe3.numerics import safe_cholesky
    M = torch.tensor([[0., 0.], [0., 1.]], dtype=torch.float64, requires_grad=True)
    L, ok = safe_cholesky(M, rounds=2, jitter_mode='absolute')
    assert ok
    derivative, = torch.autograd.grad(L.square().sum(), M)
    torch.testing.assert_close(derivative, torch.eye(2, dtype=M.dtype), rtol=1e-8, atol=1e-8)


def test_spd_inverse_reaches_fallback_without_inverting_failed_factor() -> None:
    from vfe3.numerics import safe_spd_inverse
    M = torch.diag(torch.tensor([1., -1e-6], dtype=torch.float64))
    expected = torch.linalg.inv(M + 1e-5 * torch.eye(2, dtype=M.dtype))
    torch.testing.assert_close(safe_spd_inverse(M), expected)


def test_cholesky_recovery_directional_derivative_matches_finite_difference() -> None:
    from vfe3.numerics import safe_cholesky
    M = torch.diag(torch.tensor([1., 0.], dtype=torch.float64)).requires_grad_()
    direction = torch.tensor([[0., 1.], [1., 0.]], dtype=M.dtype)
    def value(matrix: torch.Tensor) -> torch.Tensor:
        return safe_cholesky(matrix, rounds=3, jitter_mode="absolute")[0].sum()
    gradient, = torch.autograd.grad(value(M), M)
    step = 1e-7
    finite_difference = (value(M.detach()+step*direction)-value(M.detach()-step*direction))/(2*step)
    torch.testing.assert_close((gradient*direction).sum(), finite_difference, rtol=1e-6, atol=1e-7)
