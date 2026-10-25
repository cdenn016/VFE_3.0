r"""Type-level frame dropout (``phi_type_dropout``, 2026-10-25), tiny CPU model.

Off by default and in evaluation the encoder is unchanged; in training every occurrence of a dropped type in a
sequence reads the identity frame (phi = 0) and kept positions are untouched; the frequency form spares frequent
types; the validator refuses invalid settings.
"""
from __future__ import annotations

import pytest
import torch

from tests.profiles import tiny_config
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import phi_type_drop_probabilities, phi_type_keep_mask


def _model(**over: object) -> VFEModel:
    torch.manual_seed(0)
    return VFEModel(tiny_config(**over))


def test_default_and_eval_leave_the_encoder_unchanged() -> None:
    model = _model(phi_type_dropout=0.5)
    ids = torch.randint(0, model.cfg.vocab_size, (3, 6))
    model.eval()
    assert torch.equal(model.prior_bank.encode(ids).phi, model.prior_bank.phi_embed[ids])
    off = _model()
    off.train()
    assert torch.equal(off.prior_bank.encode(ids).phi, off.prior_bank.phi_embed[ids])


def test_training_drops_whole_types_per_sequence() -> None:
    model = _model(phi_type_dropout=0.5)
    model.train()
    ids = torch.tensor([[1, 2, 1, 3, 2, 1], [4, 4, 5, 4, 5, 6]])
    torch.manual_seed(1)
    phi = model.prior_bank.encode(ids).phi
    table = model.prior_bank.phi_embed[ids]
    dropped = (phi == 0).all(-1)                                   # (B, N)
    kept = torch.isclose(phi, table).all(-1)
    assert bool((dropped | kept).all())
    for b in range(ids.shape[0]):
        for v in ids[b].unique():
            where = ids[b] == v
            assert bool(dropped[b][where].all()) or bool(kept[b][where].all())
    torch.manual_seed(1)
    mask = phi_type_keep_mask(model.prior_bank, ids)              # the same draw the encoder made
    assert torch.equal(phi, table * mask.unsqueeze(-1).to(table.dtype))


def test_frequency_form_spares_frequent_types() -> None:
    model = _model(phi_type_dropout=0.2, phi_type_dropout_freq_scale=1e-2, decode_unigram_prior=True)
    bank = model.prior_bank
    counts = torch.ones(model.cfg.vocab_size)
    counts[0] = 1e6                                                # one very frequent type
    bank.set_unigram_log_prior(counts)
    p = phi_type_drop_probabilities(bank)
    assert float(p[0]) < 0.01 * 0.2 + 1e-6
    assert float(p[1:].min()) > 0.19


@pytest.mark.parametrize("over", [dict(phi_type_dropout=1.0), dict(phi_type_dropout=-0.1),
                                  dict(phi_type_dropout=0.1, phi_type_dropout_freq_scale=0.0),
                                  dict(phi_type_dropout=0.1, phi_type_dropout_freq_scale=1e-5,
                                       decode_unigram_prior=False)])
def test_invalid_settings_are_refused(over: dict) -> None:
    with pytest.raises(ValueError):
        tiny_config(**over)
