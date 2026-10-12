"""Audit 2026-10-12 items 5 and 6b: the baseline launcher's data check and the matched clip scope.

Tiny CPU models only (vocabulary 11, K=8, two heads of width 4, two layers, 6-token windows).
"""

from tests.profiles import test_config
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import torch
from torch.utils.data import DataLoader

import train_baseline as tb
import vfe3.train as T
from vfe3.baseline import BaselineConfig, BaselineTransformer
from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows

V, K, H, L, E, N, B = 11, 8, 2, 2, 2, 6, 2


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _base_dir(tmp_path: Path, *, dataset: Optional[str] = "wikitext-103",
              provenance: Optional[Dict[str, Any]] = None) -> str:
    run = tmp_path / "base"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"config": {}, "dataset": dataset}), encoding="utf-8")
    if provenance is not None:
        (run / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    return str(run)


_PROV = {"seed": 6, "data_seed": 3, "max_tokens": None, "tokenizer_tag": "tiktoken"}


def _check(base: str, **over: Any) -> None:
    kwargs: Dict[str, Any] = dict(dataset="wikitext-103", seeds=(6,), data_seed=3, max_tokens=None)
    kwargs.update(over)
    tb.check_base_data(base, **kwargs)


def test_matching_launch_passes(tmp_path: Path) -> None:
    _check(_base_dir(tmp_path, provenance=_PROV))


@pytest.mark.parametrize("over, needle", [
    ({"dataset": "wiki-en"}, "dataset"),                       # cl100k corpus: dataset and tokenizer differ
    ({"max_tokens": 4096}, "max_tokens"),
    ({"data_seed": 4}, "data seed 4"),
    ({"data_seed": None, "seeds": (7,)}, "data seed 7"),       # None resolves to the model seed
])
def test_disagreement_with_the_base_record_raises(tmp_path: Path, over: Dict[str, Any], needle: str) -> None:
    with pytest.raises(ValueError, match=needle):
        _check(_base_dir(tmp_path, provenance=_PROV), **over)


def test_tokenizer_disagreement_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tokenizer 'tiktoken_cl100k'"):
        _check(_base_dir(tmp_path, dataset=None, provenance=_PROV), dataset="wiki-en")


def test_model_seed_rule_matches_a_base_that_followed_it(tmp_path: Path) -> None:
    base = _base_dir(tmp_path, provenance={**_PROV, "data_seed": 6})
    _check(base, data_seed=None, seeds=(6, 7, 8))
    with pytest.raises(ValueError, match="data seed 3"):
        _check(base, data_seed=3)


def test_unrecorded_fields_are_logged_not_refused(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="train_baseline"):
        _check(_base_dir(tmp_path), data_seed=99, max_tokens=5)
    assert "unchecked" in caplog.text and "data_seed" in caplog.text
    with pytest.raises(ValueError, match="dataset"):                 # config.json still carries the dataset
        _check(str(tmp_path / "base"), dataset="wiki-en")


def _vcfg(**over: Any) -> VFE3Config:
    values: Dict[str, Any] = dict(
        vocab_size=V, embed_dim=K, n_heads=H, batch_size=B, max_seq_len=N, eval_stride=3,
        n_layers=L, n_e_steps=1, warmup_steps=0, max_steps=6, log_interval=0, eval_interval=0,
        checkpoint_interval=0, generate_embedding_figures=False, pos_phi="learned", use_block_mlp=True,
        block_mlp_expansion=E, beta_attention_prior="causal_alibi", family="gaussian_diagonal",
        gauge_group="block_glk", attention_mode="projected_predictive", attention_residual=True,
        decode_mode="linear", predictive_readout="kernel", content_link_edges="off",
        norm_type_block="layernorm", layernorm_affine=True, skip_belief_sigma_update=True)
    values.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


@pytest.mark.parametrize("scope", ["role", "group"])
def test_matched_arm_copies_the_clip_scope_and_the_loop_reads_it(
    scope: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bcfg = BaselineConfig.matched_to(_vcfg(grad_clip_scope=scope))
    assert bcfg.grad_clip_scope == scope
    seen: List[str] = []
    real = T._train_step_with_precision_retry

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs["grad_clip_scope"])
        return real(*args, **kwargs)

    monkeypatch.setattr(T, "_train_step_with_precision_retry", spy)
    torch.manual_seed(0)
    model = BaselineTransformer(bcfg)
    stream = (torch.arange(97) * 7 + 3) % V
    loader = DataLoader(TokenWindows(stream, N), batch_size=B, shuffle=False, drop_last=True)
    T.train(model, loader, bcfg, n_steps=1, grad_clip=bcfg.grad_clip, device=torch.device("cpu"),
            generate_samples=False)
    assert seen == [scope]


def test_clip_scope_validation_and_inactivity() -> None:
    with pytest.raises(ValueError, match="grad_clip_scope"):
        BaselineConfig(grad_clip_scope="global")
    assert "grad_clip_scope" in BaselineConfig(grad_clip=None).inactive_fields()
    assert "grad_clip_scope" not in BaselineConfig(grad_clip=1.0).inactive_fields()
    # A saved baseline config without the field reloads with the scope it ran under (role).
    payload = BaselineConfig().serialize()
    payload.pop("grad_clip_scope")
    assert BaselineConfig.from_serialized(payload, source="old").grad_clip_scope == "role"
