r"""Remediation of the 2026-10-31 audit, package ``launchers-data``: the training launcher and the text generator.

A010 (config-launcher-2): with both route switches of ``train_vfe3.py`` on, the gauge switch is applied on top of the
exact route (the frame bound the exact route keeps is dropped and every gauge flag is checked), and the config.md
header records full covariance from the resolved config.

A064 (data-pipeline-1): a resume follows the checkpoint's data order, so the data seed recorded in config.md and
provenance is the seed of the restored generator, and a difference from ``DATA_SEED`` is logged with both seeds.

A112 (baseline-generation-3): ``generate_text.py`` loads a conventional-baseline run, the baseline's sampler takes
``top_p`` and refuses non-finite logits as ``VFEModel.generate`` does, and the GPT-2 control refuses a run whose
tokens are not GPT-2 ids.

A354 (baseline-generation-4): a prompt token that never occurs in the training stream is reported, read from the
run's unigram table and its recorded training token count.

Tiny CPU fixtures only; the launcher tests replace the model, loaders and artifacts by stubs.
"""
from __future__ import annotations

import json
import logging
import math
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch
from torch.utils.data import DataLoader

import generate_text
import train_vfe3
from tests.profiles import config_values, tiny_config
from vfe3.data.datasets import TokenWindows


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


# ---------------------------------------------------------------------------------------------------------------- A010

class _Stop(Exception):
    r"""Raised by the stubbed banner: the launcher has resolved the config and written config.md."""


def _launcher_values(**overrides: Any) -> Dict[str, Any]:
    values = config_values(**overrides)
    values.pop("seed")                                       # the launcher dictionary never sets the seed
    return values


def _stub_launch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, values: Dict[str, Any], *, exact: bool,
                 gauge: bool) -> Dict[str, Any]:
    r"""Run ``_run_once`` up to the banner with stubs; return the resolved cfg and the config.md header."""
    import vfe3.config_markdown as config_markdown
    import vfe3.data.datasets as datasets_mod
    import vfe3.model.model as model_mod
    import vfe3.run_artifacts as run_artifacts_mod

    captured: Dict[str, Any] = {}

    class StubModel:
        def __init__(self, cfg: Any) -> None:
            captured["cfg"] = cfg

        def to(self, device: Any) -> "StubModel":
            return self

    def record(run_dir: Any, source: Any, resolved: Any, *, header: Any = None) -> None:
        captured["header"] = dict(header)

    def stop(*args: Any, **kwargs: Any) -> str:
        raise _Stop

    monkeypatch.setattr(model_mod, "VFEModel", StubModel)
    monkeypatch.setattr(run_artifacts_mod, "RunArtifacts", lambda *args, **kwargs: object())
    monkeypatch.setattr(config_markdown, "write_config_markdown", record)
    monkeypatch.setattr(datasets_mod, "tokens_per_char", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_vfe3, "seed_everything", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_vfe3, "_select_loader", lambda dataset, cfg, split="train": None)
    monkeypatch.setattr(train_vfe3, "_run_dir", lambda cfg, dataset, run_root=None: str(tmp_path / "run"))
    monkeypatch.setattr(train_vfe3, "_banner", stop)
    monkeypatch.setattr(train_vfe3, "config", values)
    monkeypatch.setattr(train_vfe3, "principled_exact_route", exact)
    monkeypatch.setattr(train_vfe3, "fully_gauge_covariant", gauge)
    monkeypatch.setattr(train_vfe3, "DEVICE", "cpu")
    monkeypatch.setattr(train_vfe3, "MAX_TOKENS", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(_Stop):
            train_vfe3._run_once(6, logging.getLogger("audit-20261031-launcher"), run_root=str(tmp_path))
    return captured


def test_both_route_switches_run_the_strictly_covariant_exact_route(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from vfe3.exact_route import principled_exact_route_failures
    from vfe3.gauge_covariant import gauge_covariance_failures

    values = _launcher_values(phi_frame_condition_max=20.0)
    with caplog.at_level(logging.INFO, logger="audit-20261031-launcher"):
        captured = _stub_launch(monkeypatch, tmp_path, values, exact=True, gauge=True)
    cfg = captured["cfg"]
    assert cfg.phi_frame_condition_max is None                       # the gauge switch drops the kept bound
    assert gauge_covariance_failures(cfg) == []
    assert principled_exact_route_failures(cfg) == []
    assert captured["header"]["fully_gauge_covariant"] is True
    assert captured["header"]["launch overrides"]["phi_frame_condition_max"] is None
    line = next(r.getMessage() for r in caplog.records if "overriding" in r.getMessage())
    assert "fully_gauge_covariant=True" in line and "phi_frame_condition_max=20.0 -> None" in line


def test_exact_route_alone_keeps_the_bound_and_records_no_full_covariance(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from vfe3.gauge_covariant import gauge_covariance_failures

    captured = _stub_launch(monkeypatch, tmp_path, _launcher_values(phi_frame_condition_max=20.0), exact=True,
                            gauge=False)
    cfg = captured["cfg"]
    assert cfg.phi_frame_condition_max == 20.0
    assert gauge_covariance_failures(cfg) == ["no_phi_mstep_chart_cap"]
    assert captured["header"]["fully_gauge_covariant"] is False
    unbounded = _stub_launch(monkeypatch, tmp_path, _launcher_values(phi_frame_condition_max=None), exact=True,
                             gauge=False)
    assert unbounded["header"]["fully_gauge_covariant"] is True     # the documented implication


def test_switches_off_record_the_resolved_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = _stub_launch(monkeypatch, tmp_path, _launcher_values(), exact=False, gauge=False)
    assert captured["header"]["fully_gauge_covariant"] is False
    assert captured["header"]["launch overrides"] == {}


# ---------------------------------------------------------------------------------------------------------------- A064

def _generator_state(seed: int, draws: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    for _ in range(draws):
        torch.randperm(16, generator=generator)
    return generator.get_state().clone()


def _stub_full_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, data_seed: int,
                   resume_from: Any) -> Dict[str, Any]:
    r"""``_run_once`` through train and finalize with stubs; ``train`` restores a seed-3 epoch generator when a
    resume is requested, as :func:`vfe3.train.train` does from the checkpoint's data state."""
    import vfe3.config_markdown as config_markdown
    import vfe3.data.datasets as datasets_mod
    import vfe3.model.model as model_mod
    import vfe3.run_artifacts as run_artifacts_mod

    captured: Dict[str, Any] = {}

    class StubModel:
        def __init__(self, cfg: Any) -> None:
            captured["cfg"] = cfg

        def to(self, device: Any) -> "StubModel":
            return self

    def select(dataset: str, cfg: Any, split: str = "train") -> Any:
        if split != "train":
            return None
        return SimpleNamespace(generator=torch.Generator().manual_seed(train_vfe3._effective_data_seed(cfg)))

    def train(model: Any, loader: Any, cfg: Any, **kwargs: Any) -> List[float]:
        if cfg.resume_from is not None:
            loader.generator.set_state(_generator_state(3))
        return [1.0]

    def finalize(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        captured["finalize"] = kwargs
        return {"test_ppl": None}

    def record(run_dir: Any, source: Any, resolved: Any, *, header: Any = None) -> None:
        captured["header"] = dict(header)

    values = _launcher_values(resume_from=resume_from)
    monkeypatch.setattr(model_mod, "VFEModel", StubModel)
    monkeypatch.setattr(run_artifacts_mod, "RunArtifacts", lambda *args, **kwargs: object())
    monkeypatch.setattr(run_artifacts_mod, "finalize_run", finalize)
    monkeypatch.setattr(config_markdown, "write_config_markdown", record)
    monkeypatch.setattr(datasets_mod, "tokens_per_char", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_vfe3, "seed_everything", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_vfe3, "_select_loader", select)
    monkeypatch.setattr(train_vfe3, "_run_dir", lambda cfg, dataset, run_root=None: str(tmp_path / "run"))
    monkeypatch.setattr(train_vfe3, "_banner", lambda *args, **kwargs: "")
    monkeypatch.setattr(train_vfe3, "train", train)
    monkeypatch.setattr(train_vfe3, "evaluate", lambda *args, **kwargs: {
        "ce": 1.0, "ppl": math.e, "bits_per_token": 1.0 / math.log(2.0), "bpc": None})
    monkeypatch.setattr(train_vfe3, "config", values)
    monkeypatch.setattr(train_vfe3, "principled_exact_route", False)
    monkeypatch.setattr(train_vfe3, "fully_gauge_covariant", False)
    monkeypatch.setattr(train_vfe3, "DATA_SEED", data_seed)
    monkeypatch.setattr(train_vfe3, "DEVICE", "cpu")
    monkeypatch.setattr(train_vfe3, "MAX_TOKENS", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_vfe3._run_once(6, logging.getLogger("audit-20261031-data-seed"), run_root=str(tmp_path))
    return captured


def test_resume_records_the_restored_data_seed_and_warns(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    checkpoint = tmp_path / "step_4.pt"
    torch.save({"step": 4, "data_state": {"epoch_start_generator_state": _generator_state(3, draws=1),
                                          "batches_consumed": 4, "epoch": 0}}, checkpoint)
    assert train_vfe3._checkpoint_data_seed(checkpoint) == 3
    with caplog.at_level(logging.WARNING, logger="audit-20261031-data-seed"):
        captured = _stub_full_run(monkeypatch, tmp_path, data_seed=4, resume_from=str(checkpoint))
    assert captured["header"]["data seed"] == 3                     # config.md names the order the run follows
    assert captured["finalize"]["data_seed"] == 3                   # and so does provenance.json
    warned = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("3" in message and "4" in message and "data" in message for message in warned)


def test_fresh_run_records_the_configured_data_seed(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="audit-20261031-data-seed"):
        captured = _stub_full_run(monkeypatch, tmp_path, data_seed=4, resume_from=None)
    assert captured["header"]["data seed"] == 4 and captured["finalize"]["data_seed"] == 4
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_checkpoint_data_seed_is_none_for_a_sequential_or_unreadable_checkpoint(tmp_path: Path) -> None:
    sequential = tmp_path / "sequential.pt"
    torch.save({"step": 2, "data_state": {"epoch_start_generator_state": None}}, sequential)
    assert train_vfe3._checkpoint_data_seed(sequential) is None
    broken = tmp_path / "broken.pt"
    broken.write_bytes(b"not a checkpoint")
    assert train_vfe3._checkpoint_data_seed(broken) is None
    assert train_vfe3._checkpoint_data_seed(tmp_path / "missing.pt") is None


def _shuffled_loader(loader_seed: int) -> DataLoader:
    tokens = torch.randint(0, 6, (480,), generator=torch.Generator().manual_seed(123))
    return DataLoader(TokenWindows(tokens, 8), batch_size=4, shuffle=True, drop_last=True,
                      generator=torch.Generator().manual_seed(loader_seed))


def test_train_resume_restores_the_checkpoint_generator_seed(tmp_path: Path) -> None:
    r"""The mechanism the launcher reads: ``train`` restores the checkpoint's epoch generator, seed included."""
    from vfe3.model.model import VFEModel
    from vfe3.run_artifacts import RunArtifacts
    from vfe3.train import train

    cfg = tiny_config(vocab_size=6, max_seq_len=8, batch_size=4, warmup_steps=1, max_steps=4, checkpoint_interval=2,
                      log_interval=0, eval_interval=0, deterministic=False)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    train(model, _shuffled_loader(3), cfg, n_steps=2, artifacts=RunArtifacts(tmp_path / "run", cfg, model))
    checkpoint = tmp_path / "run" / "checkpoints" / "step_2.pt"
    assert train_vfe3._checkpoint_data_seed(checkpoint) == 3
    resumed_loader = _shuffled_loader(4)
    assert resumed_loader.generator.initial_seed() == 4
    train(VFEModel(cfg), resumed_loader, cfg, n_steps=4, resume_from=checkpoint)
    assert resumed_loader.generator.initial_seed() == 3


# ---------------------------------------------------------------------------------------------------------------- A112

_V = 7


def _table(seed: int = 0) -> torch.Tensor:
    return torch.randn(_V, _V, generator=torch.Generator().manual_seed(seed))


def _baseline_model(table: torch.Tensor) -> Any:
    r"""A tiny BaselineTransformer whose forward returns ``table[token]`` at every position, and whose last-position
    decode (``_next_token_logits``, the seam ``generate`` reads) returns ``table`` at the last token."""
    from vfe3.baseline import BaselineConfig, BaselineTransformer

    model = BaselineTransformer(BaselineConfig(vocab_size=_V, embed_dim=4, n_heads=2, n_layers=1, mlp_expansion=2,
                                               max_seq_len=5, batch_size=2, deterministic=False)).eval()
    model.forward = lambda token_ids, *args, **kwargs: table[token_ids]          # (B, N, V)
    model._next_token_logits = lambda window: table[window[:, -1]]               # (B, V)
    return model


def _vfe_model(table: torch.Tensor) -> Any:
    from vfe3.model.model import VFEModel

    model = VFEModel(tiny_config(vocab_size=_V)).eval()
    model.forward_beliefs = lambda context, return_logits, decode_last: (None, table[context[:, -1]].unsqueeze(1))
    return model


@pytest.mark.parametrize("temperature, top_k, top_p", [(1.0, None, 0.8), (0.7, 3, 0.6), (1.3, None, None)])
def test_baseline_generate_takes_top_p_and_draws_the_vfe_sampler_tokens(
        temperature: float, top_k: Any, top_p: Any) -> None:
    table = _table(1)
    baseline, vfe = _baseline_model(table), _vfe_model(table)
    prompt = torch.tensor([[3, 0]])
    for seed in (0, 11):
        torch.manual_seed(seed)
        reference = vfe.generate(prompt, 10, temperature=temperature, greedy=False, top_k=top_k, top_p=top_p)
        torch.manual_seed(seed)
        out = baseline.generate(prompt, 10, temperature=temperature, greedy=False, top_k=top_k, top_p=top_p)
        assert torch.equal(out, reference)


def test_baseline_generate_refuses_non_finite_logits_and_bad_arguments() -> None:
    table = _table(2)
    table[4] = float("nan")
    baseline = _baseline_model(table)
    with pytest.raises(ValueError, match="NaN"):
        baseline.generate(torch.tensor([[4]]), 1, greedy=True)
    blank = _table(3)
    blank[2] = float("-inf")
    with pytest.raises(ValueError, match="no finite value"):
        _baseline_model(blank).generate(torch.tensor([[2]]), 1, greedy=True)
    good = _baseline_model(_table(4))
    with pytest.raises(ValueError, match="top_p"):
        good.generate(torch.tensor([[1]]), 1, top_p=1.5)
    with pytest.raises(ValueError, match="top_k"):
        good.generate(torch.tensor([[1]]), 1, top_k=_V + 1)
    with pytest.raises(ValueError, match="nonempty"):
        good.generate(torch.zeros(1, 0, dtype=torch.long), 1, greedy=True)


# The training stream never holds ids 5 and 6, so they have zero training count (A354).
_STREAMS = {"train": torch.arange(97) % 5, "validation": (torch.arange(29) * 2) % _V,
            "test": (torch.arange(23) * 5 + 1) % _V}


def _baseline_loader(dataset: str, cfg: Any, *, split: str, data_seed: int, max_tokens: Any = None) -> DataLoader:
    if split == "train":
        return DataLoader(TokenWindows(_STREAMS["train"], cfg.max_seq_len), batch_size=cfg.batch_size,
                          shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(data_seed))
    return DataLoader(TokenWindows(_STREAMS[split], cfg.max_seq_len, pad_final=True, stride=cfg.eval_stride),
                      batch_size=2, shuffle=False)


@pytest.fixture
def baseline_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    r"""A finalized tiny conventional-baseline run written by ``train_baseline.run_baseline``."""
    import train_baseline as tb
    import vfe3.run_artifacts as ra
    import vfe3.train as T
    from vfe3.baseline import BaselineConfig

    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    monkeypatch.setattr(tb, "_select_loader", _baseline_loader)
    monkeypatch.setattr(tb, "_tokens_per_char", lambda *args, **kwargs: None)
    monkeypatch.setattr(tb, "_tokenizer_tag", lambda dataset: None)
    cfg = BaselineConfig(vocab_size=_V, embed_dim=4, n_heads=2, n_layers=1, mlp_expansion=2, max_seq_len=6,
                         batch_size=2, eval_stride=3, max_steps=4, warmup_steps=1, lr=3e-3, log_interval=0,
                         eval_interval=2, checkpoint_interval=0, seed=5)
    return tb.run_baseline(cfg, dataset="tiny-synthetic", out_root=tmp_path / "baselines", seed=3, data_seed=None,
                           label="tiny", device="cpu", generate_samples=False,
                           logger=logging.getLogger("audit-20261031-baseline"))


def test_generate_text_loads_and_samples_a_conventional_baseline_run(
        baseline_run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.baseline import BaselineTransformer

    assert generate_text.newest_run_dir(baseline_run.parent.parent) == baseline_run
    loaded = generate_text.load_run(baseline_run, "cpu")
    assert isinstance(loaded.model, BaselineTransformer)
    assert loaded.model_family == "conventional_transformer" and loaded.dataset == "tiny-synthetic"
    monkeypatch.setattr(generate_text, "prompt_token_ids", lambda text, dataset, **kwargs: [1, 2])
    monkeypatch.setattr(generate_text, "decode_ids", lambda ids, dataset: " ".join(map(str, ids)))
    generations = generate_text.generate_for_prompts(loaded, ["p"], 4, greedy=False, temperature=1.0, top_k=None,
                                                     top_p=0.9, seed=0)
    assert len(generations[0].token_ids) == 4
    assert all(0 <= token < _V for token in generations[0].token_ids)


@pytest.mark.parametrize("dataset", ["wiki-en", "wikitext-103-word"])
def test_gpt2_control_refuses_a_run_not_tokenized_with_gpt2(
        dataset: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    loaded = generate_text.LoadedRun(torch.nn.Linear(1, 1), SimpleNamespace(max_seq_len=8, eval_stride=None),
                                     dataset, tmp_path, 1, 10.0)
    for name, value in dict(RUN_DIR=str(tmp_path), PROMPTS=("p",), SELF_CE=True, BASELINE="gpt2", MAX_NEW_TOKENS=3,
                            REFERENCE_DRAWS=2, BLOCK_TOKENS=None, DEVICE="cpu").items():
        monkeypatch.setattr(generate_text, name, value)
    monkeypatch.setattr(generate_text, "validated_run_dir", lambda path: path)
    monkeypatch.setattr(generate_text, "load_run", lambda run_dir, device: loaded)

    def unreachable(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the GPT-2 control was built for a run with foreign token ids")

    monkeypatch.setattr(generate_text, "HuggingFaceBaseline", unreachable)
    monkeypatch.setattr(generate_text, "generate_for_prompts", unreachable)
    with pytest.raises(SystemExit, match="gpt2"):
        generate_text.main()


# ---------------------------------------------------------------------------------------------------------------- A354

def test_zero_count_ids_are_read_from_the_unigram_table() -> None:
    from vfe3.model.model import VFEModel

    model = VFEModel(tiny_config(vocab_size=_V, decode_unigram_prior=True))
    counts = torch.tensor([5.0, 0.0, 3.0, 1.0, 0.0, 9.0, 2.0])
    model.prior_bank.set_unigram_log_prior(counts)
    assert generate_text.zero_count_ids(model, n_train_tokens=int(counts.sum())) == frozenset({1, 4})
    assert generate_text.zero_count_ids(model, n_train_tokens=None) is None
    plain = VFEModel(tiny_config(vocab_size=_V, decode_unigram_prior=False))
    assert generate_text.zero_count_ids(plain, n_train_tokens=20) is None


def test_prompt_with_an_unseen_token_warns() -> None:
    pytest.importorskip("tiktoken")
    ids = generate_text.prompt_token_ids("The history of the city", "wikitext-103")
    assert ids[0] == 464                                            # bare "The", not the corpus's " The" (383)
    with pytest.warns(UserWarning, match="464") as record:
        generate_text.prompt_token_ids("The history of the city", "wikitext-103", unseen_ids=frozenset({464}))
    assert "leading space" in str(record[0].message)                # the WikiText hint, on the GPT-2 corpus only
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert generate_text.prompt_token_ids(" The history", "wikitext-103", unseen_ids=frozenset({464}))[0] == 383


def test_load_run_reads_the_unseen_ids_of_a_baseline_run(baseline_run: Path) -> None:
    loaded = generate_text.load_run(baseline_run, "cpu")
    provenance = json.loads((baseline_run / "provenance.json").read_text(encoding="utf-8"))
    counts = torch.bincount(_STREAMS["train"], minlength=_V)
    assert provenance["train_data_n_tokens"] == int(counts.sum())
    assert loaded.unseen_ids == frozenset(int(i) for i in torch.nonzero(counts == 0).flatten()) == {5, 6}
