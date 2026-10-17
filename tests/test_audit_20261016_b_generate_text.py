r"""Audit 2026-10-16 F17: the ``generate_text`` GPT-2 control decodes with the run's own sampler.

``HuggingFaceBaseline.generate`` took the argmax at every step while the VFE continuation was sampled, so the
self-CE deltas of the two rows compared different samplers (greedy text sits at the mode and scores a lower
self-CE). The control now applies :func:`generate_text.sample_next_token`, the map ``VFEModel.generate`` applies to
its last-position logits, and is reseeded per prompt like :func:`generate_text.generate_for_prompts`. The per-prompt
baseline delta is taken against the baseline's own held-out reference. Tiny CPU stand-ins replace GPT-2, which the
CPU interpreter does not have; the method body is model-agnostic.
"""
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch

import generate_text
from tests.profiles import tiny_config
from vfe3.model.model import VFEModel

_V = 7


def _table(seed: int = 0, scale: float = 1.0) -> torch.Tensor:
    r"""(V, V) next-token logits indexed by the last context token."""
    return scale * torch.randn(_V, _V, generator=torch.Generator().manual_seed(seed))


class _LastTokenLM(torch.nn.Module):
    r"""HuggingFace-shaped causal LM whose logits at each position are ``table[token]``."""

    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        self.table = torch.nn.Parameter(table.clone(), requires_grad=False)

    def forward(self, inputs: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=self.table[inputs])                   # (B, L, V)


def _baseline(table: torch.Tensor, context: int = 5) -> generate_text.HuggingFaceBaseline:
    baseline = object.__new__(generate_text.HuggingFaceBaseline)
    baseline.name = "stand-in"
    baseline._model = _LastTokenLM(table)
    baseline.context = context
    return baseline


def _vfe_model(table: torch.Tensor) -> VFEModel:
    r"""A tiny VFEModel whose decode is replaced by the same last-token logits, to pin its sampler."""
    model = VFEModel(tiny_config(vocab_size=_V)).eval()

    def forward_beliefs(context: torch.Tensor, return_logits: bool, decode_last: bool) -> Any:
        return None, table[context[:, -1]].unsqueeze(1)                     # (B, 1, V)

    model.forward_beliefs = forward_beliefs
    return model


def test_greedy_control_is_the_argmax_loop() -> None:
    table = _table()
    prompt = [1, 4, 2]
    expected = list(prompt)
    for _ in range(9):
        expected.append(int(table[expected[-1]].argmax()))
    out = _baseline(table).generate(prompt, 9, greedy=True, temperature=1.0, top_k=None, top_p=None, seed=None)
    assert out == expected[len(prompt):]


@pytest.mark.parametrize("temperature, top_k, top_p",
                         [(1.0, None, None), (0.7, 3, None), (1.3, None, 0.8), (1.0, 4, 0.6)])
def test_sampled_control_draws_what_the_model_sampler_draws(
        temperature: float, top_k: Any, top_p: Any) -> None:
    table = _table(seed=1)
    model = _vfe_model(table)
    prompt = [3, 0]
    for seed in (0, 11):
        torch.manual_seed(seed)
        reference = model.generate(torch.tensor([prompt]), 12, temperature=temperature, greedy=False,
                                   top_k=top_k, top_p=top_p)[0, len(prompt):].tolist()
        torch.manual_seed(seed + 1000)                                       # the control reseeds itself
        out = _baseline(table).generate(prompt, 12, greedy=False, temperature=temperature, top_k=top_k,
                                        top_p=top_p, seed=seed)
        assert out == reference


def test_sampled_control_depends_on_the_seed() -> None:
    baseline = _baseline(_table(seed=2, scale=0.1))                          # near-uniform rows
    draws = {tuple(baseline.generate([1], 16, greedy=False, temperature=1.0, top_k=None, top_p=None, seed=seed))
             for seed in (0, 1, 2)}
    assert len(draws) > 1


class _StubBaseline:
    calls: List[Dict[str, Any]] = []

    def __init__(self, name: str, device: str) -> None:
        self.name = name
        self.context = 1024

    def generate(self, prompt_ids: List[int], max_new_tokens: int, **kwargs: Any) -> List[int]:
        _StubBaseline.calls.append(dict(kwargs))
        return [4, 5, 6][:max_new_tokens]

    def parameter_count(self) -> int:
        return 124_000_000


def _stub_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, json_output: bool) -> None:
    vfe_model = torch.nn.Linear(1, 1)
    loaded = generate_text.LoadedRun(vfe_model, SimpleNamespace(max_seq_len=8, eval_stride=None),
                                     "wikitext-103", tmp_path, 1, 10.0)
    settings = dict(RUN_DIR=str(tmp_path), PROMPTS=("p",), SELF_CE=True, BASELINE="gpt2", GREEDY=False,
                    TEMPERATURE=0.7, TOP_K=3, TOP_P=0.9, SEED=5, MAX_NEW_TOKENS=3, REFERENCE_DRAWS=2,
                    BLOCK_TOKENS=None, JSON_OUTPUT=json_output, DEVICE="cpu")
    for name, value in settings.items():
        monkeypatch.setattr(generate_text, name, value)
    _StubBaseline.calls = []
    monkeypatch.setattr(generate_text, "HuggingFaceBaseline", _StubBaseline)
    monkeypatch.setattr(generate_text, "validated_run_dir", lambda path: path)
    monkeypatch.setattr(generate_text, "load_run", lambda run_dir, device: loaded)
    monkeypatch.setattr(generate_text, "generate_for_prompts",
                        lambda *args, **kwargs: [generate_text.Generation("p", "text", [1, 2, 3])])
    monkeypatch.setattr(generate_text, "prompt_token_ids", lambda text, dataset: [0, 1])
    monkeypatch.setattr(generate_text, "decode_ids", lambda ids, dataset: "decoded")
    monkeypatch.setattr(generate_text, "held_out_token_ids",
                        lambda *args, **kwargs: [torch.zeros(10, dtype=torch.long)] * 2)

    def score_streams(forward: Any, *args: Any, **kwargs: Any) -> generate_text.SelfCEScore:
        if isinstance(forward, _StubBaseline):
            return generate_text.SelfCEScore(generated=[2.0], reference=[1.5, 1.7])
        return generate_text.SelfCEScore(generated=[3.0], reference=[2.5, 2.7])

    monkeypatch.setattr(generate_text, "score_streams", score_streams)


def test_main_passes_the_sampler_and_reports_the_baseline_against_its_own_reference(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _stub_main(monkeypatch, tmp_path, json_output=False)
    generate_text.main()
    assert _StubBaseline.calls == [dict(greedy=False, temperature=0.7, top_k=3, top_p=0.9, seed=5)]
    out = capsys.readouterr().out
    assert "gpt2    2.000  (delta +0.400," in out                             # 2.0 - 1.6, not 2.0 - 2.6


def test_json_labels_the_baseline_sampler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                          capsys: pytest.CaptureFixture) -> None:
    _stub_main(monkeypatch, tmp_path, json_output=True)
    generate_text.main()
    payload = json.loads(capsys.readouterr().out)
    baseline = payload["self_ce"]["baseline"]
    assert baseline["sampler"] == payload["sampler"] == "temperature=0.7, top_k=3, top_p=0.9, seed=5"
    assert baseline["generated_delta"] == pytest.approx([0.4])


def test_greedy_main_passes_no_seed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub_main(monkeypatch, tmp_path, json_output=True)
    monkeypatch.setattr(generate_text, "GREEDY", True)
    generate_text.main()
    assert _StubBaseline.calls[0]["greedy"] is True and _StubBaseline.calls[0]["seed"] is None
