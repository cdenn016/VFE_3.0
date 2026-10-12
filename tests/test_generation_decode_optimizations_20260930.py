r"""Tiny generation checks for validation decisions and scoped fixed-bank preparation."""

from tests.profiles import test_config

import inspect
from typing import Any, Optional

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vfe3.model.model import VFEModel


def _tiny_model(decode_mode: str = "diagonal_chunked") -> VFEModel:
    family = "gaussian_full" if decode_mode == "full_chunked" else "gaussian_diagonal"
    cfg = test_config(vocab_size=7, embed_dim=4, n_heads=2, n_layers=1, n_e_steps=1,
                    max_seq_len=4, batch_size=2, max_steps=1, warmup_steps=1, family=family,
                    attention_mode="projected_predictive",
                    predictive_covariance="moment_matched",
                    decode_mode=decode_mode, decode_chunk_size=3, untie_decode_bank=True,
                    decode_coordinates="local", decode_unigram_prior=False,
                    use_block_mlp=False, norm_type_block="none", seed=6)
    torch.manual_seed(6)
    return VFEModel(cfg).eval()


class _BankMeanCounter(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.preparations = 0

    def __torch_dispatch__(
        self,
        func:   Any,
        types:  tuple[type, ...],
        args:   tuple[Any, ...]          = (),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> Any:
        if (func == torch.ops.aten.mean.dim and args[0].shape == (7, 4)
                and list(args[1]) == [0]):
            self.preparations += 1
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("greedy,top_k,top_p", [(True, None, None), (False, None, None),
                                               (False, 2, .8)])
def test_healthy_generation_uses_one_validation_decision(
    monkeypatch: pytest.MonkeyPatch,
    greedy:      bool,
    top_k:       Optional[int],
    top_p:       Optional[float],
) -> None:
    model = _tiny_model()
    logits = torch.tensor([[[0., 1., 2., 3., 4., 5., 6.]],
                           [[6., 5., 4., 3., 2., 1., 0.]]])
    monkeypatch.setattr(model, "forward_beliefs", lambda *args, **kwargs: (None, logits))
    decisions, scalar_bools = [], []
    original_tolist, original_bool = torch.Tensor.tolist, torch.Tensor.__bool__

    def tracked_tolist(tensor: torch.Tensor) -> Any:
        if inspect.currentframe().f_back.f_code.co_name == "generate":
            decisions.append(tuple(tensor.shape))
        return original_tolist(tensor)

    def tracked_bool(tensor: torch.Tensor) -> bool:
        if inspect.currentframe().f_back.f_code.co_name == "generate":
            scalar_bools.append(tuple(tensor.shape))
        return original_bool(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", tracked_tolist)
    monkeypatch.setattr(torch.Tensor, "__bool__", tracked_bool)
    model.generate(torch.tensor([[1, 2], [2, 1]]), 3, greedy=greedy, top_k=top_k, top_p=top_p)
    assert scalar_bools == []
    assert decisions == [(2,) if greedy else (3,)] * 3


@pytest.mark.parametrize("greedy", [False, True])
@pytest.mark.parametrize("kind,message", [
    ("nan", r"generation logits contain NaN or \+inf values in rows \[0\]"),
    ("posinf", r"generation logits contain NaN or \+inf values in rows \[0\]"),
    ("empty", r"generation logits have no finite value in rows \[0\]"),
])
def test_generation_preserves_original_logit_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    greedy:      bool,
    kind:        str,
    message:     str,
) -> None:
    model = _tiny_model()
    logits = torch.zeros(2, 1, 7)
    logits[0, 0, 0] = {"nan": float("nan"), "posinf": float("inf"), "empty": float("-inf")}[kind]
    if kind == "empty":
        logits[0] = float("-inf")
    monkeypatch.setattr(model, "forward_beliefs", lambda *args, **kwargs: (None, logits))
    with pytest.raises(ValueError, match=message):
        model.generate(torch.tensor([[1], [2]]), 1, greedy=greedy, top_k=2, top_p=.8)


@pytest.mark.parametrize("kind", ["partial_mask", "division_overflow"])
def test_sampling_rejects_invalid_retained_logits_before_nucleus(
    monkeypatch: pytest.MonkeyPatch,
    kind:        str,
) -> None:
    model = _tiny_model()
    logits = torch.zeros(2, 1, 7)
    if kind == "partial_mask":
        logits[0, 0, -1] = float("-inf")
        temperature = 1.
    else:
        logits[0, 0, 0] = torch.finfo(torch.float32).max
        temperature = .25
    monkeypatch.setattr(model, "forward_beliefs", lambda *args, **kwargs: (None, logits))
    with pytest.raises(ValueError, match=r"generation retained logits contain non-finite values in rows \[0\]"):
        model.generate(torch.tensor([[1], [2]]), 1, temperature=temperature, top_p=.5)


def test_generation_accepts_greedy_and_filtered_negative_infinity(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _tiny_model()
    logits = torch.tensor([[[0., 1., 2., float("-inf"), float("-inf"), float("-inf"), float("-inf")]]])
    monkeypatch.setattr(model, "forward_beliefs", lambda *args, **kwargs: (None, logits))
    prompt = torch.tensor([[1]])
    assert model.generate(prompt, 1, greedy=True)[0, -1] == 2
    assert model.generate(prompt, 1, top_k=1, top_p=.5)[0, -1] == 2


@pytest.mark.parametrize("decode_mode", ["diagonal_chunked", "full_chunked"])
@pytest.mark.parametrize("greedy", [False, True])
def test_generation_reuses_bank_preparation_with_exact_uncached_parity(
    monkeypatch: pytest.MonkeyPatch,
    greedy:      bool,
    decode_mode: str,
) -> None:
    model = _tiny_model(decode_mode)
    prompt = torch.tensor([[1, 2], [3, 4]])
    expected, expected_logits = prompt, []
    torch.manual_seed(31)
    uncached_counter = _BankMeanCounter()
    with torch.no_grad(), uncached_counter:
        for _ in range(4):
            _, logits = model.forward_beliefs(expected[:, -4:], return_logits=True, decode_last=True)
            expected_logits.append(logits.clone())
            last = logits[:, 0]
            if greedy:
                next_token = last.argmax(-1, keepdim=True)
            else:
                scaled = last / .7
                kth = scaled.topk(2, dim=-1).values[:, -1:]
                scaled = scaled.masked_fill(scaled < kth, float("-inf"))
                sorted_logits, indices = scaled.sort(descending=True, dim=-1)
                probabilities = sorted_logits.softmax(-1)
                remove = probabilities.cumsum(-1) - probabilities >= .8
                scaled = scaled.masked_fill(remove.scatter(-1, indices, remove), float("-inf"))
                next_token = torch.multinomial(scaled.softmax(-1), 1)
            expected = torch.cat((expected, next_token), dim=-1)
    actual_logits = []
    original_forward = model.forward_beliefs

    def tracked_forward(*args: Any, **kwargs: Any) -> Any:
        result = original_forward(*args, **kwargs)
        actual_logits.append(result[1].clone())
        return result

    monkeypatch.setattr(model, "forward_beliefs", tracked_forward)
    cached_counter = _BankMeanCounter()
    torch.manual_seed(31)
    with cached_counter:
        actual = model.generate(prompt, 4, greedy=greedy, temperature=.7, top_k=2, top_p=.8)
    assert uncached_counter.preparations == 4
    assert cached_counter.preparations == 1
    assert not hasattr(model.prior_bank, "_fixed_decode_bank_cache")
    assert torch.equal(actual, expected)
    assert all(torch.equal(actual_logit, expected_logit)
               for actual_logit, expected_logit in zip(actual_logits, expected_logits))


def test_generation_discards_fixed_bank_scope_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _tiny_model()
    prompt = torch.tensor([[1, 2]])
    original_forward = model.forward_beliefs
    calls = 0

    def interrupted_forward(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected forward interruption")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward_beliefs", interrupted_forward)
    first_counter = _BankMeanCounter()
    with first_counter, pytest.raises(RuntimeError, match="injected forward interruption"):
        model.generate(prompt, 3, greedy=True)
    assert first_counter.preparations == 1
    assert not hasattr(model.prior_bank, "_fixed_decode_bank_cache")
    monkeypatch.setattr(model, "forward_beliefs", original_forward)
    with torch.no_grad():
        model.prior_bank._decode_mu_table().add_(.125)
    second_counter = _BankMeanCounter()
    with second_counter:
        regenerated = model.generate(prompt, 2, greedy=True)
    assert second_counter.preparations == 1
    assert not hasattr(model.prior_bank, "_fixed_decode_bank_cache")
    assert regenerated.shape == (1, 4)
