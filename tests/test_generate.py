"""Tests for VFEModel.generate -- the additive autoregressive sampler.

generate reuses VFEModel.forward (targets=None -> logits (B, N, V)), takes the
last-position logits, turns them into a next token (greedy or sampled), appends,
and repeats. It never touches the training/loss branch, so it cannot corrupt
training; that training-isolation is the safety oracle below.

Forward-comparison oracles (greedy == forward+argmax, top_k membership) hold only
for the FIRST generated token: generate's step-1 internal forward sees exactly the
prompt, so logits[:, -1, :] there equals forward(prompt)[:, -1, :]. Step 2+
conditions on a longer sequence and has no relation to forward(prompt). Those
oracles therefore use max_new_tokens=1.
"""

import os

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel

PROJECTED_DEVICE = torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))


def _tiny_model(seed: int = 0, **overrides) -> VFEModel:
    """A tiny fixed-seed VFEModel. vocab_size is kept comfortably larger than any
    top_k used below so torch.topk never sees k > vocab_size."""
    base = dict(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, e_q_mu_lr=0.05, e_phi_lr=0.0, seed=seed)
    base.update(overrides)
    cfg = VFE3Config(**base)
    torch.manual_seed(seed)
    return VFEModel(cfg)


def _projected_tiny_model(seed: int = 0, **overrides) -> VFEModel:
    """A tiny legal projected-canonical model for decode/generation consumers."""
    base = dict(
        vocab_size=16,
        embed_dim=4,
        n_heads=2,
        max_seq_len=8,
        batch_size=2,
        n_layers=1,
        n_e_steps=1,
        e_step_update="gradient",
        oracle_unroll_grad=True,
        e_q_mu_lr=0.05,
        e_phi_lr=0.0,
        family="gaussian_diagonal",
        transport_mode="flat",
        decode_mode="full_chunked",
        encode_mode="canonical_content_projected",
        lambda_alpha_mode="constant",
        pos_phi="learned",
        seed=seed,
        max_steps=1,
    )
    base.update(overrides)
    torch.manual_seed(seed)
    return VFEModel(VFE3Config(**base)).to(PROJECTED_DEVICE)


def test_shape_in_vocab_and_prompt_preserved():
    model = _tiny_model()
    V = model.cfg.vocab_size
    B, N0 = 2, 3
    prompt = torch.randint(0, V, (B, N0))
    out = model.generate(prompt, max_new_tokens=5)
    assert out.shape == (B, N0 + 5)
    assert (out >= 0).all() and (out < V).all()            # every id in [0, V)
    assert torch.equal(out[:, :N0], prompt)                # prompt preserved


def test_greedy_is_deterministic():
    model = _tiny_model()
    V = model.cfg.vocab_size
    prompt = torch.randint(0, V, (2, 3))
    a = model.generate(prompt, max_new_tokens=4, greedy=True)
    b = model.generate(prompt, max_new_tokens=4, greedy=True)
    assert torch.equal(a, b)


def test_greedy_equals_forward_argmax_first_token():
    # The pin: generate's first greedy token == argmax of the last-position logits
    # from a direct forward on the SAME prompt (no hidden divergence). First token
    # only -- step 2+ conditions on a longer sequence. Prompt <= max_seq_len so the
    # internal forward and this comparison forward see identical input.
    model = _tiny_model()
    V = model.cfg.vocab_size
    prompt = torch.randint(0, V, (2, 3))
    logits = model(prompt)                                 # (B, N0, V), targets=None
    expected = logits[:, -1, :].argmax(dim=-1)             # (B,)
    out = model.generate(prompt, max_new_tokens=1, greedy=True)
    assert torch.equal(out[:, -1], expected)


def test_generate_decodes_only_last_position(monkeypatch):
    model = _tiny_model()
    V = model.cfg.vocab_size
    prompt = torch.randint(0, V, (2, 3))
    empty = torch.empty((2, 0), dtype=torch.long)
    with pytest.raises(ValueError, match=r"decode_last=True requires a nonempty token context"):
        model.forward_beliefs(empty, return_logits=True, decode_last=True)
    with pytest.raises(ValueError, match=r"generate requires a nonempty token context"):
        model.generate(empty, max_new_tokens=1, greedy=True)
    original_forward_beliefs = model.forward_beliefs
    calls = []

    def tracked_forward_beliefs(
        token_ids,
        *,
        return_logits=False,
        decode_last=False,
        **kwargs,
    ):
        belief, logits = original_forward_beliefs(
            token_ids,
            return_logits=return_logits,
            decode_last=decode_last,
            **kwargs,
        )
        # forward_beliefs no longer takes a `training` kwarg, and nothing under vfe3/ reads
        # self.training: there is no train/eval branch to select, so record the decode shape only.
        calls.append((return_logits, decode_last, tuple(logits.shape)))
        return belief, logits

    monkeypatch.setattr(model, "forward_beliefs", tracked_forward_beliefs)
    out = model.generate(prompt, max_new_tokens=2, greedy=True)

    assert out.shape == (2, 5)
    assert calls == [
        (True, True, (2, 1, V)),
        (True, True, (2, 1, V)),
    ]


def test_projected_decode_last_matches_full_last_position_for_multitoken_context():
    """The last query and its frame must be sliced together, never from the first position."""
    model = _projected_tiny_model(seed=17)
    model.eval()
    prompt = torch.tensor([[2, 7, 4], [5, 1, 9]], device=PROJECTED_DEVICE)
    capture: dict = {}

    _, full_logits = model.forward_beliefs(
        prompt,
        return_logits=True,
        capture=capture,
    )
    _, last_logits = model.forward_beliefs(
        prompt,
        return_logits=True,
        decode_last=True,
    )

    context = capture["canonical_frame"]
    assert full_logits is not None and full_logits.shape == (2, 3, model.cfg.vocab_size)
    assert last_logits is not None and last_logits.shape == (2, 1, model.cfg.vocab_size)
    assert not torch.allclose(context.inverse[:, :1], context.inverse[:, -1:])
    assert torch.allclose(last_logits, full_logits[:, -1:], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(last_logits, full_logits[:, :1], atol=1e-6, rtol=1e-6)


def test_projected_generation_recomputes_context_after_sequence_growth():
    """Every greedy step must decode the last position of its newly grown same-forward context."""
    model = _projected_tiny_model(seed=23)
    model.eval()
    prompt = torch.tensor([[2, 7, 4], [5, 1, 9]], device=PROJECTED_DEVICE)
    expected = prompt.clone()

    for _ in range(2):
        capture: dict = {}
        _, full_logits = model.forward_beliefs(
            expected,
            return_logits=True,
            capture=capture,
        )
        assert full_logits is not None
        context = capture["canonical_frame"]
        assert context.inverse.shape[1] == expected.shape[1]
        next_token = full_logits[:, -1].argmax(dim=-1, keepdim=True)
        expected = torch.cat([expected, next_token], dim=1)

    generated = model.generate(prompt, max_new_tokens=2, greedy=True)
    assert torch.equal(generated, expected)


def test_generate_rejects_nonfinite_logit_rows(monkeypatch):
    model = _tiny_model()
    V = model.cfg.vocab_size
    prompt = torch.randint(0, V, (2, 3))
    injected = {"logits": torch.zeros(2, 1, V)}

    def fake_forward_beliefs(
        token_ids,
        *,
        return_logits=False,
        decode_last=False,
        **kwargs,
    ):
        assert return_logits and decode_last
        return None, injected["logits"]

    monkeypatch.setattr(model, "forward_beliefs", fake_forward_beliefs)

    injected["logits"] = torch.zeros(2, 1, V)
    injected["logits"][0, 0, 0] = float("nan")
    injected["logits"][1, 0, 1] = float("inf")
    with pytest.raises(
        ValueError,
        match=r"generation logits contain NaN or \+inf values in rows \[0, 1\]",
    ):
        model.generate(prompt, max_new_tokens=1, greedy=True)

    injected["logits"] = torch.zeros(2, 1, V)
    injected["logits"][0] = float("-inf")
    with pytest.raises(ValueError, match=r"generation logits have no finite value in rows \[0\]"):
        model.generate(prompt, max_new_tokens=1, greedy=False, top_k=2)

    injected["logits"] = torch.zeros(2, 1, V)
    injected["logits"][0, 0, 4:] = float("-inf")
    out = model.generate(prompt, max_new_tokens=1, greedy=False, top_k=2)
    assert out.shape == (2, 4)                              # filtered-out -inf remains valid
    with pytest.raises(
        ValueError,
        match=r"generation retained logits contain non-finite values in rows \[0\]",
    ):
        model.generate(prompt, max_new_tokens=1, greedy=False)


def test_greedy_ignores_temperature_topk_topp():
    # Greedy returns before any temperature/top_k/top_p logic: a wild temperature
    # and aggressive top_k/top_p alongside greedy=True must not change the result.
    model = _tiny_model()
    V = model.cfg.vocab_size
    prompt = torch.randint(0, V, (2, 3))
    plain = model.generate(prompt, max_new_tokens=4, greedy=True)
    wild = model.generate(prompt, max_new_tokens=4, greedy=True,
                          temperature=137.0, top_k=1, top_p=0.01)
    assert torch.equal(plain, wild)


def test_top_k_one_is_argmax_first_token():
    # top_k=1 (not greedy) leaves a single survivor -> softmax mass 1.0 -> multinomial
    # returns it deterministically; it equals the argmax token. No seed needed.
    model = _tiny_model()
    V = model.cfg.vocab_size
    prompt = torch.randint(0, V, (2, 3))
    logits = model(prompt)
    expected = logits[:, -1, :].argmax(dim=-1)
    out = model.generate(prompt, max_new_tokens=1, greedy=False, top_k=1)
    assert torch.equal(out[:, -1], expected)


def test_top_k_membership_first_token():
    # The first sampled token must lie among the k largest of the last-position logits.
    model = _tiny_model()
    V = model.cfg.vocab_size
    k = 3
    prompt = torch.randint(0, V, (2, 3))
    logits = model(prompt)
    topk_ids = logits[:, -1, :].topk(k, dim=-1).indices     # (B, k)
    torch.manual_seed(0)
    out = model.generate(prompt, max_new_tokens=1, greedy=False, top_k=k)
    chosen = out[:, -1]                                     # (B,)
    assert (chosen.unsqueeze(-1) == topk_ids).any(dim=-1).all()


def test_top_p_and_temperature_paths_run_in_vocab():
    model = _tiny_model()
    V = model.cfg.vocab_size
    prompt = torch.randint(0, V, (2, 3))
    torch.manual_seed(0)
    out_p = model.generate(prompt, max_new_tokens=3, greedy=False, top_p=0.9)
    out_t = model.generate(prompt, max_new_tokens=3, greedy=False, temperature=0.7)
    for out in (out_p, out_t):
        assert out.shape == (2, 3 + 3)
        assert (out >= 0).all() and (out < V).all()


def test_prompt_longer_than_max_seq_len_does_not_error():
    # A prompt longer than max_seq_len must not error: the loop truncates to the last
    # max_seq_len tokens before each forward. The returned sequence keeps the FULL prompt.
    model = _tiny_model()
    V = model.cfg.vocab_size
    L = model.cfg.max_seq_len
    long_prompt = torch.randint(0, V, (2, L + 4))           # longer than max_seq_len
    out = model.generate(long_prompt, max_new_tokens=2, greedy=True)
    assert out.shape == (2, (L + 4) + 2)
    assert torch.equal(out[:, : L + 4], long_prompt)        # full prompt preserved
    assert (out >= 0).all() and (out < V).all()
def test_generate_is_training_isolated():
    # The safety oracle: generate changes no parameter and does not break training.
    model = _tiny_model()
    V = model.cfg.vocab_size
    before = model.prior_bank.mu_embed.detach().clone()
    prompt = torch.randint(0, V, (2, 3))
    torch.manual_seed(0)
    model.generate(prompt, max_new_tokens=4, greedy=False, top_k=2, top_p=0.9)
    after = model.prior_bank.mu_embed.detach().clone()
    assert torch.equal(before, after)                       # no parameter changed
    # training forward still produces a finite loss after a generate call
    tokens = torch.randint(0, V, (2, 4))
    targets = torch.randint(0, V, (2, 4))
    _, loss, _ = model(tokens, targets)
    assert torch.isfinite(loss)


def test_generate_rejects_invalid_sampler_args():
    # The sampler validates its knobs up front. A negative max_new_tokens would silently no-op
    # (empty loop, prompt returned unchanged);
    # temperature<=0, out-of-range top_k, and top_p outside (0, 1] previously failed late or produced
    # invalid probabilities.
    model = _tiny_model()
    V = model.cfg.vocab_size
    prompt = torch.randint(0, V, (2, 3))
    with pytest.raises(ValueError):
        model.generate(prompt, -1)
    with pytest.raises(ValueError):
        model.generate(prompt, 4, temperature=0.0)
    for bad_k in (0, V + 1):
        with pytest.raises(ValueError):
            model.generate(prompt, 4, top_k=bad_k)
    for bad_p in (0.0, 1.5):
        with pytest.raises(ValueError):
            model.generate(prompt, 4, top_p=bad_p)
    # max_new_tokens=0 stays valid (>= 0): the prompt comes back unchanged
    assert torch.equal(model.generate(prompt, 0), prompt)
    # greedy ignores the sampler knobs, so a nonstandard temperature must still run under greedy
    out = model.generate(prompt, 2, greedy=True, temperature=0.0)
    assert out.shape == (2, 3 + 2)


def test_generate_kwargs_order():
    # audit C18 (2026-07-01): the keyword-only block was reordered to the mandated convention
    # (defined float, defined bool, then Optionals); every knob stays keyword-passable in any order.
    model = _tiny_model()
    V = model.cfg.vocab_size
    prompt = torch.randint(0, V, (1, 3))
    torch.manual_seed(0)
    out = model.generate(prompt, max_new_tokens=2, greedy=False, top_k=2, top_p=0.9, temperature=0.7)
    assert out.shape == (1, 3 + 2)
    assert (out >= 0).all() and (out < V).all()
