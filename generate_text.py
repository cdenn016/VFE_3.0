r"""Click-to-run: load a finalized run's best model and generate text.

Points at a run directory written by ``train_vfe3.py`` (``config.json`` + ``best_model.pt``),
rebuilds the selected model through the same validated load boundary the figure pipeline uses
(:mod:`vfe3.viz.run_loading`), and autoregressively extends prompts with
:meth:`vfe3.model.model.VFEModel.generate`.

The prompt is tokenized with the tokenizer the run's corpus cache was built with -- GPT-2 BPE for
``wikitext-103``, cl100k for the ``wiki-*`` corpora, the word vocabulary for the ``-word`` corpora --
so generated ids map back to text consistently.

Two properties of the sampler are worth knowing before reading the output. First, there is no
incremental belief/KV cache: every generated token re-runs the full forward (encode -> E-step ->
decode) over the whole context window, so cost grows as
``max_new_tokens * forward(max_seq_len)``. Second, the context is truncated to the last
``cfg.max_seq_len`` tokens at each step, so a prompt longer than that window is generated from its
tail alone even though the returned sequence keeps the full prompt.

Precision is left exactly as the run configured it: the weights stay fp32 and the model applies its
own ``cfg.amp_dtype`` autocast internally (a bf16 E-step with fp32 decode/CE islands), so there is
no dtype knob here to set.

``--self-ce`` adds the diagnostic that separates a sampler artifact from a model property. Each
generated continuation is scored teacher-forced against its own prompt, and compared with held-out
corpus slices of the same length that carry an equal window of preceding text. Both go through the
identical scorer, which anchors its blocks at the end of the stream so the scored tail keeps as much
context as ``max_seq_len`` allows and every tail token is counted exactly once. A self-CE well below
the held-out CE means the model has a sharp mode its own sampler keeps falling into; a self-CE at or
above it means the loop is not a preferred mode at all, and greedy is walking a flat, uninformative
distribution. Note that the held-out reference is the model's own, not a ceiling: this script scores
one model against itself, and says nothing about how its held-out CE compares to another model's.

Usage
-----
    python generate_text.py
    python generate_text.py --prompt "The history of the city" --max-new-tokens 64
    python generate_text.py --greedy
    python generate_text.py --prompt "..." --temperature 0.8 --top-k 40 --top-p 0.95 --seed 1
    python generate_text.py --self-ce --greedy --max-new-tokens 128
    python generate_text.py --self-ce --reference-draws 8 --split test
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence

import torch
import torch.nn.functional as F

RUN_ROOT = "vfe3_runs"
DEFAULT_RUN_DIR = str(Path(RUN_ROOT) / "31.04_wikitext-103_K256_4-layer_noHM")

# WikiText-flavored openers used when no --prompt is supplied.
DEFAULT_PROMPTS = (
    "The history of the city",
   # "In 1923 , the company",
   # "The film was released on",
)

PROMPT_PLACEHOLDER = "<PROMPT>"


class LoadedRun(NamedTuple):
    """A rebuilt model together with the identity of the run it came from."""

    model:   "torch.nn.Module"
    cfg:     object
    dataset: str
    run_dir: Path
    best_step:    Optional[int]
    best_val_ppl: Optional[float]


class Generation(NamedTuple):
    """One prompt's decoded continuation, with the exact token ids behind it.

    The ids are kept so ``--self-ce`` scores what the sampler actually emitted rather than a
    decode-then-re-encode round trip, which BPE does not guarantee to be an identity.
    """

    prompt:    str
    text:      str
    token_ids: List[int]


class SelfCEScore(NamedTuple):
    """Teacher-forced cross-entropy of each continuation against a length-matched corpus reference."""

    generated: List[float]      # one nats/token per prompt
    reference: List[float]      # one nats/token per held-out draw

    def summary(self) -> "tuple[float, float, float]":
        r"""Reference mean and (min, max), so the draw-to-draw spread stays visible."""
        return (sum(self.reference) / len(self.reference),
                min(self.reference), max(self.reference))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    r"""Parse the command line; sampling knobs mirror :meth:`VFEModel.generate`."""
    parser = argparse.ArgumentParser(
        description="Load a finalized run's best model and generate text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR,
                        help="run directory holding config.json and best_model.pt")
    parser.add_argument("--prompt", action="append", default=None, metavar="TEXT",
                        help=f"prompt to continue; repeatable, and may contain {PROMPT_PLACEHOLDER} "
                             "to mark where the generated continuation is spliced in "
                             "(default: a few WikiText openers)")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="number of tokens to generate per prompt")
    parser.add_argument("--greedy", action="store_true",
                        help="take the argmax each step and ignore temperature/top-k/top-p")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="logit temperature on the sampled path")
    parser.add_argument("--top-k", type=int, default=10, help="keep only the k largest logits")
    parser.add_argument("--top-p", type=float, default=None, help="nucleus sampling threshold")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed (ignored when --greedy)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="torch device for the forward pass")
    parser.add_argument("--self-ce", action="store_true",
                        help="score each generated continuation teacher-forced and compare it "
                             "against held-out corpus text of matching length")
    parser.add_argument("--reference-draws", type=int, default=4,
                        help="held-out slices to average the reference cross-entropy over "
                             "(with --self-ce)")
    parser.add_argument("--split", default="validation",
                        help="corpus split for the held-out reference (with --self-ce)")
    parser.add_argument("--baseline", choices=("none", "gpt2"), default="none",
                        help="also run this model through the identical protocol, to calibrate the "
                             "self-CE delta against a reference LM (requires --self-ce)")
    parser.add_argument("--block-tokens", type=int, default=None, metavar="N",
                        help="scoring block length for --baseline (default: the baseline's own "
                             "context window). The run's model always scores at its trained "
                             "max_seq_len, whose positional table does not extrapolate")
    parser.add_argument("--json", action="store_true",
                        help="emit results as JSON instead of formatted text")
    return parser.parse_args(argv)


def validated_run_dir(path: Path) -> Path:
    r"""Return ``path`` if it holds the finalized-run artifacts, else raise a precise error."""
    required = ("config.json", "best_model.pt")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"run directory {path} is incomplete; missing {', '.join(missing)}")
    if (path / "best_model.pt").stat().st_size == 0:
        raise ValueError(f"run directory {path} has an empty best_model.pt")
    return path


def _run_summary(run_dir: Path) -> "tuple[Optional[int], Optional[float]]":
    r"""Best-selection step and validation perplexity from ``summary.json``, when present."""
    path = run_dir / "summary.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    step = payload.get("best_step")
    ppl = payload.get("best_val_ppl")
    return (step if type(step) is int else None,
            float(ppl) if isinstance(ppl, (int, float)) and not isinstance(ppl, bool) else None)


def load_run(run_dir: Path, device: str) -> LoadedRun:
    r"""Rebuild the selected best model for ``run_dir`` through the validated load boundary.

    The config is deserialized with :func:`vfe3.config.config_from_serialized`, which migrates the
    frozen field set it was written with, and the weights pass the same
    executable-code/validation-data identity contract the offline figure pipeline enforces.
    """
    from vfe3.model.model import VFEModel
    from vfe3.viz.run_loading import (
        load_best_model_state,
        load_run_config,
        load_run_selection_contract,
    )

    cfg, dataset = load_run_config(run_dir)
    code_identity, selection_identity = load_run_selection_contract(run_dir)
    # Construction only draws RNG for parameter init, every one of which the checkpoint overwrites;
    # fork_rng keeps that draw from perturbing the caller's global stream (mirrors make_figures.py).
    with torch.random.fork_rng(devices=[]):
        model = VFEModel(cfg)
    state = load_best_model_state(
        run_dir / "best_model.pt",
        cfg,
        model.state_dict(),
        code_identity,
        selection_identity,
        map_location="cpu",
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device=device)
    model.eval()
    best_step, best_val_ppl = _run_summary(run_dir)
    return LoadedRun(model, cfg, dataset, run_dir, best_step, best_val_ppl)


def prompt_token_ids(text: str, dataset: str) -> List[int]:
    r"""Encode ``text`` with the tokenizer that built ``dataset``'s cache."""
    from vfe3.data.datasets import tiktoken_encoding_name

    name = tiktoken_encoding_name(dataset)
    if name == "word":
        raise ValueError(
            f"dataset {dataset!r} is word-level; this script encodes prompts with a BPE tokenizer")
    import tiktoken

    # encode_ordinary, not encode: the default disallowed_special="all" raises on a literal
    # "<|endoftext|>" in user text, and training never prepends or reserves special tokens anyway.
    return tiktoken.get_encoding(name).encode_ordinary(text)


def decode_ids(token_ids: Sequence[int], dataset: str) -> str:
    r"""Decode ``token_ids`` back to text with the run corpus's own decoder."""
    from vfe3.data.datasets import get_tiktoken_decoder

    decoder = get_tiktoken_decoder(dataset)
    if decoder is None:
        raise RuntimeError(f"no decoder is available for dataset {dataset!r}")
    return decoder(list(token_ids))


@torch.no_grad()
def generate_for_prompts(
    loaded:        LoadedRun,
    prompts:       Sequence[str],
    max_new_tokens: int,

    *,
    greedy:      bool,
    temperature: float,
    top_k:       Optional[int],
    top_p:       Optional[float],
    seed:        Optional[int],
) -> List[Generation]:
    r"""Return one decoded continuation per prompt, prompt text excluded."""
    device = next(loaded.model.parameters()).device
    generations: List[Generation] = []
    for prompt in prompts:
        ids = prompt_token_ids(prompt, loaded.dataset)
        if not ids:
            raise ValueError("prompt encodes to zero tokens; supply non-empty text")
        context = torch.tensor([ids], dtype=torch.long, device=device)
        if seed is not None:
            torch.manual_seed(seed)
        out = loaded.model.generate(
            context,
            max_new_tokens,
            temperature=temperature,
            greedy=greedy,
            top_k=top_k,
            top_p=top_p,
        )
        continuation_ids = out[0, context.shape[1]:].tolist()
        generations.append(Generation(
            prompt=prompt,
            text=decode_ids(continuation_ids, loaded.dataset),
            token_ids=continuation_ids,
        ))
    return generations


@torch.no_grad()
def stream_ce(
    model:   "torch.nn.Module",
    prefix:  torch.Tensor,          # (P,) the context the text was produced from; may be empty
    tail:    torch.Tensor,          # (T,) the tokens actually being scored
    seq_len: int,
) -> float:
    r"""Teacher-forced mean cross-entropy (nats/token) of ``tail`` given ``prefix``.

    Blocks are anchored at the END of ``prefix + tail``, so the scored tail always carries as much
    preceding context as ``seq_len`` allows. Scoring the tail cold instead would let the
    no-context first position dominate whenever ``tail`` is short, which is an artifact of the
    measurement rather than a property of the text. Each tail token is scored exactly once, so two
    texts of equal length are directly comparable.
    """
    n_prefix = int(prefix.shape[0])
    n_tail = int(tail.shape[0])
    if n_tail < 1:
        raise ValueError("stream_ce needs at least one scored token")
    combined = torch.cat([prefix, tail]) if n_prefix else tail
    device = next(model.parameters()).device
    total_nats = 0.0
    total_tokens = 0
    end = n_prefix + n_tail - 1     # last target index; position i is scored as the target of i-1
    while end > n_prefix:
        start = max(0, end - seq_len)
        inputs = combined[start:end].unsqueeze(0).to(device)          # (1, end - start)
        targets = combined[start + 1:end + 1].unsqueeze(0).to(device)
        logits = model(inputs)                                        # targets=None inference path
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            targets.reshape(-1),
            reduction="none",
        )
        # per_token[k] scores the token at index start + 1 + k; drop the prefix-predicting head.
        offset = max(0, n_prefix - start - 1)
        scored = per_token[offset:]
        total_nats += float(scored.sum())
        total_tokens += int(scored.numel())
        end = start
    if total_tokens == 0:
        raise ValueError("stream_ce scored no tokens")
    return total_nats / total_tokens


def held_out_token_ids(
    dataset:  str,
    split:    str,
    n_tokens: int,
    draws:    int,

    *,
    warmup: int,
) -> List[torch.Tensor]:
    r"""``draws`` disjoint corpus slices of ``n_tokens`` scored tokens plus ``warmup`` of context.

    Tail positions are FIXED (spaced across the whole stream) and the warmup extends *backwards*
    from them, so changing the block length changes only how much context each scored token sees,
    never WHICH text is scored. That is what makes two block lengths comparable: any difference
    between them is context, not a different sample of corpus.
    """
    from vfe3.data.datasets import load_cached_tokens

    if n_tokens < 1:
        raise ValueError("the held-out reference needs at least one token per draw")
    if draws < 1:
        raise ValueError("the held-out reference needs at least one draw")
    # Load uncapped and space the tails across the WHOLE stream. Taking the first draws * span
    # instead would score one contiguous region, and WikiText documents differ enough in difficulty
    # that the head of a split is not a representative reference.
    stream = load_cached_tokens(dataset, split=split)
    total = int(stream.shape[0])
    step = total // (draws + 1)
    if step < max(0, warmup):
        raise ValueError(
            f"{dataset}/{split} ({total} tokens) is too short to place {draws} reference tails "
            f"with {warmup} tokens of warmup each; reduce --reference-draws or --block-tokens")
    # Fewer tails than requested is a coverage cap, so it is reported rather than silently
    # absorbed: the caller prints len(reference) back.
    available = min(draws, max(0, (total - n_tokens) // step))
    if available < 1:
        raise ValueError(
            f"{dataset}/{split} holds {total} tokens, too few for one {n_tokens}-token reference")
    return [
        stream[(i + 1) * step - max(0, warmup):(i + 1) * step + n_tokens]
        for i in range(available)
    ]


@torch.no_grad()
def score_streams(
    forward:          object,             # inputs -> (1, L, V) logits, and a .parameters() iterator
    prompt_ids:       Sequence[List[int]],
    continuation_ids: Sequence[List[int]],
    draws:            Sequence[torch.Tensor],
    max_new_tokens:   int,
    seq_len:          int,
) -> SelfCEScore:
    r"""Score each continuation and the SHARED held-out draws through one scorer.

    ``draws`` is passed in rather than fetched here so a baseline scores the identical corpus
    slices, which is what makes two models' numbers comparable at all.
    """
    generated = [
        stream_ce(forward, torch.tensor(prefix, dtype=torch.long),
                  torch.tensor(tail, dtype=torch.long), seq_len)
        for prefix, tail in zip(prompt_ids, continuation_ids)
    ]
    reference = [
        stream_ce(forward, draw[:-max_new_tokens], draw[-max_new_tokens:], seq_len)
        for draw in draws
    ]
    return SelfCEScore(generated=generated, reference=reference)


class HuggingFaceBaseline:
    r"""A HuggingFace causal LM behind the ``(1, L, V)`` logits call the scorer expects.

    The tiktoken ``gpt2`` encoding is id-for-id the GPT-2 BPE, so the ids the run's own tokenizer
    produces are valid GPT-2 inputs with no re-tokenization; the held-out slices and the prompts are
    therefore scored on exactly the text the VFE model saw.
    """

    def __init__(self, name: str, device: str) -> None:
        from transformers import GPT2LMHeadModel

        self.name = name
        self._model = GPT2LMHeadModel.from_pretrained(name).to(device).eval()
        self.context = int(self._model.config.n_positions)

    def parameters(self):
        return self._model.parameters()

    @torch.no_grad()
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._model(inputs).logits

    @torch.no_grad()
    def generate(self, prompt_ids: List[int], max_new_tokens: int) -> List[int]:
        r"""Greedy with a full recompute per step -- the same protocol ``VFEModel.generate`` uses."""
        device = next(self.parameters()).device
        seq = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            logits = self._model(seq[:, -self.context:]).logits[:, -1, :]
            seq = torch.cat([seq, logits.argmax(dim=-1, keepdim=True)], dim=1)
        return seq[0, len(prompt_ids):].tolist()

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def render(prompt: str, continuation: str) -> str:
    r"""Join a prompt and its continuation, honoring an explicit ``<PROMPT>`` splice marker."""
    if PROMPT_PLACEHOLDER in prompt:
        return prompt.replace(PROMPT_PLACEHOLDER, continuation)
    return prompt + continuation


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_dir = validated_run_dir(Path(args.run_dir))

    loaded = load_run(run_dir, args.device)
    prompts = list(args.prompt) if args.prompt else list(DEFAULT_PROMPTS)

    sampler = "greedy" if args.greedy else (
        f"temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, seed={args.seed}")
    if args.self_ce and args.max_new_tokens < 2:
        raise SystemExit("--self-ce needs at least two generated tokens to score")
    if args.self_ce and args.reference_draws < 1:
        raise SystemExit("--reference-draws must be at least 1")
    if args.baseline != "none" and not args.self_ce:
        raise SystemExit("--baseline requires --self-ce: the control is the self-CE delta")
    if args.block_tokens is not None and args.baseline == "none":
        raise SystemExit("--block-tokens sizes the --baseline scorer; the run's own model always "
                         "scores at its trained max_seq_len")

    generations = generate_for_prompts(
        loaded,
        prompts,
        args.max_new_tokens,
        greedy=args.greedy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=None if args.greedy else args.seed,
    )
    scores = None
    baseline_scores = None
    baseline = None
    baseline_texts: Optional[List[str]] = None
    seq_len = 0
    baseline_seq_len = 0
    if args.self_ce:
        seq_len = int(loaded.cfg.max_seq_len)
        prompt_ids = [prompt_token_ids(generation.prompt, loaded.dataset)
                      for generation in generations]
        if args.baseline != "none":
            baseline = HuggingFaceBaseline(args.baseline, args.device)
            if args.block_tokens is not None and args.block_tokens > baseline.context:
                raise SystemExit(
                    f"--block-tokens {args.block_tokens} exceeds {args.baseline}'s own context "
                    f"window of {baseline.context}; its position embeddings do not extend")
            baseline_seq_len = (args.block_tokens if args.block_tokens is not None
                                else baseline.context)
        # Drawn once and shared, so both models score the identical corpus slices. Warming to the
        # LARGER block keeps the longer-block model supplied; the shorter-block one ignores the rest.
        warmup = max(seq_len, baseline_seq_len)
        draws = held_out_token_ids(
            loaded.dataset, args.split, args.max_new_tokens, args.reference_draws, warmup=warmup)
        scores = score_streams(
            loaded.model, prompt_ids, [g.token_ids for g in generations],
            draws, args.max_new_tokens, seq_len)
        if baseline is not None:
            baseline_ids = [baseline.generate(ids, args.max_new_tokens) for ids in prompt_ids]
            baseline_texts = [decode_ids(ids, loaded.dataset) for ids in baseline_ids]
            baseline_scores = score_streams(
                baseline, prompt_ids, baseline_ids, draws, args.max_new_tokens, baseline_seq_len)

    if args.json:
        payload = {
            "run_dir":        str(run_dir),
            "dataset":        loaded.dataset,
            "best_step":      loaded.best_step,
            "best_val_ppl":   loaded.best_val_ppl,
            "sampler":        sampler,
            "max_new_tokens": args.max_new_tokens,
            "results": [
                {"prompt":       generation.prompt,
                 "continuation": generation.text,
                 "text":         render(generation.prompt, generation.text),
                 "n_tokens":     len(generation.token_ids)}
                for generation in generations
            ],
        }
        if scores is not None:
            mean, low, high = scores.summary()
            payload["self_ce"] = {
                "units":              "nats_per_token",
                "split":              args.split,
                "reference_mean":     mean,
                "reference_min":      low,
                "reference_max":      high,
                "reference_ppl_mean": math.exp(mean),
                "reference_draws":    len(scores.reference),
                "block_tokens":       seq_len,
                "generated":          scores.generated,
                "generated_delta":    [value - mean for value in scores.generated],
            }
            if baseline_scores is not None:
                base_mean, base_low, base_high = baseline_scores.summary()
                payload["self_ce"]["baseline"] = {
                    "name":               args.baseline,
                    "parameters":         baseline.parameter_count(),
                    "block_tokens":       baseline_seq_len,
                    "reference_mean":     base_mean,
                    "reference_min":      base_low,
                    "reference_max":      base_high,
                    "reference_ppl_mean": math.exp(base_mean),
                    "generated":          baseline_scores.generated,
                    "generated_delta":    [value - base_mean
                                           for value in baseline_scores.generated],
                    "continuations":      baseline_texts,
                }
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return

    ppl = "n/a" if loaded.best_val_ppl is None else f"{loaded.best_val_ppl:.3f}"
    print(f"\nrun:     {run_dir}")
    print(f"dataset: {loaded.dataset}   best_step: {loaded.best_step}   best_val_ppl: {ppl}")
    print(f"device:  {args.device}   sampler: {sampler}"
          f"   max_new_tokens: {args.max_new_tokens}\n")
    if scores is not None:
        mean, low, high = scores.summary()
        print("self-CE (teacher-forced nats/token; one scorer and one shared held-out reference)")
        print(f"  held-out {args.split}: {mean:.3f}  [{low:.3f}, {high:.3f}] over "
              f"{len(scores.reference)} x {args.max_new_tokens} tokens   ppl {math.exp(mean):.2f}")
        print("  delta < 0 means the model finds its own output less surprising than real text")
        # Each model gets its own held-out number: they score the same draws but are different
        # models, and the delta is only interpretable against a model's own baseline.
        run_label = run_dir.name if len(run_dir.name) <= 36 else run_dir.name[:35] + "…"
        run_mean = sum(scores.generated) / len(scores.generated)
        print(f"\n  {'model':<38}{'block':>6}{'held-out':>10}{'self-CE':>10}{'delta':>9}{'ppl':>9}")
        print(f"  {run_label:<38}{seq_len:>6}{mean:>10.3f}{run_mean:>10.3f}{run_mean - mean:>+9.3f}"
              f"{math.exp(run_mean):>9.2f}")
        if baseline_scores is not None:
            base_mean = sum(baseline_scores.generated) / len(baseline_scores.generated)
            base_ref = baseline_scores.summary()[0]
            base_label = f"{args.baseline} ({baseline.parameter_count() / 1e6:.0f}M)"
            print(f"  {base_label:<38}{baseline_seq_len:>6}{base_ref:>10.3f}{base_mean:>10.3f}"
                  f"{base_mean - base_ref:>+9.3f}{math.exp(base_mean):>9.2f}")
        print()
    for index, generation in enumerate(generations):
        print(f"--- prompt: {generation.prompt!r}")
        print(render(generation.prompt, generation.text))
        if scores is not None:
            mean = scores.summary()[0]
            value = scores.generated[index]
            print(f"  self-CE {value:.3f}  (delta {value - mean:+.3f}, ppl {math.exp(value):.2f})")
            if baseline_scores is not None:
                base_value = baseline_scores.generated[index]
                print(f"  {args.baseline:<7} {base_value:.3f}  (delta {base_value - mean:+.3f}, "
                      f"ppl {math.exp(base_value):.2f})")
                # Printed because the loop-vs-no-loop question is the whole point of the control:
                # a low self-CE means nothing without seeing what the baseline actually produced.
                print(f"  {args.baseline} continuation:")
                print(baseline_texts[index] if baseline_texts is not None else "")
        print()


if __name__ == "__main__":
    main()
