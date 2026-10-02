r"""Gradient accumulation (grad_accum_steps) for the VFE_3.0 M-step.

The oracle: for a batch split into K equal microbatches with EQUAL counted-token
counts (no ignore_index, so every position counts), the accumulated ``.grad`` after
K microbatch backwards (each loss / K) equals the ``.grad`` from one backward on the
full batch. This holds because the model's CE and the extra F terms are MEANS over the
batch axis, and there is no cross-sequence dependency in the E-step (the ``for b in
range(B)`` loop), so ``(1/K) * sum_k mean_k == mean_full`` for equal-sized microbatches.
"""

from tests.profiles import test_config

import math
from types import SimpleNamespace

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer, lr_lambda, train, train_step


def _cfg(**over) -> VFE3Config:
    base = dict(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, e_q_mu_lr=0.3,
                m_p_mu_lr=0.05, m_p_sigma_lr=0.01, m_phi_lr=0.0, warmup_steps=2, max_steps=20)
    base.update(over)
    return test_config(**base)


def _full_batch_no_ignore(B=8, N=8, V=6, seed=0):
    # Every position counts (no -100), so the K equal microbatches carry EQUAL token
    # counts -- the precondition of the accum == full-batch grad oracle.
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, V, (B, N), generator=g)
    targets = torch.randint(0, V, (B, N), generator=g)
    return tokens, targets


def _grad_snapshot(model: VFEModel):
    pb = model.prior_bank
    return {
        "mu":    pb.mu_embed.grad.detach().clone(),
        "sigma": pb.sigma_log_embed.grad.detach().clone(),
        "phi":   pb.phi_embed.grad.detach().clone(),
    }


@pytest.mark.parametrize("K", [2, 4])
def test_accum_grad_equals_full_batch_grad(K):
    # THE KEY ORACLE: accumulated .grad over K equal-token microbatches (each loss/K)
    # == .grad from one backward on the full batch. No optimizer.step between the two
    # measurements -- pure gradient comparison on a single model.
    torch.manual_seed(0)
    cfg = _cfg()
    model = VFEModel(cfg)
    tokens, targets = _full_batch_no_ignore(B=8, N=8, V=6, seed=0)

    # Full-batch single backward.
    model.zero_grad(set_to_none=True)
    _, loss_full, _ = model(tokens, targets)
    loss_full.backward()
    full = _grad_snapshot(model)

    # K equal microbatches, each loss/K, accumulating into .grad.
    model.zero_grad(set_to_none=True)
    tok_chunks = torch.chunk(tokens, K, dim=0)
    tgt_chunks = torch.chunk(targets, K, dim=0)
    assert len(tok_chunks) == K                       # B divisible by K (equal microbatches)
    accum_loss = 0.0
    for tc, gc in zip(tok_chunks, tgt_chunks):
        _, loss_mb, _ = model(tc, gc)
        (loss_mb / K).backward()
        accum_loss += float(loss_mb.detach()) / K
    accum = _grad_snapshot(model)

    for name in ("mu", "sigma", "phi"):
        assert torch.allclose(accum[name], full[name], atol=1e-5, rtol=1e-4), name
    # The accumulated (mean-over-K) loss equals the full-batch mean loss too.
    assert accum_loss == pytest.approx(float(loss_full.detach()), abs=1e-5)


@pytest.mark.parametrize("K", [2, 4])
def test_train_step_accum_grad_matches_full_batch(K):
    # THE PRODUCTION-PATH ORACLE: drive the REAL train_step at K>1 and assert its accumulated
    # .grad equals the full-batch single-backward grad. (The sibling
    # test_accum_grad_equals_full_batch_grad reimplements the chunking inline to pin the math
    # premise; this one pins train_step's own accumulation branch.) grad_clip=0.0 skips the clip
    # (train_step checks grad_clip > 0), so .grad is the raw accumulated gradient; zero_grad is at
    # the START of train_step and optimizer.step() mutates params (not .grad), so the post-call
    # .grad is the accumulation computed at the SAME params as the full-batch snapshot.
    torch.manual_seed(0)
    cfg = _cfg()
    model = VFEModel(cfg)
    tokens, targets = _full_batch_no_ignore(B=8, N=8, V=6, seed=0)

    model.zero_grad(set_to_none=True)
    _, loss_full, _ = model(tokens, targets)
    loss_full.backward()
    full = _grad_snapshot(model)

    opt = build_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, cfg))
    train_step(model, opt, sched, tokens, targets, grad_clip=0.0, grad_accum_steps=K)
    accum = _grad_snapshot(model)

    for name in ("mu", "sigma", "phi"):
        assert torch.allclose(accum[name], full[name], atol=1e-5, rtol=1e-4), name


def test_estep_grad_metrics_are_microbatch_mean_not_last():
    class _ScriptedEstepGrad(torch.nn.Module):
        def __init__(self, inner: VFEModel) -> None:
            super().__init__()
            self.inner = inner
            self.cfg = inner.cfg
            self.records = [
                {"mu": 1.0, "sigma": 10.0},
                {"mu": 3.0},
            ]

        def forward(self, tokens, targets=None, *, estep_grad_out=None,
                    return_decode_stats=False):
            assert estep_grad_out == {}
            estep_grad_out.update(self.records.pop(0))
            return self.inner(
                tokens, targets, return_decode_stats=return_decode_stats)

    torch.manual_seed(0)
    cfg = _cfg()
    inner = VFEModel(cfg)
    model = _ScriptedEstepGrad(inner)
    optimizer = build_optimizer(inner, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_lambda(step, cfg))
    tokens, targets = _full_batch_no_ignore(B=4, N=8, V=6, seed=0)
    metrics = {}

    train_step(model, optimizer, scheduler, tokens, targets, grad_clip=1.0,
               grad_accum_steps=2, metrics_out=metrics)

    assert metrics["estep_grad_norm_mu_microbatch_mean"] == pytest.approx(2.0)
    assert metrics["estep_grad_norm_sigma_microbatch_mean"] == pytest.approx(10.0)


def test_train_step_k1_byte_identical_to_single_step_path():
    # grad_accum_steps=1 must reproduce the current train_step EXACTLY: same grads, same
    # optimizer/scheduler step, same returned loss. Two identically-seeded models, one
    # stepped with the explicit default kwarg, one with no kwarg at all.
    tokens, targets = _full_batch_no_ignore(B=8, N=8, V=6, seed=1)

    torch.manual_seed(0)
    cfg_a = _cfg()
    model_a = VFEModel(cfg_a)
    opt_a = build_optimizer(model_a, cfg_a)
    sch_a = torch.optim.lr_scheduler.LambdaLR(opt_a, lambda s: lr_lambda(s, cfg_a))
    loss_a = train_step(model_a, opt_a, sch_a, tokens, targets, grad_clip=1.0)

    torch.manual_seed(0)
    cfg_b = _cfg()
    model_b = VFEModel(cfg_b)
    opt_b = build_optimizer(model_b, cfg_b)
    sch_b = torch.optim.lr_scheduler.LambdaLR(opt_b, lambda s: lr_lambda(s, cfg_b))
    loss_b = train_step(model_b, opt_b, sch_b, tokens, targets, grad_clip=1.0, grad_accum_steps=1)

    assert loss_a == loss_b
    assert torch.equal(model_a.prior_bank.mu_embed, model_b.prior_bank.mu_embed)
    assert torch.equal(model_a.prior_bank.sigma_log_embed, model_b.prior_bank.sigma_log_embed)
    assert torch.equal(model_a.prior_bank.phi_embed, model_b.prior_bank.phi_embed)
    assert sch_a.last_epoch == sch_b.last_epoch == 1


def test_optimizer_and_scheduler_step_once_per_K_microbatches():
    # A "step" stays an OPTIMIZER step: with grad_accum_steps=K, train() takes N optimizer
    # steps over n_steps=N train_step calls, each consuming K microbatches. The scheduler
    # advances once per optimizer step (last_epoch == N), NOT once per microbatch (N*K).
    K = 4
    N = 5
    torch.manual_seed(0)
    cfg = _cfg(grad_accum_steps=K)
    model = VFEModel(cfg)

    opt = build_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, cfg))

    # Drive train_step directly N times; each must take exactly ONE optimizer + scheduler step.
    for _ in range(N):
        tokens, targets = _full_batch_no_ignore(B=8, N=8, V=6, seed=0)
        train_step(model, opt, sched, tokens, targets, grad_clip=1.0, grad_accum_steps=K)

    # AdamW records a per-parameter optimizer-step counter in opt.state; after N
    # train_step calls (each K microbatches) it must read N, not N*K -- the optimizer
    # stepped once per train_step, not once per microbatch forward.
    step_counts = {int(s["step"]) for s in opt.state.values()}
    assert step_counts == {N}                  # every param advanced exactly N optimizer steps
    assert sched.last_epoch == N               # scheduler advanced once per optimizer step


def test_train_loop_runs_under_grad_accum():
    # End-to-end: train() with grad_accum_steps>1 runs n_steps optimizer steps and returns
    # n_steps finite losses (one per optimizer step), splitting each pulled batch into K.
    from torch.utils.data import DataLoader
    from vfe3.data.datasets import TokenWindows

    torch.manual_seed(0)
    cfg = _cfg(grad_accum_steps=4, batch_size=8)
    model = VFEModel(cfg)
    base = torch.arange(3).repeat(400)
    ds = TokenWindows(base[:400].to(torch.long), 8)
    loader = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(0))
    losses = train(model, loader, cfg, n_steps=5)
    assert len(losses) == 5
    assert all(math.isfinite(x) for x in losses)


def test_config_grad_accum_steps_validation():
    assert test_config(grad_accum_steps=1).grad_accum_steps == 1
    test_config(grad_accum_steps=4)                             # accepted
    with pytest.raises(ValueError):
        test_config(grad_accum_steps=0)
    with pytest.raises(ValueError):
        test_config(grad_accum_steps=-3)


def test_uneven_microbatch_token_counts_warns():
    # C8 (audit 2026-07-01, deferred guard): with grad_accum_steps>1 and UNEVEN counted-token
    # microbatches, the n_mb/n_tot weight is exact for the token-mean CE but only approximate for
    # the non-CE regularizers -- train_step must warn loudly in exactly that regime. The equal-token
    # default stays silent (and byte-identical to the full-batch gradient, pinned by
    # test_train_step_accum_grad_matches_full_batch above).
    import warnings

    torch.manual_seed(0)
    cfg = _cfg()
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, cfg))
    tokens, targets = _full_batch_no_ignore(B=4, N=8, V=6, seed=0)
    targets = targets.clone()
    targets[0, :4] = -100                                      # chunk 0 loses 4 counted tokens -> spread > 0
    with pytest.warns(RuntimeWarning, match="non-CE regularizers"):
        train_step(model, opt, sched, tokens, targets, grad_clip=1.0, grad_accum_steps=2)

    # Equal-token microbatches (the default unpadded regime): NO such warning may fire.
    torch.manual_seed(0)
    model2 = VFEModel(cfg)
    opt2 = build_optimizer(model2, cfg)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda s: lr_lambda(s, cfg))
    tokens2, targets2 = _full_batch_no_ignore(B=4, N=8, V=6, seed=0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        train_step(model2, opt2, sched2, tokens2, targets2, grad_clip=1.0, grad_accum_steps=2)
    assert not [w for w in rec if "non-CE regularizers" in str(w.message)]


def test_train_step_indivisible_batch_raises():
    # The equal-token oracle requires B % K == 0. A non-divisible batch would silently
    # give unequal microbatches (torch.chunk(5, 4) -> sizes [2,2,1], only 3 chunks), so
    # train_step rejects it loudly rather than miscomputing the /K normalization.
    torch.manual_seed(0)
    cfg = _cfg()
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, cfg))
    tokens = torch.randint(0, 6, (5, 8))                       # B=5 not divisible by K=4
    targets = torch.randint(0, 6, (5, 8))
    with pytest.raises(ValueError):
        train_step(model, opt, sched, tokens, targets, grad_clip=1.0, grad_accum_steps=4)


class _ScriptedScoredTokenModel(torch.nn.Module):
    """A one-parameter objective with CE/count encoded by each microbatch's token value."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.cfg = SimpleNamespace(
            m_phi_update_mode="adamw",
            phi_mstep_max_matrix_norm=None,
        )

    def forward(self, tokens, targets=None, *, estep_grad_out=None,
                return_decode_stats=False):
        value = tokens[0, 0].to(self.weight.dtype)
        ce = self.weight * value
        if return_decode_stats:
            from vfe3.model.prior_bank import DecodeCEResult
            count = tokens[0, 1].to(dtype=torch.int64)
            return None, ce, DecodeCEResult(ce=ce.detach(), scored_tokens=count)
        return None, ce, ce.detach()


def _scripted_optimizer(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return optimizer, scheduler


def test_accumulation_uses_decoder_counts_for_exact_token_level_objective():
    model = _ScriptedScoredTokenModel()
    optimizer, scheduler = _scripted_optimizer(model)
    # Microbatch 1: (CE,count)=(1,1); microbatch 2: (CE,count)=(3,3).
    tokens = torch.tensor([[1, 1], [3, 3]])
    targets = torch.zeros_like(tokens)

    loss = train_step(
        model, optimizer, scheduler, tokens, targets,
        grad_clip=0.0, grad_accum_steps=2,
    )

    assert loss == pytest.approx((1.0 * 1 + 3.0 * 3) / 4)
    assert model.weight.grad.item() == pytest.approx(2.5)


def test_zero_scored_accumulation_skips_optimizer_and_scheduler_update():
    model = _ScriptedScoredTokenModel()
    optimizer, scheduler = _scripted_optimizer(model)
    tokens = torch.tensor([[0, 0], [0, 0]])
    targets = torch.full_like(tokens, -100)
    status = {}

    loss = train_step(
        model, optimizer, scheduler, tokens, targets,
        grad_clip=0.0, grad_accum_steps=2, status_out=status,
    )

    assert loss == 0.0
    assert status["did_step"] is False
    assert scheduler.last_epoch == 0
    assert optimizer.state == {}
