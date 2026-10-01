r"""Pure-route identity probe (Task 6, PB, 2026-07-12) -- an environment-driven helper, NOT a
production entry point.

It records a byte-for-byte fingerprint of ONE deterministic forward/backward/optimizer step of the
theoretically pure route (a diagonal / flat config whose Plan-5 toggles are all left at their old-route
defaults), so the SAME script run against the feature branch and against its merge base can be compared
tensor-by-tensor with ``torch.equal``. Exact equality everywhere is the plan's pure-path guarantee:
the completed hierarchy adds capacity only under opt-in toggles and never perturbs the default path.

Usage (both variables are required)::

    VFE3_PROBE_REPO=<repo root prepended to sys.path before importing vfe3>
    VFE3_PROBE_OUT=<output .pt path>
    python tests/hierarchy_identity_probe.py

The comparison itself lives in
``tests/test_hierarchical_probabilistic_completeness_20260712.py::test_pure_route_bundle...``,
which reads the two bundles named by ``VFE3_BASELINE_BUNDLE`` / ``VFE3_FEATURE_BUNDLE``.

Only plain tensors / dicts / lists / scalars are written to the bundle (never a ``vfe3`` class
instance such as ``BeliefState``), so the bundle loads under EITHER repo without a class-version
dependency. The probe imports nothing from ``vfe3`` until ``VFE3_PROBE_REPO`` is on ``sys.path``.
"""

import os
import sys


def _plain(belief) -> dict:
    r"""Extract the always-present belief tensor fields as detached plain tensors (mu, Sigma, phi).

    The pure phi-cocycle path leaves omega at None, so recording mu / sigma / phi keeps
    the bundle schema identical across repos with no vfe3-class instance in the pickle."""
    import torch  # noqa: F401  (local: after the sys.path prepend)

    return {
        "mu":    belief.mu.detach().clone(),
        "sigma": belief.sigma.detach().clone(),
        "phi":   belief.phi.detach().clone(),
    }


def main() -> None:
    repo = os.environ.get("VFE3_PROBE_REPO")
    out = os.environ.get("VFE3_PROBE_OUT")
    if not repo or not out:
        raise SystemExit("set VFE3_PROBE_REPO (repo root) and VFE3_PROBE_OUT (output .pt) before running")

    # Prepend the target repo so ``import vfe3`` binds to the repo under test, NOT any installed copy.
    sys.path.insert(0, repo)

    import torch

    # Deterministic CPU: fixed seeds plus deterministic kernels (warn-only so a missing deterministic
    # kernel degrades to a warning instead of aborting -- both repos run the SAME torch, so the path
    # is identical either way).
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(0)
    device = torch.device("cpu")

    from tests.profiles import test_config
    from vfe3.model.model import VFEModel
    from vfe3.train import build_optimizer
    import vfe3.model.block as block

    # Fixed pure-route config from the test-owned profile. Both compared revisions must expose
    # that profile, so mutable experiment settings cannot change the fingerprint.
    cfg = test_config(
        vocab_size=9, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=2, n_e_steps=2,
        family="gaussian_diagonal", transport_mode="flat", attention_mode="transported",
        predictive_covariance="passthrough", attention_residual=False,
        relational_input_scope="head", relational_output_scope="head",
        use_block_mlp=False, use_head_mixer=False, use_cg_coupling=False,
        norm_type_block="none", norm_type_final="none", layernorm_affine=False,
        decode_mode="diagonal_chunked", decode_bias=False, decode_unigram_prior=False,
        untie_decode_bank=False, decode_temperature_learned=False, pos_phi="none",
        lambda_alpha_mode="constant", skip_belief_sigma_update=False,
        mu_init_std=0.02, sigma_init=1.0, phi_scale=0.01, beta_attention_prior="causal",
    )
    torch.manual_seed(0)
    model = VFEModel(cfg).to(device)

    # Re-seed AFTER construction so token generation does not depend on the exact amount of global RNG
    # model init consumes (the pure route consumes the same amount under either repo; the re-seed keeps
    # the probe robust regardless).
    torch.manual_seed(1234)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len), device=device)
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len), device=device)

    # Inject one state_record per layer by wrapping vfe3.model.block.e_step (a probe-local patch; the
    # production call passes state_record=None). Each E-step then fills its own dict, whose
    # ["beliefs"] list is the per-iteration belief trajectory for that layer.
    layer_records = []
    real_e_step = block.e_step

    def probing_e_step(*args, **kwargs):
        rec: dict = {}
        kwargs = dict(kwargs)
        kwargs["state_record"] = rec
        result = real_e_step(*args, **kwargs)
        layer_records.append(rec)
        return result

    block.e_step = probing_e_step
    try:
        encoded = model.prior_bank.encode(tokens)               # encode-time belief (pre pos_phi / stack)
        belief, logits = model.forward_beliefs(tokens, return_logits=True)
    finally:
        block.e_step = real_e_step

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
    loss.backward()

    named_grads = {name: p.grad.detach().clone()
                   for name, p in model.named_parameters() if p.grad is not None}

    optimizer = build_optimizer(model, cfg)
    optimizer.step()
    post_step_state = {name: p.detach().clone() for name, p in model.named_parameters()}

    per_layer_beliefs = [[_plain(b) for b in rec["beliefs"]] for rec in layer_records]

    bundle = {
        "encoded_belief":    _plain(encoded),
        "converged_belief":  _plain(belief),
        "per_layer_beliefs": per_layer_beliefs,
        "logits":            logits.detach().clone(),
        "loss":              loss.detach().clone(),
        "named_grads":       named_grads,
        "post_step_state":   post_step_state,
        "optimizer_state":   optimizer.state_dict(),
        "state_dict_keys":   sorted(model.state_dict().keys()),
        "n_layer_records":   len(layer_records),
    }
    torch.save(bundle, out)
    print(f"wrote probe bundle to {out}: {len(per_layer_beliefs)} layer record(s), "
          f"{len(named_grads)} grad tensor(s), {len(post_step_state)} param tensor(s)")


if __name__ == "__main__":
    main()
