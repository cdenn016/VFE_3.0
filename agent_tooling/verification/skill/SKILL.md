---
name: verification
description: Run evidence-gated, multi-view verification for code, mathematics, sources, and experiments.
---

# Verification

Use this skill for a verifier request, audit, proof, correctness claim, experimental result, or source-backed factual claim. The ledger separates an LLM's assessment from evidence that can close a claim.

## Workflow

1. Start the control plane from the repository root:

   ```{{VERIFICATION_GATE_SHELL}}
   {{VERIFICATION_GATE_COMMAND}} start --cwd . --ledger .verification/ledger.json --mode closure
   ```

   The repository must be a Git worktree with a concrete `HEAD`. The command records `git:<HEAD>:sha256:<digest>` in both the ledger and activation marker. The digest binds the Git index plus tracked and nonignored untracked worktree content while excluding `.git` and `.verification`.

2. Add one ledger claim per check. A queued `CANDIDATE` may be recorded without fabricated criteria, views, or comparison results. Assessed claims include their artifact revision, domain, severity, aggregate 0 to 20 criterion scores, two or more uniquely identified view scores, calibration kind, comparison record, evidence, counterevidence, verifiers, and open obligations. Every evidence and counterevidence entry has a stable `id`. Each comparison records candidate descriptions, and each match records the responsible view, outcome, criterion scores, and result location. Read [the contract](references/contract.md) before assigning a state.

3. Read the criterion file for each domain in scope: [code](references/criteria-code.md), [mathematics](references/criteria-math.md), [evidence](references/criteria-evidence.md), [experiments](references/criteria-experiment.md), and [general](references/criteria-general.md). Dispatch independent domain roles and record every result at the criterion level.

4. Begin with two views and set `escalation_target` to 2. Adaptively escalate to four or eight for small margins, high dispersion, criterion disagreement, or high severity. Record each applicable reason in `escalation_triggers` using `small_margin`, `high_dispersion`, `criterion_disagreement`, or `high_severity`; do not silently omit the reason and do not invent numeric thresholds. Any trigger and every high or critical claim requires a target of 4 or 8. Record `high_severity` for high or critical claims and `criterion_disagreement` for unresolved disagreement; an unresolved disagreement after four views requires target 8. Reverse A/B order for each pairwise comparison. For more than four candidates, use a balanced pivot tournament. Every terminal state requires exactly one structured `verifier-adjudicator` record linked to known view IDs, evidence IDs, and a result location. `EVIDENCE_VERIFIED` links a current domain-eligible supporting evidence ID, while `REFUTED` links a current domain-eligible `supports: false` counterevidence ID. Critical and high closure also requires structured `verifier-skeptic` linkage.

5. Do not close a claim by vote. Missing evidence or unresolved disagreement yields `INCONCLUSIVE`, never majority-vote acceptance. LLM judgment alone may support only `LLM_SUPPORTED`, never `EVIDENCE_VERIFIED`. Set `evidence_invalidated` to true when retaining stale evidence solely as audit history; invalidated evidence cannot support either `EVIDENCE_VERIFIED` or `REFUTED`.

6. Run the deterministic check after the ledger is complete:

   ```{{VERIFICATION_GATE_SHELL}}
   {{VERIFICATION_GATE_COMMAND}} validate .verification/ledger.json --cwd .
   ```

   Repair every reported error. In the final response, name `.verification/ledger.json` as the validated ledger.

## State and evidence rules

Use `CANDIDATE` before assessment and `LLM_SUPPORTED` for a reasoned but unverified assessment only in `triage` mode. `closure` mode rejects both intermediate states: missing eligible evidence or unresolved disagreement requires `INCONCLUSIVE` with a specific open obligation. `EVIDENCE_VERIFIED` requires current eligible evidence and `REFUTED` requires current counterevidence. Code and experiment correctness require current mechanical or reproduced-output evidence. Mathematical correctness requires a derivation or formal proof; numerical evidence alone is insufficient. The active hook and `validate` command fail closed if the activation revision, ledger revision, or live Git artifact no longer agrees. See [the contract](references/contract.md) for the exact closure requirements.
