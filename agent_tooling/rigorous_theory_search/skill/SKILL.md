---
name: rigorous-theory-search
description: Use when research-level conjectures or unsolved problems require complete constructions, full proofs, or derivations, or when constructing research-level effective theories in gauge or information geometry, ELBO/VFE, or renormalization; do not use for routine calculations, routine ELBO derivations, routine RG transformations, standard summaries, reviews, or code work.
---

# Rigorous theory search

Use this protocol for research-level proof and construction programs whose quantifiers, global compatibility, physical interpretation, or effective-theory closure are load-bearing. Do not use it for textbook exercises, routine calculations, routine ELBO derivations, routine RG transformations, routine code work, ordinary summaries, manuscript peer review, or literature review alone.

## Load the protocol progressively

Always read [problem-contract.md](references/problem-contract.md), [portfolio-search.md](references/portfolio-search.md), [proof-obligations.md](references/proof-obligations.md), and [output-contract.md](references/output-contract.md). Load [physics-information-audit.md](references/physics-information-audit.md) for gauge, statistical, ELBO/VFE, pullback, or emergent-time claims. Load [renormalization-effective-theory.md](references/renormalization-effective-theory.md) for coarse-graining or scale claims. Load [runtime-adapters.md](references/runtime-adapters.md) only to map roles onto available capabilities. Before any release, load [adversarial-verification.md](references/adversarial-verification.md).

Create a durable run with [scaffold_run.py](scripts/scaffold_run.py), then fill its linked artifacts: [problem-contract.json](assets/templates/problem-contract.json), [approach-registry.json](assets/templates/approach-registry.json), [claim-ledger.json](assets/templates/claim-ledger.json), [dependency-dag.json](assets/templates/dependency-dag.json), [counterexample-register.md](assets/templates/counterexample-register.md), [construction-or-strongest-theorem.md](assets/templates/construction-or-strongest-theorem.md), [adversarial-report.json](assets/templates/adversarial-report.json), [release.json](assets/templates/release.json), and [final-report.md](assets/templates/final-report.md). Run [validate_run.py](scripts/validate_run.py) at checkpoints and release. Structural validation cannot establish mathematical truth.

## Discovery and certification

Discovery proposes; certification proves. Record an affirmative-existence request only as `SEARCH_PRIOR_AFFIRMATIVE` in `problem-contract.target.search_priors`. It may allocate search effort, but it is forbidden from premises, assumptions, evidence, claims, dependency edges, certificates, adversarial conclusions, and reports.

Discovery maintains a dynamic mechanism-diverse portfolio without broadcasting a favored narrative. Certification accepts a mathematical claim only with direct `DERIVATION`, `FORMAL_PROOF`, or a hypothesis-mapped `APPLICABLE_THEOREM`. A refutation needs a `COUNTEREXAMPLE` or `NONEXISTENCE_PROOF` matched to the target quantifiers. Computation, symbolic output without side conditions, figures, citations, and solver agreement do not close a theorem.

## Required lifecycle

1. Scaffold the run, freeze the exact problem contract, compute the canonical full-target SHA-256, and bind every JSON and Markdown artifact to its digest-derived contract ID.
2. Establish nonempty typed domains, well-defined maps and functionals, regularity, measures, boundary conditions, symmetries, and equivalence.
3. Open a dynamic mechanism-diverse portfolio without broadcasting a favorite route.
4. Accept only a proved lemma, typed construction, rigorous counterexample, or named obstruction with a precise removal obligation.
5. Certify one atomic claim at a time and maintain an acyclic dependency DAG where each edge points from a claim to what it depends on.
6. Cross-pollinate only mature routes. Give every hybrid a new family ID and certify an explicit interface claim before reusing parent results.
7. Assemble the global or effective theory, including overlap, correction, boundary, and anomaly terms, and distinguish exact versus truncated results.
8. Run every applicable domain audit, an independent reconstruction from the frozen contract, and oracle erasure of the search prior.
9. Release exactly one terminal status.

Continue only while a materially new mechanism, representation, invariant, bridge, obstruction-removal method, or assumption set remains admissible. Retire repetition. When novelty is exhausted, report the strongest verified result and unresolved obligations as `INCONCLUSIVE`; never fabricate closure or loop indefinitely. When parallel role separation is unavailable, use a sequential fallback with labeled passes from the same frozen contract and durable artifacts. This preserves role separation and auditability, not independent evidence, consensus, or corroboration.

## Search-pressure safeguards

The neutral control produced no false-closure rationale in forty final answers: all 240 graded checks passed, including five constructive-theorem runs. The safeguards below therefore derive from observed pressure forms, successful rebuttals, and nonproof boilerplate, not invented failed rationalizations or case answers.

| Pressure or risk | Required safeguard |
| --- | --- |
| Affirmative pressure | Isolate it as the search prior and perform oracle erasure before certification. |
| Finite-to-universal overreach | Treat finite or numerical checks as tests; supply a scope-matched proof or weaken the claim. |
| Local or gauge-fixed to global overreach | Prove overlap, cocycle, quotient, topology, and invariance conditions. |
| Representation, pullback, or curve parameter overreach | Run the typed-curve and Fisher-duration gate. Distinguish a vertical fixed-fiber curve, a section-induced lift of a base curve, and connection-dependent horizontal transport. State whether duration is curve-intrinsic or connection-relative. Before calling accumulated Fisher length a duration coordinate, declare the regular stratum and curve class, require positive accumulated length on every nontrivial subinterval, and require nonzero speed for a regular arc-length coordinate; then state orientation, origin, global-clock, and operational-bridge obligations. |
| Stochastic replacement or ontology overreach | Even after a counterexample, state the conditional replacement theorem: complete retained joint-law equality; joint-kernel, factorization, and conditional-independence consistency; every likelihood and ELBO term including collapse corrections; base measures, support, absolute continuity, and integrability; then label ontological completeness as a separate modeling or operational bridge postulate. |
| Projected RG overreach | Define exact coarse-graining first; prove ansatz closure or report the truncation and residual. |
| Defensive nonproof boilerplate | Acknowledge difficulty briefly, then return a lemma, construction, counterexample, named obstruction, or explicit unresolved obligation. |

## Release

Release exactly one of `COMPLETE_AFFIRMATIVE`, `COMPLETE_NEGATIVE`, or `INCONCLUSIVE`. Here an ancestor means a transitive dependency reached by following `from -> to` edges from the target. Every terminal status rejects `CANDIDATE` and `LLM_SUPPORTED` ancestors. Both complete statuses require every ancestor to be `EVIDENCE_VERIFIED`; a negative certificate that does not depend on an affirmative route must model its certificate lemmas as the target's dependencies instead of retaining irrelevant proof-route edges. `COMPLETE_AFFIRMATIVE` also requires the target to be `EVIDENCE_VERIFIED`. `COMPLETE_NEGATIVE` requires the target to be `REFUTED` by the frozen `negative_certificate_kind`: one valid counterexample closes a universal target, an existential target uses a scope-matched nonexistence proof, and a mixed target declares which applies instead of relying on inference. `INCONCLUSIVE` names the strongest verified theorem, the minimal unresolved obligation set, affected routes, and closure evidence still needed.

Every terminal status requires substantive artifact-backed attacks, independent reconstruction, and oracle erasure covering the target and every ancestor. Completion booleans are not evidence. Erasure removes the prior, scans for paraphrased dependence, and rechecks the full dependency closure. Passing shows only that the prior was unnecessary; it does not prove the theorem. Release only after the structural validator reports no errors, while retaining its limitation that it cannot establish mathematical truth.
