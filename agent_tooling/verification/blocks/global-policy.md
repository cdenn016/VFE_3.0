<!-- BEGIN VERIFICATION GLOBAL POLICY -->
## Evidence-gated verification

Use the installed `verification` skill whenever the user requests a verifier, audit, proof,
correctness determination, experimental-result check, or source-backed factual claim. Record one
claim per check in the skill's claim ledger and validate that ledger before reporting closure.

The five claim states are precise. `CANDIDATE` is queued but unassessed. `LLM_SUPPORTED` has a
reasoned model assessment but lacks closure evidence. `EVIDENCE_VERIFIED` is closed by current,
eligible supporting evidence. `REFUTED` is closed by current, eligible contradicting evidence.
`INCONCLUSIVE` is terminal for the present attempt and names at least one open obligation.
`CANDIDATE` and `LLM_SUPPORTED` are triage states and are not valid closure outcomes.

Apply the evidence hierarchy for the claim's domain. Code and experiment claims require current
mechanical checks or reproduced outputs. Mathematics claims require a derivation or formal proof;
numerical agreement alone does not prove them. Evidence, research, source, and general factual
claims require a current primary source or reproduced source record. LLM judgment may support
triage, but it cannot by itself verify or refute a claim, and agreement among agents is not closure.

Evidence is fresh only for the ledger's recorded artifact revision, configuration, inputs, and
environment. A source edit, changed dependency or configuration, new experiment, or differing
artifact revision invalidates affected evidence and requires re-verification. Test totals and
failure counts must come from current machine-readable output such as JUnit XML, never memory or a
visual progress line. Missing eligible evidence or unresolved disagreement yields `INCONCLUSIVE`,
not a majority-vote result.
<!-- END VERIFICATION GLOBAL POLICY -->
