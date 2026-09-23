<!-- BEGIN VERIFICATION PROJECT POLICY -->
## Project verification control plane

For this repository, invoke the installed `verification` skill for audits, proofs, correctness
claims, experimental results, and source-backed factual claims. Use its validated claim ledger as
the closure record. Every claim is exactly one of `CANDIDATE` (queued), `LLM_SUPPORTED` (reasoned
triage without closure evidence), `EVIDENCE_VERIFIED` (closed by current supporting evidence),
`REFUTED` (closed by current contradicting evidence), or `INCONCLUSIVE` (an open obligation remains).
Closure mode rejects `CANDIDATE` and `LLM_SUPPORTED`.

Read executable paths and active configuration rather than relying on comments. Code and
experiment closure requires current mechanical or reproduced-output evidence, with commands,
exit status, configuration, and artifact revision recorded. Mathematical closure requires a
derivation or formal proof; numerical checks are supporting evidence only. Research, source, and
general factual closure requires a current primary source or reproduced-source record. LLM output,
agent consensus, comments, and remembered results cannot independently close a claim.

Evidence is revision-bound. Any relevant edit, configuration or dependency change, new run, input
change, or artifact-revision mismatch invalidates the affected evidence and requires re-running the
verification. Pytest totals and failures must be read from current machine-readable output such as
JUnit XML. Missing eligible evidence or unresolved disagreement is `INCONCLUSIVE` with a specific
open obligation. Before reporting closure, run the verification skill's deterministic ledger
validator and name the validated ledger in the final response.
<!-- END VERIFICATION PROJECT POLICY -->
