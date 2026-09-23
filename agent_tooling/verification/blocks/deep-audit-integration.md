<!-- BEGIN VERIFICATION DEEP AUDIT INTEGRATION -->
## Deep-audit verification-ledger integration

For `deep-audit`, this block replaces the free-form single-verifier stage and any report step that
derives findings from prose summaries. Preserve the base-investigator wave, the scope-selected
domain-expert wave, and their parallel dispatch rules. Their outputs are candidate inputs, not
verified findings.

After all investigator and expert waves return, invoke the installed `verification` skill in
closure mode and create `.verification/ledger.json` at the audited artifact revision. Convert every
candidate finding into a separate ledger claim with its domain, severity, current source location,
active-configuration reachability, evidence, counterevidence, and open obligations. Use the skill's
domain criteria: executable code paths and experiments need current mechanical or reproduced-output
evidence; mathematics needs a derivation or formal proof; source-backed claims need current primary
or reproduced-source evidence. LLM agreement, comments, and investigator votes cannot close a
claim.

Run independent verification views and the skill's adaptive escalation. Every high or critical
claim must record the `high_severity` trigger, receive four or eight views, and include both
`verifier-skeptic` and `verifier-adjudicator`. The skeptic challenges reachability, assumptions,
counterexamples, and severity; the adjudicator may assign `EVIDENCE_VERIFIED` or `REFUTED` only
from eligible evidence accepted by the deterministic validator. Missing evidence or unresolved
disagreement is `INCONCLUSIVE`. Retain the existing audit skeptic/defender challenge for every
high or critical claim that remains evidence-verified, and record its disposition as ledger-linked
evidence rather than as an independent source of closure.

Validate the completed ledger before rendering the audit report. Generate verifier verdicts,
severity totals, challenge outcomes, and the surviving punch list from the validated ledger, not
from investigator prose. Report only current machine-readable test evidence: record the command,
artifact revision, exit status, and JUnit or equivalent tests/failures/errors/skips counts. If the
audited artifact changes after validation, mark affected evidence invalidated, re-run verification,
and render the report again. Do not leave a verification activation marker after a successful
validated closure.
<!-- END VERIFICATION DEEP AUDIT INTEGRATION -->
