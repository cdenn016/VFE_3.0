# Durable output contract

Use `docs/derivations/YYYY-MM-DD-<slug>/` inside a repository unless repository instructions or the user specify another path. Outside a repository, use `rigorous-theory-search/YYYY-MM-DD-<slug>/`. Keep one contract ID and schema `rigorous-theory-search/v1` across the run.

The directory must contain all nine artifacts:

- `problem-contract.json`
- `approach-registry.json`
- `claim-ledger.json`
- `dependency-dag.json`
- `counterexample-register.md`
- `construction-or-strongest-theorem.md`
- `adversarial-report.json`
- `release.json`
- `final-report.md`

Evidence records use safe relative artifact paths, never absolute paths, drive-qualified paths, backslashes, NULs, or parent traversal. Each path must resolve beneath the run directory to a regular file through no symbolic-link or junction component. Record the computed SHA-256 from the referenced file together with its support polarity, exact scope, and side conditions. The validator reads the bytes and rejects a missing file or digest mismatch. Use one schema, full target digest, and contract identifier throughout. Each Markdown artifact begins with `<!-- rigorous-theory-search-metadata {canonical JSON} -->` carrying `schema_version`, `contract_id`, and `target_digest`. Do not copy the affirmative search prior into assumptions, evidence, claims, dependency endpoints, adversarial results, construction prose, or the final report; any such occurrence is prior leakage.

A checkpoint requires `terminal_status: null` and may retain candidate claims. A release contains exactly one terminal status and its explicit quantifier-sensitive certificate. A release also replaces every scaffold Markdown body, supplies a nonempty nonsentinel mechanism portfolio, and records artifact-backed attacks, independent reconstruction, and oracle erasure covering the target and all transitive dependencies. Replace the candidate report with these exact final-report headings and no additional headings:

```markdown
# Rigorous theory search report

## Frozen contract
## Terminal status
## Certificate
## Strongest verified result
## Dependency closure
## Independent reconstruction
## Oracle erasure
## Unresolved obligations
## Scope and limitations
```

The certificate section names the target claim, evidence artifacts, assumptions, quantifiers, and dependency closure. The strongest-result section states only what the evidence proves. The unresolved-obligations section remains present even when empty. The limitations section separates theorem, construction, modeling postulate, operational identification, physical interpretation, analogy, and numerical observation.

Run the structural validator at checkpoints and in release mode. Version 1 validates the current run-package snapshot: schema and full-target binding, dependency structure, terminal-state compatibility, per-kind evidence eligibility and polarity, artifact containment, hash syntax and path safety, and recomputed file hashes. The former limitation phrase "does not verify file bytes" no longer applies because the validator hashes the referenced bytes. It has no append-only history and makes no claim about status transitions between snapshots. It does not reproduce derivations, establish source applicability, or prove mathematical truth. Those remain certification and adversarial obligations.
