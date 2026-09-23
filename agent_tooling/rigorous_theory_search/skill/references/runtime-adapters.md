# Runtime capability adapters

The mathematical status of a result cannot depend on vendor, worker count, or parallelism. Select available capabilities at runtime and preserve the same frozen contract, role boundaries, evidence rules, and artifacts.

## Claude Code

Use Claude Code's current delegation interface, such as its `Agent` or `Task` capability, for isolated solver, falsifier, defender, checker, and reconstruction assignments. Use `TaskOutput` or `SendMessage` only for coordination. Pass bounded briefs and artifact paths rather than a favored proof narrative. Record whether workers share a filesystem, transcript, or inherited context.

## Codex

Use Codex `spawn_agent` for bounded independent roles, `send_message` for nonturn coordination, `followup_task` for a new turn on an existing role, and `wait_agent` for completion. If available, use `interrupt_agent` only to stop obsolete work and `list_agents` to audit active ownership. Do not treat agents with shared history or shared files as independent corroboration.

## Shared-agent environments

In a shared-agent runtime, assign nonoverlapping artifact ownership and disclose shared filesystem/history, inherited prompts, model identity, and any communication between roles. A shared-agent result can supply several adversarial views, but shared provenance limits independence. Never manufacture agreement by relabeling one trace as several solvers.

## Sequential fallback

When delegation is unavailable, use a sequential fallback: run labeled solver, falsifier, defender, reconstruction, and adjudicator passes from the same frozen contract, clearing the favored narrative between passes as far as the runtime permits. Persist each pass before beginning the next. This preserves procedural separation and auditability, not independent evidence, consensus, or corroboration.
