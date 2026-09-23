# Code Criteria

Assess implementation claims against the current checked-out artifact. Score each applicable criterion from 0 to 20: execution of the claimed path, input and output behavior, boundary and failure behavior, regression coverage, configuration reachability, and reproducibility of the command or harness.

The code verifier runs or inspects a mechanical check tied to the current artifact revision. The skeptic searches for an unexercised branch, active-config mismatch, counterexample input, stale binary, or test that does not reach the claim. Record commands, test identifiers, exit statuses, and output locations as evidence rather than summarizing them from memory.

Only current `mechanical` or `reproduced_output` evidence can close a code correctness claim. `LLM_SUPPORTED` is triage-only. If closure is requested and execution is unavailable, use `INCONCLUSIVE` and state the missing command, environment, or fixture as an open obligation.
