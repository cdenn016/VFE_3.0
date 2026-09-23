# Mechanism-diverse portfolio search

Index approach families by mechanism, never wording, solver identity, or field label. Do not broadcast a favorite route. Allocate effort toward unresolved obligations and underexplored mechanisms, and give early solvers only the frozen contract plus their assigned obligation.

Each registry entry records a family ID, representation, core mechanism, invariant or obstruction, target obligations, bridge, failure test, certified results, open gaps, novelty fingerprint, and disposition. Require every return in this form:

```text
family ID
LEMMA | CONSTRUCTION | COUNTEREXAMPLE | OBSTRUCTION
exact statement or object
types
quantifiers
assumptions
target obligations
derivation path
failure test
open gap
novelty fingerprint
proposed disposition
```

Reject status prose, confidence, examples without a quantified role, and claims that compatibility is routine. The novelty fingerprint is:

```text
representation + mechanism + invariant/obstruction + bridge + assumption set + target obligations
```

New notation, a different worker, another example, or greater confidence is not novelty. Preserve retired and refuted routes with the exact reason so they are not rediscovered. Retire a route when its core obligation is refuted, its required hypothesis violates the contract, or a follow-up repeats its fingerprint without removing a gap.

A route is mature only after it has a certified lemma or typed object, an explicit unresolved gap, and a serious failure test. Cross-pollinate mature routes only. Give each hybrid a new family ID, state an interface claim with exact input and output types, and test whether the parents' assumptions are jointly satisfiable. Hybrids inherit no verification state. Continue while an admissible new fingerprint can address an open obligation; otherwise release the strongest result without extending the search by repetition.
