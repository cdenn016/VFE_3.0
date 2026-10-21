# Gauge-group buildouts: GL(m,C), SU(N), and other candidates (plan, 2026-10-05)

Planning document only. No code was changed. Line numbers are against `891cb1a`. Gauge fixing is
out of scope. Three read-only planning passes produced the material below; the load-bearing claims
marked (verified) were spot-checked against the code. Nothing was executed (no torch in the
planning environment), so every numeric claim is a test to write, not a result.

All proposals are additive: existing configs, checkpoint parameter names, and the `vfe3` package
are unchanged. New group names must avoid the retired set
`{"glk", "tied_block_glk", "so_k", "sp", "sp_n"}` (`config.py:81`, verified).

Update 2026-10-05: both head mixers (`use_head_mixer`, `use_priorbank_head_evidence_mixer`) were
removed in PR #45 (`389413b`). Every head-mixer item below was dropped: the mixer-label and
auto-disable seams, the `tied` builder attribute that existed only to drive them, `block_glmc`
test 8, `su_n` D8, R3, M3 and the mixer test, and the Plan A mixer-label test. Config line
numbers below are still against `891cb1a`.

Prerequisite (2026-10-05): no new group is built until the existing groups are complete; see
`docs/implementation/so-n-mixed-towers-and-frame-tying-plan-2026-10-05.md` (SO(N) mixed irrep
towers, then a frame-tying axis for `block_glk` and `so_n`).

Prerequisite status (2026-10-05): Phases 0-2 of that plan are merged, so new groups are unblocked.
The record is in `docs/2026-10-05.md`. The plan's Phase 3 and three Phase 1-2 items remain deferred.

- **Phase 0 (PR #46, `96c2273`).**
  - A deterministic canonical SO(3) irrep basis (`irrep_basis`, default `"canonical"`), which also
    lifts the l <= 6 cap.
  - A generator fingerprint stored in checkpoints and checked on load, and a CG re-check after load.
  - A certified Frobenius Gram for `so_n`.
  - Skew-symmetric groups are exempt from the fail-closed transport radius.
- **Phase 1 (PR #47, `3508579`).**
  - `so_n` heads may differ in size, with one irrep block per head and `n_heads` equal to the
    block count; for example `[("l3", 3), ("l8", 1)]`, K=38.
  - The relational operators are stored per run of equal-size heads.
  - Mixed towers run on both projected routes.
- **Phase 2 (PR #48, `30922c6`).**
  - A `gauge_tying` field and a `GaugeGroup.tying` attribute.
  - Tied `block_glk`: U = I_H ⊗ g, n_gen = d^2. Its per-block metrics, `pullback_group`,
    `fisher_group` and `group_log` are refused.
  - Untied `so_n`: one SO(N) element per block. CG coupling is refused and the per-block
    preconditioners are accepted.
- **Deferred.**
  - A compact fast path for tied `block_glk`; it runs on the dense route.
  - One host transfer per scoring call for mixed towers; R runs cost R transfers.
  - The ablation-cell and posthoc-report smoke tests.
  - Phase 3: compact per-run frame storage, a deterministic basis for SO(N) with N >= 4, and CG
    coupling across mixed towers.

Effect on this plan:
- **Survey #9 is done.** Untied `so_n` irreps per head ship as `gauge_tying="per_head"`.
- **Survey #2 is now a config, not a new group.** `block_so` (untied SO(d)^H) is
  `gauge_group="so_n"`, `group_n=d`, `irrep_spec=[("l1", H)]`, `gauge_tying="per_head"`. The block
  basis is exactly SO(d) in its defining representation on each head.
- **`su_n` should reuse the Phase 0-2 machinery.** Concretely, it needs:
  - the per-head `irrep_spec` partition, so its "one equal block per head" rule should be dropped;
  - the `gauge_tying` axis;
  - the fingerprint and canonical-basis pattern for R1;
  - the skew chart exemption, which removes R2 as a raised error.
- **Line numbers are stale.** The `config.py`, `transport.py`, `model.py`, `metrics.py` and
  `train.py` line numbers in this plan predate these merges.

## 0. Shared conclusions

- **Beliefs stay real.** Every group is realified onto real Gaussian beliefs. Complex-valued
  beliefs (circular CN(mu, R)) are the real Gaussians with Sigma commuting with a complex structure
  J. They would require a new family, conj-aware sandwiches everywhere, no bf16 kernels, and they
  break the float64 guards. Recommendation: do not build them. If the restriction is wanted, add a
  real `gaussian_proper` family (Sigma commutes with J) instead.
- **Complex dtype must not enter transport.** `stable_matrix_exp_pair` maps any non-float32/64
  dtype to float32 (`transport.py:711`), and the certified inverse calls `.double()`
  (`transport.py:988, 993`). Both silently drop imaginary parts (verified). All runtime exponentials
  stay real, on realified blocks. exp commutes with realification, so this is exact.
- **The irrep-tower seams are keyed on the name `so_n`.** Add an irrep-tower map
  `{"so_n": "so", "su_n": "su"}` (or an `algebra` attribute on `register_group`, read without
  building the group, like `declared_invariant_families`, `groups.py:375-382`) for the tower seams.
- **The fast paths are keyed on the elementary basis.** The compact (H,d,d) transport gate
  (`transport.py:1153-1159`), content-factor reuse (`model.py:1010`), the norm shortcut
  (`metrics.py:137`), `train.py:2347`, `semantics/run.py:411` and `group_log`
  (`gauge_optim.py:565`) all check `elementary_block_shape`. Every new group is correct on the
  dense route but slower. A generic "packed block codec" (section 1, D5) is the shared way to give
  non-elementary block groups the compact path.
- **phi is an AdamW-updated table.** `retract_phi` has no callers outside its own module
  (verified). `pullback_group` and `fisher_group` accept `block_glk` only, and fail closed for new
  groups (`config.py:1267, 1292`). So retraction defaults and Killing-metric degeneracy on central
  directions are not practical blockers.

## 1. GL(m,C) per head: `block_glmc`

**Group.** GL(m,C)^H realified, d_head = 2m (must be even). It is a real subgroup of `block_glk` at
the same d_head, with 2m^2 generators per head versus 4m^2: an inductive bias, not added
expressivity.

**Design decisions.**

- **D1. Registration.** `_build_block_glmc(K, n_heads, *, dtype, device)` registered next to
  `_build_block_glk` (`groups.py:342-372`). Fields:
  - `skew_symmetric=False`, `algebra=None`, `irrep_dims=[d_head]*H`;
  - `pullback_capability=None` (fail closed);
  - `frobenius_gram_diagonal=True`, `frobenius_gram_uniform=2.0`.

  With two positional arguments, `build_group` (`model.py:141-165`) needs no change.
- **D2. Interleaved layout.** Use J = blockdiag([[0,-1],[1,0]]), so complex coordinate k is the
  real pair (2k, 2k+1). RoPE writes exactly R = cI + sJ on those pairs (`rope.py:88-89`,
  verified). In this layout each RoPE block is diag(e^{i theta_k}), which lies in U(1)^m inside
  GL(m,C).
  - The folded frame stays inside the group, and RoPE becomes standard complex RoPE.
  - Unlike `so_n`, there is no structure-group warning (`config.py:1323`).
  - Reject the [Re; Im]-halves layout.
- **D3. Basis.** Generators are {E_ij (x) I_2, E_ij (x) J} with entries ±1 and no 1/sqrt 2 scaling.
  Coordinate order is (h, re/im, i, j), and the coordinates are exactly (Re, Im) of the complex
  algebra element.
  - Gram = 2I exactly, so `gram_diagonal_uniform == 2.0` is an exact certificate.
  - Initial matrix energy matches `block_glk`.
  - Coordinate-norm caps correspond to a sqrt 2 larger matrix norm (precedent: `so_n`). Document
    this.
- **D4. Exponential.** Real `matrix_exp` on realified blocks. Reject a complex exp at runtime (see
  section 0). It could be added later as a private packed-(A,B) kernel that realifies before any
  guard, for at most about 2x flops.
- **D5. Packed block codec (milestone 2).** A capability object on `GaugeGroup`:

  ```python
  @dataclass(frozen=True)
  class PackedBlockCodec:
      layout: str; n_heads: int; block_dim: int
      def to_blocks(self, phi): ...    # (..., n_gen) -> (..., H, d, d), differentiable
      def from_blocks(self, Z): ...    # exact on the algebra
  ```

  For `block_glmc`, `from_blocks` gives A = (Z[0::2,0::2] + Z[1::2,1::2]) / 2 and
  B = (Z[1::2,0::2] - Z[0::2,1::2]) / 2.
  - The `block_glk` branches stay literally unchanged and are byte-identical.
  - Add `elif` branches in `embed_algebra` (`groups.py:147-174`), the compact transport gate
    (`transport.py:1153-1194`), the `compose_bch` compact branch (`lie_ops.py:600-618`),
    `retraction.py:808-812`, and `model.py:757, 1010`.
  - The dense basis costs 2H^3 d^4 bytes (8 MB at K=64/H=4, about 1 GB at K=256/H=8). M2 is
    therefore required at scale; warn above about 64 MB until then.
- **D6. Centre C* = R+ x U(1).** `project_phi_to_slk` and `clamp_phi_trace` (`lie_ops.py:812-873`)
  use real traces, and tr(E_ij (x) J) = 0. So they remove only R+ and give |det_C| = 1 with the phase
  kept. That is the correct volume control, with no code change.
- **D7. J-blind learned maps (open question).** `raw_score_q/k`, `T`, `O`
  (`relational_operators.py:183-194`) and the block MLP are unconstrained real maps, so the complex
  bias lives only in the frames. An optional C-linear variant is research (M4).

**Diagnostics.**

- **`per_head_circularity(sigma, irrep_dims)`.** Split Sigma into R = J-commuting part and
  P = J-anticommuting part. Report the singular values of R^{-1/2} P R^{-T/2}. These are
  GL(m,C)-invariant per agent, in [0,1], and 0 for proper beliefs. Complex128 is acceptable here
  because this is diagnostic only. For diagonal Sigma it reduces to |s_2k - s_2k+1| / (s_2k + s_2k+1).
- **`per_head_complex_phase(exp_phi)`.** arg det_C, the U(1) class function that logdet cannot see.
- Wire both into `metrics.py`, and gate a summary in `posthoc/covariance.py:403` on the group name.

**Config.** No new fields. Rules:
- d_head even and >= 2;
- `group_n`, `irrep_spec`, `use_cg_coupling`, `pullback_group` and `fisher_group` are rejected
  (existing paths);
- also reject `m_phi_group_retraction="group_log"` at config time.

**Tests** (`tests/test_block_glmc.py`, tiny, CPU, float64):
1. Generator shape and entries in {0, ±1}.
2. [G_a, J] = 0.
3. Gram = 2I and the norm route.
4. Bracket closure.
5. exp(M) equals realify(matrix_exp(A + iB)) and commutes with J.
6. KL pushforward invariance (`test_gauge_groups.py:56`) and `check_admissible`.
7. Config accept/reject paths.
8. RoPE blocks and the folded frame commute with J.
9. `project_phi_to_slk` gives |det_C| = 1 with the phase kept.
10. Circularity: in range, GL(m,C)-invariant, 0 for proper Sigma, and the diagonal closed form.
11. Tiny forward/backward and a train step, with and without RoPE.

M2 adds:
- codec round-trip and parity with dense `embed_algebra`;
- compact vs dense transport parity;
- compact BCH parity;
- `block_glk` byte-identity regression.

**Milestones.**
- **M1, dense route** (~250 LOC + ~300 LOC tests): generator, builder, config seam, metrics,
  tests 1-11.
- **M2, codec and compact fast paths** (~350 LOC).
- **M3:** posthoc report column, viz/semantics branches, docs.
- **M4 (research):** C-linear relational/MLP option, a `gaussian_proper` family, and a complex-exp
  kernel benchmark.

**Ablations.** `block_glk` at equal K and at half K; log circularity to see whether the improper
part is used.

## 2. SU(N) irrep tower: `su_n`

**Group.** SU(N) with N = `group_n` decoupled from K. It is a tied gauge like `so_n` (one phi in
su(N) drives every block), with `skew_symmetric=True`.

**Design decisions.**

- **D1. Basis.** `generate_sun(N)` returns complex128 (N^2-1, N, N), X_a = i lambda_a (generalised
  Gell-Mann).
  - The first N(N-1)/2 generators are exactly `generate_son(N)`'s L_jk, so phi contains so(N) as a
    sub-slice.
  - tr(X_a^H X_b) = 2 delta_ab, the same scale as `so_n`, so retraction defaults carry over.
  - Structure constants are real.
- **D2. Irreps (p,q).** Traceless tensors in Sym^p(C^N) (x) Sym^q(conj C^N).
  - The dimension (code) is C(p+N-1,p)C(q+N-1,q) - C(p+N-2,p-1)C(q+N-2,q-1).
  - The Weyl formula is used in tests.
  - The Leibniz action applies X on fundamental slots and conj X on antifundamental slots.
  - The invariant subspace has a real orthonormal basis, so the existing `_invariant_basis` logic is
    reused through kron(S_p, S_q) plus a mixed contraction.
- **D3. Frobenius–Schur type from the label**, checked numerically at build.
  - N >= 3: p = q is real type; otherwise complex type.
  - N = 2: only q = 0 is allowed. Even p is real type; odd p is quaternionic (spinors).
- **D4. Real form per block.**
  - Complex and quaternionic types are realified to 2d: [[A,-B],[B,A]], which is skew and
    irreducible over R.
  - Real type gets a real basis at dimension d, via the antilinear J = conj composed with the
    real slot-swap permutation P. P_W is a real symmetric involution: eigen-split it and rotate
    the -1 eigenspace by i.
- **D5. Build-time checks** (raise on failure). These are added to the existing dimension and
  bracket-homomorphism checks:
  - imaginary residual;
  - skewness;
  - Casimir -sum rho^2 = 4 C2(p,q) I, where
    C2 = 1/2[(p^2+q^2)(N-1)/N + 2pq/N + (p+q)(N-1)].
    Check values: fundamental (N^2-1)/(2N); adjoint N.

  Test-only checks: real-commutant dimension (1 / 2 / 4) and FS type.
- **D6. New field.** `irrep_types: Optional[List[str]] = None` on `GaugeGroup` (additive).
- **D7. Centre.** Z_N acts on (p,q) as omega^{p-q}. An adjoint-only tower is PSU(N). This is
  documentation only.
- **D8. CG.** `cg_intertwiners` / `cg_selection` are type-agnostic. The only change is a `"su"`
  branch in `cg._defining` (`cg.py:47`).
  - SU(3) 8 (x) 8 -> 8 has real multiplicity 2 (f and d). The self-pair rule keeps the symmetric
    d-path, a live quadratic self-channel that SO(3) `l1` lacks.
  - Labels with p < q are rejected for N >= 3, because (1,0) and (0,1) are R-equivalent.

**File changes.**

- **`generators.py`:** `generate_sun`.
- **`irreps.py`:**
  - arity-aware `_parse_label` / `register_irrep(..., arity=1)`, so `so:l` stays byte-identical;
  - complex-aware `structure_constants` (real lstsq on `view_as_real`);
  - `_leibniz_action(..., conj_slots=0)`;
  - `_mixed_contraction`, `_mixed_invariant_basis`, `_su_real_form`, `_realify`, `su_fs_type`,
    `su_casimir`, `_su_pq_dim` (real dim), `_su_pq_build`;
  - `register_irrep("su:pq", ..., arity=2)`;
  - `_guard_cost(N, p+q)`.
- **`cg.py:47`:** the `"su"` branch.
- **`groups.py`:** factor `_build_irrep_tower(name, algebra, G_def, ...)` out of `_build_so_n`, with
  `so_n` output identical. Add `_build_su_n` and `irrep_types`.
- **`config.py`:** `_SUPPORTED_GAUGE_GROUPS`, and tower-map membership in place of `== "so_n"`
  at 1003, 1030, 1042-1052, 1066-1075, 1143, 1323 and 1380.
- **`metrics.py:1008`:** dispatch on `skew_symmetric` instead of `name == "so_n"`.
- **Docstrings:** `cg_coupling.py:81-100`.

**Config.** Reuse `group_n` and `irrep_spec`. The label grammar is `pq<p>_<q>`:

| Label | Irrep | Real dim |
|---|---|---|
| `pq0_0` | trivial | 1 |
| `pq1_0` | fundamental | 2N |
| `pq1_1` | adjoint | N^2 - 1 |
| `pq2_0` | Sym^2 | N(N+1) |

Validation:
- N = 2 requires q = 0; N >= 3 requires p >= q;
- the cost guard;
- dims sum to embed_dim;
- one block per head (blocks may differ in size since 2026-10-05; `n_heads` = block count).

Example: `gauge_group="su_n", group_n=3, irrep_spec=[("pq1_1", 2)], embed_dim=16, n_heads=2`.

**Tests** (`tests/test_sun_irreps.py`, additions to `tests/test_cg.py`):

1. Basis properties, and the leading slice equals `generate_son`.
2. Real structure constants.
3. Weyl dimension equals the subtraction formula. Spot values: SU(3) 3, 8, 6, 15, 27, 10;
   SU(4) adjoint 15; SU(5) adjoint 24.
4. Built real dims: SU(3) `pq1_0` 6, `pq1_1` 8, `pq2_0` 12, `pq2_2` 27; SU(2) `pq1_0` 4,
   `pq2_0` 3.
5. Skew, homomorphism, and Casimir: SU(3) 4/3, 3, 10/3, 8; SU(2) spin-1/2 3/4.
6. Commutant dimension / FS type. SU(2) `pq2_0` is R-equivalent to SO(3) `l1`.
7. Centre action.
8. Builder and admissibility.
9. Config errors.
10. `so_n` bitwise regression.
11. CG multiplicity 2 for 8 (x) 8 -> 8, and `CGCoupling` keeps d and drops f.
12. Projected train step, equivariance and phi finite-difference checks, mirroring
    `tests/test_son_irreps.py:167-250`.

**Risks.**

- **R1, basis determinism.** Generators are rebuilt at load, not persisted. Learned mu/phi tables
  are tied to the eigh/QR-chosen irrep basis, and degenerate eigenspaces make SU worse.
  - Mitigation: a canonical deterministic construction plus a stored fingerprint test.
  - Done for `so_n` (Phase 0: canonical SO(3) basis, `basis_fingerprint`, CG re-check after load);
    `su_n` needs its own canonical construction under the same fingerprint seam.
- **R2.** The phi chart radius is hit sooner for higher irreps (Dynkin index). Resolved for skew
  groups by the Phase 0 chart exemption, which `su_n` inherits through `skew_symmetric=True`.
- **R3.** The real-form construction must fail closed, never silently realify a real-type block
  into a reducible 2d block.
- **R4.** Tensor-power cost is about N^{p+q}. The CG guard d_a d_b d_c <= 5000 excludes 27 (x) 27.

**Milestones.**
- **M1** (~250 LOC + ~150 LOC tests): real-type only (p = q, N >= 3), plus builder, config and
  metrics.
- **M2** (~120 LOC): complex-type realification, FS and Casimir checks, CG `"su"` branch.
- **M3** (optional): SU(2) spinors, Young-tableau labels (Lambda^k for N >= 4), and shared
  deterministic-basis hardening with `so_n`.

**First experiment.** SU(3) adjoint-only `[("pq1_1", 2)]`, embed_dim 16, n_heads 2. Compare with
`so_n` N=8 `[("l1", 2)]` (same K, 28 vs 8 generators) and `so_n` N=3 `l1` heads. Note that this
run is PSU(3).

## 3. Other groups: survey and top picks

Score = usefulness x interest / cost (each 1-5).

| # | Candidate | Use | Int | Cost | Score | Route |
|---|---|---|---|---|---|---|
| 1 | `block_co`: CO(d)^H = R+ x SO(d) | 4 | 4 | 1.2 | 13.3 | block-local basis |
| 2 | `block_so`: untied SO(d)^H (available as untied `so_n` with `[("l1", H)]`) | 4 | 3 | 1 | 12 | block-local basis |
| 3 | `block_diag` (torus) / `block_scale` | 4 | 3 | 1 | 12 | block-local basis |
| 4 | `block_sl`: SL(d)^H | 3 | 3 | 1 | 9 | block-local basis |
| 5 | `block_borel`: B+(d)^H (+ unipotent / Heisenberg) | 3 | 3.5 | 1.2 | 8.8 | block-local basis |
| 6 | `block_lorentz`: SO(p,q)^H | 2.5 | 4 | 1.3 | 7.7 | block-local basis |
| 7 | Mixed per-head groups | 3 | 3 | 1.5 | 6 | block-local basis + list field |
| 8 | `block_sp`: Sp(d,R)^H, J on adjacent pairs | 2 | 3.5 | 1.2 | 5.8 | block-local basis |
| 9 | Untied `so_n` irreps per head (done: `gauge_tying="per_head"`) | 2.5 | 3 | 1.5 | 5 | irrep tower |
| 10 | Spin(N) spinor irreps | 2 | 4.5 | 3 | 3 | new irrep family |
| 11 | G2 in SO(7) | 1.5 | 4 | 2.5 | 2.4 | new invariant subspaces |
| 12 | Aff(K) | 3 | 4 | 5 | 2.4 | breaks K x K frames |
| 13 | O(N) with reflections | 1.5 | 2 | 2.5 | 1.2 | discrete per-token component |
| 14 | Permutations / discrete | 1 | 2 | 3 | <1 | not differentiable |

**Notes.**

- **CO(d).**
  - exp is surjective.
  - Each block is lambda R, so cond(Omega_h) = 1 even though the group is non-compact.
  - It preserves angles and eigenvalue ratios: a per-token, per-head gain (salience/precision)
    plus a rotation.
  - RoPE stays inside the group.
- **Untied SO(d).** Fills the gap between tied `so_n` `[("l1", H)]` and `block_glk`. It is
  compact with det = 1, and gives a clean "rotations only" ablation.
- **Torus / scale.**
  - The only candidate that keeps diagonal Sigma diagonal. It can therefore truthfully declare the
    default `gaussian_diagonal` family invariant.
  - No registered group does today, so `family_group_invariant` (`run_artifacts.py:4616`) is False
    for the default config.
  - It is the cheapest "do frames need mixing?" baseline.
- **SL(d).**
  - Volume preserving, so the projected_gradient receiver-energy log-det shift
    (`viz/extract.py:807-809`) is exactly zero.
  - It formalises the partial `project_phi_to_slk` already applied to positional coordinates
    (`model.py:773`).
- **Borel.**
  - exp is a diffeomorphism.
  - It preserves the coordinate flag, a hierarchical ordering of channels, and acts transitively
    on SPD matrices (Cholesky).
  - Iwasawa factorisation GL = SO B+ means `block_so` + `block_borel` decompose `block_glk`.
- **SO(p,q).** Preserves mu^T eta mu, which links to hyperbolic (Lorentz-model) embeddings for
  hierarchies. Boosts make cond unbounded. It needs p/q config fields.
- **Sp(d,R).** A per-head re-introduction of the retired `sp` (retired for projected-route block
  structure, not for a scientific failure; see `docs/2026-10-02.md`). RoPE lies in U(d/2), inside
  Sp. Williamson eigenvalues are invariants.
- **Aff(K), O(N), discrete.** Aff(K) would need homogeneous (K+1) x (K+1) frames through transport,
  the relational maps, the MLP and decode. O(N) and discrete groups need non-differentiable
  per-token components. Not recommended.

**Shared infrastructure first** (~150 LOC + ~150 LOC tests).

1. **`generators.py`.**
   - Orthonormal local bases: `local_basis_{so,co,diag,scale,sl,borel}(d)`.
   - `generate_block_local(K, n_heads, local, *, dtype, device)` with a contiguous per-head
     partition.
2. **`groups.py`.**
   - `_build_block_local(name, K, n_heads, local, *, skew, dtype, device)`, with
     `frobenius_gram_uniform=1.0`, `phi_coordinate_layout=None`, and an optional
     `block_local_basis` field.
3. **`config.py`.**
   - Extend `_SUPPORTED_GAUGE_GROUPS`.
   - A `_check_block_local_group()` seam.
   - Generalise the RoPE warning to a builder attribute `contains_pair_rotations` (False for diag,
     scale and borel).
4. **Before any large-K run:** per-head embedding in `embed_algebra`,
   `einsum("...ha,aij->...hij")` plus a slice-assign. This replaces the O(n_gen K^2) dense einsum.
   The `block_sl` dense basis at K=64/H=2 is 8.4M floats. The packed codec from section 1 D5 is the
   natural shared form of this step.

**Plan A: `block_so` + `block_co` (top pick).** Two builders `(K, n_heads, *, dtype, device)`,
built in float64 and then cast. No config fields.

Tests (`tests/test_block_local_groups.py`):
1. Structure.
2. Bracket closure.
3. exp lands in the group (U^T U = I and det = 1; U^T U = lambda^2 I per block).
4. `check_admissible`, with `gaussian_diagonal` as the negative control.
5. Config accept/reject paths.
6. Projected train step, equivariance and phi finite-difference checks (template
   `tests/test_son_irreps.py:206-235`).

Size: ~40 LOC + ~120 LOC tests.

**Plan B: `block_diag` + `block_scale` — DROPPED (2026-10-05, user decision; kept for reference).**
- Declare `("gaussian_full", "gaussian_diagonal")` invariant, so the purity report gives True for
  the default family.
- Optionally declare `laplace_diagonal` via an additive `LaplaceDiagonal.from_covariance`.
- Optional fast path: `phi_coordinate_layout="block_diagonal_entries"` with elementwise exp(±phi).
- RoPE warning set.

Size: ~30 LOC (+~40 optional) + ~80 LOC tests.

**Plan C: `block_sl`, then `block_borel` / `block_unipotent`.**
- Helmert-orthonormal traceless diagonal (Gram = I).
- `pos_phi_project_slk` becomes inert under `block_sl`.
- Equivariance test asserting zero log-det shift.

Size: ~50 LOC + ~100 LOC tests.

## 4. Recommended order

1. **Shared seams** (the section 0 irrep-tower map) and the block-local infrastructure
   (section 3, steps 1-3).
2. **Plan A** (`block_so`, `block_co`). This gives the cheap untied ablation ladder:
   SO -> CO -> SL -> GL. Plan B (diagonal/scale) is dropped.
3. **`block_glmc` M1**, then the packed codec. The codec is shared with section 3 step 4 and serves
   every non-elementary block group.
4. **`su_n` M1** (adjoint-only SU(3)), then M2. The `so_n` basis-determinism check (R1) is done
   (Phase 0).
5. **Later:** Plan C, `block_sp`, `block_lorentz`, mixed per-head groups; SU(2) spinors and Spin(N).
