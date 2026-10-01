# Test configuration isolation results, October 1, 2026

Tests now own complete local configuration mappings through `tests.profiles`. Three workers migrated disjoint test groups; the root agent handled profiles, configuration-isolation regressions, integration, and documentation. No production configuration or model change was authored by this task.

The whole CPU suite ran once with alternate constructor defaults, class attributes, dataclass metadata, and complete alternate training, ablation, and scaling dictionaries. Its JUnit report records 3,473 cases: 3,447 passed, ten failed, sixteen skipped, and zero errors in 647.332 seconds. All ten failing cases passed after their fixes in the recorded whole-file followups; their individual followup outcomes are listed in `test-summary.json`.

A later focused run included the concurrent covariance-centering feature and its new configuration field. It recorded 84 cases: 83 passed and one default-relative warning test failed. That test now locally selects the dataclass metadata its notice assertion requires. The entire nine-case centering module then passed under the same alternate defaults. These reports are separate runs; this folder does not claim a subsequent all-green whole-suite execution. The unrelated feature implementation changed during the session.

The fixes retain numerical goldens, thresholds, exact decoder exclusion checks, gradient assertions, and zero-device CUDA gates. The two decoder exclusion fixtures use deterministic tiny dyadic inputs and explicit precision so their exact comparisons do not depend on ambient initialization or GEMM rounding. The mixed-generation provenance fixture selects its own process and disk identities. Historical missing-field tests exercise the real loader with local fallback policies.

`profile-red.xml` records the two original failing tiny-profile reproductions. `profile-green.xml` records six passing isolation regressions. All 194 Python files under tests passed AST parsing; direct `VFE3Config` construction remains in the complete factory and in the launcher test receiving a complete local dictionary. The test and README whitespace diff check passed. GPU and external lanes were excluded.

Replay from the repository root in PowerShell, using the CUDA-capable interpreter for torch while explicitly selecting CPU:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:VFE3_TEST_DEVICE = 'cpu'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMBA_NUM_THREADS = '1'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
& 'C:/anaconda/python.exe' 'docs/test-results/config-isolation-2026-10-01/run_alternate_defaults.py'
```

The saved runners reproduce the validation setup against the current checkout, so a new run may differ from these historical outputs. Replay output is written under `.test-output/config-isolation`. Only this task's original scratch directory was removed after archiving these results.


Publication verification: 75 focused CPU cases passed with alternate defaults against the clean origin/main source, with zero failures, errors, or skips. The publication excludes the uncommitted covariance-centering implementation and its test module.
