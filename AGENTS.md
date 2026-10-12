# VFE_3.0 repository instructions

This is the active gauge-theoretic VFE transformer codebase (import package `vfe3`). It succeeded `V3_Transformer`, which is retired; do not edit or run anything there. Follow the user's global instructions. The README is the reference for architecture, configuration, and entrypoints.

## Running and testing

- Entrypoints are click-to-run: edit the configuration dictionary at the top of the script (`train_vfe3.py`, `ablation.py`, `scaling.py`, `make_figures.py`) and run it directly. There is no CLI configuration. The dictionaries hold the user's live settings; do not change them unless asked.
- Torch work uses `C:/anaconda/python.exe` (`$CUDA_PYTHON`). Bare `python` has CPU-only torch.
- The CPU lanes run through `run_cpu_tests.py`, which sets `VFE3_TEST_DEVICE=cpu`, hides CUDA, pins one thread per worker, and runs workers in parallel. For focused checks, run the relevant files under `tests/` directly.
- GPU tests need both the CUDA interpreter and `VFE3_TEST_DEVICE=cuda`; `check_gpu_tests.py` sets this up.
- Tests take their configuration from `tests/profiles.py` (`test_config`, `tiny_config`, `config_values`), never from `vfe3/config.py` defaults. A new config field needs an explicit test baseline value there; `tests/test_config_isolation.py` reports omissions.
- Every new `VFE3Config` field (toggle, mode or tunable value) must also be added to the `config` dictionary of `train_vfe3.py`, at its default value and in the group matching `vfe3/config.py`, with an inline comment listing its valid values. Adding the key is not a change to the user's live settings; changing an existing key's value is.

## Conventions

- Keep checkpoint parameter names and the `vfe3` import package stable; saved runs depend on them.
- Real-corpus loaders are cache-only and never substitute synthetic data for a missing corpus.
- Record each day's changes in `docs/<date>.md`.
