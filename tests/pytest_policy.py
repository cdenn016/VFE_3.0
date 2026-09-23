import os

import pytest
import torch


SLOW_TESTS: frozenset[str] = frozenset()

CUDA_TESTS: frozenset[str] = frozenset({
    'test_laplace_family.py::test_laplace_cuda_matches_cpu',
    'test_p3_pairwise_stats_reuse_20260711.py::test_p3_cuda_filtering_and_mm_reuse_smoke',
})

CUDA_MIRROR_TESTS: frozenset[str] = frozenset({
    'test_divergence.py::test_safe_kl_clamp_bounds_and_nan',
    'test_free_energy.py::test_free_energy_entropy_exact_for_deep_finite_prior',
    'test_retraction.py::test_full_retraction_stays_spd',
})

EXTERNAL_TESTS: frozenset[str] = frozenset()

RESOURCE_GROUPS: dict[str, str] = {
    **{key: "cuda" for key in CUDA_TESTS},
}


def node_key(nodeid: str) -> str:
    """Return the file/function policy key for a collected pytest node ID."""
    tail = nodeid.replace("\\", "/").split("/")[-1]
    return tail.split("[")[0]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run the slow figure/artifact integration tests (skipped by default)",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_slow = config.getoption("--runslow")
    skip_slow = pytest.mark.skip(reason="slow integration test; pass --runslow to run it")
    mirror_cuda = torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu")).type == "cuda"

    for item in items:
        key = node_key(item.nodeid)
        if key in SLOW_TESTS:
            item.add_marker(pytest.mark.slow)
        if key in CUDA_TESTS or (mirror_cuda and key in CUDA_MIRROR_TESTS):
            item.add_marker(pytest.mark.cuda)
        if key in EXTERNAL_TESTS:
            item.add_marker(pytest.mark.external)
        if key in RESOURCE_GROUPS:
            item.add_marker(pytest.mark.xdist_group(name=RESOURCE_GROUPS[key]))
        elif mirror_cuda and key in CUDA_MIRROR_TESTS:
            item.add_marker(pytest.mark.xdist_group(name="cuda"))
        if key in SLOW_TESTS and not run_slow:
            item.add_marker(skip_slow)
