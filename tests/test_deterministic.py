import os
import subprocess
import sys
from typing import Optional

import torch

from vfe3.config import VFE3Config


def test_deterministic_config_field_default_and_settable() -> None:
    assert VFE3Config().deterministic is True
    assert VFE3Config(deterministic=False).deterministic is False


def test_seed_everything_uses_config_deterministic_value() -> None:
    from vfe3.runtime import seed_everything
    cfg = VFE3Config()
    seed_everything(0, deterministic=cfg.deterministic)
    assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    seed_everything(0, deterministic=False)


def test_seed_everything_deterministic_toggle_sets_flags() -> None:
    from vfe3.runtime import seed_everything
    was = torch.are_deterministic_algorithms_enabled()
    was_cudnn = torch.backends.cudnn.deterministic
    try:
        seed_everything(0, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    finally:                                                       # restore so this does not leak into other tests
        torch.use_deterministic_algorithms(was)
        torch.backends.cudnn.deterministic = was_cudnn


def test_seed_everything_true_then_false_is_reversible() -> None:
    from vfe3.runtime import seed_everything
    seed_everything(1, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
    seed_everything(1, deterministic=False)
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True


def _cublas_restore_probe(initial_value: Optional[str]) -> str:
    env = os.environ.copy()
    if initial_value is None:
        env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        env["CUBLAS_WORKSPACE_CONFIG"] = initial_value
    code = (
        "import os\n"
        "from vfe3.runtime import seed_everything\n"
        "seed_everything(1, deterministic=True)\n"
        "assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'\n"
        "seed_everything(1, deterministic=False)\n"
        "print(os.environ.get('CUBLAS_WORKSPACE_CONFIG', '<absent>'))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=True,
        env=env,
        text=True,
    )
    return result.stdout.strip()


def test_seed_everything_restores_initially_absent_cublas_environment() -> None:
    assert _cublas_restore_probe(None) == "<absent>"


def test_seed_everything_restores_preexisting_cublas_environment() -> None:
    assert _cublas_restore_probe(":16:8") == ":16:8"
