"""Regression coverage for shared analysis utilities in the independent VFE copy."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from vfe3 import process_utils


def test_isolated_python_preserves_child_environment_and_streams(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    script = tmp_path / 'analysis.py'
    script.write_text(
        'import json, os, sys\n'
        'print(json.dumps([os.getcwd(), os.environ["_TEST_ANALYSIS_CHILD"], '
        'os.environ["KMP_DUPLICATE_LIB_OK"], os.environ["PYTHONUNBUFFERED"]]))\n'
        'print("child stderr", file=sys.stderr)\n'
        'sys.exit(7)\n', encoding='utf-8',
    )
    status = process_utils.run_isolated_python(str(script), '_TEST_ANALYSIS_CHILD', 10.0, 'test analysis')
    captured = capsys.readouterr()
    assert status == 7
    assert json.loads(captured.out) == [str(tmp_path), '1', 'TRUE', '1']
    assert captured.err == 'child stderr\n'


def test_isolated_python_has_bounded_timeout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    script = tmp_path / 'waiting_analysis.py'
    script.write_text('import time\ntime.sleep(5)\n', encoding='utf-8')
    assert process_utils.run_isolated_python(str(script), '_TEST_ANALYSIS_CHILD', 0.25, 'test analysis') == 1
    assert 'process tree' in capsys.readouterr().err


def test_process_helper_import_keeps_scientific_runtime_out_of_parent() -> None:
    result = subprocess.run(
        [sys.executable, '-c', 'import sys; from vfe3.process_utils import run_isolated_python; '
         'assert not {"torch", "numpy", "matplotlib"}.intersection(sys.modules)'],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_canonical_digest_preserves_scaling_bytes() -> None:
    from vfe3.artifact_utils import canonical_json_sha256
    expected = hashlib.sha256('{"a":-0.0,"z":[1,2.5],"μ":"é"}'.encode('utf-8')).hexdigest()
    value = {'μ': 'é', 'z': [1, 2.5], 'a': -0.0}
    assert canonical_json_sha256(value) == expected
    for name in ('scaling', 'scaling_analysis'):
        assert importlib.import_module(name)._canonical_json_sha256(value) == expected


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_canonical_digest_rejects_nonfinite_numbers(value: float) -> None:
    from vfe3.artifact_utils import canonical_json_sha256
    with pytest.raises(ValueError):
        canonical_json_sha256({'value': value})
