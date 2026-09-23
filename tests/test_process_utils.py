import subprocess

import pytest

from vfe3 import path_utils, process_utils


class _FakeProcess:
    pid = 123
    returncode = 0
    _handle = 456

    def __init__(self, *, timeout_once: bool = False) -> None:
        self.timeout_once = timeout_once
        self.calls = 0

    def communicate(self, timeout=None):
        self.calls += 1
        if self.timeout_once and self.calls == 1:
            raise subprocess.TimeoutExpired(["worker"], timeout)
        return "stdout", "stderr"


class _InterruptingProcess(_FakeProcess):
    def communicate(self, timeout=None):
        self.calls += 1
        if self.calls == 1:
            raise KeyboardInterrupt
        return "stdout", "stderr"


class _NeverReapingProcess(_FakeProcess):
    def __init__(self) -> None:
        super().__init__()
        self.timeouts = []
        self.kills = 0

    def communicate(self, timeout=None):
        self.calls += 1
        self.timeouts.append(timeout)
        raise subprocess.TimeoutExpired(["worker"], timeout)

    def kill(self):
        self.kills += 1


class _Gate:
    def __init__(self) -> None:
        self.value = None
        self.closed = False

    def write(self, value):
        self.value = value

    def flush(self):
        return None

    def close(self):
        self.closed = True


class _FailingGate(_Gate):
    def flush(self):
        raise OSError("gate broke")


def test_run_process_tree_returns_success_without_signaling_exited_root(monkeypatch):
    captured = {}
    process = _FakeProcess()

    def _popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return process

    killed = []
    monkeypatch.setattr(process_utils.os, "name", "posix")
    monkeypatch.setattr(process_utils.subprocess, "Popen", _popen)
    monkeypatch.setattr(
        process_utils,
        "_kill_process_tree",
        lambda child, job: killed.append((child, job)),
    )

    completed = process_utils.run_process_tree(["worker"], capture_output=True, text=True)

    assert captured["start_new_session"] is True
    assert completed.stdout == "stdout"
    assert killed == []


def test_run_process_tree_terminates_whole_group_before_timeout_escapes(monkeypatch):
    process = _FakeProcess(timeout_once=True)
    monkeypatch.setattr(process_utils.os, "name", "posix")
    monkeypatch.setattr(process_utils.subprocess, "Popen", lambda *_args, **_kwargs: process)
    killed = []
    monkeypatch.setattr(
        process_utils,
        "_kill_process_tree",
        lambda child, job: killed.append((child, job)),
    )

    with pytest.raises(subprocess.TimeoutExpired):
        process_utils.run_process_tree(["worker"], timeout=1.0)

    assert killed == [(process, None)]
    assert process.calls == 2


def test_run_process_tree_terminates_and_reaps_after_base_exception(monkeypatch):
    process = _InterruptingProcess()
    monkeypatch.setattr(process_utils.os, "name", "posix")
    monkeypatch.setattr(process_utils.subprocess, "Popen", lambda *_args, **_kwargs: process)
    killed = []
    monkeypatch.setattr(
        process_utils,
        "_kill_process_tree",
        lambda child, job: killed.append((child, job)),
    )

    with pytest.raises(KeyboardInterrupt):
        process_utils.run_process_tree(["worker"])

    assert killed == [(process, None)]
    assert process.calls == 2


def test_timeout_cleanup_never_uses_an_unbounded_reap(monkeypatch):
    process = _NeverReapingProcess()
    monkeypatch.setattr(process_utils.os, "name", "posix")
    monkeypatch.setattr(process_utils.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(process_utils, "_kill_process_tree", lambda *_args: None)

    with pytest.raises(subprocess.TimeoutExpired):
        process_utils.run_process_tree(["worker"], timeout=1.0)

    assert process.timeouts == [
        1.0,
        process_utils._PROCESS_REAP_TIMEOUT_SECONDS,
        process_utils._PROCESS_REAP_TIMEOUT_SECONDS,
    ]
    assert process.kills == 1


def test_owned_output_child_rejects_reparse_point_before_use(tmp_path, monkeypatch):
    redirected = tmp_path / "figures"
    redirected.mkdir()
    monkeypatch.setattr(
        path_utils,
        "path_is_reparse_point",
        lambda path: path == redirected,
    )

    with pytest.raises(ValueError, match="symlink, junction, or reparse point"):
        path_utils.prepare_owned_output_child(
            tmp_path,
            "figures",
            role="single-run figure",
        )


def test_windows_workload_is_released_only_after_job_assignment(monkeypatch):
    process = _FakeProcess()
    gate = _Gate()
    process.stdin = gate
    captured = {}
    events = []

    class _Job:
        def assign(self, child):
            events.append(("assigned", child.stdin.value))

        def close(self):
            events.append(("closed", None))

    def _popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return process

    monkeypatch.setattr(process_utils.os, "name", "nt")
    monkeypatch.setattr(process_utils.subprocess, "Popen", _popen)
    monkeypatch.setattr(process_utils, "_WindowsJob", _Job)

    completed = process_utils.run_process_tree(["worker", "argument"], text=True)

    assert captured["command"][-3:] == ["vfe3-process-gate", "worker", "argument"]
    assert "sys.argv[2:]" in process_utils._WINDOWS_GATED_LAUNCHER
    assert events[0] == ("assigned", None)
    assert events[-1] == ("closed", None)
    assert gate.value == "1" and gate.closed is True
    assert completed.returncode == 0


def test_windows_assignment_failure_uses_bounded_cleanup(monkeypatch):
    process = _FakeProcess()
    process.stdin = _Gate()
    cleaned = []

    class _Job:
        def assign(self, _child):
            raise OSError("assignment failed")

        def close(self):
            return None

    monkeypatch.setattr(process_utils.os, "name", "nt")
    monkeypatch.setattr(process_utils.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(process_utils, "_WindowsJob", _Job)
    monkeypatch.setattr(
        process_utils,
        "_terminate_and_reap_after_interruption",
        lambda child, job: cleaned.append((child, job)),
    )

    with pytest.raises(OSError, match="contain the child"):
        process_utils.run_process_tree(["worker"])

    assert cleaned == [(process, None)]


def test_windows_gate_failure_uses_bounded_cleanup(monkeypatch):
    process = _FakeProcess()
    process.stdin = _FailingGate()
    cleaned = []

    class _Job:
        def assign(self, _child):
            return None

        def close(self):
            return None

    job = _Job()
    monkeypatch.setattr(process_utils.os, "name", "nt")
    monkeypatch.setattr(process_utils.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(process_utils, "_WindowsJob", lambda: job)
    monkeypatch.setattr(
        process_utils,
        "_terminate_and_reap_after_interruption",
        lambda child, owner: cleaned.append((child, owner)),
    )

    with pytest.raises(OSError, match="gate broke"):
        process_utils.run_process_tree(["worker"])

    assert cleaned == [(process, job)]


def test_persistent_windows_worker_is_released_only_after_job_assignment(monkeypatch):
    events = []

    class _OrderedGate(_Gate):
        def write(self, value):
            events.append(("released", value))
            super().write(value)

    process = _FakeProcess()
    process.stdin = _OrderedGate()
    captured = {}

    class _Job:
        def assign(self, child):
            events.append(("assigned", child.stdin.value))

        def close(self):
            return None

    def _popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return process

    monkeypatch.setattr(process_utils.os, "name", "nt")
    monkeypatch.setattr(process_utils.subprocess, "Popen", _popen)
    monkeypatch.setattr(process_utils, "_WindowsJob", _Job)

    tree = process_utils.spawn_process_tree(
        ["worker", "argument"], stdin=subprocess.PIPE, text=True,
    )

    assert captured["command"][-3:] == ["vfe3-process-gate", "worker", "argument"]
    assert events == [("assigned", None), ("released", "1")]
    assert process.stdin.closed is False
    assert tree.process is process


def test_persistent_windows_assignment_failure_never_releases_workload(monkeypatch):
    process = _FakeProcess()
    process.stdin = _Gate()
    cleaned = []
    captured = {}

    class _Job:
        def assign(self, _child):
            raise OSError("assignment failed")

        def close(self):
            return None

    def _popen(command, **kwargs):
        captured["command"] = command
        return process

    monkeypatch.setattr(process_utils.os, "name", "nt")
    monkeypatch.setattr(process_utils.subprocess, "Popen", _popen)
    monkeypatch.setattr(process_utils, "_WindowsJob", _Job)
    monkeypatch.setattr(
        process_utils,
        "_terminate_and_reap_after_interruption",
        lambda child, owner: cleaned.append((child, owner)),
    )

    with pytest.raises(OSError, match="contain the persistent child"):
        process_utils.spawn_process_tree(
            ["worker"], stdin=subprocess.PIPE, text=True,
        )

    assert "vfe3-process-gate" in captured["command"]
    assert process.stdin.value is None
    assert cleaned == [(process, None)]


class _ExitedRoot:
    pid = 321

    def __init__(self):
        self.kills = 0

    def poll(self):
        return 0

    def kill(self):
        self.kills += 1

    def wait(self, timeout=None):
        return 0


def test_nonzero_taskkill_with_exited_root_is_never_clean(monkeypatch):
    root = _ExitedRoot()

    class _Taskkill:
        def wait(self, timeout=None):
            return 5

    class _Job:
        def terminate(self):
            raise OSError("job terminate failed")

        def close(self):
            raise OSError("job close failed")

    monkeypatch.setattr(process_utils.os, "name", "nt")
    monkeypatch.setattr(process_utils.subprocess, "Popen", lambda *_a, **_k: _Taskkill())

    status = process_utils.ProcessTree(root, _Job()).terminate(reason="timeout", timeout=0.1)

    assert status["root_reaped"] is True
    assert status["tree_termination_confirmed"] is False
    assert "taskkill failed with exit code 5" in status["cleanup_error"]
    assert "job close failed" in status["cleanup_error"]


def test_taskkill_timeout_with_exited_root_is_never_clean(monkeypatch):
    root = _ExitedRoot()

    class _Taskkill:
        def __init__(self):
            self.killed = False

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(["taskkill"], timeout)

        def kill(self):
            self.killed = True

    monkeypatch.setattr(process_utils.os, "name", "nt")
    monkeypatch.setattr(process_utils.subprocess, "Popen", lambda *_a, **_k: _Taskkill())

    status = process_utils.ProcessTree(root).terminate(reason="timeout", timeout=0.05)

    assert status["root_reaped"] is True
    assert status["tree_termination_confirmed"] is False
    assert status["cleanup_error"]


def test_posix_exited_root_never_signals_stale_process_group(monkeypatch):
    root = _ExitedRoot()
    signaled = []
    monkeypatch.setattr(process_utils.os, "name", "posix")
    monkeypatch.setattr(
        process_utils.os,
        "killpg",
        lambda pid, sig: signaled.append((pid, sig)),
        raising=False,
    )
    monkeypatch.setattr(process_utils.signal, "SIGKILL", 9, raising=False)

    status = process_utils.ProcessTree(root).terminate(reason="close", timeout=0.05)

    assert signaled == []
    assert status["root_reaped"] is True
    assert status["tree_termination_confirmed"] is False
    assert "ownership" in status["cleanup_error"]
