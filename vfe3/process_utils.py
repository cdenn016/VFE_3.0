"""Subprocess helpers that contain every descendant in one disposable process tree."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import monotonic, sleep
from ctypes import wintypes
from typing import Mapping, Optional, Sequence


_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_BASIC_PROCESS_ID_LIST = 3
_ERROR_MORE_DATA = 234
_JOB_DRAIN_POLL_SECONDS = 0.005
_PROCESS_REAP_TIMEOUT_SECONDS = 10.0
_TASKKILL_TIMEOUT_SECONDS = 10.0
_WINDOWS_GATED_LAUNCHER = (
    "import os,subprocess,sys; "
    "gate=os.read(sys.stdin.fileno(),1); "
    "sys.exit(125) if gate != b'1' else None; "
    "sys.exit(126) if len(sys.argv) < 3 or sys.argv[1] != 'vfe3-process-gate' else None; "
    "child=subprocess.Popen(sys.argv[2:]); "
    "sys.exit(child.wait())"
)


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION_STRUCT(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _JOBOBJECT_BASIC_PROCESS_ID_LIST(ctypes.Structure):
    """Header of the job's live-PID listing; only the two counts are read back here."""

    _fields_ = [
        ("NumberOfAssignedProcesses", wintypes.DWORD),
        ("NumberOfProcessIdsInList", wintypes.DWORD),
        ("ProcessIdList", ctypes.c_size_t * 1),
    ]


class _WindowsJob:
    """Own one Windows Job Object whose closure kills all assigned descendants."""

    def __init__(self) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        information = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION_STRUCT()
        information.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            error = ctypes.get_last_error()
            kernel32.CloseHandle(handle)
            raise ctypes.WinError(error)
        self._kernel32 = kernel32
        self._handle = handle

    def assign(self, process: subprocess.Popen[str]) -> None:
        if not self._kernel32.AssignProcessToJobObject(self._handle, process._handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def terminate(self, exit_code: int = 1) -> None:
        if self._handle and not self._kernel32.TerminateJobObject(self._handle, exit_code):
            raise ctypes.WinError(ctypes.get_last_error())

    def active_process_count(self) -> Optional[int]:
        """Processes still assigned to the job (0 once drained, None when the query fails)."""
        if not self._handle:
            return 0
        listing = _JOBOBJECT_BASIC_PROCESS_ID_LIST()
        returned = wintypes.DWORD(0)
        ok = self._kernel32.QueryInformationJobObject(
            self._handle,
            _JOB_OBJECT_BASIC_PROCESS_ID_LIST,
            ctypes.byref(listing),
            ctypes.sizeof(listing),
            ctypes.byref(returned),
        )
        # A job holding more than one live PID overflows the single-slot listing, but Windows still
        # fills both counts before failing with ERROR_MORE_DATA -- that is a valid "not drained yet".
        if not ok and ctypes.get_last_error() != _ERROR_MORE_DATA:
            return None
        return int(listing.NumberOfAssignedProcesses)

    def close(self) -> None:
        if self._handle:
            handle = self._handle
            self._handle = None
            if not self._kernel32.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())


def _kill_process_tree(
    process: subprocess.Popen[str],
    job:     Optional[_WindowsJob],

    *,
    timeout: float = _TASKKILL_TIMEOUT_SECONDS,
) -> bool:
    """Terminate a child tree and return only after platform confirmation."""
    if timeout <= 0.0:
        raise ValueError("process-tree termination timeout must be positive")
    if os.name == "nt":
        deadline = monotonic() + timeout
        if job is not None:
            try:
                job.terminate()
            except OSError:
                pass
            else:
                # TerminateJobObject is ASYNCHRONOUS: it signals every assigned process and returns
                # immediately, so the root can be reaped while a grandchild (the gated launcher
                # spawns the real workload as its own child) is still alive holding the inherited
                # stderr handle. Deleting that log then fails with WinError 32. Confirm the job has
                # actually drained before reporting the tree terminated.
                while True:
                    remaining_processes = job.active_process_count()
                    if remaining_processes == 0:
                        return True
                    if remaining_processes is None or monotonic() >= deadline:
                        return False
                    sleep(min(_JOB_DRAIN_POLL_SECONDS, max(0.0, deadline - monotonic())))
        taskkill = subprocess.Popen(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            returncode = taskkill.wait(timeout=max(0.001, deadline - monotonic()))
        except subprocess.TimeoutExpired as exc:
            try:
                taskkill.kill()
                remaining = deadline - monotonic()
                if remaining > 0.0:
                    taskkill.wait(timeout=remaining)
            except BaseException:
                pass
            raise TimeoutError("taskkill did not finish within its bounded timeout") from exc
        if returncode != 0:
            raise OSError(f"taskkill failed with exit code {returncode}")
        return True
    if process.poll() is not None:
        raise OSError(
            "process-group ownership unavailable after root exit; killpg skipped"
        )
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return True


def _terminate_and_reap_after_interruption(
    process: subprocess.Popen[str],
    job:     Optional[_WindowsJob],
) -> None:
    """Best-effort tree termination and reaping without masking an active ``BaseException``."""
    try:
        _kill_process_tree(process, job)
    except BaseException:
        try:
            process.kill()
        except BaseException:
            pass
    try:
        process.communicate(timeout=_PROCESS_REAP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.communicate(timeout=_PROCESS_REAP_TIMEOUT_SECONDS)
        except BaseException:
            pass
    except BaseException:
        pass


@dataclass
class ProcessTree:
    """One subprocess and the platform containment primitive that owns its descendants."""

    process: subprocess.Popen
    _job: Optional[_WindowsJob] = None

    def terminate(self, *, reason: str, timeout: float) -> dict[str, object]:
        """Terminate only this tree and return bounded, structured cleanup diagnostics."""
        if timeout <= 0.0:
            raise ValueError("process-tree cleanup timeout must be positive")
        started = monotonic()
        deadline = started + timeout
        process = self.process
        status: dict[str, object] = {
            "pid": int(process.pid),
            "reason": str(reason),
            "terminated": True,
            "reaped": False,
            "root_reaped": False,
            "tree_termination_confirmed": False,
            "returncode": process.poll(),
            "cleanup_error": None,
            "elapsed_s": 0.0,
        }
        cleanup_errors: list[str] = []
        try:
            try:
                status["tree_termination_confirmed"] = bool(_kill_process_tree(
                    process,
                    self._job,
                    timeout=max(0.001, deadline - monotonic()),
                ))
            except BaseException as exc:
                cleanup_errors.append(f"{type(exc).__name__}: {exc}")
                if process.poll() is None:
                    try:
                        process.kill()
                    except BaseException as kill_exc:
                        cleanup_errors.append(f"{type(kill_exc).__name__}: {kill_exc}")
            remaining = deadline - monotonic()
            if process.poll() is None and remaining > 0.0:
                try:
                    process.wait(timeout=remaining)
                except subprocess.TimeoutExpired as exc:
                    cleanup_errors.append(f"{type(exc).__name__}: post-kill wait exceeded {timeout:g}s")
                except BaseException as exc:
                    cleanup_errors.append(f"{type(exc).__name__}: {exc}")
            elif process.poll() is None:
                cleanup_errors.append("post-kill wait budget exhausted")
            status["returncode"] = process.poll()
            status["root_reaped"] = status["returncode"] is not None
            status["reaped"] = status["root_reaped"]
        finally:
            if self._job is not None:
                try:
                    self._job.close()
                except BaseException as exc:
                    cleanup_errors.append(f"{type(exc).__name__}: {exc}")
                self._job = None
            status["cleanup_error"] = " | ".join(cleanup_errors) or None
            status["elapsed_s"] = monotonic() - started
        return status


def spawn_process_tree(
    command: Sequence[str],

    **popen_kwargs: object,
) -> ProcessTree:
    """Start a persistent subprocess behind the Windows pre-containment gate."""
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
        raise TypeError("command must be a nonempty sequence of strings")
    if not command:
        raise ValueError("command must be a nonempty sequence of strings")
    if any(not isinstance(argument, str) or not argument for argument in command):
        raise ValueError("command must contain only nonempty strings")
    kwargs = dict(popen_kwargs)
    if os.name == "nt":
        if kwargs.get("stdin") is not subprocess.PIPE:
            raise ValueError("persistent Windows process trees require stdin=subprocess.PIPE")
        kwargs["creationflags"] = (
            int(kwargs.get("creationflags", 0))
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
        launch_command = [
            sys.executable,
            "-I",
            "-S",
            "-c",
            _WINDOWS_GATED_LAUNCHER,
            "vfe3-process-gate",
            *command,
        ]
    else:
        kwargs["start_new_session"] = True
        launch_command = list(command)
    process = subprocess.Popen(launch_command, **kwargs)
    job: Optional[_WindowsJob] = None
    if os.name == "nt":
        try:
            job = _WindowsJob()
            job.assign(process)
        except OSError as exc:
            if job is not None:
                try:
                    job.close()
                except BaseException:
                    pass
            _terminate_and_reap_after_interruption(process, None)
            raise OSError("could not contain the persistent child in a Windows Job Object") from exc
        gate = process.stdin
        if gate is None:
            _terminate_and_reap_after_interruption(process, job)
            raise OSError("persistent Windows process gate was not created")
        try:
            gate.write("1" if kwargs.get("text") else b"1")
            gate.flush()
        except BaseException:
            _terminate_and_reap_after_interruption(process, job)
            raise
    return ProcessTree(process=process, _job=job)


def run_process_tree(
    command: Sequence[str],

    *,
    creationflags:  int                         = 0,
    capture_output: bool                        = False,
    text:           bool                        = False,
    cwd:            Optional[str]               = None,
    env:            Optional[Mapping[str, str]] = None,
    timeout:        Optional[float]             = None,
    encoding:       Optional[str]               = None,
    errors:         Optional[str]               = None,
) -> 'subprocess.CompletedProcess[bytes] | subprocess.CompletedProcess[str]':
    r"""Run ``command`` in a process tree that is destroyed on timeout or parent exit.

    Windows descendants are assigned to a Job Object with
    ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``. POSIX descendants share a fresh process group. This
    prevents native plotting workers, UMAP helpers, or xdist workers from surviving a timeout.
    """
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
        raise TypeError("command must be a nonempty sequence of strings")
    if not command:
        raise ValueError("command must be a nonempty sequence of strings")
    if any(not isinstance(argument, str) for argument in command):
        raise TypeError("command must contain only strings")
    if any(not argument for argument in command):
        raise ValueError("command must not contain empty strings")

    stdout = subprocess.PIPE if capture_output else None
    stderr = subprocess.PIPE if capture_output else None
    popen_kwargs: dict[str, object] = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = (
            int(creationflags) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    else:
        popen_kwargs["start_new_session"] = True

    launch_command = (
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            _WINDOWS_GATED_LAUNCHER,
            "vfe3-process-gate",
            *command,
        ]
        if os.name == "nt"
        else list(command)
    )
    process = subprocess.Popen(
        launch_command,
        cwd=cwd,
        env=(dict(env) if env is not None else None),
        stdin=(subprocess.PIPE if os.name == "nt" else None),
        stdout=stdout,
        stderr=stderr,
        text=text,
        encoding=encoding,
        errors=errors,
        **popen_kwargs,
    )
    job: Optional[_WindowsJob] = None
    try:
        if os.name == "nt":
            try:
                job = _WindowsJob()
                job.assign(process)
            except OSError as exc:
                if job is not None:
                    job.close()
                job = None
                _terminate_and_reap_after_interruption(process, None)
                raise OSError(
                    "could not contain the child in a Windows Job Object"
                ) from exc
            gate = process.stdin
            if gate is None:
                _terminate_and_reap_after_interruption(process, job)
                raise OSError("Windows process gate was not created")
            try:
                gate.write("1" if text else b"1")
                gate.flush()
                gate.close()
            except BaseException:
                _terminate_and_reap_after_interruption(process, job)
                raise
            process.stdin = None
        try:
            output, error = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            cleanup_errors = []
            try:
                _kill_process_tree(process, job)
            except BaseException as cleanup_exc:
                cleanup_errors.append(cleanup_exc)
                if job is not None:
                    try:
                        job.close()
                    except BaseException as close_exc:
                        cleanup_errors.append(close_exc)
                    job = None
            try:
                output, error = process.communicate(timeout=_PROCESS_REAP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired as reap_exc:
                cleanup_errors.append(reap_exc)
                try:
                    process.kill()
                    output, error = process.communicate(
                        timeout=_PROCESS_REAP_TIMEOUT_SECONDS,
                    )
                except BaseException as kill_exc:
                    cleanup_errors.append(kill_exc)
                    output = error = None
            except BaseException as reap_exc:
                cleanup_errors.append(reap_exc)
                output = error = None
            exc.output = output
            exc.stderr = error
            add_note = getattr(exc, "add_note", None)
            if callable(add_note):
                for cleanup_error in cleanup_errors:
                    add_note(
                        "process-tree cleanup diagnostic: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            raise
        except BaseException:
            _terminate_and_reap_after_interruption(process, job)
            raise
        completed = subprocess.CompletedProcess(list(command), process.returncode, output, error)
        return completed
    finally:
        if job is not None:
            job.close()


def run_isolated_python(
    script_path: str,
    sentinel:    str,
    timeout:     float,
    label:       str,
) -> int:
    """Run an analysis script with isolated native runtimes and bounded child lifetime."""
    script = Path(script_path).resolve()
    environment = os.environ.copy()
    environment[sentinel] = "1"
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    environment["PYTHONUNBUFFERED"] = "1"
    try:
        completed = run_process_tree(
            [sys.executable, str(script)],
            cwd=str(script.parent),
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        print(
            f"isolated worker exceeded {timeout:g}s and was terminated with its process tree",
            file=sys.stderr,
        )
        return 1
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"isolated {label} worker could not start: {exc}", file=sys.stderr)
        return 1
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return int(completed.returncode)
