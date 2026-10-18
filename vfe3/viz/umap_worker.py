r"""Crash-isolated UMAP subprocess for :func:`vfe3.viz.figures.umap_embed`.

umap-learn's numba/llvmlite native layer can fault inside a long-running process, so every embedding
runs in a separate interpreter that this module starts, feeds through a JSON line protocol and tears
down. Moved out of :mod:`vfe3.viz.figures` and reduced to one reusable worker (audit 2026-10-17
item 76).
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from threading import Event, RLock
from types import TracebackType
from typing import Dict, List, Optional
from uuid import uuid4

import numpy as np

from vfe3.process_utils import ProcessTree, spawn_process_tree


# Worker source for the ISOLATED UMAP embedding subprocess (see UMAPWorker).
# init="pca" skips UMAP's spectral eigensolver, which on disconnected / near-degenerate belief
# clouds fails to converge (tiny eigengap) and silently falls back to uniform-random init; PCA is
# deterministic and data-aware. n_jobs=1 matches what random_state already forces, so it also
# silences UMAP's "n_jobs overridden" warning. Same compute, no warning cascade.
_UMAP_WORKER_SRC = (
    "import json, os, sys, traceback\n"
    "import numpy as np\n"
    "import umap\n"
    "for line in sys.stdin:\n"
    "    request = json.loads(line)\n"
    "    try:\n"
    "        X = np.load(request['input'])\n"
    "        reducer = umap.UMAP(n_neighbors=int(request['n_neighbors']),\n"
    "                            min_dist=float(request['min_dist']),\n"
    "                            n_components=int(request['n_components']), init='pca',\n"
    "                            random_state=int(request['seed']), n_jobs=1)\n"
    "        np.save(request['output'], reducer.fit_transform(X))\n"
    "        response = {'ok': True}\n"
    "    except BaseException:\n"
    "        response = {'ok': False, 'error': traceback.format_exc()[-4000:]}\n"
    "    status_tmp = request['status'] + '.tmp'\n"
    "    with open(status_tmp, 'w', encoding='utf-8') as handle:\n"
    "        json.dump(response, handle)\n"
    "    os.replace(status_tmp, request['status'])\n"
)


_STDERR_UNLINK_POLL_SECONDS = 0.005      # Windows handle-release lag after a drained process tree


def _numba_cache_dir() -> Optional[str]:
    r"""A PERSISTENT numba cache for the UMAP workers; None means "inherit numba's own default".

    The worker used to point ``NUMBA_CACHE_DIR`` at its request tempdir, which ``close`` then
    removed, so umap-learn's numba kernels were JIT-compiled from scratch on every report and the
    cache never survived a single run. An explicitly set ``NUMBA_CACHE_DIR`` in the parent
    environment wins; otherwise the first writable candidate is used.
    """
    import os
    import tempfile
    inherited = os.environ.get("NUMBA_CACHE_DIR")
    if inherited:
        return inherited
    candidates = (Path.home() / ".cache" / "vfe3" / "numba",
                  Path(tempfile.gettempdir()) / "vfe3-numba-cache")
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return str(candidate)
        except OSError:
            continue
    return None


@dataclass(eq=False)
class _UMAPWorkerRecord:
    """Identity-bearing lifecycle state for one owned worker process."""

    tree: ProcessTree
    stderr_handle: object
    identity: str = field(default_factory=lambda: uuid4().hex)
    retiring: bool = False
    retired: bool = False
    requests_submitted: int = 0
    retired_event: Event = field(default_factory=Event)
    cleanup_status: Optional[Dict[str, object]] = None

    @property
    def process(self):
        return self.tree.process


class UMAPWorker:
    r"""A lazily started, crash-isolated UMAP interpreter that a caller may reuse across embeddings.

    The interpreter starts on the first :meth:`embed` and serves every later request until
    :meth:`close`. Its only caller (:func:`vfe3.viz.figures.umap_embed`, from the semantics report)
    asks for one seeded embedding at a time, so the multi-worker pool and ``embed_many`` that served
    the retired multi-seed protocol were removed (audit 2026-10-17 item 76).
    """

    def __init__(
        self,

        *,
        timeout:         float = 1200.0,
        cleanup_timeout: float = 5.0,
    ) -> None:
        if timeout <= 0.0:
            raise ValueError("UMAP worker timeout must be positive")
        if cleanup_timeout <= 0.0:
            raise ValueError("UMAP cleanup timeout must be positive")
        self.timeout = timeout
        self.cleanup_timeout = cleanup_timeout
        self._counter = 0
        self._workers: List[_UMAPWorkerRecord] = []
        self._lifecycle_lock = RLock()
        self._closing = False
        self._request_paths: set[str] = set()
        self.cleanup_statuses: List[Dict[str, object]] = []
        self._workdir = None

    def _record_cleanup_failure(self, *, reason: str, path: str, exc: BaseException) -> dict:
        status: Dict[str, object] = {
            "pid": None,
            "reason": reason,
            "terminated": False,
            "reaped": False,
            "root_reaped": False,
            "tree_termination_confirmed": False,
            "returncode": None,
            "cleanup_error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": 0.0,
            "path": path,
        }
        with self._lifecycle_lock:
            self.cleanup_statuses.append(status)
        return status

    def _cleanup_request_path(self, path: str) -> None:
        try:
            os.remove(path)
        except FileNotFoundError:
            with self._lifecycle_lock:
                self._request_paths.discard(path)
        except OSError as exc:
            self._record_cleanup_failure(reason="request_cleanup", path=path, exc=exc)
        else:
            with self._lifecycle_lock:
                self._request_paths.discard(path)

    def _ensure(self) -> None:
        r"""Guarantee one live interpreter."""
        import subprocess
        import sys
        import tempfile
        with self._lifecycle_lock:
            if self._closing:
                raise RuntimeError("UMAP worker pool is closing")
            if self._workdir is None:
                self._workdir = tempfile.mkdtemp(prefix="vfe3_umap_")
            cache_dir = _numba_cache_dir()
            if not self._workers:
                worker_env = os.environ.copy()
                if cache_dir is None:
                    worker_env.pop("NUMBA_CACHE_DIR", None)
                else:
                    worker_env["NUMBA_CACHE_DIR"] = cache_dir
                handle = open(os.path.join(self._workdir, f"stderr_{uuid4().hex}.log"), "w+b")
                try:
                    tree = spawn_process_tree(
                        [sys.executable, "-c", _UMAP_WORKER_SRC],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=handle,
                        text=True,
                        env=worker_env,
                    )
                except BaseException:
                    handle.close()
                    raise
                self._workers.append(_UMAPWorkerRecord(tree=tree, stderr_handle=handle))

    def _error_tail(self, record: _UMAPWorkerRecord) -> str:
        stderr_handle = record.stderr_handle
        try:
            stderr_handle.flush()
        except (OSError, ValueError):
            return ""
        try:
            with open(stderr_handle.name, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                handle.seek(max(0, handle.tell() - 1000), os.SEEK_SET)
                return handle.read().decode(errors="replace")
        except (OSError, ValueError):
            return ""

    def _retire(
        self,
        record: _UMAPWorkerRecord,
        *,
        reason: str,
    ) -> Optional[Dict[str, object]]:
        """Retire one identity exactly once, without holding the lock while waiting."""
        with self._lifecycle_lock:
            if record.retired:
                return record.cleanup_status
            if record.retiring:
                owner = False
            else:
                record.retiring = True
                owner = True
        if not owner:
            record.retired_event.wait(self.cleanup_timeout * 2.0 + 0.1)
            return record.cleanup_status

        from time import monotonic
        retirement_started = monotonic()

        proc = record.process
        handle = record.stderr_handle
        cleanup_errors: List[str] = []
        returncode = proc.poll()
        status: Dict[str, object] = {
            "pid": int(proc.pid),
            "reason": reason,
            "terminated": True,
            "reaped": returncode is not None,
            "root_reaped": returncode is not None,
            "tree_termination_confirmed": False,
            "returncode": returncode,
            "cleanup_error": None,
            "elapsed_s": 0.0,
        }
        try:
            try:
                status = record.tree.terminate(reason=reason, timeout=self.cleanup_timeout)
            except BaseException as exc:
                cleanup_errors.append(f"{type(exc).__name__}: {exc}")
                returncode = proc.poll()
                status.update(
                    reaped=returncode is not None,
                    root_reaped=returncode is not None,
                    returncode=returncode,
                )

            # Only after bounded tree termination: close streams on daemon helpers sharing the
            # remaining cleanup budget. TextIOWrapper.close() may flush behind a concurrent submit.
            stdin = proc.stdin
            proc.stdin = None
            try:
                stderr_path = handle.name
            except BaseException as exc:
                cleanup_errors.append(f"stderr path: {type(exc).__name__}: {exc}")
                stderr_path = None

            from threading import Thread
            close_deadline = monotonic() + max(
                0.0,
                self.cleanup_timeout - float(status.get("elapsed_s", 0.0)),
            )
            pending_closes = []
            for label, stream in (("stdin", stdin), ("stderr", handle)):
                if stream is None:
                    continue
                completed = Event()
                close_errors: List[BaseException] = []

                def _close_stream(
                    owned_stream=stream,
                    errors=close_errors,
                    done=completed,
                ) -> None:
                    try:
                        owned_stream.close()
                    except BaseException as exc:
                        errors.append(exc)
                    finally:
                        done.set()

                closer = Thread(
                    target=_close_stream,
                    name=f"umap-{label}-close-{proc.pid}",
                    daemon=True,
                )
                try:
                    closer.start()
                except BaseException as exc:
                    cleanup_errors.append(f"{label} close start: {type(exc).__name__}: {exc}")
                    continue
                pending_closes.append((label, completed, close_errors))

            for label, completed, close_errors in pending_closes:
                remaining = max(0.0, close_deadline - monotonic())
                if not completed.wait(remaining):
                    cleanup_errors.append(f"{label} close exceeded cleanup budget")
                elif close_errors:
                    exc = close_errors[0]
                    cleanup_errors.append(f"{label} close: {type(exc).__name__}: {exc}")

            if stderr_path is not None:
                # Windows releases a terminated tree's INHERITED handles a few milliseconds after the
                # job reports itself drained (the gated launcher hands its stderr to the real worker,
                # so the log outlives both). A single unlinked attempt therefore lost the race and
                # published WinError 32 on every worker. Retry inside the caller's cleanup budget --
                # the same policy close() already applies to the scratch directory itself.
                remove_deadline = retirement_started + self.cleanup_timeout
                while True:
                    try:
                        os.remove(stderr_path)
                    except FileNotFoundError:
                        break
                    except OSError as exc:
                        remaining = remove_deadline - monotonic()
                        if remaining <= 0.0:
                            cleanup_errors.append(f"{type(exc).__name__}: {exc}")
                            break
                        Event().wait(min(_STDERR_UNLINK_POLL_SECONDS, remaining))
                    else:
                        break
        finally:
            status["elapsed_s"] = monotonic() - retirement_started
            existing_error = status.get("cleanup_error")
            status["cleanup_error"] = " | ".join(
                part for part in (existing_error, *cleanup_errors) if part
            ) or None
            with self._lifecycle_lock:
                self._workers[:] = [
                    active for active in self._workers if active.identity != record.identity
                ]
                record.cleanup_status = status
                record.retired = True
                record.retiring = False
                self.cleanup_statuses.append(status)
                record.retired_event.set()
        return status

    def _submit(self, record: _UMAPWorkerRecord, request: Dict[str, object]) -> None:
        import json
        from threading import Thread
        proc = record.process
        stream = proc.stdin
        if stream is None:
            raise OSError("UMAP worker input is unavailable")
        payload = json.dumps(request) + "\n"
        completed = Event()
        outcome: List[BaseException] = []

        def _write_request() -> None:
            try:
                stream.write(payload)
                stream.flush()
            except BaseException as exc:
                outcome.append(exc)
            finally:
                completed.set()

        writer = Thread(target=_write_request, name=f"umap-submit-{proc.pid}", daemon=True)
        writer.start()
        if not completed.wait(self.timeout):
            self._retire(record, reason="submit_timeout")
            raise TimeoutError(f"UMAP worker request submission exceeded {self.timeout:g} seconds")
        if outcome:
            exc = outcome[0]
            tail = self._error_tail(record)
            self._retire(record, reason="pipe_closed")
            if "ModuleNotFoundError" in tail and "umap" in tail:
                raise ImportError("umap_embed needs umap-learn (pip install umap-learn)") from exc
            if isinstance(exc, (BrokenPipeError, OSError, ValueError)):
                raise OSError(f"UMAP worker pipe closed: {tail[-500:]}") from exc
            raise exc
        with self._lifecycle_lock:
            if record.retired or record.retiring:
                raise OSError("UMAP worker retired during request submission")
            record.requests_submitted += 1

    def _collect(self, record: _UMAPWorkerRecord, status: str, fout: str) -> np.ndarray:
        r"""Block on one submitted request; the timeout starts here."""
        import json
        from threading import Thread
        proc = record.process
        ready = Event()
        canceled = Event()

        def _watch() -> None:
            while not canceled.is_set():
                if os.path.exists(status) or proc.poll() is not None:
                    ready.set()
                    return
                canceled.wait(0.01)

        watcher = Thread(target=_watch, name=f"umap-status-{proc.pid}", daemon=True)
        watcher.start()
        with self._lifecycle_lock:
            first_request = record.requests_submitted == 1
        startup_grace = min(0.25, self.cleanup_timeout) if first_request else 0.0
        finished = ready.wait(self.timeout + startup_grace)
        canceled.set()
        watcher.join(timeout=self.cleanup_timeout)
        if not finished and not os.path.exists(status):
            self._retire(record, reason="timeout")
            raise TimeoutError(f"UMAP worker exceeded {self.timeout:g} seconds")
        if not os.path.exists(status):
            returncode = proc.poll()
            tail = self._error_tail(record)
            self._retire(record, reason="worker_exit")
            if "ModuleNotFoundError" in tail and "umap" in tail:
                raise ImportError("umap_embed needs umap-learn (pip install umap-learn)")
            raise OSError(f"UMAP worker exited (rc={returncode}): {tail[-500:]}")
        with open(status, encoding="utf-8") as handle:
            response = json.load(handle)
        if not response.get("ok"):
            error = str(response.get("error", "unknown worker failure"))
            if "ModuleNotFoundError" in error and "umap" in error:
                raise ImportError("umap_embed needs umap-learn (pip install umap-learn)")
            raise OSError(f"UMAP embedding worker failed: {error[-500:]}")
        return np.load(fout)

    def embed(
        self,
        features: np.ndarray,                # (N, D) features to embed

        *,
        n_neighbors:  int,
        min_dist:     float,
        n_components: int,
        seed:         int,
    ) -> np.ndarray:                         # (N, n_components) seeded embedding
        r"""Embed ``features`` under one seed in the isolated interpreter (single-threaded fit)."""
        self._ensure()
        with self._lifecycle_lock:
            assert self._workdir is not None
            self._counter += 1
            stem = os.path.join(self._workdir, f"request_{self._counter}")
            record = self._workers[0]
        fin, fout, status = f"{stem}_in.npy", f"{stem}_out.npy", f"{stem}_status.json"
        paths = [fin, fout, status, f"{status}.tmp"]
        with self._lifecycle_lock:
            self._request_paths.update(paths)
        try:
            np.save(fin, features)
            self._submit(record, dict(input=fin, output=fout, status=status,
                                      n_neighbors=n_neighbors, min_dist=min_dist,
                                      n_components=n_components, seed=int(seed)))
            return self._collect(record, status, fout)
        finally:
            for path in paths:
                self._cleanup_request_path(path)

    def close(self) -> List[Dict[str, object]]:
        from time import monotonic
        import shutil
        statuses: List[Dict[str, object]] = []
        with self._lifecycle_lock:
            self._closing = True
            workers = tuple(reversed(self._workers))
            workdir = self._workdir
            self._workdir = None
        for record in workers:
            status = self._retire(record, reason="close")
            if status is not None:
                statuses.append(status)
        with self._lifecycle_lock:
            request_paths = tuple(self._request_paths)
        for path in request_paths:
            self._cleanup_request_path(path)
        if workdir is not None:
            # The numba cache lives OUTSIDE this tree on purpose; only the request scratch goes.
            # Windows can retain a terminated worker's file handles briefly. Retry only inside
            # the caller's explicit cleanup budget and publish the terminal failure, if any.
            cleanup_deadline = monotonic() + self.cleanup_timeout
            while True:
                try:
                    shutil.rmtree(workdir)
                    break
                except FileNotFoundError:
                    break
                except OSError as exc:
                    remaining = cleanup_deadline - monotonic()
                    if remaining <= 0.0:
                        statuses.append(self._record_cleanup_failure(
                            reason="workdir_cleanup", path=workdir, exc=exc,
                        ))
                        break
                    Event().wait(min(0.01, remaining))
        with self._lifecycle_lock:
            self._closing = False
        return statuses

    def __enter__(self) -> "UMAPWorker":
        return self

    def __exit__(
        self,
        _exc_type: Optional[type[BaseException]],
        _exc:      Optional[BaseException],
        _tb:       Optional[TracebackType],
    ) -> None:
        self.close()
