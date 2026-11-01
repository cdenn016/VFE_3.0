import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Tuple

import torch


_RATE_FLOOR_S = 1e-9
# The segment of a timed step after its last mark (the whole step when it has none), up to finish_step.
STEP_TAIL_PHASE = "other"


class _TimingEvent(Protocol):
    def record(self, stream: object) -> None: ...
    def synchronize(self) -> None: ...
    def elapsed_time(self, end_event: "_TimingEvent") -> float: ...


@dataclass(frozen=True)
class TrainTimingSample:
    train_step_ms_mean:      float
    train_steps_per_s:       float
    train_step_tokens_per_s: float
    # Mean milliseconds per timed step of each segment that TrainingTimer.mark closed, and of the STEP_TAIL_PHASE tail;
    # the values sum to train_step_ms_mean (audit 2026-10-31 perf-live-path-5).
    phase_ms_mean:           Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineTimingSample:
    pipeline_tokens_per_s: float
    wall_clock_s:           float


class TrainingTimer:
    def __init__(
        self,
        device: torch.device,

        *,
        clock:               Optional[Callable[[], float]]              = None,
        cuda_event_factory:  Optional[Callable[..., object]]            = None,
        cuda_stream_factory: Optional[Callable[[torch.device], object]] = None,
        wall_clock_offset:   float                                      = 0.0,
    ) -> None:
        self._device = device
        # Seconds already elapsed before this timer started (a resumed run's checkpointed clock).
        self._wall_clock_offset = float(wall_clock_offset)
        self._clock = time.perf_counter if clock is None else clock
        self._is_cuda = device.type == "cuda"
        self._event_factory: Optional[Callable[..., _TimingEvent]] = None
        self._stream: Optional[object] = None
        if self._is_cuda:
            self._event_factory = torch.cuda.Event if cuda_event_factory is None else cuda_event_factory
            stream_factory = torch.cuda.current_stream if cuda_stream_factory is None else cuda_stream_factory
            self._stream = stream_factory(device)
        self._train_origin = self._clock()
        self._pipeline_origin = self._train_origin
        self._active_step: Optional[tuple[_TimingEvent, _TimingEvent]] = None
        self._event_pairs: list[tuple[_TimingEvent, _TimingEvent]] = []
        # CUDA: the marks of the active step, and those of each finished step of the window (aligned with _event_pairs).
        self._active_marks: List[Tuple[str, _TimingEvent]] = []
        self._step_marks: List[List[Tuple[str, _TimingEvent]]] = []
        self._cpu_step_origin: Optional[float] = None
        self._cpu_mark: Optional[float] = None               # CPU: host time of the active step's last mark or start
        self._cpu_phase_seconds: Dict[str, float] = {}
        self._cpu_train_seconds = 0.0
        self._train_steps = 0
        self._train_tokens = 0
        self._pipeline_tokens = 0

    def start_step(self) -> None:
        if self._is_cuda:
            assert self._event_factory is not None
            start = self._event_factory(enable_timing=True)
            end = self._event_factory(enable_timing=True)
            start.record(self._stream)
            self._active_step = (start, end)
            self._active_marks = []
        else:
            self._cpu_step_origin = self._clock()
            self._cpu_mark = self._cpu_step_origin

    def mark(
        self,
        phase: str,                        # the segment that ends here, e.g. "forward"
    ) -> None:
        r"""Close the active step's current segment, which began at its previous mark or at the step start, as ``phase``.

        Outside a step this is a no-op. On CUDA it records one timing event on the timer's stream and reads nothing:
        :meth:`sample_train_window` resolves the events after the synchronization it already performs, so a mark adds
        no host synchronization (audit 2026-10-31 perf-live-path-5). A phase closed more than once in a step (the
        microbatches of gradient accumulation, a replayed attempt) accumulates.
        """
        if self._is_cuda:
            if self._active_step is None:
                return
            assert self._event_factory is not None
            event = self._event_factory(enable_timing=True)
            event.record(self._stream)
            self._active_marks.append((phase, event))
        elif self._cpu_mark is not None:
            now = self._clock()
            self._cpu_phase_seconds[phase] = self._cpu_phase_seconds.get(phase, 0.0) + (now - self._cpu_mark)
            self._cpu_mark = now

    def finish_step(
        self,
        *,
        n_tokens: int,
    ) -> None:
        if self._is_cuda:
            assert self._active_step is not None
            start, end = self._active_step
            end.record(self._stream)
            self._event_pairs.append((start, end))
            self._step_marks.append(self._active_marks)
            self._active_marks = []
            self._active_step = None
        else:
            assert self._cpu_step_origin is not None and self._cpu_mark is not None
            now = self._clock()
            self._cpu_train_seconds += now - self._cpu_step_origin
            self._cpu_phase_seconds[STEP_TAIL_PHASE] = (self._cpu_phase_seconds.get(STEP_TAIL_PHASE, 0.0)
                                                        + (now - self._cpu_mark))
            self._cpu_step_origin = None
            self._cpu_mark = None
        self._train_steps += 1
        self._train_tokens += int(n_tokens)
        self._pipeline_tokens += int(n_tokens)

    def sample_train_window(self) -> TrainTimingSample:
        if self._is_cuda:
            self._event_pairs[-1][1].synchronize()
            total_seconds = sum(
                float(start.elapsed_time(end)) for start, end in self._event_pairs
            ) / 1000.0
            # Each step's segments run start -> mark_1 -> ... -> mark_k -> end, so they sum to the step's span.
            phase_seconds: Dict[str, float] = {}
            for (start, end), marks in zip(self._event_pairs, self._step_marks):
                previous = start
                for phase, event in (*marks, (STEP_TAIL_PHASE, end)):
                    phase_seconds[phase] = (phase_seconds.get(phase, 0.0)
                                            + float(previous.elapsed_time(event)) / 1000.0)
                    previous = event
        else:
            total_seconds = self._cpu_train_seconds
            phase_seconds = dict(self._cpu_phase_seconds)
        denominator = max(total_seconds, _RATE_FLOOR_S)
        sample = TrainTimingSample(
            train_step_ms_mean=(total_seconds * 1000.0) / self._train_steps,
            train_steps_per_s=self._train_steps / denominator,
            train_step_tokens_per_s=self._train_tokens / denominator,
            phase_ms_mean={phase: seconds * 1000.0 / self._train_steps for phase, seconds in phase_seconds.items()},
        )
        self._event_pairs.clear()
        self._step_marks.clear()
        self._cpu_phase_seconds = {}
        self._cpu_train_seconds = 0.0
        self._train_steps = 0
        self._train_tokens = 0
        return sample

    def sample_pipeline_window(self) -> PipelineTimingSample:
        if self._is_cuda:
            assert self._event_factory is not None
            boundary = self._event_factory(enable_timing=False)
            boundary.record(self._stream)
            boundary.synchronize()
        now = self._clock()
        elapsed = now - self._pipeline_origin
        return PipelineTimingSample(
            pipeline_tokens_per_s=self._pipeline_tokens / max(elapsed, _RATE_FLOOR_S),
            wall_clock_s=self._wall_clock_offset + now - self._train_origin,
        )

    def elapsed_wall_clock_s(self) -> float:
        """Cumulative wall-clock seconds, including a resumed run's checkpointed elapsed time."""
        return self._wall_clock_offset + self._clock() - self._train_origin

    def reset_pipeline_window(self) -> None:
        self._pipeline_origin = self._clock()
        self._pipeline_tokens = 0
