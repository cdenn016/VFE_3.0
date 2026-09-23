import time
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import torch


_RATE_FLOOR_S = 1e-9


class _TimingEvent(Protocol):
    def record(self, stream: object) -> None: ...
    def synchronize(self) -> None: ...
    def elapsed_time(self, end_event: "_TimingEvent") -> float: ...


@dataclass(frozen=True)
class TrainTimingSample:
    train_step_ms_mean:      float
    train_steps_per_s:       float
    train_step_tokens_per_s: float


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
    ) -> None:
        self._device = device
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
        self._cpu_step_origin: Optional[float] = None
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
        else:
            self._cpu_step_origin = self._clock()

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
            self._active_step = None
        else:
            assert self._cpu_step_origin is not None
            self._cpu_train_seconds += self._clock() - self._cpu_step_origin
            self._cpu_step_origin = None
        self._train_steps += 1
        self._train_tokens += int(n_tokens)
        self._pipeline_tokens += int(n_tokens)

    def sample_train_window(self) -> TrainTimingSample:
        if self._is_cuda:
            self._event_pairs[-1][1].synchronize()
            total_seconds = sum(
                float(start.elapsed_time(end)) for start, end in self._event_pairs
            ) / 1000.0
        else:
            total_seconds = self._cpu_train_seconds
        denominator = max(total_seconds, _RATE_FLOOR_S)
        sample = TrainTimingSample(
            train_step_ms_mean=(total_seconds * 1000.0) / self._train_steps,
            train_steps_per_s=self._train_steps / denominator,
            train_step_tokens_per_s=self._train_tokens / denominator,
        )
        self._event_pairs.clear()
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
            wall_clock_s=now - self._train_origin,
        )

    def reset_pipeline_window(self) -> None:
        self._pipeline_origin = self._clock()
        self._pipeline_tokens = 0
