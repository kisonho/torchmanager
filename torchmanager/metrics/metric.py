import multiprocessing as mp
from multiprocessing.managers import SyncManager
from multiprocessing.process import BaseProcess

from torchmanager_core import abc, torch, API_VERSION, Version, raise_error
from torchmanager_core.typing import Any, Callable, Generic, TypeVar, cast

MetricFn = TypeVar("MetricFn", bound=Callable[[Any, Any], torch.Tensor] | None)

__all__ = ["BaseMetric", "Metric", "metric", "metric_fn"]


def _record_metric_results(record_queue: Any, results: Any) -> None:
    """
    Consume recorded metric tensors from a queue and append them to shared storage.

    This function runs in a background process created by `BaseMetric`. The
    training process only needs to enqueue detached CPU tensors, while the
    recorder process performs the append operation against the shared results
    container. A `None` item is treated as a shutdown sentinel and causes the
    loop to exit after acknowledging the queued task.

    - Parameters:
        - record_queue: A multiprocessing queue that receives metric tensors
          shaped for later concatenation.
        - results: A shared list-like object, usually a manager-backed list,
          that stores the appended metric history.
    """
    while True:
        metric = record_queue.get()
        if metric is None:
            record_queue.task_done()
            break
        results.append(metric)
        record_queue.task_done()


class BaseMetric(torch.nn.Module, abc.ABC):
    """
    The basic metric class

    * extends: `torch.nn.Module`
    * implements: `torchmanager_core.protocols.Resulting`
    * abstract methods: `forward`
    * Metric tensor is released from memory as soon as the result returned

    - Properties:
        - result: The `torch.Tensor` of average metric results
        - results: An optional `torch.Tensor` of all metric results
    """
    __count: int
    __result: torch.Tensor | float
    _record_manager: SyncManager | None
    _record_process: BaseProcess | None
    _record_queue: mp.JoinableQueue | None
    _results: list[torch.Tensor]
    _target: str | None

    @property
    def result(self) -> torch.Tensor:
        if self.__count > 0:
            result = self.__result / self.__count
            return result if isinstance(result, torch.Tensor) else torch.tensor(result)
        else:
            return torch.tensor(torch.nan)

    @property
    def results(self) -> torch.Tensor | None:
        results = self._snapshot_results()
        if len(results) > 0:
            return torch.concat(results)
        else:
            return None

    def __init__(self, target: str | None = None) -> None:
        """
        Constructor

        - Parameters:
            - metric_fn: An optional `Callable` metrics function that accepts `Any` kind of prediction input and target and returns a metric `torch.Tensor`. A `call` method must be overriden if this parameter is set as `None`.
            - target: A `str` of target name in `input` and `target` during direct calling
        """
        super().__init__()
        self.__count = 0
        self.__result = 0
        self._results = []
        self._record_manager = None
        self._record_process = None
        self._record_queue = None
        self._target = target

    def __getstate__(self) -> dict[str, Any]:
        self._shutdown_recorder(persist_results=True)
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        for key, value in state.items():
            setattr(self, key, value)
        self._record_manager = None
        self._record_process = None
        self._record_queue = None

    def __call__(self, input: Any, target: Any) -> torch.Tensor:
        # unpack input and target
        input = input[self._target] if self._target is not None and isinstance(input, dict) else input
        target = target[self._target] if self._target is not None and isinstance(target, dict) else target

        # call
        m: torch.Tensor = super().__call__(input, target)
        self.record(m)
        return m

    def convert(self, from_version: Version) -> None:
        if from_version < API_VERSION:
            self.__count = 0
            self.__result = 0
        pass

    @abc.abstractmethod
    def forward(self, input: Any, target: Any) -> torch.Tensor:
        """
        Forward the current result method

        - Parameters:
            - input: The prediction, or `y_pred`, in `Any` kind
            - target: The label, or `y_true`, in `Any` kind
        - Returns: The metric in `torch.Tensor`
        """
        ...

    def record(self, m: torch.Tensor) -> None:
        """
        Record the metric value.

        - Parameters:
            - m: A metric value in `torch.Tensor`
        """
        metric = m.detach()
        self.__result = metric if self.__count == 0 else self.__result + metric
        self.__count += 1
        self._ensure_recorder()
        assert self._record_queue is not None, raise_error(RuntimeError("Metric recorder queue is not initialized."))
        self._record_queue.put_nowait(metric.unsqueeze(0).cpu())

    def reset(self) -> None:
        """Reset the current results list"""
        self._shutdown_recorder()
        self.__count = 0
        self.__result = 0
        self._results.clear()

    def _ensure_recorder(self) -> None:
        """
        Lazily start the background recorder process.

        The recorder is only created when the first metric value is recorded.
        This keeps initialization cheap for metrics that may never be used in a
        run. Once started, the recorder owns the append-only history container,
        while the main process keeps only the running sum and count needed for
        constant-time access to `result`.

        If the recorder is already alive, this method returns immediately.
        Otherwise it creates:
        - a multiprocessing manager for shared list storage,
        - a joinable queue for metric payloads, and
        - a spawned background process that appends queued tensors into the
          shared results list.
        """
        if self._record_process is not None and self._record_process.is_alive():
            return

        ctx = mp.get_context("spawn")
        self._record_manager = ctx.Manager()
        self._results = cast(list[torch.Tensor], self._record_manager.list(self._results))
        self._record_queue = ctx.JoinableQueue()
        self._record_process = ctx.Process(target=_record_metric_results, args=(self._record_queue, self._results), daemon=True)
        self._record_process.start()

    def _shutdown_recorder(self, *, persist_results: bool = False) -> None:
        """
        Stop the background recorder and release multiprocessing resources.

        This method first waits for all queued metric items to be processed,
        sends a shutdown sentinel to the recorder process, and then joins the
        process. If the process does not exit within a short timeout, it is
        terminated as a fallback to avoid leaking background workers.

        After shutdown, the shared results store can either be preserved in a
        normal in-process list or discarded entirely:
        - When `persist_results` is `True`, the current history is copied back
          into a local Python list so the metric can be pickled or restored
          without live multiprocessing objects.
        - When `persist_results` is `False`, the history is cleared.

        - Parameters:
            - persist_results: A `bool` flag indicating whether the recorded
              history should be copied into local memory before shutting down
              the shared recorder state.
        """
        if self._record_queue is not None and self._record_process is not None:
            self._record_queue.join()
            self._record_queue.put(None)
            self._record_queue.join()
            self._record_process.join(timeout=1)
            if self._record_process.is_alive():
                self._record_process.terminate()
                self._record_process.join(timeout=1)

        if persist_results:
            self._results = self._snapshot_results(wait=False)
        else:
            self._results = []

        if self._record_queue is not None:
            self._record_queue.close()
        if self._record_manager is not None:
            self._record_manager.shutdown()
        self._record_queue = None
        self._record_process = None
        self._record_manager = None

    def _snapshot_results(self, *, wait: bool = True) -> list[torch.Tensor]:
        """
        Materialize the currently recorded metric history as a local list.

        This helper provides a consistent way to read the append-only history
        regardless of whether `_results` is a normal Python list or a
        multiprocessing manager proxy. When `wait` is enabled, the method blocks
        until all queued metric values have been appended by the background
        recorder, ensuring the returned snapshot is complete up to the point of
        the call.

        - Parameters:
            - wait: A `bool` flag indicating whether to wait for the recorder
              queue to drain before reading the stored history.
        - Returns: A `list` of recorded metric tensors in append order.
        """
        if self._record_queue is not None and wait:
            self._record_queue.join()
        if isinstance(self._results, list):
            return self._results.copy()
        else:
            return list(self._results)


class Metric(BaseMetric, Generic[MetricFn]):
    """
    The basic metric class

    * extends: `BaseMetric`
    * implements: `torchmanager_core.protocols.Resulting`
    * Metric tensor is released from memory as soon as the result returned
    """
    def __init__(self, metric_fn: MetricFn = None, target: str | None = None) -> None:
        """
        Constructor

        - Parameters:
            - metric_fn: An optional `Callable` metrics function that accepts `Any` kind of prediction input and target and returns a metric `torch.Tensor`. A `call` method must be overriden if this parameter is set as `None`.
            - target: A `str` of target name in `input` and `target` during direct calling
        """
        super().__init__(target=target)
        self._metric_fn = metric_fn

    def forward(self, input: Any, target: Any) -> torch.Tensor:
        """
        Forward the current result method

        - Parameters:
            - input: The prediction, or `y_pred`, in `Any` kind
            - target: The label, or `y_true`, in `Any` kind
        - Returns: The metric in `torch.Tensor`
        """
        # main method
        if self._metric_fn is not None:
            return self._metric_fn(input, target)
        else:
            raise NotImplementedError("metric_fn is not given.")


WrappedMetricFn = TypeVar("WrappedMetricFn", bound=Callable[[Any, Any], torch.Tensor])


class _WrappedMetric(Metric[WrappedMetricFn]):
    @property
    def wrapped_metric_fn(self) -> WrappedMetricFn:
        assert self._metric_fn is not None, raise_error(AttributeError("Metric function is not given."))
        return self._metric_fn

    def __init__(self, metric_fn: WrappedMetricFn, target: str | None = None) -> None:
        super().__init__(metric_fn, target)

    @torch.no_grad()
    def forward(self, input: Any, target: Any) -> torch.Tensor:
        return self.wrapped_metric_fn(input, target)


def metric(fn: Callable[[Any, Any], torch.Tensor]) -> _WrappedMetric:
    """
    The metric wrapping function that wrap a function into a metric

    Use as a decorator:
    >>> import torch
    >>> @metric
    >>> def some_metric_fn(input: Any, target: Any) -> torch.Tensor:
    ...    return ...
    >>> manager = (..., metric_fns={'out': some_metric_fn})
    """
    return _WrappedMetric(fn)


def metric_fn(target: str | None = None) -> Callable[[Callable[[Any, Any], torch.Tensor]], _WrappedMetric]:
    """
    The loss wrapping function that wrap a function with target and weight given into a loss

    Use as a decorator:
    >>> import torch
    >>> @metric_fn(target='out')
    >>> def some_metric_fn(input: Any, target: Any) -> torch.Tensor:
    ...    return ...
    >>> manager = (..., metric_fns={'out': some_metric_fn})
    """
    def wrapped_fn(fn_to_wrap: Callable[[Any, Any], torch.Tensor]) -> _WrappedMetric:
        return _WrappedMetric(fn_to_wrap, target=target)
    return wrapped_fn
