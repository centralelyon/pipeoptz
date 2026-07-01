"""Callback interfaces for observing pipeline optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Union

if TYPE_CHECKING:
    from .optimizer import PipelineOptimizer


class Callback:
    """Base class for callbacks invoked during optimization.

    Subclasses can override any lifecycle method. Iteration and evaluation
    indexes are zero-based, matching the convention used by Keras callbacks.
    The optimizer is available through ``self.optimizer`` after registration.
    """

    def __init__(self) -> None:
        self.optimizer: Optional[PipelineOptimizer] = None
        self.params: Dict[str, Any] = {}

    def set_optimizer(self, optimizer: PipelineOptimizer) -> None:
        """Set the optimizer associated with this callback."""
        self.optimizer = optimizer

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the configuration of the current optimization run."""
        self.params = params

    def on_optimization_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Run before the optimization strategy starts."""

    def on_optimization_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Run after optimization finishes or fails."""

    def on_iteration_begin(
        self, iteration: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Run before an optimization iteration starts."""

    def on_iteration_end(
        self, iteration: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Run after an optimization iteration finishes."""

    def on_evaluation_begin(
        self, evaluation: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Run before a candidate parameter configuration is evaluated."""

    def on_evaluation_end(
        self, evaluation: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Run after a candidate parameter configuration is evaluated."""


class _CallbackList:
    """Dispatch callback hooks for one optimization run."""

    def __init__(
        self,
        callbacks: Optional[Union[Callback, Iterable[Callback]]] = None,
        optimizer: Optional[PipelineOptimizer] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if callbacks is None:
            self.callbacks: List[Callback] = []
        elif isinstance(callbacks, Callback):
            self.callbacks = [callbacks]
        else:
            self.callbacks = list(callbacks)

        for callback in self.callbacks:
            if not isinstance(callback, Callback):
                raise TypeError("callbacks must contain only Callback instances")
            if optimizer is not None:
                callback.set_optimizer(optimizer)
            callback.set_params(dict(params or {}))

    def call(self, hook: str, *args: Any, **kwargs: Any) -> None:
        """Invoke a lifecycle hook on every registered callback."""
        for callback in self.callbacks:
            getattr(callback, hook)(*args, **kwargs)
