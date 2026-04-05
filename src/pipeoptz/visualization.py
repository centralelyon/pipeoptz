"""Visualization module for Pipeline graphs using DOT/Graphviz."""
from __future__ import annotations
import os
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .pipeline import Pipeline

from .node import Node, NodeIf, NodeFor, NodeWhile


class Visualizer:
    """Handles visualization of Pipeline graphs in DOT format and PNG images."""

    def __init__(self, pipeline: Pipeline) -> None:
        """Initialize the Visualizer with a pipeline."""
        self.pipeline = pipeline

    def to_dot(self, filepath: Optional[str] = None,
               add_optz: bool = False, show_function: bool = True, _prefix: str = "") -> str:
        """Generates a DOT language representation of the pipeline graph."""
        pass

    def to_image(self, filepath: str, dpi: int = 160,
                 add_optz: bool = False, show_function: bool = True) -> None:
        """Generates a PNG image of the pipeline graph using Graphviz."""
        pass
