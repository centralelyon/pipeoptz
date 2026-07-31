"""A Python library for creating and optimizing processing pipelines."""

from .callback import Callback
from .node import Node, NodeFor, NodeIf, NodeWhile
from .optimizer import PipelineOptimizer
from .parameter import (
           BoolParameter,
           ChoiceParameter,
           FloatParameter,
           IntParameter,
           MultiChoiceParameter,
)
from .pipeline import Pipeline
from .visualization import Visualizer

__version__ = "0.1.6"
__all__ = [
           "BoolParameter",
           "Callback",
           "ChoiceParameter",
           "FloatParameter",
           "IntParameter",
           "MultiChoiceParameter",
           "Node",
           "NodeFor",
           "NodeIf",
           "NodeWhile",
           "Pipeline",
           "PipelineOptimizer",
           "Visualizer",
]
