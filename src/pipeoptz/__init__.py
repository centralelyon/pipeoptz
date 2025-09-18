"""A Python library for creating and optimizing processing pipelines."""

from .node import Node, NodeIf, NodeFor, NodeWhile
from .parameter import IntParameter, FloatParameter, \
                       BoolParameter, ChoiceParameter, MultiChoiceParameter
from .pipeline import Pipeline
from .optimizer import PipelineOptimizer

__version__ = "0.1.4"
__all__ = ["Node", "NodeIf", "NodeFor", "NodeWhile", "IntParameter",
           "FloatParameter", "BoolParameter", "ChoiceParameter",
           "MultiChoiceParameter", "Pipeline", "PipelineOptimizer"]
