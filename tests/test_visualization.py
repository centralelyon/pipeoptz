"""Tests for the Visualizer class."""
import pytest
import sys
import os
sys.path.append(os.path.abspath("../src/"))
from pipeoptz.pipeline import Pipeline
from pipeoptz.node import Node, NodeIf
from pipeoptz.visualization import Visualizer


@pytest.fixture
def add_func():
    return lambda a, b: a + b


@pytest.fixture
def mul_func():
    return lambda a, b: a * b


@pytest.fixture
def identity_func():
    return lambda x: x


@pytest.fixture
def basic_pipeline(add_func, mul_func):
    """
    A simple linear pipeline: run_params -> add -> mul
    """
    p = Pipeline(name="basic")
    p.add_node(Node(node_id="add", func=add_func), predecessors={'a': 'run_params:x', 'b': 'run_params:y'})
    p.add_node(Node(node_id="mul", func=mul_func, fixed_params={'b': 10}), predecessors={'a': 'add'})
    return p


@pytest.fixture
def node_if_pipeline(identity_func):
    """
    A pipeline containing a NodeIf.
    """
    true_pipe = Pipeline(name="true_branch")
    true_pipe.add_node(Node(node_id="true_op", func=lambda x: x + 1), predecessors={'x': 'run_params:val'})

    false_pipe = Pipeline(name="false_branch")
    false_pipe.add_node(Node(node_id="false_op", func=lambda x: x - 1), predecessors={'x': 'run_params:val'})

    p = Pipeline(name="conditional")
    p.add_node(Node(node_id="start", func=identity_func), predecessors={'x': 'run_params:start_val'})
    node_if = NodeIf(
        node_id="conditional_node",
        condition_func=lambda c: c > 10,
        true_pipeline=true_pipe,
        false_pipeline=false_pipe
    )
    p.add_node(node_if, predecessors={'condition_func:c': 'start', 'val': 'start'})
    return p


class TestVisualizer:
    def test_visualizer_initialization(self, basic_pipeline):
        """
        Tests that a Visualizer is initialized correctly with a pipeline.
        """
        visualizer = Visualizer(basic_pipeline)
        assert visualizer.pipeline == basic_pipeline

    def test_visualizer_to_dot_generates_string(self, basic_pipeline):
        """
        Tests the to_dot method of Visualizer.
        """
        visualizer = Visualizer(basic_pipeline)
        dot_string = visualizer.to_dot()
        assert "digraph Pipeline" in dot_string
        assert '"add"' in dot_string
        assert '"mul"' in dot_string
        assert '"add" -> "mul"' in dot_string
        assert 'label="a"' in dot_string

    def test_visualizer_to_dot_with_filepath(self, basic_pipeline, tmp_path):
        """
        Tests that to_dot saves to a file when filepath is provided.
        """
        filepath = tmp_path / "test_graph.dot"
        visualizer = Visualizer(basic_pipeline)
        result = visualizer.to_dot(str(filepath))
        
        assert filepath.exists()
        with open(filepath, 'r') as f:
            content = f.read()
        assert "digraph Pipeline" in content

    def test_visualizer_to_dot_with_conditional(self, node_if_pipeline):
        """
        Tests to_dot with a pipeline containing a NodeIf.
        """
        visualizer = Visualizer(node_if_pipeline)
        dot_string = visualizer.to_dot()
        assert "digraph Pipeline" in dot_string
        assert "conditional_node" in dot_string
        assert "diamond" in dot_string

    def test_pipeline_to_dot_delegates_to_visualizer(self, basic_pipeline):
        """
        Tests that Pipeline.to_dot delegates to Visualizer.
        """
        dot_string = basic_pipeline.to_dot()
        visualizer = Visualizer(basic_pipeline)
        visualizer_dot_string = visualizer.to_dot()
        assert dot_string == visualizer_dot_string

    def test_visualizer_can_be_imported(self):
        """
        Tests that Visualizer can be imported from the main package.
        """
        from pipeoptz import Visualizer as VisualizerImported
        assert VisualizerImported is Visualizer
