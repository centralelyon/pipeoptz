"""Tests for the Visualizer class."""
import pytest
import sys
import os
import subprocess
from unittest.mock import patch
sys.path.append(os.path.abspath("../src/"))
from pipeoptz.pipeline import Pipeline
from pipeoptz.node import Node, NodeIf, NodeFor, NodeWhile
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


@pytest.fixture
def node_for_pipeline():
    loop_p = Pipeline("loop")
    loop_p.add_node(Node("inc", lambda loop_var: loop_var + 1),
                    predecessors={'loop_var': 'run_params:loop_var'})
    p = Pipeline("with_for")
    p.add_node(NodeFor("for_node", loop_p, fixed_params={'iterations': 3}),
               predecessors={'loop_var': 'run_params:start'})
    return p


@pytest.fixture
def node_while_pipeline():
    loop_p = Pipeline("loop")
    loop_p.add_node(Node("inc", lambda loop_var: loop_var + 1),
                    predecessors={'loop_var': 'run_params:loop_var'})
    p = Pipeline("with_while")
    p.add_node(NodeWhile("while_node", lambda loop_var: loop_var < 5, loop_p),
               predecessors={'loop_var': 'run_params:start'})
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
        visualizer.to_dot(str(filepath))
        
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

    def test_to_image_regenerates_missing_dot_when_png_exists(self, basic_pipeline, tmp_path):
        """An existing PNG must not prevent regeneration of its temporary DOT file."""
        image_path = tmp_path / "graph.png"
        dot_path = tmp_path / "graph.dot"
        image_path.write_bytes(b"existing image")

        def successful_dot(command, **kwargs):
            assert command[-1] == str(dot_path)
            assert dot_path.exists()
            return subprocess.CompletedProcess(command, 0)

        with patch("pipeoptz.visualization.subprocess.run", side_effect=successful_dot):
            basic_pipeline.to_image(str(image_path))

        assert not dot_path.exists()

    def test_to_image_includes_graphviz_error(self, basic_pipeline, tmp_path):
        """Graphviz diagnostics should be included in the public error."""
        image_path = tmp_path / "graph.png"
        error = subprocess.CalledProcessError(
            2,
            ["dot"],
            stderr="Error: dot: can't open graph.dot",
        )

        with patch("pipeoptz.visualization.subprocess.run", side_effect=error):
            with pytest.raises(RuntimeError, match="can't open graph.dot"):
                basic_pipeline.to_image(str(image_path))

    def test_visualizer_to_mermaid_generates_string(self, basic_pipeline):
        """
        Tests the to_mermaid method of Visualizer.
        """
        visualizer = Visualizer(basic_pipeline)
        mermaid_string = visualizer.to_mermaid()
        assert "flowchart TD" in mermaid_string
        assert 'add["add' in mermaid_string
        assert 'add -->|a| mul' in mermaid_string

    def test_visualizer_to_mermaid_with_filepath(self, basic_pipeline, tmp_path):
        """
        Tests that to_mermaid saves to a file when filepath is provided.
        """
        filepath = tmp_path / "test_graph.mmd"
        visualizer = Visualizer(basic_pipeline)
        visualizer.to_mermaid(str(filepath))

        assert filepath.exists()
        with open(filepath, 'r', encoding="utf-8") as f:
            content = f.read()
        assert "flowchart TD" in content

    def test_visualizer_to_mermaid_with_conditional(self, node_if_pipeline):
        """
        Tests to_mermaid with a pipeline containing a NodeIf.
        """
        visualizer = Visualizer(node_if_pipeline)
        mermaid_string = visualizer.to_mermaid()
        assert "flowchart TD" in mermaid_string
        assert 'conditional_node{"conditional_node' in mermaid_string
        assert '-->|True|' in mermaid_string

    def test_visualizer_can_be_imported(self):
        """
        Tests that Visualizer can be imported from the main package.
        """
        from pipeoptz import Visualizer as VisualizerImported
        assert VisualizerImported is Visualizer


# ─── NodeFor visualization ───────────────────────────────────────────────────
class TestVisualizerNodeFor:
    def test_to_dot_contains_for_loop_label(self, node_for_pipeline):
        """
        Tests that the to_dot method includes the for loop label.
        """
        dot = node_for_pipeline.to_dot()
        assert "For Loop" in dot
        assert "Mdiamond" in dot

    def test_to_dot_contains_loop_edges(self, node_for_pipeline):
        """
        Tests that the to_dot method includes edges for the loop structure.
        """
        dot = node_for_pipeline.to_dot()
        assert "start" in dot
        assert "next" in dot

    def test_to_dot_for_output_node_present(self, node_for_pipeline):
        """
        Tests that the to_dot method includes a node for the for loop output.
        """
        dot = node_for_pipeline.to_dot()
        assert "for_node_output" in dot

    def test_to_mermaid_contains_for_loop_label(self, node_for_pipeline):
        """
        Tests that the to_mermaid method includes the for loop label.
        """
        mermaid = node_for_pipeline.to_mermaid()
        assert "For Loop" in mermaid
        assert "-->|start|" in mermaid

    def test_to_mermaid_for_output_node_present(self, node_for_pipeline):
        """
        Tests that the to_mermaid method includes a node for the for loop output.
        """
        mermaid = node_for_pipeline.to_mermaid()
        assert "for_node_output" in mermaid or "For Output" in mermaid


# ─── NodeWhile visualization ─────────────────────────────────────────────────

class TestVisualizerNodeWhile:
    def test_to_dot_contains_while_loop_label(self, node_while_pipeline):
        """
        Tests that the to_dot method includes the while loop label.
        """
        dot = node_while_pipeline.to_dot()
        assert "While Loop" in dot
        assert "Mdiamond" in dot

    def test_to_dot_while_output_node_present(self, node_while_pipeline):
        """
        Tests that the to_dot method includes a node for the while loop output.
        """
        dot = node_while_pipeline.to_dot()
        assert "while_node_output" in dot

    def test_to_mermaid_contains_while_loop_label(self, node_while_pipeline):
        """
        Tests that the to_mermaid method includes the while loop label.
        """
        mermaid = node_while_pipeline.to_mermaid()
        assert "While Loop" in mermaid

    def test_to_mermaid_while_output_node_present(self, node_while_pipeline):
        """
        Tests that the to_mermaid method includes a node for the while loop output.
        """
        mermaid = node_while_pipeline.to_mermaid()
        assert "while_node_output" in mermaid or "While Output" in mermaid


# ─── Visualizer helper methods ───────────────────────────────────────────────

class TestVisualizerHelpers:
    def test_mermaid_id_empty_string_returns_node(self):
        """
        Tests that the _mermaid_id method returns 'node' when given an empty string.
        """
        assert Visualizer._mermaid_id("") == "node"

    def test_mermaid_id_digit_start_gets_prefix(self):
        """
        Tests that the _mermaid_id method adds a prefix when the input starts with a digit.
        """
        result = Visualizer._mermaid_id("123node")
        assert result.startswith("n_")

    def test_mermaid_id_special_chars_replaced(self):
        """
        Tests that the _mermaid_id method replaces special characters with underscores.
        """
        result = Visualizer._mermaid_id("node-id with space")
        assert "-" not in result
        assert " " not in result

    def test_mermaid_id_with_prefix(self):
        """
        Tests that the _mermaid_id method correctly applies a custom prefix when the input starts with a digit.
        """
        result = Visualizer._mermaid_id("step", prefix="sub_")
        assert result.startswith("sub_")

    def test_function_label_lambda(self):
        """
        Tests that the _function_label method returns 'lambda' for lambda functions.
        """
        node = Node("n", lambda x: x)
        assert Visualizer._function_label(node) == "lambda"

    def test_function_label_main_module(self):
        """
        Tests that the _function_label method returns just the function name for functions defined in the __main__ module.
        """
        def my_func(x):
            return x
        my_func.__module__ = "__main__"
        node = Node("n", my_func)
        assert Visualizer._function_label(node) == "my_func"

    def test_function_label_external_module(self):
        """
        Tests that the _function_label method returns the full module and function name for functions defined in external modules.
        """
        def my_func(x):
            return x
        my_func.__module__ = "mypackage.mymodule"
        node = Node("n", my_func)
        assert Visualizer._function_label(node) == "mypackage.mymodule.my_func"

    def test_to_dot_show_function_false_omits_func_details(self, basic_pipeline):
        """
        Tests that the to_dot method omits function details when show_function is set to False.
        """
        dot_with = basic_pipeline.to_dot(show_function=True)
        dot_without = basic_pipeline.to_dot(show_function=False)
        # With show_function=False the DOT should be shorter (no func labels)
        assert len(dot_without) < len(dot_with)

    def test_to_dot_run_params_rendered_as_ellipse(self, basic_pipeline):
        """
         Tests that run_params edges are shown as dashed ellipses labelled by the input param name.
         """
        dot = basic_pipeline.to_dot()
        assert "shape=ellipse" in dot
        # The basic_pipeline has predecessors {'a': 'run_params:x', 'b': 'run_params:y'},
        # so param nodes are labelled by the input key ('a', 'b'), not the source key.
        assert "params_a" in dot
        assert "params_b" in dot

    def test_to_mermaid_saves_to_file(self, basic_pipeline, tmp_path):
        """
        Tests that the to_mermaid method saves the Mermaid diagram to a file when a filepath is provided, and that the file contains the expected Mermaid syntax.
        """
        fp = tmp_path / "graph.mmd"
        basic_pipeline.to_mermaid(str(fp))
        assert fp.exists()
        assert "flowchart TD" in fp.read_text()

    def test_to_dot_saves_to_file(self, basic_pipeline, tmp_path):
        """
        Tests that the to_dot method saves the DOT diagram to a file when a filepath is provided, and that the file contains the expected DOT syntax.
        """
        fp = tmp_path / "graph.dot"
        basic_pipeline.to_dot(str(fp))
        assert fp.exists()
        assert "digraph Pipeline" in fp.read_text()
