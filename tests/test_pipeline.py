import pytest
import json

import sys
import os
sys.path.append(os.path.abspath("../src/"))
from pipeoptz.pipeline import Pipeline
from pipeoptz.node import Node, NodeIf


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
def cyclic_pipeline(identity_func):
    """
    A pipeline with a cycle to test error detection.
    """
    p = Pipeline(name="cyclic")
    p.add_node(Node(node_id="A", func=identity_func), predecessors={'x': 'C'})
    p.add_node(Node(node_id="B", func=identity_func), predecessors={'x': 'A'})
    p.add_node(Node(node_id="C", func=identity_func), predecessors={'x': 'B'})
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
def loop_pipeline(add_func):
    """
    A pipeline with a looping node.
    """
    p = Pipeline(name="looping")
    p.add_node(Node(node_id="data_provider", func=lambda: [1, 2, 3]), predecessors={})
    p.add_node(Node(node_id="add_one", func=add_func, fixed_params={'b': 1}), predecessors={'[a]': 'data_provider'})
    return p

@pytest.fixture
def failing_loop_pipeline():
    """
    A pipeline where one iteration of the loop will fail.
    """
    def fail_on_two(a, b):
        if a == 2:
            raise ValueError("Cannot process 2")
        return a + b

    p = Pipeline(name="failing_loop")
    p.add_node(Node(node_id="data_provider", func=lambda: [1, 2, 3]), predecessors={})
    p.add_node(Node(node_id="add_one_fail", func=fail_on_two, fixed_params={'b': 1}), predecessors={'[a]': 'data_provider'})
    return p


# --- Classes tests ---

class TestPipelineStructure:
    def test_initialization(self):
        """
        Tests if a Pipeline is initialized correctly.
        """
        p = Pipeline(name="test_pipe", description="A test pipeline")
        assert p.name == "test_pipe"
        assert p.description == "A test pipeline"
        assert p.nodes == {}
        assert p.node_dependencies == {}

    def test_add_node(self, add_func):
        """
        Tests adding a node to the pipeline.
        """
        p = Pipeline(name="test")
        node = Node(node_id="add1", func=add_func)
        p.add_node(node, predecessors={'a': 'run_params:x'})
        assert "add1" in p.nodes
        assert p.nodes["add1"] == node
        assert p.node_dependencies["add1"] == {'a': 'run_params:x'}

    def test_add_duplicate_node_id_raises_error(self, add_func):
        """
        Tests that adding a node with a duplicate id raises an error.
        """
        p = Pipeline(name="test")
        node1 = Node(node_id="add1", func=add_func)
        node2 = Node(node_id="add1", func=add_func)
        p.add_node(node1)
        with pytest.raises(ValueError, match="A node with id 'add1' already exists."):
            p.add_node(node2)

    def test_get_node(self, basic_pipeline):
        """
        Tests retrieving a node from the pipeline.
        """
        node = basic_pipeline.get_node("add")
        assert node.id == "add"
        with pytest.raises(ValueError, match="The node does not exist in the pipeline."):
            basic_pipeline.get_node("nonexistent")

    def test_static_order(self, basic_pipeline):
        """
        Tests the static_order method.
        """
        order = basic_pipeline.static_order()
        assert order == ["add", "mul"]

    def test_static_order_cycle_detection(self, cyclic_pipeline):
        """
        Tests that the static_order method detects cycles.
        """
        with pytest.raises(ValueError, match="The graph contains a cycle"):
            cyclic_pipeline.static_order()


class TestPipelineParams:
    def test_get_fixed_params(self, basic_pipeline):
        """
        Tests retrieving fixed parameters from the pipeline.
        """
        params = basic_pipeline.get_fixed_params()
        assert params == {"mul.b": 10}

    def test_set_fixed_params(self, basic_pipeline):
        """
        Tests setting fixed parameters on the pipeline.
        """
        basic_pipeline.set_fixed_params({"mul.b": 20})
        node = basic_pipeline.get_node("mul")
        assert node.fixed_params['b'] == 20

    def test_set_fixed_params_invalid_node(self, basic_pipeline):
        """
        Tests that setting fixed parameters on an invalid node raises an error.
        """
        with pytest.raises(ValueError, match="The node with id 'nonexistent' does not exist"):
            basic_pipeline.set_fixed_params({"nonexistent.b": 20})

    def test_set_fixed_params_invalid_param(self, basic_pipeline):
        """
        Tests that setting an invalid fixed parameter raises an error.
        """
        with pytest.raises(ValueError, match="Key 'c' is not a fixed parameter of node 'mul'"):
            basic_pipeline.set_fixed_params({"mul.c": 20})


class TestPipelineRun:
    def test_run_basic(self, basic_pipeline):
        """
        Tests the basic execution of the pipeline.
        """
        last_node_id, outputs, _ = basic_pipeline.run(run_params={'x': 5, 'y': 3})
        assert last_node_id == "mul"
        assert outputs['add'] == 8
        assert outputs['mul'] == 80  # 8 * 10 (fixed_param)

    def test_run_with_node_if_true_path(self, node_if_pipeline):
        """
        Tests the execution of the pipeline with a NodeIf.
        """
        last_node_id, outputs, _ = node_if_pipeline.run(run_params={'start_val': 20})
        assert last_node_id == "conditional_node"
        assert outputs['conditional_node'] == 21  # 20 + 1

    def test_run_with_node_if_false_path(self, node_if_pipeline):
        """
        Tests the execution of the pipeline with a NodeIf.
        """
        last_node_id, outputs, _ = node_if_pipeline.run(run_params={'start_val': 5})
        assert last_node_id == "conditional_node"
        assert outputs['conditional_node'] == 4  # 5 - 1

    def test_run_with_loop(self, loop_pipeline):
        """        
        Tests the execution of the pipeline with a looping node.
        """
        last_node_id, outputs, _ = loop_pipeline.run()
        assert last_node_id == "add_one"
        assert outputs['data_provider'] == [1, 2, 3]
        assert outputs['add_one'] == [2, 3, 4]  # [1+1, 2+1, 3+1]

    def test_run_no_optimize_memory(self, basic_pipeline):
        """
        Tests the execution of the pipeline with optimize_memory=False.
        """
        _, outputs, _ = basic_pipeline.run(run_params={'x': 5, 'y': 3}, optimize_memory=False)
        assert 'add' in outputs
        assert 'mul' in outputs
        assert outputs['add'] == 8


class TestPipelineSerialization:
    def test_to_dot_generates_string(self, basic_pipeline):
        """
        Tests the to_dot method.
        """""
        dot_string = basic_pipeline.to_dot()
        assert "digraph Pipeline" in dot_string
        assert '"add"' in dot_string
        assert '"mul"' in dot_string
        assert '"add" -> "mul"' in dot_string
        assert 'label="a"' in dot_string

    def test_to_and_from_json(self, basic_pipeline, tmp_path, add_func, mul_func):
        """
        Tests the to_json and from_json methods.
        """
        filepath = tmp_path / "pipeline.json"
        
        def test_resolver(type_str):
            if type_str == "test_pipeline.add_func":
                return add_func
            if type_str == "test_pipeline.mul_func":
                return mul_func
            return Pipeline._default_function_resolver(type_str)

        basic_pipeline.get_node("add").func.__module__ = "test_pipeline"
        basic_pipeline.get_node("add").func.__name__ = "add_func"
        basic_pipeline.get_node("mul").func.__module__ = "test_pipeline"
        basic_pipeline.get_node("mul").func.__name__ = "mul_func"

        basic_pipeline.to_json(filepath)
        assert filepath.exists()

        with open(filepath, 'r') as f:
            data = json.load(f)
        assert data['name'] == 'basic'
        assert len(data['nodes']) == 2
        assert len(data['edges']) == 3

        reconstructed_pipeline = Pipeline.from_json(filepath, function_resolver=test_resolver)
        assert reconstructed_pipeline.name == "basic"
        assert "add" in reconstructed_pipeline.nodes
        assert "mul" in reconstructed_pipeline.nodes
        assert reconstructed_pipeline.get_fixed_params() == {"mul.b": 10}

        last_node_id, outputs, _ = reconstructed_pipeline.run(run_params={'x': 5, 'y': 3})
        assert last_node_id == "mul"
        assert outputs['mul'] == 80

class TestPipelineMultiOutput:
    def test_run_multi_output_by_key(self, add_func):
        """
        Tests pipeline with a node that has multiple outputs, accessed by key.
        """
        p = Pipeline(name="multi_output_key")
        p.add_node(Node(node_id="multi_out", func=lambda: {'x': 1, 'y': 2}))
        p.add_node(Node(node_id="add", func=add_func), predecessors={'a': 'multi_out:x', 'b': 'multi_out:y'})
        last_node_id, outputs, _ = p.run()
        assert last_node_id == "add"
        assert outputs['add'] == 3

    def test_run_multi_output_by_index(self, add_func):
        """
        Tests pipeline with a node that has multiple outputs, accessed by index.
        """
        p = Pipeline(name="multi_output_index")
        p.add_node(Node(node_id="multi_out", func=lambda: [10, 20]))
        p.add_node(Node(node_id="add", func=add_func), predecessors={'a': 'multi_out:0', 'b': 'multi_out:1'})
        last_node_id, outputs, _ = p.run()
        assert last_node_id == "add"
        assert outputs['add'] == 30