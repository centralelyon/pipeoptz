import pytest
import json

import sys
import os
sys.path.append(os.path.abspath("../src/"))
from pipeoptz.pipeline import Pipeline, _product
from pipeoptz.node import Node, NodeIf, NodeFor, NodeWhile


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

class TestProduct:
    def test_basic_cartesian_product(self):
        """
        Tests that the _product function correctly generates the Cartesian product of two input lists, producing all possible pairs of elements from the first and second list.
        """
        result = list(_product([1, 2], ['a', 'b']))
        assert result == [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]

    def test_max_combinations_limits_output(self):
        """
        Tests that the _product function respects the max_combinations parameter by limiting the number of generated combinations to the specified maximum, ensuring that it does not produce more pairs than allowed when max_combinations is set.
        """
        result = list(_product([1, 2], [3, 4], max_combinations=2))
        assert len(result) == 2

    def test_random_sampling_returns_correct_count(self):
        """
        Tests that the _product function returns a random sample of combinations when the random parameter is set to True.
        """
        result = list(_product([1, 2, 3], [4, 5, 6], random=True, max_combinations=5))
        assert len(result) == 5
        for a, b in result:
            assert a in [1, 2, 3]
            assert b in [4, 5, 6]

    def test_random_optimize_memory_returns_correct_count(self):
        """
        Tests that the _product function returns a random sample of combinations with optimize_memory=True, ensuring that it generates the correct number of combinations while optimizing memory usage.
        """
        result = list(_product([1, 2, 3], [4, 5, 6],
                               random=True, max_combinations=4, optimize_memory=True))
        assert len(result) == 4


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
    
    def test_add_node_id_starting_with_run_params_raises(self):
        """
        Tests that adding a node with an id starting with 'run_params:' raises an error, ensuring that node identifiers do not conflict with the reserved namespace used for run parameters in the pipeline execution context.
        """
        p = Pipeline("test")
        node = Node("run_params:bad", lambda: 1)
        with pytest.raises(ValueError, match="cannot start with 'run_params:'"):
            p.add_node(node)
    
    def test_add_duplicate_sub_pipeline_raises(self):
        """
        Tests that adding the same sub-pipeline twice raises an error, ensuring that the pipeline structure maintains unique node identifiers and prevents conflicts when incorporating sub-pipelines.
        """
        sub = Pipeline("dup")
        p = Pipeline("main")
        p.add_node(sub)
        with pytest.raises(ValueError, match="already exists"):
            p.add_node(sub)


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

    def test_run_nodefor_in_pipeline(self):
        """
        Tests the execution of the pipeline with a NodeFor, ensuring that the loop executes the specified number of iterations and produces the correct final output based on the loop pipeline's logic and the fixed parameters provided for the iterations.
        """
        loop_p = Pipeline("loop")
        loop_p.add_node(Node("inc", lambda loop_var: loop_var + 1),
                        predecessors={'loop_var': 'run_params:loop_var'})
        p = Pipeline("main")
        p.add_node(NodeFor("for_node", loop_p, fixed_params={'iterations': 3}),
                   predecessors={'loop_var': 'run_params:start'})
        _, outputs, _ = p.run({'start': 0})
        assert outputs['for_node'] == 3

    def test_run_nodewhile_in_pipeline(self):
        """
        Tests the execution of the pipeline with a NodeWhile, ensuring that the loop executes until the condition is no longer met and produces the correct final output based on the loop pipeline's logic and the fixed parameters provided for the iterations.
        """
        loop_p = Pipeline("loop")
        loop_p.add_node(Node("inc", lambda loop_var: loop_var + 1),
                        predecessors={'loop_var': 'run_params:loop_var'})
        p = Pipeline("main")
        p.add_node(NodeWhile("while_node", lambda loop_var: loop_var < 5, loop_p),
                   predecessors={'loop_var': 'run_params:start'})
        _, outputs, _ = p.run({'start': 0})
        assert outputs['while_node'] == 5


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

    def test_run_optimize_memory_removes_intermediate_outputs(self, basic_pipeline):
        """
        Tests the execution of the pipeline with optimize_memory=True, ensuring that intermediate node outputs 
        are not included in the returned outputs dictionary, while still providing the final outputs of the pipeline execution. 
        """
        _, outputs, _ = basic_pipeline.run({'x': 5, 'y': 3}, optimize_memory=True)
        assert 'add' not in outputs
        assert outputs['mul'] == 80


    def test_run_debug_mode_prints_node_names(self, basic_pipeline, capsys):
        """
        Tests that running the pipeline in debug mode prints the names of the nodes as they are executed.
        """
        basic_pipeline.run({'x': 1, 'y': 2}, debug=True)
        out = capsys.readouterr().out
        assert "add" in out
        assert "mul" in out

    def test_run_skip_failed_loop_continues_on_error(self):
        """
        Tests that when skip_failed_loop is True, if an iteration of a loop node raises an exception, 
        the pipeline continues executing the remaining iterations and completes the run without crashing, while skipping the failed iteration's output.
        """
        def fail_on_two(a, b):
            if a == 2:
                raise ValueError("fail on 2")
            return a + b

        p = Pipeline("test")
        p.add_node(Node("data", lambda: [1, 2, 3]))
        p.add_node(Node("proc", fail_on_two, fixed_params={'b': 10}),
                   predecessors={'[a]': 'data'})
        _, outputs, _ = p.run(skip_failed_loop=True)
        assert 2 not in outputs['proc']
        assert 11 in outputs['proc']    # 1 + 10
        assert 13 in outputs['proc']    # 3 + 10

    def test_run_returns_timing_info(self, basic_pipeline):
        """
        Tests that the run method returns timing information for the total execution time and individual node execution times.
        """
        _, _, (total_time, node_times) = basic_pipeline.run({'x': 1, 'y': 2})
        assert total_time >= 0
        assert set(node_times.keys()) == {'add', 'mul'}

    def test_run_multiple_inputs_curly_syntax(self):
        """
        Tests the {param} syntax for fan-out over the source node's output list.
        """
        p = Pipeline("multi")
        p.add_node(Node("src", lambda: [1, 2]))
        p.add_node(Node("double", lambda a: a * 10),
                   predecessors={'{a}': 'src'})
        _, outputs, _ = p.run()
        assert outputs['double'] == [10, 20]


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

    def test_to_mermaid_generates_string(self, basic_pipeline):
        """
        Tests the to_mermaid method.
        """
        mermaid_string = basic_pipeline.to_mermaid()
        assert "flowchart TD" in mermaid_string
        assert 'add["add' in mermaid_string
        assert 'add -->|a| mul' in mermaid_string

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
    
    def test_default_resolver_lambda_raises_import_error(self):
        """
        Tests that the default function resolver raises an ImportError when trying to resolve a lambda function.
        """
        with pytest.raises(ImportError, match="lambda"):
            Pipeline._default_function_resolver("mymodule.<lambda>")

    def test_default_resolver_missing_function_raises(self):
        """
        Tests that the default function resolver raises an ImportError when trying to resolve a non-existent function.
        """
        with pytest.raises((ImportError, AttributeError)):
            Pipeline._default_function_resolver("os.nonexistent_xyz_abc")


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

class TestPipelineClearMemory:
    def test_clear_memory_forces_recomputation(self):
        """
        Tests that clear_memory forces recomputation of node outputs on subsequent runs, ensuring that cached results are invalidated and 
        the pipeline re-executes nodes to produce new outputs after memory is cleared.
        """
        call_count = [0]
        def counting(x):
            call_count[0] += 1
            return x * 2

        p = Pipeline("test")
        p.add_node(Node("n", counting), predecessors={'x': 'run_params:val'})
        p.run({'val': 5})
        p.run({'val': 5})
        assert call_count[0] == 1   # cached on second run
        p.clear_memory()
        p.run({'val': 5})
        assert call_count[0] == 2   # re-executes after clear