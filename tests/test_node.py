import pytest
import numpy as np
from unittest.mock import Mock

import sys
import os
sys.path.append(os.path.abspath("../src/"))
from pipeoptz.node import Node, NodeIf, NodeFor, NodeWhile
from pipeoptz.pipeline import Pipeline


@pytest.fixture
def simple_add_func():
    """
    A simple function that adds two numbers.
    """
    return lambda a, b: a + b

@pytest.fixture
def mock_func_with_call_tracker():
    """
    A mock function that tracks its calls.
    """
    mock = Mock(return_value="computed")
    return mock

@pytest.fixture
def true_pipeline():
    """
    A simple pipeline for the 'true' path of NodeIf.
    """
    p = Pipeline(name="true_path")
    p.add_node(Node(node_id="true_node", func=lambda x: f"true_{x}"), predecessors={'x': 'run_params:input'})
    return p

@pytest.fixture
def false_pipeline():
    """
    A simple pipeline for the 'false' path of NodeIf.
    """
    p = Pipeline(name="false_path")
    p.add_node(Node(node_id="false_node", func=lambda x: f"false_{x}"), predecessors={'x': 'run_params:input'})
    return p

@pytest.fixture
def loop_pipeline():
    """
    A simple pipeline for loop nodes.
    """
    p = Pipeline(name="loop_pipe")
    p.add_node(Node(node_id="add_one", func=lambda loop_var: loop_var + 1), predecessors={'loop_var': 'run_params:loop_var'})
    return p


# --- Class Node tests ---

class TestNode:
    def test_node_initialization(self, simple_add_func):
        """
        Tests if a Node is initialized correctly.
        """
        node = Node(node_id="add_node", func=simple_add_func, fixed_params={'a': 1})
        assert node.id == "add_node"
        assert node.func == simple_add_func
        assert node.fixed_params == {'a': 1}
        assert node.output is None
        assert node.input_hash_last_exec is None

    def test_get_id(self, simple_add_func):
        """
        Tests the get_id method.
        """
        node = Node(node_id="test_id", func=simple_add_func)
        assert node.get_id() == "test_id"

    def test_execute_simple(self, simple_add_func):
        """
        Tests that the basic execution without fixed parameters works correctly.
        """
        node = Node(node_id="add_node", func=simple_add_func)
        result = node.execute(inputs={'a': 5, 'b': 10})
        assert result == 15

    def test_execute_with_fixed_params(self, simple_add_func):
        """
        Tests execution with a mix of fixed and runtime parameters.
        """
        node = Node(node_id="add_node", func=simple_add_func, fixed_params={'a': 1})
        result = node.execute(inputs={'b': 9})
        assert result == 10


    def test_execute_no_inputs_uses_fixed_params_only(self):
        """
        Tests execution with no inputs (defaults to {}).
        """
        node = Node(node_id="n", func=lambda: 42)
        assert node.execute() == 42

    def test_memory_caching_avoids_recomputation(self, mock_func_with_call_tracker):
        """
        Tests that memory=True prevents re-execution with the same inputs.
        """
        node = Node(node_id="cache_node", func=mock_func_with_call_tracker)
        
        result1 = node.execute(inputs={'x': 1})
        assert result1 == "computed"
        assert mock_func_with_call_tracker.call_count == 1
        assert node.output == "computed"

        result2 = node.execute(inputs={'x': 1})
        assert result2 == "computed"
        assert mock_func_with_call_tracker.call_count == 1

    def test_memory_caching_recomputes_on_new_input(self, mock_func_with_call_tracker):
        """
        Tests that memory=True re-executes with different inputs.
        """
        node = Node(node_id="cache_node", func=mock_func_with_call_tracker)
        node.execute(inputs={'x': 1})
        assert mock_func_with_call_tracker.call_count == 1
        node.execute(inputs={'x': 2})
        assert mock_func_with_call_tracker.call_count == 2

    def test_memory_caching_with_numpy_array(self):
        """
        Tests caching with numpy arrays as input.
        """
        call_count = 0
        def numpy_func(arr):
            nonlocal call_count
            call_count += 1
            return np.sum(arr)

        node = Node(node_id="numpy_node", func=numpy_func)
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])

        res1 = node.execute(inputs={'arr': arr1})
        assert res1 == 6
        assert call_count == 1

        res2 = node.execute(inputs={'arr': arr1})
        assert res2 == 6
        assert call_count == 1

        res3 = node.execute(inputs={'arr': np.array([1, 2, 3])})
        assert res3 == 6
        assert call_count == 1

        res4 = node.execute(inputs={'arr': arr2})
        assert res4 == 15
        assert call_count == 2

    def test_clear_memory(self, mock_func_with_call_tracker):
        """
        Tests that clear_memory forces re-execution.
        """
        node = Node(node_id="cache_node", func=mock_func_with_call_tracker)
        
        node.execute(inputs={'x': 1})
        assert mock_func_with_call_tracker.call_count == 1
        
        node.clear_memory()
        assert node.output is None
        assert node.input_hash_last_exec is None

        node.execute(inputs={'x': 1})
        assert mock_func_with_call_tracker.call_count == 2

    def test_set_fixed_param(self, simple_add_func):
        """
        Tests setting a single fixed parameter.
        """
        node = Node(node_id="add_node", func=simple_add_func, fixed_params={'a': 1})
        node.set_fixed_param('a', 5)
        assert node.get_fixed_params()['a'] == 5

    def test_set_fixed_param_raises_error_for_new_key(self, simple_add_func):
        """
        Tests that setting a non-existent fixed parameter raises a ValueError.
        """
        node = Node(node_id="add_node", func=simple_add_func, fixed_params={'a': 1})
        with pytest.raises(ValueError, match="Key 'b' is not a fixed parameter of node 'add_node'"):
            node.set_fixed_param('b', 10)

    def test_set_fixed_params_non_dict_raises(self):
        """
        Tests that set_fixed_params raises a ValueError if the input is not a dictionary.
        """
        node = Node("n", lambda a: a, fixed_params={'a': 1})
        with pytest.raises(ValueError, match="Fixed parameters must be a dictionary"):
            node.set_fixed_params([('a', 2)])

    def test_set_fixed_params_non_string_key_raises(self):
        """
        Tests that set_fixed_params raises a ValueError if any key in the input dictionary is not a string.
        """
        node = Node("n", lambda a: a, fixed_params={'a': 1})
        with pytest.raises(ValueError, match="is not a string"):
            node.set_fixed_params({1: 2})


    def test_is_fixed_param(self, simple_add_func):
        """
        Tests the is_fixed_param method.
        """
        node = Node(node_id="add_node", func=simple_add_func, fixed_params={'a': 1})
        assert node.is_fixed_param('a') is True
        assert node.is_fixed_param('b') is False


# --- Class NodeIf tests ---

class TestNodeIf:
    def test_nodeif_initialization(self, true_pipeline, false_pipeline):
        """
        Tests if a NodeIf is initialized correctly.
        """
        def cond_func(x):
            return x > 0
        node_if = NodeIf(
            node_id="if_node",
            condition_func=cond_func,
            true_pipeline=true_pipeline,
            false_pipeline=false_pipeline,
            fixed_params={'y': 1}
        )
        assert node_if.id == "if_node"
        assert node_if.func == cond_func
        assert node_if.true_pipeline == true_pipeline
        assert node_if.false_pipeline == false_pipeline
        assert node_if.fixed_params == {'y': 1}

    def test_execute_true_path(self, true_pipeline, false_pipeline):
        """
        Tests that the 'true' pipeline is executed if the condition is true.
        """
        def cond_func(val):
            return val > 10
        node_if = NodeIf(
            node_id="if_node",
            condition_func=cond_func,
            true_pipeline=true_pipeline,
            false_pipeline=false_pipeline
        )
        
        inputs = {'condition_func:val': 20, 'input': 'world'}
        result = node_if.execute(inputs=inputs)
        
        assert result == "true_world"

    def test_execute_false_path(self, true_pipeline, false_pipeline):
        """
        Tests that the 'false' pipeline is executed if the condition is false.
        """
        def cond_func(val):
            return val > 10
        node_if = NodeIf(
            node_id="if_node",
            condition_func=cond_func,
            true_pipeline=true_pipeline,
            false_pipeline=false_pipeline
        )
        
        inputs = {'condition_func:val': 5, 'input': 'space'}
        result = node_if.execute(inputs=inputs)
        
        assert result == "false_space"

    def test_execute_does_not_mutate_caller_inputs(self, true_pipeline, false_pipeline):
        """
        Tests if the caller's dict is not mutated.
        """
        node_if = NodeIf("if", lambda val: val > 0, true_pipeline, false_pipeline)
        inputs = {'condition_func:val': 5, 'input': 'hello'}
        original = dict(inputs)
        node_if.execute(inputs)
        assert inputs == original

    def test_get_fixed_params_nested(self, true_pipeline, false_pipeline):
        """
        Tests retrieving fixed parameters from NodeIf and its sub-pipelines.
        """
        true_pipeline.get_node("true_node").fixed_params = {'z': 100}
        
        node_if = NodeIf(
            node_id="if_node",
            condition_func=lambda: True,
            true_pipeline=true_pipeline,
            false_pipeline=false_pipeline,
            fixed_params={'own_param': 42}
        )
        
        params = node_if.get_fixed_params()
        
        expected_params = {
            'own_param': 42,
            'true_pipeline': {'true_node.z': 100},
            'false_pipeline': {}
        }
        assert params == expected_params

    def test_set_fixed_params_nested(self, true_pipeline, false_pipeline):
        """
        Tests setting fixed parameters on NodeIf and its sub-pipelines.
        """
        true_pipeline.get_node("true_node").fixed_params = {'z': 0}
        false_pipeline.get_node("false_node").fixed_params = {'w': 0}

        node_if = NodeIf(
            node_id="if_node",
            condition_func=lambda: True,
            true_pipeline=true_pipeline,
            false_pipeline=false_pipeline,
            fixed_params={'own_param': 0}
        )

        new_params = {
            'own_param': 99,
            'true_pipeline': {'true_node.z': 101},
            'false_pipeline': {'false_node.w': 202}
        }
        
        node_if.set_fixed_params(new_params)

        assert node_if.fixed_params['own_param'] == 99
        assert true_pipeline.get_node("true_node").fixed_params['z'] == 101
        assert false_pipeline.get_node("false_node").fixed_params['w'] == 202

    def test_set_fixed_param_true_pipeline_key(self, true_pipeline, false_pipeline):
        """
        Tests setting a fixed parameter on the true_pipeline via set_fixed_param.
        """
        true_pipeline.get_node("true_node").fixed_params = {'z': 0}
        node_if = NodeIf("if", lambda: True, true_pipeline, false_pipeline)
        node_if.set_fixed_param('true_pipeline', {'true_node.z': 99})
        assert true_pipeline.get_node("true_node").fixed_params['z'] == 99

    def test_set_fixed_param_false_pipeline_key(self, true_pipeline, false_pipeline):
        """
        Tests setting a fixed parameter on the false_pipeline via set_fixed_param.
        """
        false_pipeline.get_node("false_node").fixed_params = {'w': 0}
        node_if = NodeIf("if", lambda: True, true_pipeline, false_pipeline)
        node_if.set_fixed_param('false_pipeline', {'false_node.w': 42})
        assert false_pipeline.get_node("false_node").fixed_params['w'] == 42


# --- Class NodeFor tests ---

class TestNodeFor:
    def test_nodefor_initialization(self, loop_pipeline):
        """
        Tests if a NodeFor is initialized correctly.
        """
        node_for = NodeFor(node_id="for_node", loop_pipeline=loop_pipeline, fixed_params={'iterations': 3})
        assert node_for.id == "for_node"
        assert node_for.loop_pipeline == loop_pipeline
        assert node_for.fixed_params == {'iterations': 3}

    def test_invalid_fixed_param_key_raises(self, loop_pipeline):
        with pytest.raises(ValueError, match="Only 'iterations' is allowed"):
            NodeFor("for", loop_pipeline, fixed_params={'not_iterations': 5})

    def test_get_fixed_params_includes_loop_pipeline_key(self, loop_pipeline):
        node_for = NodeFor("for", loop_pipeline, fixed_params={'iterations': 3})
        params = node_for.get_fixed_params()
        assert 'iterations' in params
        assert 'loop_pipeline' in params

    def test_set_fixed_params_updates_iterations(self, loop_pipeline):
        node_for = NodeFor("for", loop_pipeline, fixed_params={'iterations': 3})
        node_for.set_fixed_params({'iterations': 7})
        assert node_for.fixed_params['iterations'] == 7
    
    def test_execute_fixed_iterations(self, loop_pipeline):
        """
        Tests NodeFor execution with a fixed number of iterations.
        """
        node_for = NodeFor(node_id="for_node", loop_pipeline=loop_pipeline, fixed_params={'iterations': 3})
        result = node_for.execute(inputs={'loop_var': 0})
        assert result == 3

    def test_execute_zero_iterations_returns_loop_var_unchanged(self, loop_pipeline):
        """
        Tests that iterations=0 is valid and the loop body never runs.
        """
        node_for = NodeFor("for", loop_pipeline)
        result = node_for.execute({'iterations': 0, 'loop_var': 42})
        assert result == 42

    def test_execute_input_iterations(self, loop_pipeline):
        """
        Tests NodeFor execution with iterations from input.
        """
        node_for = NodeFor(node_id="for_node", loop_pipeline=loop_pipeline)
        result = node_for.execute(inputs={'iterations': 5, 'loop_var': 0})
        assert result == 5

    def test_execute_missing_iterations_raises_error(self, loop_pipeline):
        """
        Tests that NodeFor raises an error if 'iterations' is missing.
        """
        node_for = NodeFor(node_id="for_node", loop_pipeline=loop_pipeline)
        with pytest.raises(ValueError, match="NodeFor requires an 'iterations' input"):
            node_for.execute(inputs={'loop_var': 0})

    def test_execute_missing_loop_var_raises_error(self, loop_pipeline):
        """
        Tests that NodeFor raises an error if 'loop_var' is missing.
        """
        node_for = NodeFor(node_id="for_node", loop_pipeline=loop_pipeline, fixed_params={'iterations': 3})
        with pytest.raises(ValueError, match="NodeFor requires a 'loop_var' input"):
            node_for.execute(inputs={})


    def test_skip_failed_loop(self):
        """
        Tests if skip_failed_loop=True a failed iteration is skipped silently.
        """
        call_count = [0]
        def inc_or_fail(loop_var):
            call_count[0] += 1
            if call_count[0] == 3:
                raise ValueError("fail on 3rd call")
            return loop_var + 1

        p = Pipeline("lp")
        p.add_node(Node("step", inc_or_fail),
                   predecessors={'loop_var': 'run_params:loop_var'})
        node_for = NodeFor("for", p, fixed_params={'iterations': 4})
        node_for.set_run_params(skip_failed_loop=True)
        # calls: 1→0+1=1, 2→1+1=2, 3→fail(loop_var stays 2), 4→2+1=3
        assert node_for.execute({'loop_var': 0}) == 3



# --- Class NodeWhile tests ---

class TestNodeWhile:
    def test_nodewhile_initialization(self, loop_pipeline):
        """
        Tests if a NodeWhile is initialized correctly.
        """
        def cond_func(loop_var):
            return loop_var < 5
        node_while = NodeWhile(node_id="while_node", condition_func=cond_func, loop_pipeline=loop_pipeline)
        assert node_while.id == "while_node"
        assert node_while.func == cond_func
        assert node_while.loop_pipeline == loop_pipeline

    def test_condition_func_prefix_inputs_routed_correctly(self):
        """
        Tests that inputs with 'condition_func:' prefix are passed to the condition function.
        """
        loop_p = Pipeline("lp")
        loop_p.add_node(Node("inc", lambda loop_var: loop_var + 1),
                        predecessors={'loop_var': 'run_params:loop_var'})

        def cond(loop_var, limit):
            return loop_var < limit

        node_while = NodeWhile("w", cond, loop_p)
        result = node_while.execute({'condition_func:limit': 3, 'loop_var': 0})
        assert result == 3


    def test_get_fixed_params_includes_loop_pipeline_key(self, loop_pipeline):
        """
        Tests that get_fixed_params includes the 'loop_pipeline' key for NodeWhile.
        """
        def cond(loop_var):
            return loop_var < 5
        node_while = NodeWhile("w", cond, loop_pipeline,
                               fixed_params={'max_iterations': 10})
        params = node_while.get_fixed_params()
        assert 'max_iterations' in params
        assert 'loop_pipeline' in params

    def test_set_fixed_param_invalid_own_key_raises(self, loop_pipeline):
        """
        Tests that setting a non-existent fixed parameter on NodeWhile raises a ValueError.
        """
        def cond(loop_var):
            return False
        node_while = NodeWhile("w", cond, loop_pipeline,
                               fixed_params={'max_iterations': 5})
        with pytest.raises(ValueError, match="is not a fixed parameter"):
            node_while.set_fixed_param('nonexistent_key', 99)

    def test_execute_while_condition_true(self, loop_pipeline):
        """
        Tests NodeWhile execution until the condition is false.
        """
        def cond_func(loop_var):
            return loop_var < 5
        node_while = NodeWhile(node_id="while_node", condition_func=cond_func, loop_pipeline=loop_pipeline)
        result = node_while.execute(inputs={'loop_var': 0})
        assert result == 5

    def test_execute_max_iterations(self, loop_pipeline):
        """
        Tests NodeWhile execution with a max_iterations limit.
        """
        def cond_func(loop_var):
            return True  # Condition is always true
        node_while = NodeWhile(node_id="while_node", condition_func=cond_func, loop_pipeline=loop_pipeline, fixed_params={'max_iterations': 3})
        result = node_while.execute(inputs={'loop_var': 0})
        assert result == 3

    def test_execute_missing_loop_var_raises_error(self, loop_pipeline):
        """
        Tests that NodeWhile raises an error if 'loop_var' is missing.
        """
        def cond_func(loop_var):
            return loop_var < 5
        node_while = NodeWhile(node_id="while_node", condition_func=cond_func, loop_pipeline=loop_pipeline)
        with pytest.raises(ValueError, match="NodeWhile requires a 'loop_var' input"):
            node_while.execute(inputs={})

    def test_execute_does_not_mutate_caller_inputs(self, loop_pipeline):
        """
        Tests that NodeWhile does not delete keys from the caller's dict.
        """
        def cond(loop_var, limit):
            return loop_var < limit
        node_while = NodeWhile("w", cond, loop_pipeline,
                               fixed_params={'max_iterations': 2})
        inputs = {'condition_func:limit': 3, 'loop_var': 0}
        original = dict(inputs)
        node_while.execute(inputs)
        assert inputs == original

    def test_skip_failed_loop(self):
        """With skip_failed_loop=True, a failed iteration is skipped."""
        call_count = [0]
        def inc_or_fail(loop_var):
            call_count[0] += 1
            if call_count[0] == 2:
                raise ValueError("fail on 2nd call")
            return loop_var + 1

        p = Pipeline("lp")
        p.add_node(Node("step", inc_or_fail),
                   predecessors={'loop_var': 'run_params:loop_var'})

        def cond(loop_var):
            return loop_var < 3

        node_while = NodeWhile("w", cond, p, fixed_params={'max_iterations': 5})
        node_while.set_run_params(skip_failed_loop=True)
        result = node_while.execute({'loop_var': 0})
        assert result is not None   # did not raise
