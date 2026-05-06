# tests/test_optimizer.py

import pytest
import numpy as np
from unittest.mock import patch
import warnings

import sys
import os
sys.path.append(os.path.abspath("../src/"))
from pipeoptz.optimizer import PipelineOptimizer
from pipeoptz.pipeline import Pipeline
from pipeoptz.node import Node
from pipeoptz.parameter import (
    IntParameter,
    FloatParameter,
    ChoiceParameter,
    BoolParameter,
    MultiChoiceParameter
)

# --- Fixtures and Helpers ---

@pytest.fixture
def simple_pipeline():
    """A simple pipeline with two nodes for tests."""
    p = Pipeline(name="test_pipeline")
    # node1 multiplies input a by 2
    p.add_node(Node(node_id="node1", func=lambda a, **kwargs: a * 2, fixed_params={'a': 1, 'c': 'x'}))
    # node2 adds 1 to input b and depends on node1 output
    p.add_node(Node(node_id="node2", func=lambda b, **kwargs: b + 1, fixed_params={'b': 1, 'd': True}), predecessors={'b': 'node1'})
    return p

@pytest.fixture
def mse_loss():
    """A simple loss function (Mean Squared Error)."""
    return lambda y_pred, y_true: np.mean((y_pred - y_true)**2)

@pytest.fixture
def triplet_loss():
    """Triplet loss function."""
    return lambda anchor, positive, negative: np.mean(np.maximum(0, 1 + np.sum((anchor - positive)**2) - np.sum((anchor - negative)**2)))

@pytest.fixture
def optimizer_instance(simple_pipeline, mse_loss):
    """A base PipelineOptimizer instance."""
    opt = PipelineOptimizer(simple_pipeline, mse_loss, max_time_pipeline=1.0)
    # Register a mix of parameter types
    opt.add_param(IntParameter("node1", "a", 0, 10))
    opt.add_param(FloatParameter("node2", "b", 0, 10))
    opt.add_param(ChoiceParameter("node1", "c", ["x", "y"]))
    opt.add_param(BoolParameter("node2", "d"))
    return opt

@pytest.fixture
def sample_data():
    """Sample data for evaluation."""
    # run_params are not used in this simple pipeline
    X = [{'a': i} for i in range(3)]
    # y_true for node1.a = i
    y = [i * 2 + 1 for i in range(3)]
    y_neg = [i * 3 for i in range(3)]
    return X, y, y_neg

# --- Classes de Test ---

class TestOptimizerInitializationAndParams:
    def test_initialization(self, simple_pipeline, mse_loss):
        """
        Tests that the PipelineOptimizer initializes correctly with valid inputs.
        """
        opt = PipelineOptimizer(simple_pipeline, mse_loss, max_time_pipeline=0.5)
        # Ensure constructor assigns fields as expected
        assert opt.pipeline == simple_pipeline
        assert opt.loss == mse_loss
        assert opt.max_time_pipeline == 0.5
        assert opt.params_to_optimize == []

    def test_initialization_raises_errors(self, mse_loss):
        """
        Tests that the constructor raises errors for invalid inputs.
        """
        with pytest.raises(AssertionError):
            PipelineOptimizer("not_a_pipeline", mse_loss, 1.0)
        with pytest.raises(AssertionError):
            PipelineOptimizer(Pipeline("p"), "not_a_callable", 1.0)
        with pytest.raises(AssertionError):
            PipelineOptimizer(Pipeline("p"), mse_loss, -1.0)

    def test_add_param(self, optimizer_instance):
        """
        Tests that parameters are added correctly to the optimizer.
        """
        assert len(optimizer_instance.params_to_optimize) == 4
        assert isinstance(optimizer_instance.params_to_optimize[0], IntParameter)

    def test_set_and_get_params(self, optimizer_instance):
        """
        Tests that setting parameters updates the internal state and that retrieving them returns the correct values.
        """
        params_to_set = {"node1.a": 5, "node2.b": 8.0, "node1.c": "y", "node2.d": False}
        optimizer_instance.set_params(params_to_set)

        retrieved_params = optimizer_instance.get_params_value()
        assert retrieved_params == params_to_set

    def test_set_param_invalid_key_raises_error(self, optimizer_instance):
        """
        Tests that setting a parameter with an invalid key raises a ValueError.
        """
        with pytest.raises(ValueError, match="Parameter node1.z not found"):
            optimizer_instance.set_params({"node1.z": 1})

    def test_set_param_single_by_node_and_name(self, optimizer_instance):
        """
        Tests that set_param can update a single parameter by specifying node and parameter name.
        """
        optimizer_instance.set_param("node1", "a", 7)
        assert optimizer_instance.get_params_value()["node1.a"] == 7

    def test_set_param_invalid_raises(self, optimizer_instance):
        """
        Tests that setting a parameter with an invalid name raises a ValueError.
        """
        with pytest.raises(ValueError, match="not found"):
            optimizer_instance.set_param("node1", "nonexistent_xyz", 5)

    def test_update_pipeline_params(self, optimizer_instance):
        """
        Tests that the optimizer can update the pipeline's fixed parameters based on the current parameter values in the optimizer.
        """
        params_to_set = {"node1.a": 3, "node2.b": 7.0}
        optimizer_instance.set_params(params_to_set)
        optimizer_instance.update_pipeline_params()
        # Check pipeline sees the updated values
        pipeline_params = optimizer_instance.pipeline.get_fixed_params()
        assert pipeline_params["node1.a"] == 3
        assert pipeline_params["node2.b"] == 7.0


class TestOptimizerEvaluate:
    def test_evaluate(self, optimizer_instance, sample_data):
        """
        Tests that the evaluate method runs the pipeline with the current parameters and computes the loss correctly.
        """
        X, y, _ = sample_data
        
        optimizer_instance.set_params({"node1.a": 5, "node2.b": 5.0, "node1.c": "x", "node2.d": True})
        
        X_eval = [{} for _ in range(3)]
        y_eval = [1, 3, 5]
        # Pipeline executes node1(a=5) -> 10, then node2(b=10) -> 11
        results, loss = optimizer_instance.evaluate(X_eval, y_eval)
        
        assert results == [11, 11, 11]
        assert np.isclose(loss, ((11-1)**2 + (11-3)**2 + (11-5)**2) / 3)

    def test_evaluate_with_triplet_loss(self, simple_pipeline, triplet_loss, sample_data):
        """
        Tests that the evaluate method can handle a triplet loss function and computes it correctly based on the outputs of the pipeline.
        """
        optimizer = PipelineOptimizer(simple_pipeline, triplet_loss)
        
        optimizer.add_param(IntParameter("node1", "a", 0, 10))
        optimizer.set_params({"node1.a": 1})
        X, y, y_neg = sample_data

        # anchor=3, positive=3, negative=3
        # loss = mean(max(0, 1 + 0 - 0)) = 1
        _, loss = optimizer.evaluate([{'a': 1}], [3], [3])
        assert loss == 1.0

    def test_evaluate_with_timeout(self, optimizer_instance, sample_data):
        """
        Tests that the evaluate method handles timeout correctly.
        
        """
        X, y, _ = sample_data
        optimizer_instance.max_time_pipeline = 0.01
        # Simulate a run that exceeds time budget
        with patch.object(optimizer_instance.pipeline, 'run', return_value=("node2", {"node2": 0}, (0.02, {}))):
            _, loss = optimizer_instance.evaluate(X, y)
            assert loss == float("inf")

    def test_evaluate_early_exit_on_timeout(self, optimizer_instance, sample_data):
        """
        Tests that when a run exceeds max_time_pipeline, the evaluate method returns None for results and inf for loss, indicating an early exit due to timeout.
        """
        from unittest.mock import patch
        optimizer_instance.max_time_pipeline = 0.001
        X = [{}, {}]
        y = [1, 2]
        # Simulate a slow run and verify early exit
        with patch.object(optimizer_instance.pipeline, 'run',
                          return_value=("node2", {"node2": 0}, (1.0, {}))):
            results, loss = optimizer_instance.evaluate(X, y)
        assert loss == float("inf")
        assert None in results



class TestOptimizerMethods:
    """
    Tests that the optimize method correctly dispatches to the specified optimization method and that each method runs without errors and returns results in the expected format.
    """
    @pytest.mark.parametrize("method,kwargs", [
        ("GS", {"max_combinations": 5}),
        ("ACO", {"iterations": 2, "ants": 2}),
        ("SA", {"iterations": 5}),
        ("PSO", {"iterations": 2, "swarm_size": 3}),
        ("GA", {"generations": 2, "population_size": 3}),
        ("BO", {"iterations": 2, "init_points": 2})
    ])
    def test_optimization_methods_run_and_return_correct_format(self, optimizer_instance, sample_data, method, kwargs):
        """
        Tests that each optimization method runs and returns results in the expected format.
        """
        X, y, _ = sample_data
        # BO cannot handle multi-choice params in this setup
        if method == "BO":
            optimizer_instance.params_to_optimize = [
                p for p in optimizer_instance.params_to_optimize if not isinstance(p, MultiChoiceParameter)
            ]

        warnings.filterwarnings("ignore")
        best_params, loss_log = optimizer_instance.optimize(X, y, method=method, **kwargs)
        warnings.resetwarnings()
        
        assert isinstance(best_params, dict)
        assert isinstance(loss_log, list)
        
        expected_keys = {f"{p.node_id}.{p.param_name}" for p in optimizer_instance.params_to_optimize}
        assert set(best_params.keys()) == expected_keys
        
        num_iterations = kwargs.get("iterations", kwargs.get("generations", kwargs.get("max_combinations")))
        assert len(loss_log) == num_iterations

    def test_optimize_dispatcher_invalid_method(self, optimizer_instance, sample_data):
        """
        Tests that calling optimize with an unknown method raises a ValueError."""
        X, y, _ = sample_data
        # Unknown method should fail
        with pytest.raises(ValueError, match="Unknown optimization method: UNKNOWN"):
            optimizer_instance.optimize(X, y, method="UNKNOWN")


class TestOptimizerHelpers:
    @pytest.fixture
    def bo_param_defs(self):
        return [
            ("node1.p_int", IntParameter("node1", "p_int", 0, 10)),
            ("node1.p_float", FloatParameter("node1", "p_float", 0.0, 1.0)),
            ("node1.p_choice", ChoiceParameter("node1", "p_choice", ["A", "B", "C"])),
            ("node1.p_bool", BoolParameter("node1", "p_bool"))
        ]

    def test_encode_decode_cycle(self, bo_param_defs):
        """
        Tests that encoding a parameter dictionary to a numeric vector and then decoding it back returns the original parameter values, ensuring the encode and decode functions are consistent with each other.
        """
        params_dict = {
            "node1.p_int": 5,
            "node1.p_float": 0.5,
            "node1.p_choice": "B",
            "node1.p_bool": True
        }
        
        encoded = PipelineOptimizer._encode(params_dict, bo_param_defs)
        assert np.array_equal(encoded, np.array([5, 0.5, 1, 1]))
        decoded = PipelineOptimizer._decode(encoded, bo_param_defs)
        assert decoded["node1.p_int"] == 5
        assert decoded["node1.p_float"] == 0.5
        assert decoded["node1.p_choice"] == "B"
        assert decoded["node1.p_bool"] is True

    def test_decode_with_clipping(self, bo_param_defs):
        """
        Tests that the decode function correctly clips out-of-bounds numeric values to the valid ranges defined by the parameter definitions, ensuring that the optimizer can handle and recover from invalid numeric inputs gracefully.
        """
        encoded_out_of_bounds = np.array([15, -0.5, 5, 0.2])
        decoded = PipelineOptimizer._decode(encoded_out_of_bounds, bo_param_defs)
        assert decoded["node1.p_int"] == 10
        assert decoded["node1.p_float"] == 0.0
        assert decoded["node1.p_choice"] == "C"
        assert decoded["node1.p_bool"] is False

    def test_plot_convergence(self, optimizer_instance):
        """
        Tests that the plot_convergence method generates a valid convergence plot.
        """
        optimizer_instance.best_params_history = [
            {"node1.a": 1, "node2.b": 1.0, "node1.c": "x", "node2.d": True},
            {"node1.a": 2, "node2.b": 2.0, "node1.c": "y", "node2.d": False},
            {"node1.a": 10, "node2.b": 10.0, "node1.c": "x", "node2.d": True},
        ]
        plot = optimizer_instance.plot_convergence()
        assert isinstance(plot, np.ndarray)
        assert plot.shape == (4, 3) # 4 params, 3 iterations
        assert plot.dtype == np.uint8

    def test_plot_convergence_empty_history_returns_empty_array(self, optimizer_instance):
        """
        Tests that the plot_convergence method returns an empty array when the best_params_history is empty.
        """
        optimizer_instance.best_params_history = []
        result = optimizer_instance.plot_convergence()
        assert result.size == 0

    def test_plot_convergence_bool_parameter(self, simple_pipeline, mse_loss):
        """
        Tests that the plot_convergence method correctly visualizes boolean parameters.
        """
        opt = PipelineOptimizer(simple_pipeline, mse_loss)
        # Bool param should map to 0/255
        opt.add_param(BoolParameter("node2", "d"))
        opt.best_params_history = [
            {"node2.d": True},
            {"node2.d": False},
        ]
        plot = opt.plot_convergence()
        assert plot.shape == (1, 2)
        assert plot[0, 0] == 255   # True → max
        assert plot[0, 1] == 0     # False → min

    def test_plot_convergence_single_choice_parameter(self, simple_pipeline, mse_loss):
        """
        Tests that the plot_convergence method correctly visualizes a ChoiceParameter with only one option, ensuring that it maps all values to the midpoint (128) in the convergence plot, since there is no variability in the parameter values.
        """
        opt = PipelineOptimizer(simple_pipeline, mse_loss)
        # Single-option choice should normalize to midpoint
        opt.add_param(ChoiceParameter("node1", "c", ["only_one"]))
        opt.best_params_history = [
            {"node1.c": "only_one"},
            {"node1.c": "only_one"},
        ]
        plot = opt.plot_convergence()
        assert plot.shape == (1, 2)
        # Single choice -> normalized to 128
        assert all(v == 128 for v in plot[0])