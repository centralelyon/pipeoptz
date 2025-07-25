import pytest
import numpy as np
from unittest.mock import Mock

import sys, os
sys.path.append(os.path.abspath("../"))
from pipeoptz.node import Node, NodeIf
from pipeoptz.pipeline import Pipeline

@pytest.fixture
def simple_add_func():
    """Une fonction simple qui additionne deux nombres."""
    return lambda a, b: a + b

@pytest.fixture
def mock_func_with_call_tracker():
    """Une fonction mock qui suit ses appels."""
    mock = Mock(return_value="computed")
    return mock

@pytest.fixture
def true_pipeline():
    """Un pipeline simple pour le chemin 'true' de NodeIf."""
    p = Pipeline(name="true_path")
    p.add_node(Node(id="true_node", func=lambda x: f"true_{x}"), predecessors={'x': 'run_params:input'})
    return p

@pytest.fixture
def false_pipeline():
    """Un pipeline simple pour le chemin 'false' de NodeIf."""
    p = Pipeline(name="false_path")
    p.add_node(Node(id="false_node", func=lambda x: f"false_{x}"), predecessors={'x': 'run_params:input'})
    return p

# --- Tests pour la classe Node ---

class TestNode:
    def test_node_initialization(self, simple_add_func):
        """Teste si un Node est initialisé correctement."""
        node = Node(id="add_node", func=simple_add_func, fixed_params={'a': 1})
        assert node.id == "add_node"
        assert node.func == simple_add_func
        assert node.fixed_params == {'a': 1}
        assert node.output is None
        assert node.input_hash_last_exec is None

    def test_get_id(self, simple_add_func):
        """Teste la méthode get_id."""
        node = Node(id="test_id", func=simple_add_func)
        assert node.get_id() == "test_id"

    def test_execute_simple(self, simple_add_func):
        """Teste l'exécution de base sans paramètres fixes."""
        node = Node(id="add_node", func=simple_add_func)
        result = node.execute(inputs={'a': 5, 'b': 10})
        assert result == 15

    def test_execute_with_fixed_params(self, simple_add_func):
        """Teste l'exécution avec un mélange de paramètres fixes et d'exécution."""
        node = Node(id="add_node", func=simple_add_func, fixed_params={'a': 1})
        result = node.execute(inputs={'b': 9})
        assert result == 10

    def test_execute_raises_exception(self):
        """Teste que les exceptions de la fonction encapsulée sont propagées."""
        def error_func():
            raise ValueError("Test error")
        
        node = Node(id="error_node", func=error_func)
        with pytest.raises(ValueError, match="Test error"):
            node.execute()

    def test_memory_caching_avoids_recomputation(self, mock_func_with_call_tracker):
        """Teste que memory=True empêche la ré-exécution avec les mêmes entrées."""
        node = Node(id="cache_node", func=mock_func_with_call_tracker)
        
        # Première exécution
        result1 = node.execute(inputs={'x': 1}, memory=True)
        assert result1 == "computed"
        assert mock_func_with_call_tracker.call_count == 1
        assert node.output == "computed"

        # Seconde exécution avec les mêmes entrées
        result2 = node.execute(inputs={'x': 1}, memory=True)
        assert result2 == "computed"
        # Le compteur d'appels ne doit PAS augmenter
        assert mock_func_with_call_tracker.call_count == 1

    def test_memory_caching_recomputes_on_new_input(self, mock_func_with_call_tracker):
        """Teste que memory=True ré-exécute avec des entrées différentes."""
        node = Node(id="cache_node", func=mock_func_with_call_tracker)
        
        # Première exécution
        node.execute(inputs={'x': 1}, memory=True)
        assert mock_func_with_call_tracker.call_count == 1

        # Seconde exécution avec des entrées différentes
        node.execute(inputs={'x': 2}, memory=True)
        # Le compteur d'appels DOIT augmenter
        assert mock_func_with_call_tracker.call_count == 2

    def test_memory_caching_with_numpy_array(self):
        """Teste la mise en cache avec des tableaux numpy en entrée."""
        call_count = 0
        def numpy_func(arr):
            nonlocal call_count
            call_count += 1
            return np.sum(arr)

        node = Node(id="numpy_node", func=numpy_func)
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])

        res1 = node.execute(inputs={'arr': arr1}, memory=True)
        assert res1 == 6
        assert call_count == 1

        res2 = node.execute(inputs={'arr': arr1}, memory=True)
        assert res2 == 6
        assert call_count == 1

        res3 = node.execute(inputs={'arr': np.array([1, 2, 3])}, memory=True)
        assert res3 == 6
        assert call_count == 1 # Le hash doit être le même

        res4 = node.execute(inputs={'arr': arr2}, memory=True)
        assert res4 == 15
        assert call_count == 2

    def test_clear_memory(self, mock_func_with_call_tracker):
        """Teste que clear_memory force la ré-exécution."""
        node = Node(id="cache_node", func=mock_func_with_call_tracker)
        
        node.execute(inputs={'x': 1}, memory=True)
        assert mock_func_with_call_tracker.call_count == 1
        
        node.clear_memory()
        assert node.output is None
        assert node.input_hash_last_exec is None

        node.execute(inputs={'x': 1}, memory=True)
        # Le compteur d'appels doit augmenter car la mémoire a été vidée
        assert mock_func_with_call_tracker.call_count == 2

    def test_set_fixed_param(self, simple_add_func):
        """Teste la définition d'un seul paramètre fixe."""
        node = Node(id="add_node", func=simple_add_func, fixed_params={'a': 1})
        node.set_fixed_param('a', 5)
        assert node.get_fixed_params()['a'] == 5

    def test_set_fixed_param_raises_error_for_new_key(self, simple_add_func):
        """Teste que la définition d'un paramètre fixe non existant lève une ValueError."""
        node = Node(id="add_node", func=simple_add_func, fixed_params={'a': 1})
        with pytest.raises(ValueError, match="Key 'b' is not a fixed parameter of node 'add_node'"):
            node.set_fixed_param('b', 10)

    def test_is_fixed_param(self, simple_add_func):
        """Teste la méthode is_fixed_param."""
        node = Node(id="add_node", func=simple_add_func, fixed_params={'a': 1})
        assert node.is_fixed_param('a') is True
        assert node.is_fixed_param('b') is False


# --- Tests pour la classe NodeIf ---

class TestNodeIf:
    def test_nodeif_initialization(self, true_pipeline, false_pipeline):
        """Teste si un NodeIf est initialisé correctement."""
        cond_func = lambda x: x > 0
        node_if = NodeIf(
            id="if_node",
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
        """Teste que le pipeline 'true' est exécuté si la condition est vraie."""
        cond_func = lambda val: val > 10
        node_if = NodeIf(
            id="if_node",
            condition_func=cond_func,
            true_pipeline=true_pipeline,
            false_pipeline=false_pipeline
        )
        
        inputs = {'condition_func:val': 20, 'input': 'world'}
        result = node_if.execute(inputs=inputs)
        
        assert result == "true_world"

    def test_execute_false_path(self, true_pipeline, false_pipeline):
        """Teste que le pipeline 'false' est exécuté si la condition est fausse."""
        cond_func = lambda val: val > 10
        node_if = NodeIf(
            id="if_node",
            condition_func=cond_func,
            true_pipeline=true_pipeline,
            false_pipeline=false_pipeline
        )
        
        inputs = {'condition_func:val': 5, 'input': 'space'}
        result = node_if.execute(inputs=inputs)
        
        assert result == "false_space"

    def test_get_fixed_params_nested(self, true_pipeline, false_pipeline):
        """Teste la récupération des paramètres fixes de NodeIf et de ses sous-pipelines."""
        true_pipeline.get_node("true_node").fixed_params = {'z': 100}
        
        node_if = NodeIf(
            id="if_node",
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
        """Teste la définition des paramètres fixes sur NodeIf et ses sous-pipelines."""
        true_pipeline.get_node("true_node").fixed_params = {'z': 0}
        false_pipeline.get_node("false_node").fixed_params = {'w': 0}

        node_if = NodeIf(
            id="if_node",
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