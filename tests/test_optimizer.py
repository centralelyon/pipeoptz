# tests/test_optimizer.py

import pytest
import numpy as np
from unittest.mock import patch

import sys, os
sys.path.append(os.path.abspath("../"))
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

# --- Fixtures et Fonctions d'aide ---

@pytest.fixture
def simple_pipeline():
    """Un pipeline simple avec deux nœuds pour les tests."""
    p = Pipeline(name="test_pipeline")
    p.add_node(Node(id="node1", func=lambda a: a * 2, fixed_params={'a': 1}))
    p.add_node(Node(id="node2", func=lambda b: b + 1, fixed_params={'b': 1}), predecessors={'b': 'node1'})
    return p

@pytest.fixture
def mse_loss():
    """Une fonction de perte simple (Mean Squared Error)."""
    return lambda y_pred, y_true: np.mean((y_pred - y_true)**2)

@pytest.fixture
def optimizer_instance(simple_pipeline, mse_loss):
    """Une instance de base de PipelineOptimizer."""
    opt = PipelineOptimizer(simple_pipeline, mse_loss, max_time_pipeline=1.0)
    opt.add_param(IntParameter("node1", "a", 0, 10))
    opt.add_param(IntParameter("node2", "b", 0, 10))
    return opt

@pytest.fixture
def sample_data():
    """Données d'exemple pour l'évaluation."""
    X = [{'a': i} for i in range(3)]  # run_params ne sont pas utilisés dans ce pipeline simple
    y = [i * 2 + 1 for i in range(3)] # y_true pour node1.a = i
    return X, y

# --- Classes de Test ---

class TestOptimizerInitializationAndParams:
    def test_initialization(self, simple_pipeline, mse_loss):
        """Teste l'initialisation correcte de l'optimiseur."""
        opt = PipelineOptimizer(simple_pipeline, mse_loss, max_time_pipeline=0.5)
        assert opt.pipeline == simple_pipeline
        assert opt.loss == mse_loss
        assert opt.max_time_pipeline == 0.5
        assert opt.params_to_optimize == []

    def test_initialization_raises_errors(self, mse_loss):
        """Teste que les assertions à l'initialisation fonctionnent."""
        with pytest.raises(AssertionError):
            PipelineOptimizer("not_a_pipeline", mse_loss, 1.0)
        with pytest.raises(AssertionError):
            PipelineOptimizer(Pipeline("p"), "not_a_callable", 1.0)
        with pytest.raises(AssertionError):
            PipelineOptimizer(Pipeline("p"), mse_loss, -1.0)

    def test_add_param(self, optimizer_instance):
        """Teste l'ajout de paramètres."""
        assert len(optimizer_instance.params_to_optimize) == 2
        assert isinstance(optimizer_instance.params_to_optimize[0], IntParameter)

    def test_nb_params_possibilities(self, optimizer_instance):
        """Teste le calcul du nombre de combinaisons."""
        # IntParameter(0, 10) a une range_size de 10 (max-min)
        # Donc 10 * 10 = 100
        assert optimizer_instance.nb_params_possibilities() == 100

    def test_set_and_get_params(self, optimizer_instance):
        """Teste la définition et la récupération des valeurs de paramètres."""
        params_to_set = {"node1.a": 5, "node2.b": 8}
        optimizer_instance.set_params(params_to_set)
        
        retrieved_params = optimizer_instance.get_params_value()
        assert retrieved_params == params_to_set

    def test_set_param_invalid_key_raises_error(self, optimizer_instance):
        """Teste qu'une clé invalide lève une ValueError."""
        with pytest.raises(ValueError, match="Parameter node1.c not found"):
            optimizer_instance.set_params({"node1.c": 1})

    def test_update_pipeline_params(self, optimizer_instance):
        """Teste que le pipeline est bien mis à jour avec les paramètres de l'optimiseur."""
        params_to_set = {"node1.a": 3, "node2.b": 7}
        optimizer_instance.set_params(params_to_set)
        optimizer_instance.update_pipeline_params()
        
        pipeline_params = optimizer_instance.pipeline.get_fixed_params()
        assert pipeline_params == params_to_set


class TestOptimizerEvaluate:
    def test_evaluate(self, optimizer_instance, sample_data):
        """Teste la méthode d'évaluation."""
        X, y = sample_data
        
        # node1.a = 5, node2.b = 5
        # y_pred = 5 * 2 + 1 = 11
        # y_true = 0*2+1=1, 1*2+1=3, 2*2+1=5
        # loss = mean((11-1)^2, (11-3)^2, (11-5)^2) = mean(100, 64, 36) = 66.66
        optimizer_instance.set_params({"node1.a": 5, "node2.b": 5}) # node2.b n'est pas utilisé
        
        # Le pipeline simple ignore X, donc on peut passer des dicts vides
        X_eval = [{} for _ in range(3)]
        y_eval = [1, 3, 5]
        
        # Le pipeline exécute node1(a=5) -> 10, puis node2(b=10) -> 11
        # La sortie est toujours 11
        
        results, loss = optimizer_instance.evaluate(X_eval, y_eval)
        
        assert results == [11, 11, 11]
        assert np.isclose(loss, ((11-1)**2 + (11-3)**2 + (11-5)**2) / 3)

    def test_evaluate_with_timeout(self, optimizer_instance, sample_data):
        """Teste que l'évaluation retourne une perte infinie si le temps est dépassé."""
        X, y = sample_data
        optimizer_instance.max_time_pipeline = 0.01
        
        # Simuler un long temps d'exécution
        with patch.object(optimizer_instance.pipeline, 'run', return_value=("node2", {"node2": 0}, (0.02, {}))):
            _, loss = optimizer_instance.evaluate(X, y)
            assert loss == float("inf")


class TestOptimizerMethods:
    """
    Tests de haut niveau pour chaque méthode d'optimisation.
    L'objectif est de vérifier qu'elles s'exécutent et retournent le bon format,
    pas de valider la performance de l'algorithme lui-même.
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
        """Teste que chaque méthode d'optimisation s'exécute et retourne le bon format."""
        X, y = sample_data
        
        # BO ne supporte pas MultiChoiceParameter, donc on le retire pour ce test
        if method == "BO":
            optimizer_instance.params_to_optimize = [
                p for p in optimizer_instance.params_to_optimize if not isinstance(p, MultiChoiceParameter)
            ]

        best_params, loss_log = optimizer_instance.optimize(X, y, method=method, **kwargs)

        assert isinstance(best_params, dict)
        assert isinstance(loss_log, list)
        
        # Vérifier que les clés des paramètres sont correctes
        expected_keys = {f"{p.node_id}.{p.param_name}" for p in optimizer_instance.params_to_optimize}
        assert set(best_params.keys()) == expected_keys
        
        # Vérifier que le log de perte a la bonne longueur
        num_iterations = kwargs.get("iterations", kwargs.get("generations", kwargs.get("max_combinations")))
        assert len(loss_log) == num_iterations

    def test_optimize_dispatcher_invalid_method(self, optimizer_instance, sample_data):
        """Teste que le dispatcheur lève une erreur pour une méthode inconnue."""
        X, y = sample_data
        with pytest.raises(ValueError, match="Unknown optimization method: UNKNOWN"):
            optimizer_instance.optimize(X, y, method="UNKNOWN")


class TestOptimizerBOHelpers:
    """Tests pour les méthodes d'encodage/décodage de l'optimisation bayésienne."""
    
    @pytest.fixture
    def bo_param_defs(self):
        """Définitions de paramètres pour les tests BO."""
        return [
            ("node1.p_int", IntParameter("node1", "p_int", 0, 10)),
            ("node1.p_float", FloatParameter("node1", "p_float", 0.0, 1.0)),
            ("node1.p_choice", ChoiceParameter("node1", "p_choice", ["A", "B", "C"])),
            ("node1.p_bool", BoolParameter("node1", "p_bool"))
        ]

    def test_encode_decode_cycle(self, bo_param_defs):
        """Teste que l'encodage puis le décodage redonnent les valeurs initiales."""
        params_dict = {
            "node1.p_int": 5,
            "node1.p_float": 0.5,
            "node1.p_choice": "B",
            "node1.p_bool": True
        }

        encoded = PipelineOptimizer._encode(params_dict, bo_param_defs)
        
        # Vérifier l'encodage
        assert np.array_equal(encoded, np.array([5, 0.5, 1, 1]))

        decoded = PipelineOptimizer._decode(encoded, bo_param_defs)
        
        # Vérifier le décodage
        assert decoded["node1.p_int"] == 5
        assert decoded["node1.p_float"] == 0.5
        assert decoded["node1.p_choice"] == "B"
        assert decoded["node1.p_bool"] is True

    def test_decode_with_clipping(self, bo_param_defs):
        """Teste que le décodage gère bien le clipping pour les valeurs hors limites."""
        # Valeurs hors limites : int > 10, float < 0, choice_idx > 2
        encoded_out_of_bounds = np.array([15, -0.5, 5, 0.2])
        
        decoded = PipelineOptimizer._decode(encoded_out_of_bounds, bo_param_defs)
        
        assert decoded["node1.p_int"] == 10  # Clipped à max_value
        assert decoded["node1.p_float"] == 0.0 # Clipped à min_value
        assert decoded["node1.p_choice"] == "C" # Clipped à l'index max
        assert decoded["node1.p_bool"] is False # round(0.2) -> 0 -> False
