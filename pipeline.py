"""
This module provides classes for creating and managing customizable data processing pipelines.
It allows for the construction of complex workflows as Directed Acyclic Graphs (DAGs),
where each node represents a specific operation.
"""

import json
import importlib
import os, sys, time
from collections import deque
import numpy as np
import random as rd
from math import comb
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from scipy.stats import norm


def product(*iterables, random=False, max_combinations=0, optimize_memory=False):
    """
    Returns the cartesian product of input iterables, with an option for random sampling.

    Args:
        *iterables: Variable number of iterables to compute the product.
        random (bool): If True, returns a random sample from the product instead of all combinations.
        max_combinations (int): The maximum number of combinations to sample.
        optimize_memory (bool): Have an effect only if random is True and max_combinations > 0. 
            If True, optimizes memory usage by generating a random product
            without storing all combinations in memory. But  there is a risk of generating the same 
            value multiple times. Put to True only if max_combinations << len(all_combinations) or if there is no problem
            if the same value is repeated.

    Yields:
        Tuples representing the cartesian product of the input iterables.
    """
    len_index = [len(iterable) for iterable in iterables]
    max_combinations = max_combinations if max_combinations > 0 else np.prod(len_index)

    if random and optimize_memory:
        for i in range(max_combinations):
            yield tuple(it[rd.randrange(length)] for it, length in zip(iterables, len_index))
        return
    
    from itertools import product as it_product
    if random:
        rd_index = list(it_product(*[range(length) for length in len_index]))
        rd.shuffle(rd_index)
        for i in range(min(max_combinations, len(rd_index))):
            yield tuple(iterables[j][rd_index[i][j]] for j in range(len(iterables)))
        return
    
    prod = it_product(*iterables)
    for i in range(min(max_combinations, np.prod(len_index))):
        yield next(prod)


class Node:
    """
    Represents a single, executable step (a node) in a processing pipeline.

    A Node encapsulates a function to be executed, along with any fixed parameters
    that function requires. It can also cache its last output to avoid re-computation
    if the inputs haven't changed.

    Attributes:
        id (str): A unique identifier for the node.
        func (callable): The function to be executed by this node.
        fixed_params (dict): A dictionary of parameters that are fixed for this
            node's function and do not change during pipeline execution.
        output: Caches the result of the last execution.
        input_hash_last_exec: Caches the hash of the inputs from the last execution,
            used for memory optimization.
    """
    def __init__(self, id, func, fixed_params={}):
        """
        Initializes a Node.

        Args:
            id (str): The unique identifier for the node.
            func (callable): The function this node will execute.
            fixed_params (dict, optional): A dictionary of keyword arguments that will be
                passed to the function on every execution. Defaults to {}.
        """
        self.id = id
        self.func = func
        self.fixed_params = fixed_params
        self.output = None
        self.input_hash_last_exec = None

    def get_id(self):
        """Returns the node's unique identifier."""
        return self.id

    def clear_memory(self):
        """Clears the cached output and input hash."""
        del self.output
        del self.input_hash_last_exec
        self.output = None
        self.input_hash_last_exec = None

    def execute(self, inputs={}, memory=False):
        """
        Executes the node's function with the given inputs.

        Args:
            inputs (dict, optional): A dictionary of inputs for the node's function.
                These are typically the outputs of predecessor nodes. Defaults to {}.
            memory (bool, optional): If True, the node will cache its output and
                only re-execute if the inputs have changed since the last run.
                Defaults to False.

        Returns:
            The result of the function execution.

        Raises:
            Exception: Propagates any exception that occurs during the function's
                execution, after printing debug information.
        """
        if memory:
            to_hash = list(inputs.values())
            for i, e in enumerate(to_hash):
                if type(e) is np.ndarray:
                    to_hash[i] = e.tobytes()
            try:
                current_input_hash = hash(frozenset(to_hash))
            except TypeError:
                current_input_hash = None
        try:
            if not memory:
                return self.func(**self.fixed_params, **inputs)
            elif self.output is None or current_input_hash != self.input_hash_last_exec:
                self.output = self.func(**self.fixed_params, **inputs)
                self.input_hash_last_exec = current_input_hash
            return self.output
        except Exception as e:
            print(f"Erreur lors de l'exécution du noeud {self.id}: {e}")
            print("Paramètres fixes du noeud :", self.fixed_params)
            if inputs:
                print("Entrées du noeud :", inputs)
            raise

    def get_fixed_params(self):
        """Returns the dictionary of fixed parameters."""
        return self.fixed_params

    def set_fixed_params(self, fixed_params):
        """
        Sets the fixed parameters for the node.

        Args:
            fixed_params (dict): A dictionary of parameters to set.
        """
        if not isinstance(fixed_params, dict):
            raise ValueError("Les paramètres fixes doivent être un dictionnaire.")
        for key, value in fixed_params.items():
            if not isinstance(key, str):
                raise ValueError(f"La clé '{key}' n'est pas une chaîne de caractères.")
            self.set_fixed_param(key, value)

    def set_fixed_param(self, key, value):
        """
        Sets a single fixed parameter.

        Args:
            key (str): The name of the parameter.
            value: The value of the parameter.

        Raises:
            ValueError: If the key is not an existing fixed parameter.
        """
        if key not in self.fixed_params:
            raise ValueError(f"La clé '{key}' n'est pas un paramètre fixe du noeud '{self.id}'.")
        self.fixed_params[key] = value

    def is_fixed_param(self, key):
        """Checks if a parameter name is in the fixed parameters."""
        return key in self.fixed_params
        

class NodeIf(Node):
    """
    A conditional node that executes one of two sub-pipelines based on a condition.

    This node allows for branching logic within a pipeline. It evaluates a
    condition function and, based on the boolean result, runs either a
    'true_pipeline' or a 'false_pipeline'.

    Attributes:
        condition_func (callable): A function that returns a boolean value.
        true_pipeline (Pipeline): The pipeline to execute if the condition is True.
        false_pipeline (Pipeline): The pipeline to execute if the condition is False.
    """
    def __init__(self, id, condition_func, true_pipeline, false_pipeline, fixed_params={}):
        super().__init__(id, condition_func, fixed_params)
        self.true_pipeline = true_pipeline
        self.false_pipeline = false_pipeline
        self.skip_failed_images=False, 
        self.debug=False
    
    def set_run_params(self, skip_failed_images=False, debug=False):
        """
        Sets the run parameters for the sub-pipelines.

        Args:
            skip_failed_images (bool, optional): If True, execution of iterative nodes
                in sub-pipelines will continue even if one iteration fails.
                Defaults to False.
            debug (bool, optional): If True, enables debug printing for sub-pipelines.
                Defaults to False.
        """
        self.skip_failed_images = skip_failed_images
        self.debug = debug

    def execute(self, inputs={}, memory=False):
        """
        Evaluates the condition and executes the corresponding sub-pipeline.

        Inputs for the condition function must be prefixed with "condition_func:"
        in the predecessor mapping. The remaining inputs are passed to the
        chosen sub-pipeline as its `run_params`.

        Args:
            inputs (dict, optional): A dictionary of inputs. Defaults to {}.
            memory (bool, optional): If True, enables caching within the
                sub-pipelines. Defaults to False.

        Returns:
            The output of the final node of the executed sub-pipeline.
        """
        condition_inputs = {}
        for k in inputs:
            if k.startswith("condition_func:"):
                condition_inputs[k[15:]] = inputs[k]
        for k in condition_inputs:
            del inputs["condition_func:"+k]
        if self.func(**self.fixed_params, **condition_inputs):
            id, hist, _ = self.true_pipeline.run(run_params=inputs, optimize_memory= not memory, skip_failed_images=self.skip_failed_images, debug=self.debug)
        else:
            id, hist, _ = self.false_pipeline.run(run_params=inputs, optimize_memory= not memory, skip_failed_images=self.skip_failed_images, debug=self.debug)
        if memory:
            self.output = id, hist
        return hist[id]

    def get_fixed_params(self):
        """
        Gets the fixed parameters of the NodeIf and its sub-pipelines.

        Returns:
            dict: A dictionary containing the node's own fixed parameters and the
                  parameters of the true and false pipelines under the keys
                  "true_pipeline" and "false_pipeline".
        """
        # Retourne les paramètres fixes du noeud IF : donc fixe_params + ceux des pipelines
        true_fixed_params = self.true_pipeline.get_fixed_params()
        false_fixed_params = self.false_pipeline.get_fixed_params()
        return {**self.fixed_params, "true_pipeline": true_fixed_params, "false_pipeline": false_fixed_params}
    
    def set_fixed_params(self, fixed_params):
        """
        Sets the fixed parameters for the NodeIf and its sub-pipelines.
        It expects a dictionary that may contain "true_pipeline" and "false_pipeline" keys.
        """
        for key, value in fixed_params.items():
            if key == "true_pipeline":
                self.true_pipeline.set_fixed_params(value)
            elif key == "false_pipeline":
                self.false_pipeline.set_fixed_params(value)
            else:
                self.fixed_params[key] = value
    
    def set_fixed_param(self, key, value):
        """
        Sets a single fixed parameter on the NodeIf or its sub-pipelines.
        """
        if key == "true_pipeline":
            self.true_pipeline.set_fixed_param(value)
        elif key == "false_pipeline":
            self.false_pipeline.set_fixed_param(value)
        else:
            if key in self.fixed_params:
                raise ValueError(f"La clé '{key}' n'est pas un paramètre fixe du noeud '{self.id}'.")
            self.fixed_params[key] = value


class Pipeline:
    """
    Manages and executes a workflow of interconnected Nodes as a Directed Acyclic Graph (DAG).

    A Pipeline holds a collection of nodes and the dependencies between them.
    It determines the correct execution order, passes outputs from one node
    to the inputs of another, and provides functionality for serialization
    and visualization.

    Attributes:
        name (str): The name of the pipeline.
        description (str): A brief description of the pipeline's purpose.
        nodes (dict): A dictionary mapping node IDs to Node objects.
        node_dependencies (dict): A dictionary mapping a node ID to its
            predecessors. The format is:
            { 'target_node_id': {'target_input_name': 'source_node_id'} }
        timer (dict): Stores the execution time for each node after a run.
    """
    def __init__(self, name, description=""):
        self.name = name
        self.description = description
        self.nodes = {}  # id_noeud -> Node
        self.node_dependencies = {} # id_noeud -> {input_name: source_node_id}
        self.timer = {}

    def add_node(self, node, predecessors={}):
        """
        Adds a node or a sub-pipeline to the pipeline.

        Args:
            node (Node or Pipeline): The node or sub-pipeline to add.
            predecessors (dict, optional): A dictionary defining the inputs for this node.
                The keys are the parameter names of the node's function, and the
                values are the IDs of the nodes that provide the output.
                - For standard inputs: {'input_param': 'source_node_id'}
                - For iterative inputs (loops): {'[input_param]': 'source_node_id'}
                  The source node must produce a list. The current node will be
                  executed for each item in the list.
                - For NodeIf condition inputs: {'condition_func:param': 'source_node_id'}
                Defaults to {}.

        Raises:
            ValueError: If a node with the same ID already exists.
        """
        assert not node.get_id().startswith("run_params:"), ValueError("L'ID d'un noeud ne doit pas commencer par 'run_params:'")
        if isinstance(node, Node):
            node_id = node.get_id()
            if node_id in self.nodes:
                raise ValueError(f"Un nœud avec l'id '{node_id}' existe déjà.")
            self.nodes[node_id] = node
            self.node_dependencies[node_id] = predecessors

        elif isinstance(node, Pipeline):
            if "["+node.name+"]" in self.nodes:
                raise ValueError(f"Une pipeline avec le nom {node.name}' existe déjà.")
            self.nodes["["+node.name+"]"] = node
            self.node_dependencies["["+node.name+"]"] = predecessors

    def get_node(self, node_id):
        """Gets a node by its ID."""
        return self.nodes.get(node_id)

    def set_fixed_params(self, params):
        """Sets fixed parameters for multiple nodes in the pipeline."""
        for node_id, params in params.items():
            if node_id not in self.nodes:
                raise ValueError(f"Le noeud d'id '{node_id}' n'existe pas dans la pipeline.")
            for key, value in params.items():
                self.nodes[node_id].set_fixed_param(key, value)
    
    def get_fixed_params(self):
        """Gets the fixed parameters from all nodes in the pipeline."""
        return {node_id: node.get_fixed_params() for node_id, node in self.nodes.items()}
    
    def _get_graph_representation(self):
        """
        Construit une représentation du graphe avec les degrés entrants et les listes de successeurs.
        """
        in_degree = {node_id: 0 for node_id in self.nodes}
        successors = {node_id: [] for node_id in self.nodes}

        for node_id, deps in self.node_dependencies.items():
            for _, source_node_id in deps.items():
                if source_node_id.startswith("run_params:"):
                    continue
                elif source_node_id not in self.nodes:
                     raise ValueError(f"Le noeud source '{source_node_id}' pour '{node_id}' n'existe pas dans le pipeline.")
                elif node_id not in self.nodes: # Devrait être impossible si add_node est bien utilisé
                     raise ValueError(f"Le noeud cible '{node_id}' n'existe pas dans le pipeline.")
                else:
                    successors[source_node_id].append(node_id)
                    in_degree[node_id] += 1
        return in_degree, successors

    def static_order(self):
        """
        Calculates the topological order of nodes for execution.

        This method ensures that nodes are executed only after their dependencies
        have been met.

        Returns:
            list: A list of node IDs in a valid topological order.

        Raises:
            ValueError: If a cycle is detected in the graph.
        """
        in_degree, successors = self._get_graph_representation()
        
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        topological_order = []

        while queue:
            u = queue.popleft()
            topological_order.append(u)

            for v in successors[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if len(topological_order) != len(self.nodes):
            raise ValueError("Le graphe contient un cycle, le tri topologique est impossible.")
        return topological_order

    def run(self, run_params={}, optimize_memory=False, skip_failed_images=False, debug=False):
        """
        Executes the entire pipeline from start to finish.

        Args:
            run_params (dict, optional): Initial parameters for the first node(s)
                in the pipeline. Defaults to {}.
            optimize_memory (bool, optional): If True, outputs of nodes that are no
                longer needed by subsequent nodes will be deleted to save memory.
                Defaults to False.
            skip_failed_images (bool, optional): In an iterative node, if True,
                execution will continue even if one iteration fails. Defaults to False.
            debug (bool, optional): If True, prints the execution status of each node.
                Defaults to False.

        Returns:
            tuple: A tuple containing:
                - str: The ID of the last executed node.
                - dict: A dictionary of all node outputs.
                - tuple: A tuple with the total execution time and a dictionary
                         of individual node execution times.
        """
        node_outputs = {}
        self.timer = {}
        try:
            ordered_nodes = self.static_order()
        except ValueError as e:
            raise ValueError(f"Erreur de préparation de la pipeline: {e}")

        for i, node_id in enumerate(ordered_nodes):
            start_time = time.time()
            print(f"Exécution du noeud: {node_id}") if debug else None
            if node_id not in self.nodes:
                raise ValueError(f"Le noeud d'id: '{node_id}' a été spécifié comme dépendance mais n'a pas été ajouté au pipeline.")
            if isinstance(self.nodes[node_id], NodeIf):
                self.nodes[node_id].set_run_params(skip_failed_images, debug)

            node = self.nodes[node_id]
            inputs = {}
            loop_inputs = {}
            multiple_inputs = {}
            len_loop = float("inf")
            # node_dependencies contient les prédécesseurs de node_id
            for input_param_name, source_node_id in self.node_dependencies.get(node_id, {}).items():
                if source_node_id.startswith("run_params:"):
                    inputs[input_param_name] = run_params[source_node_id.split(":", 1)[1]]
                elif input_param_name[0]+input_param_name[-1] == "[]":
                    loop_inputs[input_param_name[1:-1]] = node_outputs[source_node_id]
                    len_loop = min(len_loop, len(node_outputs[source_node_id]))
                elif input_param_name[0]+input_param_name[-1] == "{}":
                    multiple_inputs[input_param_name[1:-1]] = node_outputs[source_node_id]
                else:
                    inputs[input_param_name] = node_outputs[source_node_id]
            
            if len_loop == float("inf") and multiple_inputs == {}:
                node_outputs[node_id] = node.execute(inputs, memory=not optimize_memory) if node_id[0]+node_id[-1] != "[]" \
                                        else node.run(inputs, optimize_memory, skip_failed_images, debug)
            elif multiple_inputs == {}:
                node_outputs[node_id] = []
                for i in range(len_loop):
                    try:
                        print(f"{node_id} en exécution itérations {i+1}/{len_loop}", end="\r") if debug else None
                        node_outputs[node_id].append(node.execute({**inputs, **{k: v[i] for k, v in loop_inputs.items()}}, memory=False) if node_id[0]+node_id[-1] != "[]" \
                                        else node.run({**inputs, **{k: v[i] for k, v in loop_inputs.items()}}, optimize_memory, skip_failed_images, debug))
                    except Exception as e:
                        if skip_failed_images:
                            print(f"Erreur dans le noeud {node_id} à l'itération {i+1}/{len_loop}: {e}") if debug else None
                            continue
                        raise e
                print() if debug else None
            elif len_loop == float("inf"):
                node_outputs[node_id] = []
                for p in product(*multiple_inputs.values()):
                    try:
                        print(f"{node_id} en exécution sur {zip(multiple_inputs.keys(),p)}", end="\r") if debug else None
                        node_outputs[node_id].append(node.execute({**inputs, **{k: v for k, v in zip(multiple_inputs.keys(), p)}}, memory=False))
                        node_outputs[node_id].append(node.execute({**inputs, **{k: v for k, v in zip(multiple_inputs.keys(), p)}}, memory=False) if node_id[0]+node_id[-1] != "[]" \
                                        else node.run({**inputs, **{k: v for k, v in zip(multiple_inputs.keys(), p)}}, optimize_memory, skip_failed_images, debug))
                    except Exception as e:
                        if skip_failed_images:
                            print(f"Erreur dans le noeud {node_id} avec les paramètres {zip(multiple_inputs.keys(),p)}: {e}") if debug else None
                            continue
                        raise e
                    print() if debug else None
            else:
                raise NotImplementedError("La combinaison de boucles et d'entrées multiples n'est pas implémentée.")

            # on efface la mémoire des noeuds qu'on n'utilise plus
            if optimize_memory:
                for dep_id in self.node_dependencies.get(node_id, {}).values():
                    still_used = False
                    for not_executed_node_id in ordered_nodes[i+1:]:
                        if dep_id in self.node_dependencies.get(not_executed_node_id, {}).values():
                            still_used = True
                            break
                    if not still_used and not dep_id.startswith("run_params:"):
                        del node_outputs[dep_id]

            last_node_id = node_id
            self.timer[node_id] = time.time() - start_time
        return last_node_id, node_outputs, (sum(self.timer.values()), self.timer)

    def to_dot(self, filepath=None, generate_png=False, png_filepath=None, _prefix=""):
        """
        Generates a DOT language representation of the pipeline graph.

        This can be used with Graphviz to visualize the pipeline structure.

        Args:
            filepath (str, optional): The path to save the .dot file. If None,
                the DOT string is returned instead. Defaults to None.
            generate_png (bool, optional): If True and `filepath` is provided,
                generates a PNG image from the DOT file using the `dot` command.
                Defaults to False.
            png_filepath (str, optional): The path for the output PNG file. If None,
                it's derived from the `filepath`. Defaults to None.
        """
        def escape_id(nid): return f"{_prefix}{nid}"

        dot_lines = []
        is_top = filepath is not None
        if is_top:
            dot_lines.append("digraph Pipeline {")
            dot_lines.append('  rankdir=TB;')  # vertical layout
            dot_lines.append('  node [fontsize=12 fontname="Helvetica"];')

        for node_id, node in self.nodes.items():
            full_id = escape_id(node_id)

            if isinstance(node, NodeIf):
                if node.func.__module__ == "__main__":
                    func_label = node.func.__name__
                elif node.func.__name__ == "<lambda>":
                    func_label = "lambda"
                else:
                    func_label = f"{node.func.__module__}.{node.func.__name__}"
                # IF block as dashed cluster without label
                dot_lines.append(f'  subgraph cluster_{full_id} {{')
                dot_lines.append('    style=dashed;')

                # IF node in diamond with small function text
                dot_lines.append(f'    "{full_id}" [shape=diamond, label=< <B>{node_id}</B><BR/><FONT POINT-SIZE="10">{func_label}</FONT> >];')

                # inline true_pipeline + false_pipeline inside the IF cluster
                dot_lines.append(node.true_pipeline.to_dot(None, _prefix=full_id + "_T_"))
                dot_lines.append(node.false_pipeline.to_dot(None, _prefix=full_id + "_F_"))

                # Draw edges from IF block to entry nodes of true/false pipelines
                true_first = node.true_pipeline.static_order()[0]
                false_first = node.false_pipeline.static_order()[0]
                true_last = node.true_pipeline.static_order()[-1]
                false_last = node.false_pipeline.static_order()[-1]

                dot_lines.append(f'    "{full_id}" -> "{full_id}_T_{true_first}" [label="True", tailport=s];')
                dot_lines.append(f'    "{full_id}" -> "{full_id}_F_{false_first}" [label="False", tailport=s];')
                
                dot_lines.append(f'    "{full_id}_output" [shape=diamond, label=< <FONT POINT-SIZE="10"> If Output</FONT> >];')
                dot_lines.append(f'    "{full_id}_T_{true_last}" -> "{full_id}_output" [tailport=s];')
                dot_lines.append(f'    "{full_id}_F_{false_last}" -> "{full_id}_output" [tailport=s];')

                dot_lines.append('  }')  # close cluster

            elif isinstance(node, Pipeline):
                dot_lines.append(f'  subgraph cluster_{full_id} {{')
                dot_lines.append(f'    label="SubPipeline: {node.name}"; style=filled; color=lightgrey;')
                dot_lines.append(node.to_dot(None, _prefix=full_id + "_"))
                dot_lines.append('  }')
            else:
                func_module = node.func.__module__
                func_name = node.func.__name__
                func_label = f"{func_module}.{func_name}" if func_module != '__main__' else func_name
                dot_lines.append(f'  "{full_id}" [shape=box, label=< <B>{node_id}</B><BR/><FONT POINT-SIZE="10">{func_label}</FONT> >];')

        # Connections
        for to_id, deps in self.node_dependencies.items():
            for input_name, from_id in deps.items():
                if from_id.startswith("run_params:"):
                    continue
                from_label = escape_id(from_id)
                to_label = escape_id(to_id)
                if isinstance(self.nodes[to_id], NodeIf) and input_name.startswith("condition_func:"):
                    dot_lines.append(f'  "{from_label}" -> "{to_label}" [label="{input_name[15:]}", tailport=s, headport=w, arrowhead=normal];')
                elif isinstance(self.nodes[from_id], NodeIf):
                    dot_lines.append(f'  "{from_label}_output" -> "{to_label}" [label="{input_name}"];')
                else:
                    dot_lines.append(f'  "{from_label}" -> "{to_label}" [label="{input_name}"];')

        if is_top:
            dot_lines.append("}")
            dot_str = "\n".join(dot_lines)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(dot_str)

            if generate_png:
                if png_filepath is None:
                    png_filepath = os.path.splitext(filepath)[0] + ".png"
                os.system(f'dot -Tpng -Gdpi=200 "{filepath}" -o "{png_filepath}"')
        else:
            return "\n".join(dot_lines)

    def to_json(self, filepath):
        """
        Serializes the pipeline's structure to a JSON file.

        This saves the nodes, their parameters, and their connections, allowing
        the pipeline to be reconstructed later.

        Args:
            filepath (str): The path to save the JSON file.
        """
        def serialize_node(node):
            if isinstance(node, NodeIf):
                return {
                    "id": node.id,
                    "type": "NodeIf",
                    "condition_type": f"{node.func.__module__}.{node.func.__name__}",
                    "true_pipeline": serialize_pipeline(node.true_pipeline),
                    "false_pipeline": serialize_pipeline(node.false_pipeline),
                    "fixed_params": node.fixed_params
                }
            elif isinstance(node, Pipeline):
                return {
                    "id": node.name,
                    "type": "SubPipeline",
                    "pipeline": serialize_pipeline(node)
                }
            else:  # Node
                func_module = node.func.__module__
                func_name = node.func.__name__
                return {
                    "id": node.id,
                    "type": f"{func_module}.{func_name}" if func_module != "__main__" else func_name,
                    "fixed_params": node.fixed_params
                }

        def serialize_pipeline(pipe):
            return {
                "name": pipe.name,
                "description": pipe.description,
                "nodes": [serialize_node(pipe.nodes[nid]) for nid in pipe.static_order()],
                "edges": [
                    {"from_node": src, "to_node": dst, "to_input": param}
                    for dst, deps in pipe.node_dependencies.items()
                    for param, src in deps.items()
                ]
            }

        pipeline_json = serialize_pipeline(self)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(pipeline_json, f, indent=4)

    @staticmethod
    def _default_function_resolver(type_str):
        """
        Default function resolver for `from_json`.
        Resolves a function from a string like 'module.function'.
        """
        if "." in type_str:
            module_name, function_name = type_str.rsplit('.', 1)
        else:
            module_name, function_name = '__main__', type_str
        if function_name == '<lambda>':
            raise ImportError("Impossible derésoudre une fonction lambda depuis un fichier JSON.")
        if module_name == 'builtins':
            return getattr(importlib.import_module(module_name), function_name)
        if module_name not in sys.modules:
            module = importlib.import_module(module_name)
        else:
            module = sys.modules[module_name]
        if not hasattr(module, function_name):
            raise ImportError(f"Le module '{module_name}' n'a pas de fonction '{function_name}'")
        return getattr(module, function_name)

    @classmethod
    def from_json(cls, filepath, function_resolver=None):
        """
        Creates a Pipeline instance from a JSON definition file.

        Args:
            filepath (str): The path to the JSON file.
            function_resolver (callable, optional): A function that takes a
                type string (e.g., 'module.function') and returns the
                corresponding callable. If None, a default resolver is used.
                Defaults to None.

        Returns:
            Pipeline: The reconstructed Pipeline instance.
        """
        if function_resolver is None:
            resolver = cls._default_function_resolver
        else:
            resolver = function_resolver

        with open(filepath, 'r', encoding='utf-8') as f:
            pipeline_def = json.load(f)

        def build_pipeline(pipeline_data):
            pipeline_instance = cls(name=pipeline_data["name"], description=pipeline_data["description"])
            nodes_data = pipeline_data["nodes"]
            edges_data = pipeline_data["edges"]

            for node_data in nodes_data:
                node_id = node_data["id"]
                node_type = node_data["type"]
                fixed_params = node_data.get("fixed_params", {})

                predecessors = {
                    edge["to_input"]: edge["from_node"]
                    for edge in edges_data if edge["to_node"] == node_id
                }

                if node_type == "NodeIf":
                    condition_func = resolver(node_data["condition_type"])
                    true_pipeline = build_pipeline(node_data["true_pipeline"])
                    false_pipeline = build_pipeline(node_data["false_pipeline"])

                    node = NodeIf(
                        id=node_id,
                        condition_func=condition_func,
                        true_pipeline=true_pipeline,
                        false_pipeline=false_pipeline,
                        fixed_params=fixed_params
                    )
                elif node_type == "SubPipeline":
                    sub_pipeline = build_pipeline(node_data["pipeline"])
                    pipeline_instance.add_node(sub_pipeline, predecessors)
                    continue  # Ne pas réajouter
                else:
                    func = resolver(node_type)
                    node = Node(node_id, func, fixed_params)

                pipeline_instance.add_node(node, predecessors)

            return pipeline_instance

        return build_pipeline(pipeline_def)


class Parameter:
    """
    Base class for parameters used in pipeline nodes.

    This abstract class defines the interface for all parameter types.
    It should not be instantiated directly. Subclasses must implement
    methods for setting, getting, and generating random values, as well
    as describing their parametric space.

    Attributes:
        node_id (str): The ID of the node this parameter belongs to.
        param_name (str): The name of the parameter.
        value: The current value of the parameter.
    """
    def __init__(self, node_id, param_name):
        """Initializes the base parameter."""
        self.node_id = node_id
        self.param_name = param_name
        self.value = None
    
    def get_parametric_space(self):
        """
        Returns the parametric space of the parameter.
        This should be overridden in subclasses.
        """
        raise NotImplementedError("Cette méthode doit être implémentée dans les sous-classes.")

    def set_value(self, value):
        """
        Sets the value of the parameter.
        This should be overridden in subclasses.
        """
        self.value = value
    
    def get_value(self):
        """
        Returns the current value of the parameter.
        This should be overridden in subclasses.
        """
        return self.value

    def get_random_value(self, set_value=False):
        """
        Returns a random value for the parameter.
        This should be overridden in subclasses.
        """
        raise NotImplementedError("Cette méthode doit être implémentée dans les sous-classes.")


class IntParameter(Parameter):
    """
    Represents an integer parameter with a defined range.

    This parameter type is used for node inputs that must be an integer
    between a specified minimum and maximum value.

    Attributes:
        node_id (str): The ID of the node this parameter belongs to.
        param_name (str): The name of the parameter.
        min_value (int): The minimum allowed value for the parameter.
        max_value (int): The maximum allowed value for the parameter.
        value (int): The current value of the parameter.
    """
    MAXINT = 2**63 - 1
    MININT = -2**63

    def __init__(self, node_id, param_name, min_value, max_value):
        """
        Initializes an IntParameter.

        Args:
            node_id (str): The ID of the node this parameter belongs to.
            param_name (str): The name of the parameter.
            min_value (int): The minimum allowed value.
            max_value (int): The maximum allowed value.

        Raises:
            ValueError: If min_value or max_value are not integers, or if
                        min_value is greater than max_value.
        """
        if not isinstance(min_value, (int, np.integer)) or not isinstance(max_value, (int, np.integer)):
            raise ValueError("Les valeurs min_value et max_value doivent être des entiers.")
        if min_value > max_value:
            raise ValueError("min_value ne peut pas être supérieur à max_value.")
        super(IntParameter, self).__init__(node_id, param_name)
        self.min_value = max(min_value, self.MININT)
        self.max_value = min(max_value, self.MAXINT)
        self.get_random_value(True)
    
    def get_parametric_space(self):
        """
        Returns the description of the parameter's search space.

        Returns:
            dict: A dictionary describing the parameter type, dimension, and range size.
        """
        if self.min_value == self.MININT:
            range_size = float('inf')
        elif self.max_value == self.MAXINT:
            range_size = float('inf')
        else:
            range_size = self.max_value - self.min_value
        return {"type": int, "dim": 1, "range_size": range_size}
    
    def set_value(self, value):
        """
        Sets the value of the parameter, ensuring it is within the valid range.

        Args:
            value (int): The integer value to set.

        Raises:
            ValueError: If the value is not an integer or is outside the defined range.
        """
        if not isinstance(value, (int, np.integer)):
            raise ValueError(f"La valeur doit être un entier, mais 'value' est de type {type(value)}.")
        elif value < self.min_value or value > self.max_value:
            raise ValueError(f"La valeur doit être comprise entre {self.min_value} et {self.max_value}.")
        self.value = value

    def get_value(self):
        """Returns the current value of the parameter."""
        return self.value
    
    def get_random_value(self, set_value=False):
        """
        Generates a random integer within the defined range.

        Args:
            set_value (bool, optional): If True, sets the parameter's current value
                to the new random value. Defaults to False.

        Returns:
            int: A random integer.
        """
        r = rd.randint(self.min_value, self.max_value)
        if set_value:
            self.set_value(r)
        return r


class FloatParameter(Parameter):
    """
    Represents a floating-point parameter with a defined range.

    This parameter type is used for node inputs that must be a float
    between a specified minimum and maximum value.

    Attributes:
        node_id (str): The ID of the node this parameter belongs to.
        param_name (str): The name of the parameter.
        min_value (float): The minimum allowed value for the parameter.
        max_value (float): The maximum allowed value for the parameter.
        value (float): The current value of the parameter.
    """
    MAXFLOAT = 1.7976931348623157e+308
    MINFLOAT = -1.7976931348623157e+308

    def __init__(self, node_id, param_name, min_value, max_value):
        """
        Initializes a FloatParameter.

        Args:
            node_id (str): The ID of the node this parameter belongs to.
            param_name (str): The name of the parameter.
            min_value (float): The minimum allowed value.
            max_value (float): The maximum allowed value.

        Raises:
            ValueError: If min_value or max_value are not floats, or if
                        min_value is greater than max_value.
        """
        if not isinstance(min_value, (float, int, np.integer, np.floating)) or not isinstance(max_value, (float, int, np.integer, np.floating)):
            raise ValueError("Les valeurs min_value et max_value doivent être des entiers.")
        if min_value > max_value:
            raise ValueError("min_value ne peut pas être supérieur à max_value.")
        super(FloatParameter, self).__init__(node_id, param_name)
        self.min_value = max(float(min_value), self.MINFLOAT)
        self.max_value = min(float(max_value), self.MAXFLOAT)
        self.get_random_value(True)

    def get_parametric_space(self):
        """
        Returns the description of the parameter's search space.

        Returns:
            dict: A dictionary describing the parameter type and range size.
        """
        if self.min_value == self.MINFLOAT:
            range_size = float('inf')
        elif self.max_value == self.MAXFLOAT:
            range_size = float('inf')
        else:
            range_size = self.max_value - self.min_value
        return {"type": float, "range_size": range_size}

    def set_value(self, value):
        """
        Sets the value of the parameter, ensuring it is within the valid range.

        Args:
            value (float): The float value to set.

        Raises:
            ValueError: If the value is not a float or is outside the defined range.
        """
        if not isinstance(value, (float, int, np.integer, np.floating)):
            raise ValueError(f"La valeur doit être un float, mais 'value' est de type {type(value)}.")
        elif value < self.min_value or value > self.max_value:
            raise ValueError(f"La valeur doit être comprise entre {self.min_value} et {self.max_value}.")
        self.value = float(value)
    
    def get_value(self):
        """Returns the current value of the parameter."""
        return self.value

    def get_random_value(self, set_value=False):
        """
        Generates a random float within the defined range.

        Args:
            set_value (bool, optional): If True, sets the parameter's current value
                to the new random value. Defaults to False.

        Returns:
            float: A random float.
        """
        r = rd.uniform(self.min_value, self.max_value)
        if set_value:
            self.set_value(r)
        return r


class ChoiceParameter(Parameter):
    """
    Represents a parameter that must be chosen from a predefined list of options.

    This is used for categorical parameters where the input must be one of the
    given choices.

    Attributes:
        node_id (str): The ID of the node this parameter belongs to.
        param_name (str): The name of the parameter.
        choices (list): The list of valid options for this parameter.
        value: The currently selected choice.
    """
    def __init__(self, node_id, param_name, choices):
        """
        Initializes a ChoiceParameter.

        Args:
            node_id (str): The ID of the node this parameter belongs to.
            param_name (str): The name of the parameter.
            choices (list): A list of the possible values for the parameter.
        """
        super(ChoiceParameter, self).__init__(node_id, param_name)
        self.choices = choices
        self.get_random_value(True)
    
    def get_parametric_space(self):
        """
        Returns the description of the parameter's search space.

        Returns:
            dict: A dictionary describing the parameter type and the number of choices.
        """
        return {"type": None, "range_size": len(self.choices)}
    
    def set_value(self, value):
        """
        Sets the value of the parameter, ensuring it is a valid choice.

        Args:
            value: The value to set.

        Raises:
            ValueError: If the value is not in the list of allowed choices.
        """
        if value not in self.choices:
            raise ValueError(f"Value must be one of the following options: {self.choices}.")
        self.value = value

    def get_value(self):
        """Returns the current value of the parameter."""
        return self.value
    
    def get_random_value(self, set_value=False):
        """
        Generates a random value from the list of choices.

        Args:
            set_value (bool, optional): If True, sets the parameter's current value
                to the new random value. Defaults to False.

        Returns:
            A random value from the choices.
        """
        r = rd.choice(self.choices)
        if set_value:
            self.set_value(r)
        return r
    

class MultiChoiceParameter(Parameter):
    """
    Represents a parameter where multiple options can be chosen from a list.

    This allows for selecting a sub-list of items, with constraints on the
    minimum and maximum number of selections.

    Attributes:
        node_id (str): The ID of the node this parameter belongs to.
        param_name (str): The name of the parameter.
        choices (list): The list of all possible options.
        min_choices (int): The minimum number of items to select.
        max_choices (int): The maximum number of items to select.
        value (list): The currently selected list of choices.
    """
    def __init__(self, node_id, param_name, choices, min_choices=1, max_choices=None):
        """
        Initializes a MultiChoiceParameter.

        Args:
            node_id (str): The ID of the node this parameter belongs to.
            param_name (str): The name of the parameter.
            choices (list): A list of all possible values.
            min_choices (int, optional): The minimum number of choices to select.
                Defaults to 1.
            max_choices (int or None, optional): The maximum number of choices.
                - If None, the number of choices is fixed to `min_choices`.
                - If 0, there is no upper limit (up to all choices).
                - Otherwise, it's the specific upper limit.
                Defaults to None.

        Raises:
            ValueError: If constraints are invalid (e.g., min > max).
        """
        if max_choices is not None and max_choices != 0 and min_choices > max_choices:
            raise ValueError("min_choices cannot be greater than max_choices.")
        if min_choices < 1:
            raise ValueError("min_choices must be at least 1.")
        super(MultiChoiceParameter, self).__init__(node_id, param_name)
        self.choices = choices
        self.min_choices = min_choices
        if max_choices is None:
            self.max_choices = self.min_choices
        elif max_choices == 0:
            self.max_choices = len(choices)
        else:
            self.max_choices = max_choices

        n = len(self.choices)
        self._C =  [comb(n, k) for k in range(self.min_choices, self.max_choices + 1)]
        self.get_random_value(True)

    def get_parametric_space(self):
        """
        Returns the description of the parameter's search space.

        The range size is the total number of possible combinations.

        Returns:
            dict: A dictionary describing the parameter type and range size.
        """
        return {"type": None, "range_size": sum(self._C)}

    def set_value(self, value):
        """
        Sets the value of the parameter, ensuring it is a valid sub-list of choices.

        Args:
            value (list): The list of selected values.

        Raises:
            ValueError: If the value is not a list, violates size constraints,
                        or contains items not in the allowed choices.
        """
        if not isinstance(value, list):
            raise ValueError(f"Value must be a list, but received type {type(value)}.")
        if len(value) < self.min_choices or len(value) > self.max_choices:
            raise ValueError(f"The list of values must contain between {self.min_choices} and {self.max_choices} items.")
        for v in value:
            if v not in self.choices:
                raise ValueError(f"The value '{v}' is not in the allowed choices: {self.choices}.")
        self.value = value

    def get_value(self):
        """Returns the current list of selected values."""
        return self.value
    
    def get_random_value(self, set_value=False):
        """
        Generates a random sub-list of choices that respects the size constraints.

        Args:
            set_value (bool, optional): If True, sets the parameter's current value
                to the new random value. Defaults to False.

        Returns:
            list: A random list of choices.
        """
        k = rd.choices(range(self.min_choices, self.max_choices + 1), weights=self._C, k=1)[0]
        r = rd.sample(self.choices, k=k)
        if set_value:
            self.set_value(r)
        return r


class BoolParameter(Parameter):
    """
    Represents a boolean parameter (True or False).

    Attributes:
        node_id (str): The ID of the node this parameter belongs to.
        param_name (str): The name of the parameter.
        value (bool): The current value of the parameter.
    """
    def __init__(self, node_id, param_name):
        """
        Initializes a BoolParameter.

        Args:
            node_id (str): The ID of the node this parameter belongs to.
            param_name (str): The name of the parameter.
        """
        super(BoolParameter, self).__init__(node_id, param_name)
        self.get_random_value(True)

    def get_parametric_space(self):
        """
        Returns the description of the parameter's search space.

        Returns:
            dict: A dictionary describing the parameter type and range size (2).
        """
        return {"type": bool, "range_size": 2}

    def set_value(self, value):
        """
        Sets the value of the parameter.

        Args:
            value (bool): The boolean value to set.

        Raises:
            ValueError: If the value is not a boolean.
        """
        if not isinstance(value, bool):
            raise ValueError(f"Value must be a boolean, but received type {type(value)}.")
        self.value = value

    def get_value(self):
        """Returns the current value of the parameter."""
        return self.value

    def get_random_value(self, set_value=False):
        """
        Generates a random boolean value.

        Args:
            set_value (bool, optional): If True, sets the parameter's current value
                to the new random value. Defaults to False.

        Returns:
            bool: A random boolean (True or False).
        """
        r = rd.choice([True, False])
        if set_value:
            self.set_value(r)
        return r


class PipelineOptimizer:
    """
    Provides a framework for optimizing pipeline parameters using various algorithms.

    This class defines the interface for different optimization strategies like
    Grid Search, Bayesian Optimization, Genetic Algorithms, etc. The methods
    are placeholders and should be implemented in subclasses or by integrating
    with external optimization libraries.

    Attributes:
        pipeline (Pipeline): The pipeline instance to be optimized.
        params_to_optimize (list): A list of Parameter objects to be tuned.
    """
    def __init__(self, pipeline, loss_function, max_time_pipeline, X, y):
        """
        Initializes a PipelineOptimizer.

        Args:
            pipeline (Pipeline): The pipeline instance to be optimized.
            loss_function (callable): A function that takes the pipeline's output and the
                expected output, and returns a numerical loss value.
            max_time_pipeline (float): The maximum time allowed for a single pipeline run (in seconds).
            X (list): A list of dictionaries, where each dictionary represents the `run_params`
                for a pipeline execution during optimization.
            y (list): A list of expected outputs corresponding to each `run_params` in `X`.
        """
        assert isinstance(pipeline, Pipeline), "pipeline must be an instance of Pipeline"
        assert callable(loss_function), "loss_function must be a callable function"
        assert isinstance(X, list), "X must be a list"
        assert isinstance(y, list), "y must be a list"
        assert len(X) == len(y), "X and y must have the same length"
        assert len(X) > 0, "X must not be empty"
        assert all(isinstance(run_params, dict) for run_params in X), "All items in X must be dictionaries"
        assert isinstance(max_time_pipeline, (int, float)), "max_time_pipeline must be a number"
        assert max_time_pipeline > 0, "max_time_pipeline must be a positive number"

        self.pipeline = pipeline
        self.params_to_optimize = []
        self.max_time_pipeline = max_time_pipeline
        self.X = X
        self.y = y
        self.loss = loss_function

    def add_param(self, param):
        """Adds a parameter to the list of parameters that the optimizer will tune."""
        self.params_to_optimize.append(param)

    def nb_params_possibilities(self):
        """Returns the total number of possible combinations of parameter values."""
        return np.prod([p.get_parametric_space()["range_size"] for p in self.params_to_optimize])

    def set_params(self, values:dict):
        """
        Sets the values of the parameters to optimize.

        Args:
            values (dict): A dictionary where keys are parameter names (node_id.param_name)
                               and values are the values to set.
        """
        for param_str, value in values.items():
            node_id, param_name = param_str.split('.')
            found = False
            for param_obj in self.params_to_optimize:
                if param_obj.node_id == node_id and param_obj.param_name == param_name:
                    param_obj.set_value(value)
                    found = True
                    break
            if not found:
                raise ValueError(f"Parameter {param_str} not found in the parameters to optimize.")

    def set_param(self, node_id, param_name, value):
        """
        Sets the value of a specific parameter in the pipeline.

        Args:
            node_id (str): The ID of the node containing the parameter.
            param_name (str): The name of the parameter to set.
            value: The value to set for the parameter.
        """
        for param in self.params_to_optimize:
            if param.node_id == node_id and param.param_name == param_name:
                param.set_value(value)
                return
        raise ValueError(f"Parameter {node_id}.{param_name} not found in the parameters to optimize.")

    def update_pipeline_params(self):
        """Updates the pipeline with the current parameter values."""
        params = {}
        for param in self.params_to_optimize:
            if not param.node_id in params:
                params[param.node_id] = {}
            params[param.node_id][param.param_name] = param.get_value()
        self.pipeline.set_fixed_params(params)

    def get_params_value(self):
        """
        Returns the current values of the parameters to optimize.

        Returns:
            dict: A dictionary where keys are parameter names (node_id.param_name)
                  and values are their current values.
        """
        values = {}
        for param in self.params_to_optimize:
            values[f"{param.node_id}.{param.param_name}"] = param.get_value()
        return values

    def evaluate(self):
        """
        Evaluates the current parameters of the pipeline.

        This method runs the pipeline with the current parameter values on the `X` and `y` for each run,
        and returns the outputs and the average loss compared to the expected values.

        Returns:
            dict: The outputs of the pipeline after running it with the current parameters.
        """
        self.update_pipeline_params()
        
        # Run the pipeline and return the outputs
        results = []
        loss = 0.
        for i, run_param in enumerate(self.X):
            index, res, t = self.pipeline.run(run_param, optimize_memory=True)
            results.append(res[index])
            loss += self.loss(results[-1], self.y[i])
            if t[0] > self.max_time_pipeline:
                return results, float("inf")
        loss /= i+1
        return results, loss

    def optimize_ACO(self, iterations=100, ants=20, alpha=1.0, beta=1.0, evaporation_rate=0.3, param_sampling=20, verbose=False):
        """
        Ant Colony Optimization (ACO) with real use of beta for heuristic guidance.

        Args:
            iterations (int): Number of iterations.
            ants (int): Number of ants per iteration.
            alpha (float): Importance of pheromone.
            beta (float): Importance of heuristic (1 / estimated loss).
            evaporation_rate (float): Rate at which pheromones evaporate.
            param_sampling (int): Number of random values to sample for each parameter.

        Returns:
            best_params (dict): Best parameter configuration.
            loss_log (list): Best loss after each iteration.
        """
        # Generate candidate values
        param_names = [f"{p.node_id}.{p.param_name}" for p in self.params_to_optimize]
        domain_values = {name: set() for name in param_names}
        for p, n in zip(self.params_to_optimize, param_names):
            if isinstance(p, IntParameter) and (p.max_value - p.min_value < param_sampling-1):
                domain_values[n] = list(range(p.min_value, p.max_value + 1))
            elif isinstance(p, ChoiceParameter) and len(p.choices) < param_sampling:
                domain_values[n] = p.choices
            elif isinstance(p, BoolParameter):
                domain_values[n] = [True, False]
            else:
                while len(domain_values[n]) < param_sampling:
                    domain_values[n].add(p.get_random_value())
                domain_values[n] = list(domain_values[n])

        # Initialize pheromones and heuristic tables
        pheromones = {
            name: {val: 1.0 for val in domain_values[name]}
            for name in param_names
        }

        # Store previous heuristic (average inverse loss)
        heuristics = {
            name: {val: 1.0 for val in domain_values[name]}
            for name in param_names
        }

        best_params = None
        best_loss = float("inf")
        loss_log = []

        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}", end="\r") if verbose else None
            solutions = []
            losses = []

            for _ in range(ants):
                candidate = {}
                for name in param_names:
                    values = domain_values[name]
                    pher = np.array([pheromones[name][v] for v in values])
                    heur = np.array([heuristics[name][v] for v in values])
                    probs = (pher ** alpha) * (heur ** beta)
                    probs /= probs.sum()
                    selected = np.random.choice(values, p=probs)
                    candidate[name] = selected

                self.set_params(candidate)
                _, loss = self.evaluate()

                solutions.append(candidate)
                losses.append(loss)

                for name in param_names:
                    val = candidate[name]
                    heuristics[name][val] = max(1e-6, 1.0 / (1e-6 + loss))  # éviter division par zéro

                if loss < best_loss:
                    best_loss = loss
                    best_params = candidate.copy()

            # Evaporation
            for name in pheromones:
                for val in pheromones[name]:
                    pheromones[name][val] *= (1.0 - evaporation_rate)

            # Dépôt de phéromone par la meilleure fourmi
            best_idx = int(np.argmin(losses))
            for name, val in solutions[best_idx].items():
                pheromones[name][val] += 1.0 / (1.0 + losses[best_idx])

            loss_log.append(best_loss)

        self.set_params(best_params)
        self.update_pipeline_params()
        return best_params, loss_log

    def optimize_SA(self, iterations=1000, initial_temp=1.0, cooling_rate=0.95, verbose=False):
        """
        Optimizes the pipeline using Simulated Annealing (SA).

        Args:
            iterations (int): Number of iterations.
            initial_temp (float): Initial temperature.
            cooling_rate (float): Cooling factor.

        Returns:
            dict: Best parameters found.
            list: Loss log over iterations.
        """
        current_params = {
            f"{p.node_id}.{p.param_name}": p.get_random_value()
            for p in self.params_to_optimize
        }
        self.set_params(current_params)
        _, current_loss = self.evaluate()

        best_params = current_params.copy()
        best_loss = current_loss

        temperature = initial_temp
        loss_log = [best_loss]

        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}", end="\r") if verbose else None
            candidate = current_params.copy()
            # Mutate a random parameter
            name = rd.choice(list(candidate.keys()))
            param = next(p for p in self.params_to_optimize if f"{p.node_id}.{p.param_name}" == name)
            candidate[name] = param.get_random_value()

            self.set_params(candidate)
            _, candidate_loss = self.evaluate()

            delta = candidate_loss - current_loss
            if delta < 0 or np.exp(-delta / temperature) > rd.random():
                current_params = candidate
                current_loss = candidate_loss
                if candidate_loss < best_loss:
                    best_loss = candidate_loss
                    best_params = candidate.copy()

            loss_log.append(best_loss)
            temperature *= cooling_rate

        self.set_params(best_params)
        self.update_pipeline_params()
        return best_params, loss_log

    def optimize_PSO(self, iterations=100, swarm_size=20, inertia=0.5, cognitive=1.5, social=1.5, verbose=False):
        """
        Optimizes the pipeline using Particle Swarm Optimization (PSO).

        Args:
            iterations (int): Number of iterations.
            swarm_size (int): Number of particles.
            inertia (float): Inertia weight.
            cognitive (float): Cognitive parameter.
            social (float): Social parameter.

        Returns:
            dict: Best parameters found.
            list: Loss log over iterations.
        """
        param_names = [f"{p.node_id}.{p.param_name}" for p in self.params_to_optimize]
        particles = []
        velocities = []
        personal_best = []
        personal_best_loss = []
        loss_log = []

        # Initialize swarm
        for _ in range(swarm_size):
            particle = {name: p.get_random_value() for name, p in zip(param_names, self.params_to_optimize)}
            velocity = {name: 0.0 for name in param_names}

            self.set_params(particle)
            _, loss = self.evaluate()

            particles.append(particle)
            velocities.append(velocity)
            personal_best.append(particle.copy())
            personal_best_loss.append(loss)

        best_loss = min(personal_best_loss)
        best_particle = personal_best[np.argmin(personal_best_loss)].copy()

        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}", end="\r") if verbose else None
            for i in range(swarm_size):
                new_particle = {}
                for name in param_names:
                    r1 = rd.random()
                    r2 = rd.random()
                    p_val = particles[i][name]
                    pb_val = personal_best[i][name]
                    gb_val = best_particle[name]

                    # Velocity update (heuristic): not strictly numeric, sample mutation
                    if p_val != pb_val:
                        velocities[i][name] = cognitive * r1
                    elif p_val != gb_val:
                        velocities[i][name] += social * r2
                    else:
                        velocities[i][name] *= inertia

                    if rd.random() < velocities[i][name]:
                        param = next(p for p in self.params_to_optimize if f"{p.node_id}.{p.param_name}" == name)
                        new_particle[name] = param.get_random_value()
                    else:
                        new_particle[name] = particles[i][name]

                self.set_params(new_particle)
                _, loss = self.evaluate()

                if loss < personal_best_loss[i]:
                    personal_best[i] = new_particle.copy()
                    personal_best_loss[i] = loss

                    if loss < best_loss:
                        best_loss = loss
                        best_particle = new_particle.copy()

                particles[i] = new_particle

            loss_log.append(best_loss)

        self.set_params(best_particle)
        self.update_pipeline_params()
        return best_particle, loss_log

    def optimize_GA(self, generations=50, population_size=20, mutation_rate=0.1, crossover_rate=0.7, verbose=False):
        """
        Optimizes the pipeline using Genetic Algorithm (GA).

        Args:
            generations (int): Number of generations.
            population_size (int): Number of individuals in the population.
            mutation_rate (float): Probability of mutation.
            crossover_rate (float): Probability of crossover.

        Returns:
            dict: Best parameters found.
            list: Loss log over generations.
        """
        param_names = [f"{p.node_id}.{p.param_name}" for p in self.params_to_optimize]
        loss_log = []

        def random_individual():
            return {name: p.get_random_value() for name, p in zip(param_names, self.params_to_optimize)}

        def crossover(parent1, parent2):
            return {
                name: parent1[name] if rd.random() < 0.5 else parent2[name]
                for name in param_names
            }

        def mutate(individual):
            for name in param_names:
                if rd.random() < mutation_rate:
                    param = next(p for p in self.params_to_optimize if f"{p.node_id}.{p.param_name}" == name)
                    individual[name] = param.get_random_value()
            return individual

        # Initial population
        population = [random_individual() for _ in range(population_size)]
        evaluated = []
        for ind in population:
            self.set_params(ind)
            _, loss = self.evaluate()
            evaluated.append((ind, loss))

        best_individual = min(evaluated, key=lambda x: x[1])[0].copy()
        best_loss = min(evaluated, key=lambda x: x[1])[1]

        for gen in range(generations):
            print(f"Generation {gen+1}/{generations}", end="\r") if verbose else None
            new_population = []
            evaluated.sort(key=lambda x: x[1])
            parents = [ind for ind, _ in evaluated[:population_size//2]]

            while len(new_population) < population_size:
                if rd.random() < crossover_rate:
                    p1, p2 = rd.sample(parents, 2)
                    child = crossover(p1, p2)
                    child = mutate(child)
                else:
                    child = mutate(rd.choice(parents).copy())
                new_population.append(child)

            evaluated = []
            for ind in new_population:
                self.set_params(ind)
                _, loss = self.evaluate()
                evaluated.append((ind, loss))

                if loss < best_loss:
                    best_loss = loss
                    best_individual = ind.copy()

            loss_log.append(best_loss)

        self.set_params(best_individual)
        self.update_pipeline_params()
        return best_individual, loss_log
    
    def optimize_grid_search(self, max_combinations=1000, param_sampling=None, verbose=False):
        """
        Exhaustively searches all possible parameter combinations (within a limited budget).

        Args:
            max_combinations (int): Maximum number of combinations to evaluate.
            param_sampling (int): Number of random values to sample for each parameter.

        Returns:
            dict: Best parameters found.
            list: Loss log over evaluated combinations.
        """
        if param_sampling is None:
            param_sampling = max(10, max_combinations // len(self.params_to_optimize))
        param_names = [f"{p.node_id}.{p.param_name}" for p in self.params_to_optimize]
        param_values = []
        for p in self.params_to_optimize:
            if isinstance(p, IntParameter) and (p.max_value - p.min_value < param_sampling - 1):
                param_values.append(list(range(p.min_value, p.max_value + 1)))
            elif isinstance(p, ChoiceParameter) and len(p.choices) < param_sampling:
                param_values.append(p.choices)
            elif isinstance(p, BoolParameter):
                param_values.append([True, False])
            else:
                values= set()
                while len(values) < param_sampling:
                    values.add(p.get_random_value())
                param_values.append(list(values))

        # On a réécrit la fonction product pour gérer les grands espaces de recherche sans tous les expliciter d'un coup
        if np.prod([len(v) for v in param_values])>>4 > max_combinations:
            combinations = product(*param_values, random=True, max_combinations=max_combinations, optimize_memory=True)
        else:
            combinations = product(*param_values, random=True, max_combinations=max_combinations)

        best_loss = float("inf")
        best_params = None
        loss_log = []

        for i, combo in enumerate(combinations):
            print(f"Iteration {i+1}/{max_combinations}", end="\r") if verbose else None
            params = dict(zip(param_names, combo))
            self.set_params(params)
            _, loss = self.evaluate()

            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()

            loss_log.append(best_loss)

        self.set_params(best_params)
        self.update_pipeline_params()
        return best_params, loss_log
   
    @staticmethod
    def _encode(params, param_defs):
        """ 
        Encodes a dictionary of parameters into a numpy array.
        This is used for optimization algorithms that require numerical input like optimize_BO.

        Args:
            params (dict): Dictionary of parameters where keys are parameter names
                           (node_id.param_name) and values are the parameter values.
            param_defs (list): List of tuples (name, Parameter) defining the parameters.

        Returns:
            numpy.ndarray: Encoded array of parameters.
        """
        encoded = []
        for name, p in param_defs:
            val = params[name]
            if isinstance(p, BoolParameter):
                encoded.append(int(val))
            elif isinstance(p, ChoiceParameter):
                encoded.append(p.choices.index(val))
            else:
                encoded.append(val)
        return np.array(encoded)
    
    @staticmethod
    def _decode(x, param_defs):
        """
        Decodes a numpy array into a dictionary of parameters.
        This is used for optimization algorithms that require numerical input like optimize_BO.

        Args:
            x (numpy.ndarray): Encoded array of parameters.
            param_defs (list): List of tuples (name, Parameter) defining the parameters.

        Returns:
            dict: Decoded dictionary of parameters.
        """
        params = {}
        for i, (name, p) in enumerate(param_defs):
            if isinstance(p, BoolParameter):
                params[name] = bool(round(x[i]))
            elif isinstance(p, ChoiceParameter):
                idx = int(round(np.clip(x[i], 0, len(p.choices)-1)))
                params[name] = p.choices[idx]
            elif isinstance(p, IntParameter):
                params[name] = int(round(np.clip(x[i], p.min_value, p.max_value)))
            elif isinstance(p, FloatParameter):
                params[name] = float(np.clip(x[i], p.min_value, p.max_value))
        return params

    def optimize_BO(self, iterations=50, init_points=5, noise_level=0, n_candidates=None, verbose=False):
        """
        Bayesian Optimization using Gaussian Process and Expected Improvement (EI),
        with input normalization and robust handling.

        Args:
            iterations (int): Number of optimization steps.
            init_points (int): Initial random samples before BO starts.
            noise_level (float): Noise level for the Gaussian Process. Lower it is more precise the model but can lead to overfitting.
            n_candidates (int): Number of candidates to evaluate at each step. Suggested to be 100 times the number of parameters.
                                if None, defaults to 100 * number of parameters.

        Returns:
            best_params (dict): Best parameter configuration.
            loss_log (list): Best loss per iteration.
        """
        from sklearn.exceptions import ConvergenceWarning
        import warnings
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        # Les paramètres optimisables en liste (on exclut MultiChoiceParameter pour simplifier l'implémentation)
        assert any(not isinstance(p, MultiChoiceParameter) for p in self.params_to_optimize)
        param_defs = [(f"{p.node_id}.{p.param_name}", p) for p in self.params_to_optimize]

        n_candidates = 100 * len(param_defs) if n_candidates is None else n_candidates

        # Échantillonnage initial aléatoire
        X_raw = []
        Y = []
        for _ in range(init_points):
            sample = {}
            for name, p in param_defs:
                sample[name] = p.get_random_value()
            self.set_params(sample)
            _, loss = self.evaluate()
            X_raw.append(self._encode(sample, param_defs))
            Y.append(loss)

        X_raw = np.array(X_raw)
        Y = np.array(Y)

        # Normalisation de X
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        # Modèle GP
        kernel = C(1.0) * Matern(nu=2.5)
        if noise_level > 0:
            kernel += WhiteKernel(noise_level=noise_level)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)

        best_idx = np.argmin(Y)
        best_x = X_raw[best_idx]
        best_loss = Y[best_idx]
        loss_log = [best_loss]

        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}", end="\r") if verbose else None
            gp.fit(X, Y)

            # Génère des candidats aléatoires
            candidates_raw = []
            for _ in range(n_candidates):
                cand = {name: p.get_random_value() for name, p in param_defs}
                candidates_raw.append(self._encode(cand, param_defs))
            candidates_raw = np.array(candidates_raw)
            candidates = scaler.transform(candidates_raw)

            # EI (Expected Improvement)
            mu, sigma = gp.predict(candidates, return_std=True)
            sigma = np.maximum(sigma, 1e-8)
            improvement = best_loss - mu
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

            x_next = candidates_raw[np.argmax(ei)]
            params = self._decode(x_next, param_defs)
            self.set_params(params)
            _, loss = self.evaluate()

            # Mise à jour
            X = np.vstack([X, scaler.transform([x_next])])
            Y = np.append(Y, loss)

            if loss < best_loss:
                best_loss = loss
                best_x = x_next
            loss_log.append(best_loss)

        self.set_params(self._decode(best_x, param_defs))
        self.update_pipeline_params()
        return self._decode(best_x, param_defs), loss_log
        
    def optimize(self, method, verbose=False, **kwargs):
        """
        Optimizes the pipeline using the specified method.

        Args:
            method (str): The optimization method to use (e.g., "grid_search", "BO", "ACO", "SA", "GA").
            **kwargs: Additional arguments for the chosen optimization method.

        Returns:
            tuple: A tuple containing:
                - dict: The best parameters found.
                - list: A log of the loss values during optimization.
        """
        if method == "grid_search":
            return self.optimize_grid_search(verbose=verbose,**kwargs)
        elif method == "BO":
            return self.optimize_BO(verbose=verbose, **kwargs)
        elif method == "ACO":
            return self.optimize_ACO(verbose=verbose, **kwargs)
        elif method == "SA":
            return self.optimize_SA(verbose=verbose, **kwargs)
        elif method == "GA":
            return self.optimize_GA(verbose=verbose, **kwargs)
        elif method == "PSO":
            return self.optimize_PSO(verbose=verbose, **kwargs)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
