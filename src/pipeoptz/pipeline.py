"""Defines the Pipeline class, which manages and executes a workflow of interconnected Nodes."""
from __future__ import annotations
import json
import importlib
import os
import sys
import time
from random import randrange, shuffle
from itertools import product as it_product
from collections import deque
from typing import Callable, Any, Union, Dict, List, \
                   Tuple, Deque, Iterator, Optional, Iterable
from .node import Node, NodeIf, NodeFor, NodeWhile



def _product(*iterables: Iterable[Any], random: bool = False, max_combinations: int = 0, \
             optimize_memory: bool = False) -> Iterator[Tuple[Any, ...]]:
    """
    Returns the cartesian product of input iterables, with an option for random sampling.

    Args:
        *iterables: Variable number of iterables to compute the product.
        random (bool): If True, returns a random sample from the product 
            instead of all combinations.
        max_combinations (int): The maximum number of combinations to sample.
        optimize_memory (bool): Have an effect only if random is True and max_combinations > 0. 
            If True, optimizes memory usage by generating a random product
            without storing all combinations in memory. But  there is a risk of generating the same 
            value multiple times. Put to True only if max_combinations << len(all_combinations) 
            or if there is no problem in the case of the same value is repeated.

    Yields:
        Tuples representing the cartesian product of the input iterables.
    """
    len_index = [len(iterable) for iterable in iterables]
    prod_len_index = 1
    for length in len_index:
        prod_len_index *= length
    max_combinations = max_combinations if max_combinations > 0 else prod_len_index

    if random and optimize_memory:
        for i in range(max_combinations):
            yield tuple(it[randrange(length)] for it, length in zip(iterables, len_index))
        return

    if random:
        rd_index = list(it_product(*[range(length) for length in len_index]))
        shuffle(rd_index)
        for i in range(min(max_combinations, len(rd_index))):
            yield tuple(iterables[j][rd_index[i][j]] for j in range(len(iterables)))
        return

    prod = it_product(*iterables)
    for i in range(min(max_combinations, prod_len_index)):
        try:
            yield next(prod)
        except StopIteration:
            return


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
            {'target_node_id': {'target_input_name': 'source_node_id'}}
        timer (dict): Stores the execution time for each node after a run.
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name: str = name
        self.description: str = description
        self.nodes: Dict[str, Union[Node, Pipeline]] = {}
        self.node_dependencies: Dict[str, Dict[str, str]] = {}
        self.timer: Dict[str, float] = {}

    def add_node(self, node: Union[Node, Pipeline], predecessors: Dict[str, str] = None) -> None:
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
        if predecessors is None:
            predecessors = {}
        assert not node.get_id().startswith("run_params:"), \
            "The ID of a node cannot start with 'run_params:'"
        if isinstance(node, Node):
            node_id = node.get_id()
            if node_id in self.nodes:
                raise ValueError(f"A node with id '{node_id}' already exists.")
            self.nodes[node_id] = node
            self.node_dependencies[node_id] = predecessors

        elif isinstance(node, Pipeline):
            if "["+node.name+"]" in self.nodes:
                raise ValueError(f"A pipeline with the name '{node.name}' already exists.")
            self.nodes["["+node.name+"]"] = node
            self.node_dependencies["["+node.name+"]"] = predecessors

    def get_node(self, node_id: str) -> Union[Node, Pipeline]:
        """Gets a node by its ID."""
        if node_id not in self.nodes:
            raise ValueError("The node does not exist in the pipeline.")
        return self.nodes[node_id]

    def get_nodes(self) -> Dict[str, Union[Node, Pipeline]]:
        """Gets all nodes in the pipeline."""
        return self.nodes

    def set_fixed_params(self, params: Dict[str, Any]) -> None:
        """Sets fixed parameters for multiple nodes in the pipeline."""
        for node_id, value in params.items():
            node_id, param = node_id.split('.', 1)
            if node_id not in self.nodes:
                raise ValueError(f"The node with id '{node_id}' does not exist in the pipeline.")
            self.nodes[node_id].set_fixed_param(param, value)

    def get_fixed_params(self) -> Dict[str, Any]:
        """Gets the fixed parameters from all nodes in the pipeline."""
        params = {}
        for node_id, node in self.nodes.items():
            for param, value in node.get_fixed_params().items():
                params[f"{node_id}.{param}"] = value
        return params

    def _get_graph_representation(self) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
        """
        Build a graph representation with in-degrees and successor lists.
        """
        in_degree: Dict[str, int] = {node_id: 0 for node_id in self.nodes}
        successors: Dict[str, List[str]] = {node_id: [] for node_id in self.nodes}

        for node_id, deps in self.node_dependencies.items():
            for _, source_node_id in deps.items():
                if source_node_id.startswith("run_params:"):
                    continue
                if source_node_id not in self.nodes:
                    if ":" in source_node_id[-3:] and \
                       source_node_id.rsplit(":", 1)[0] in self.nodes:
                        continue
                    raise ValueError(f"The source node '{source_node_id}' for \
                                     '{node_id}' does not exist in the pipeline")
                if node_id not in self.nodes:
                    raise ValueError(f"The target node '{node_id}' does not exist in the pipeline.")
                successors[source_node_id].append(node_id)
                in_degree[node_id] += 1
        return in_degree, successors

    def static_order(self) -> List[str]:
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
        queue: Deque[str] = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        topological_order: List[str] = []

        while queue:
            u = queue.popleft()
            topological_order.append(u)
            for v in successors[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if len(topological_order) != len(self.nodes):
            raise ValueError("The graph contains a cycle, topological sort is impossible.")
        return topological_order

    def run(self, run_params: Union[Dict[str, Any]] = None, \
            optimize_memory: bool = False, skip_failed_loop: bool = False, \
            debug: bool = False) -> Tuple[str, Dict[str, Any], Tuple[float, Dict[str, float]]]:
        """
        Executes the entire pipeline from start to finish.

        Args:
            run_params (dict, optional): Initial parameters for the first node(s)
                in the pipeline. Defaults to {}.
            optimize_memory (bool, optional): If True, outputs of nodes that are no
                longer needed by subsequent nodes will be deleted to save memory.
                Defaults to False.
            skip_failed_loop (bool, optional): In an iterative node, if True,
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
        if run_params is None:
            run_params = {}
        node_outputs: Dict[str, Any] = {}
        self.timer = {}
        try:
            ordered_nodes = self.static_order()
        except ValueError as e:
            raise ValueError(f'Error preparing the pipeline: {e}') from e

        for i, node_id in enumerate(ordered_nodes):
            start_time = time.time()
            if debug:
                print(f"Executing node: {node_id}")
            if node_id not in self.nodes:
                raise ValueError(f"The node with id: '{node_id}' was specified as \
                                 a dependency but has not been added to the pipeline.")

            if isinstance(self.nodes[node_id], (NodeIf, NodeFor, NodeWhile)):
                self.nodes[node_id].set_run_params(skip_failed_loop, debug)
            node = self.nodes[node_id]
            inputs: Dict[str, Any] = {}
            loop_inputs: Dict[str, Any] = {}
            multiple_inputs: Dict[str, Any] = {}
            len_loop: Union[int, float] = float("inf")
            # node_dependencies contains the predecessors of node_id
            for input_param_name, source_node_id in self.node_dependencies.get(node_id, {}).items():
                if source_node_id.startswith("run_params:"):
                    inputs[input_param_name] = run_params[source_node_id.split(":", 1)[1]]
                elif input_param_name[0]+input_param_name[-1] == "[]":
                    loop_inputs[input_param_name[1:-1]] = node_outputs[source_node_id]
                    len_loop = min(len_loop, len(node_outputs[source_node_id]))
                elif input_param_name[0]+input_param_name[-1] == "{}":
                    multiple_inputs[input_param_name[1:-1]] = node_outputs[source_node_id]
                elif ":" in source_node_id[-3:] and \
                     not source_node_id.startswith("condition_func:"):
                    source_node_id, key = source_node_id.rsplit(":", 1)
                    key = int(key) if key.isdigit() else key
                    inputs[input_param_name.split(":", 1)[0]] = node_outputs[source_node_id][key]
                else:
                    inputs[input_param_name] = node_outputs[source_node_id]

            if len_loop == float("inf") and not multiple_inputs:
                node_outputs[node_id] = node.execute(inputs)
            elif not multiple_inputs:
                node_outputs[node_id] = []
                for i in range(len_loop):
                    try:
                        if debug:
                            print(f"Executing node: {node_id} iteration {i+1}/{len_loop}", end="\r")
                        run_params = {**inputs, **{k: v[i] for k, v in loop_inputs.items()}}
                        if node_id[0]+node_id[-1] != "[]":
                            node_outputs[node_id].append(node.execute(run_params))
                        else:
                            node_outputs[node_id].append(\
                                node.run(run_params, optimize_memory, skip_failed_loop, debug))
                    except Exception as e:
                        if skip_failed_loop:
                            print(f"Error in node {node_id} at iteration {i+1}/{len_loop}: {e}")
                            continue
                        raise e
                if debug:
                    print()
            elif len_loop == float("inf"):
                node_outputs[node_id] = []
                for p in _product(*multiple_inputs.values()):
                    try:
                        if debug:
                            print(f"Executing node: {node_id} with parameters \
                                  {dict(zip(multiple_inputs.keys(), p))}", end="\r")
                        run_params = {**inputs, **dict(zip(multiple_inputs.keys(), p))}
                        if node_id[0]+node_id[-1] != "[]":
                            node_outputs[node_id].append(node.execute(run_params))
                        else:
                            node_outputs[node_id].append(\
                                node.run(run_params, optimize_memory, skip_failed_loop, debug))
                    except Exception as e:
                        if skip_failed_loop:
                            print(f"Error in node {node_id} with parameters \
                                  {dict(zip(multiple_inputs.keys(),p))}: {e}")
                            continue
                        raise e
                    if debug:
                        print()
            else:
                raise NotImplementedError("Combining loops and multiple inputs is not implemented.")

            # If optimize_memory is True, delete outputs of nodes that are no longer needed
            # We check if the output of a predecessor node (dep_id) is still needed by any
            # of the subsequent nodes in the topological order. If not, we delete it.
            if optimize_memory:
                for dep_id in self.node_dependencies.get(node_id, {}).values():
                    still_used = False
                    for not_executed_node_id in ordered_nodes[i+1:]:
                        if dep_id in self.node_dependencies.get(not_executed_node_id, {}).values():
                            still_used = True
                            break
                    if not still_used and not dep_id.startswith("run_params:"):
                        del node_outputs[dep_id]
                        node.clear_memory()

            last_node_id = node_id
            self.timer[node_id] = time.time() - start_time

        if optimize_memory:
            for node_id in node_outputs:
                self.nodes[node_id].clear_memory()
        return last_node_id, node_outputs, (sum(self.timer.values()), self.timer)

    def to_dot(self, filepath: Optional[str] = None, \
               add_optz: bool = False, show_function: bool = True, _prefix: str = "") -> str:
        """
        Generates a DOT language representation of the pipeline graph.

        This can be used with Graphviz to visualize the pipeline structure.

        Args:
            filepath (str, optional): The path to save the .dot file. 
                If None, no .dot file is saved.
        
        Returns:
            the DOT string of the pipeline
        """
        def escape_id(nid: str) -> str:
            return f"{_prefix}{nid}"

        dot_lines: List[str] = []
        dot_lines.append("digraph Pipeline {" if _prefix == "" else "subgraph Pipeline {")
        dot_lines.append('  rankdir=TB;')  # vertical layout
        dot_lines.append('  node [fontsize=12 fontname="Helvetica"];')

        last_node_id = self.static_order()[-1] if self.static_order() else None

        for node_id, node in self.nodes.items():
            full_id = escape_id(node_id).replace(" ", "_")
            is_last = node_id == last_node_id

            if isinstance(node, NodeIf):
                if node.func.__module__ == "__main__":
                    func_label = node.func.__name__
                else:
                    func_label = f"{node.func.__module__}.{node.func.__name__}"
                if node.func.__name__ == "<lambda>":
                    func_label = "lambda"
                dot_lines.append(f'  subgraph cluster_{full_id} {{')
                dot_lines.append('    style=dashed;')
                if show_function:
                    dot_lines.append(f'    "{full_id}" [shape=diamond, label=< <B>{node_id}</B> '+
                                     f'<BR/><FONT POINT-SIZE=\"10\">{func_label}</FONT> >];')
                else:
                    dot_lines.append(f'    "{full_id}" [shape=diamond, \
                                     label=< <B>{node_id}</B> >];')
                dot_lines.append(node.true_pipeline.to_dot(None, _prefix=full_id + "_T_"))
                dot_lines.append(node.false_pipeline.to_dot(None, _prefix=full_id + "_F_"))
                true_first = node.true_pipeline.static_order()[0]
                false_first = node.false_pipeline.static_order()[0]
                true_last = node.true_pipeline.static_order()[-1]
                false_last = node.false_pipeline.static_order()[-1]
                dot_lines.append(f'    "{full_id}" -> "{full_id}_T_{true_first}" '+
                                 '[label="True", tailport=s];')
                dot_lines.append(f'    "{full_id}" -> "{full_id}_F_{false_first}" '+
                                 '[label="False", tailport=s];')
                dot_lines.append(f'    "{full_id}_output" [shape=diamond, '+
                                 'label=< <FONT POINT-SIZE="10"> If Output</FONT> >];')
                dot_lines.append(f'    "{full_id}_T_{true_last}" -> "{full_id}_output" '+
                                 '[tailport=s];')
                dot_lines.append(f'    "{full_id}_F_{false_last}" -> "{full_id}_output" '+
                                 '[tailport=s];')
                dot_lines.append('  }')
            elif isinstance(node, NodeFor):
                dot_lines.append(f'  subgraph cluster_{full_id} {{')
                dot_lines.append('    style=dashed;')
                dot_lines.append(f'    "{full_id}" [shape=Mdiamond, label=< <B>{node_id}</B><BR/>'+
                                 '<FONT POINT-SIZE="10">For Loop</FONT> >];')
                dot_lines.append(node.loop_pipeline.to_dot(None, _prefix=full_id + "_L_"))
                loop_first = node.loop_pipeline.static_order()[0]
                loop_last = node.loop_pipeline.static_order()[-1]
                dot_lines.append(f'    "{full_id}" -> "{full_id}_L_{loop_first}" '+
                                 '[label="start", tailport=s];')
                dot_lines.append(f'    "{full_id}_L_{loop_last}" -> "{full_id}" '+
                                 '[label="next"];')
                dot_lines.append(f'    "{full_id}_output" [shape=diamond, '+
                                 'label=< <FONT POINT-SIZE="10"> For Output</FONT> >];')
                dot_lines.append(f'    "{full_id}_L_{loop_last}" -> "{full_id}_output";')
                dot_lines.append('  }')
            elif isinstance(node, NodeWhile):
                dot_lines.append(f'  subgraph cluster_{full_id} {{')
                dot_lines.append('    style=dashed;')
                dot_lines.append(f'    "{full_id}" [shape=Mdiamond, label=< <B>{node_id}</B><BR/>'+
                                 '<FONT POINT-SIZE="10">While Loop</FONT> >];')
                dot_lines.append(node.loop_pipeline.to_dot(None, _prefix=full_id + "_L_"))
                loop_first = node.loop_pipeline.static_order()[0]
                loop_last = node.loop_pipeline.static_order()[-1]
                dot_lines.append(f'    "{full_id}" -> "{full_id}_L_{loop_first}" '+
                                 '[label="start", tailport=s];')
                dot_lines.append(f'    "{full_id}_L_{loop_last}" -> "{full_id}" '+
                                 '[label="next"];')
                dot_lines.append(f'    "{full_id}_output" [shape=diamond, '+
                                 'label=< <FONT POINT-SIZE="10"> While Output</FONT> >];')
                dot_lines.append(f'    "{full_id}_L_{loop_last}" -> "{full_id}_output";')
                dot_lines.append('  }')
            elif isinstance(node, Pipeline):
                dot_lines.append(f'  subgraph cluster_{full_id} {{')
                dot_lines.append(f'    label="SubPipeline: {node.name}"; '+
                                 'style=filled; color=lightgrey;')
                dot_lines.append(node.to_dot(None, _prefix=full_id + "_"))
                dot_lines.append('  }')
            elif add_optz or not node_id.startswith("[optz]"):
                func_module = node.func.__module__
                func_name = node.func.__name__
                func_label = func_name
                if func_module != '__main__':
                    func_label = func_module + "." + func_label
                shape = "doubleoctagon" if is_last and _prefix == "" else "box"
                if show_function:
                    dot_lines.append(f'    "{full_id}" [shape={shape}, label=< <B>{node_id}</B> '+
                                     f'<BR/><FONT POINT-SIZE=\"10\">{func_label}</FONT> >];')
                else:
                    dot_lines.append(f'    "{full_id}" [shape={shape}, '+
                                     f'label=< <B>{node_id}</B> >];')
                if (param_keys := list(node.get_fixed_params().keys())) != []:
                    dot_lines[-1] = dot_lines[-1][:-3] + '<BR/><FONT POINT-SIZE="8">'+\
                                    f'<I>({", ".join(param_keys)})</I></FONT> >];'

        for to_id, deps in self.node_dependencies.items():
            for input_name, from_id in deps.items():
                from_label = escape_id(from_id).replace(" ", "_")
                to_label = escape_id(to_id).replace(" ", "_")
                label_text = f"{input_name}"
                if from_id.startswith("run_params:"):
                    if _prefix != "":
                        continue
                    input_label = input_name.split(":")[-1]
                    dot_lines.append(f'  {{ rank=source; "params_{input_label}"; }}')
                    dot_lines.append(f'  "params_{input_label}" [shape=ellipse, style=dashed, '+
                                     f'label=< <FONT POINT-SIZE="10">{input_label}</FONT> >];')
                    dot_lines.append(f'  "params_{input_label}" -> "{to_label}" '+
                                     f'[label="{input_label}", fontsize=10, style=dashed];')
                elif isinstance(self.nodes[from_id], (NodeIf, NodeFor, NodeWhile)):
                    dot_lines.append(f'  "{from_label}_output" -> "{to_label}" '+
                                     f'[label="{label_text}", fontsize=9];')
                elif isinstance(self.nodes[to_id], (NodeIf, NodeWhile)) and \
                     input_name.startswith("condition_func:"):
                    dot_lines.append(f'  "{from_label}" -> "{to_label}" '+
                                     f'[label="{label_text[15:]}", fontsize=9, headport=w];')
                elif add_optz or not from_label.startswith("[optz]"):
                    dot_lines.append(f'  "{from_label}" -> "{to_label}" '+
                                     f'[label="{label_text}", fontsize=9];')

        dot_lines.append("}")
        dot_str = "\n".join(dot_lines)
        if filepath is not None:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(dot_str)
        return "\n".join(dot_lines)

    def to_image(self, filepath: str, dpi: int = 160, \
                 add_optz: bool = False, show_function: bool = True) -> None:
        """Generates a PNG image of the pipeline graph using Graphviz."""
        delete = False
        if filepath is None or not os.path.exists(filepath):
            self.to_dot(os.path.splitext(filepath)[0] + ".dot", \
                        add_optz=add_optz, show_function=show_function)
            delete = True
        try:
            res = os.system(f'dot -Tpng -Gdpi={dpi} \
                            "{os.path.splitext(filepath)[0] + ".dot"}" -o "{filepath}"')
        except Exception as e:
            raise RuntimeError("Error during PNG generation.\n"+
                               "Do you have graphviz installed?") from e
        if res:
            print("Error during PNG generation")
        if delete:
            os.remove(os.path.splitext(filepath)[0] + ".dot")

    def to_json(self, filepath: str) -> None:
        """
        Serializes the pipeline's structure to a JSON file.

        This saves the nodes, their parameters, and their connections, allowing
        the pipeline to be reconstructed later.

        Args:
            filepath (str): The path to save the JSON file.
        """
        def serialize_node(node: Union[Node, Pipeline]) -> Dict[str, Any]:
            if isinstance(node, NodeIf):
                func_mod = node.func.__module__
                func_name = node.func.__name__
                return {
                    "id": node.id,
                    "type": "NodeIf",
                    "condition_type": f"{func_mod}.{func_name}",
                    "true_pipeline": serialize_pipeline(node.true_pipeline),
                    "false_pipeline": serialize_pipeline(node.false_pipeline),
                    "fixed_params": node.fixed_params
                }
            if isinstance(node, NodeFor):
                return {
                    "id": node.id,
                    "type": "NodeFor",
                    "loop_pipeline": serialize_pipeline(node.loop_pipeline),
                    "fixed_params": node.fixed_params
                }
            if isinstance(node, NodeWhile):
                return {
                    "id": node.id,
                    "type": "NodeWhile",
                    "condition_type": f"{node.func.__module__}.{node.func.__name__}",
                    "loop_pipeline": serialize_pipeline(node.loop_pipeline),
                    "fixed_params": node.fixed_params
                }
            if isinstance(node, Pipeline):
                return {
                    "id": node.name,
                    "type": "SubPipeline",
                    "pipeline": serialize_pipeline(node)
                }
            # Node
            func_mod = node.func.__module__
            func_name = node.func.__name__
            return {
                "id": node.id,
                "type": f"{func_mod}.{func_name}" if func_mod != "__main__" else func_name,
                "fixed_params": node.fixed_params
            }

        def serialize_pipeline(pipe: Pipeline) -> Dict[str, Any]:
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
    def _default_function_resolver(type_str: str) -> Callable[..., Any]:
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
    def from_json(cls, filepath: str, \
                  function_resolver: Optional[Callable[[str], Any]] = None) -> Pipeline:
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

        def build_pipeline(pipeline_data: Dict[str, Any]) -> Pipeline:
            pipeline_instance = cls(pipeline_data["name"], pipeline_data["description"])
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
                    node = NodeIf(
                        node_id,
                        resolver(node_data["condition_type"]),
                        build_pipeline(node_data["true_pipeline"]),
                        build_pipeline(node_data["false_pipeline"]),
                        fixed_params
                    )
                elif node_type == "NodeFor":
                    loop_pipeline = build_pipeline(node_data["loop_pipeline"])
                    node = NodeFor(
                        node_id,
                        loop_pipeline,
                        fixed_params
                    )
                elif node_type == "NodeWhile":
                    loop_pipeline = build_pipeline(node_data["loop_pipeline"])
                    node = NodeWhile(
                        node_id,
                        resolver(node_data["condition_type"]),
                        loop_pipeline,
                        fixed_params
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

    def run_single_node(self, node_id: str, inputs: Union[None, Dict[str, Any]] = None, \
                        change_memory: bool = False) -> Any:
        """Executes a single node within the pipeline."""
        inputs = {**inputs,
                  **{input_param: self.nodes[source_node_id].output
                        for input_param, source_node_id in self.node_dependencies[node_id].items()}}
        return self.nodes[node_id].execute(inputs, memory=change_memory)

    def clear_memory(self) -> None:
        """Clears the memory of all nodes in the pipeline."""
        for node in self.nodes.values():
            node.clear_memory()
