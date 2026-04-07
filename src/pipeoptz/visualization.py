"""Visualization module for Pipeline graphs using DOT/Graphviz."""
from __future__ import annotations
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .pipeline import Pipeline

from .node import Node, NodeIf, NodeFor, NodeWhile


class Visualizer:
    """Handles visualization of Pipeline graphs in DOT format and PNG images."""

    def __init__(self, pipeline: Pipeline) -> None:
        """Initialize the Visualizer with a pipeline."""
        self.pipeline = pipeline

    def to_dot(self, filepath: Optional[str] = None,
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

        last_node_id = self.pipeline.static_order()[-1] if self.pipeline.static_order() else None

        for node_id, node in self.pipeline.nodes.items():
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
            elif isinstance(node, type(self.pipeline)):
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

        for to_id, deps in self.pipeline.node_dependencies.items():
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
                elif isinstance(self.pipeline.nodes[from_id], (NodeIf, NodeFor, NodeWhile)):
                    dot_lines.append(f'  "{from_label}_output" -> "{to_label}" '+
                                     f'[label="{label_text}", fontsize=9];')
                elif isinstance(self.pipeline.nodes[to_id], (NodeIf, NodeWhile)) and \
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

    @staticmethod
    def _mermaid_id(node_id: str, prefix: str = "") -> str:
        """Builds a Mermaid-safe identifier."""
        safe_id = re.sub(r"[^0-9A-Za-z_]", "_", f"{prefix}{node_id}")
        if safe_id == "":
            return "node"
        if safe_id[0].isdigit():
            return f"n_{safe_id}"
        return safe_id

    @staticmethod
    def _mermaid_text(text: str) -> str:
        """Escapes label text for Mermaid."""
        return text.replace('"', "'").replace("<lambda>", "lambda")

    @staticmethod
    def _function_label(node: Node) -> str:
        """Returns a readable function label for a node."""
        func_module = node.func.__module__
        func_name = node.func.__name__
        if func_name == "<lambda>":
            return "lambda"
        if func_module == "__main__":
            return func_name
        return f"{func_module}.{func_name}"

    @classmethod
    def _add_mermaid_node(cls, state: Dict[str, Any], node_id: str,
                          label: str, shape: str = "box") -> None:
        """Adds a Mermaid node definition once."""
        if node_id in state["defined_nodes"]:
            return
        wrappers = {
            "box": ('["', '"]'),
            "diamond": ('{"', '"}'),
            "terminal": ('(["', '"])'),
        }
        prefix, suffix = wrappers[shape]
        state["lines"].append(f'    {node_id}{prefix}{cls._mermaid_text(label)}{suffix}')
        state["defined_nodes"].add(node_id)

    @staticmethod
    def _add_mermaid_edge(state: Dict[str, Any], source: str,
                          target: str, label: Optional[str] = None) -> None:
        """Adds a Mermaid edge definition once."""
        edge = f"{source}|{label}|{target}" if label is not None else f"{source}|{target}"
        if edge in state["defined_edges"]:
            return
        if label is None:
            state["lines"].append(f"    {source} --> {target}")
        else:
            state["lines"].append(f'    {source} -->|{label}| {target}')
        state["defined_edges"].add(edge)

    def to_mermaid(self, filepath: Optional[str] = None,
                   add_optz: bool = False, show_function: bool = True,
                   _prefix: str = "", _state: Optional[Dict[str, Any]] = None) -> str:
        """
        Generates a Mermaid flowchart representation of the pipeline graph.

        Args:
            filepath (str, optional): The path to save the Mermaid text file.
                If None, no file is saved.

        Returns:
            the Mermaid flowchart string of the pipeline
        """
        is_root = _state is None
        if _state is None:
            _state = {
                "lines": ["flowchart TD"],
                "defined_nodes": set(),
                "defined_edges": set(),
            }

        last_node_id = self.pipeline.static_order()[-1] if self.pipeline.static_order() else None

        for node_id, node in self.pipeline.nodes.items():
            full_id = self._mermaid_id(node_id, _prefix)
            is_last = node_id == last_node_id and _prefix == ""

            if isinstance(node, NodeIf):
                label = node_id
                if show_function:
                    label += f" | {self._function_label(node)}"
                self._add_mermaid_node(_state, full_id, label, shape="diamond")

                Visualizer(node.true_pipeline).to_mermaid(
                    None, add_optz, show_function, full_id + "_T_", _state
                )
                Visualizer(node.false_pipeline).to_mermaid(
                    None, add_optz, show_function, full_id + "_F_", _state
                )

                output_id = self._mermaid_id(f"{node_id}_output", _prefix)
                self._add_mermaid_node(_state, output_id, "If Output")

                true_order = node.true_pipeline.static_order()
                false_order = node.false_pipeline.static_order()
                if true_order:
                    self._add_mermaid_edge(
                        _state, full_id, self._mermaid_id(true_order[0], full_id + "_T_"), "True"
                    )
                    self._add_mermaid_edge(
                        _state, self._mermaid_id(true_order[-1], full_id + "_T_"), output_id
                    )
                else:
                    self._add_mermaid_edge(_state, full_id, output_id, "True")

                if false_order:
                    self._add_mermaid_edge(
                        _state, full_id, self._mermaid_id(false_order[0], full_id + "_F_"), "False"
                    )
                    self._add_mermaid_edge(
                        _state, self._mermaid_id(false_order[-1], full_id + "_F_"), output_id
                    )
                else:
                    self._add_mermaid_edge(_state, full_id, output_id, "False")

            elif isinstance(node, NodeFor):
                self._add_mermaid_node(_state, full_id, f"{node_id} | For Loop")
                Visualizer(node.loop_pipeline).to_mermaid(
                    None, add_optz, show_function, full_id + "_L_", _state
                )

                output_id = self._mermaid_id(f"{node_id}_output", _prefix)
                self._add_mermaid_node(_state, output_id, "For Output")
                loop_order = node.loop_pipeline.static_order()
                if loop_order:
                    loop_first = self._mermaid_id(loop_order[0], full_id + "_L_")
                    loop_last = self._mermaid_id(loop_order[-1], full_id + "_L_")
                    self._add_mermaid_edge(_state, full_id, loop_first, "start")
                    self._add_mermaid_edge(_state, loop_last, full_id, "next")
                    self._add_mermaid_edge(_state, loop_last, output_id)
                else:
                    self._add_mermaid_edge(_state, full_id, output_id)

            elif isinstance(node, NodeWhile):
                label = f"{node_id} | While Loop"
                if show_function:
                    label += f" | {self._function_label(node)}"
                self._add_mermaid_node(_state, full_id, label, shape="diamond")
                Visualizer(node.loop_pipeline).to_mermaid(
                    None, add_optz, show_function, full_id + "_L_", _state
                )

                output_id = self._mermaid_id(f"{node_id}_output", _prefix)
                self._add_mermaid_node(_state, output_id, "While Output")
                loop_order = node.loop_pipeline.static_order()
                if loop_order:
                    loop_first = self._mermaid_id(loop_order[0], full_id + "_L_")
                    loop_last = self._mermaid_id(loop_order[-1], full_id + "_L_")
                    self._add_mermaid_edge(_state, full_id, loop_first, "start")
                    self._add_mermaid_edge(_state, loop_last, full_id, "next")
                    self._add_mermaid_edge(_state, loop_last, output_id)
                else:
                    self._add_mermaid_edge(_state, full_id, output_id)

            elif isinstance(node, type(self.pipeline)):
                Visualizer(node).to_mermaid(None, add_optz, show_function, full_id + "_", _state)

            elif add_optz or not node_id.startswith("[optz]"):
                label = node_id
                if show_function:
                    label += f" | {self._function_label(node)}"
                param_keys = list(node.get_fixed_params().keys())
                if param_keys:
                    label += f" | params: {', '.join(param_keys)}"
                shape = "terminal" if is_last else "box"
                self._add_mermaid_node(_state, full_id, label, shape=shape)

        for to_id, deps in self.pipeline.node_dependencies.items():
            for input_name, from_id in deps.items():
                to_label = self._mermaid_id(to_id, _prefix)
                label_text = input_name
                if from_id.startswith("run_params:"):
                    if _prefix != "":
                        continue
                    input_label = input_name.split(":")[-1]
                    param_id = self._mermaid_id(f"params_{input_label}")
                    self._add_mermaid_node(_state, param_id, input_label)
                    self._add_mermaid_edge(_state, param_id, to_label, input_label)
                elif isinstance(self.pipeline.nodes[from_id], (NodeIf, NodeFor, NodeWhile)):
                    from_label = self._mermaid_id(f"{from_id}_output", _prefix)
                    self._add_mermaid_edge(_state, from_label, to_label, label_text)
                elif isinstance(self.pipeline.nodes[to_id], (NodeIf, NodeWhile)) and \
                     input_name.startswith("condition_func:"):
                    from_label = self._mermaid_id(from_id, _prefix)
                    self._add_mermaid_edge(_state, from_label, to_label, label_text[15:])
                elif add_optz or not from_id.startswith("[optz]"):
                    from_label = self._mermaid_id(from_id, _prefix)
                    self._add_mermaid_edge(_state, from_label, to_label, label_text)

        mermaid_str = "\n".join(_state["lines"])
        if is_root and filepath is not None:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(mermaid_str)
        return mermaid_str

    def to_image(self, filepath: str, dpi: int = 160,
                 add_optz: bool = False, show_function: bool = True) -> None:
        """Generates a PNG image of the pipeline graph using Graphviz."""
        delete = False
        if filepath is None or not os.path.exists(filepath):
            self.to_dot(os.path.splitext(filepath)[0] + ".dot",
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
