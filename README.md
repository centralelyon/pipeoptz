# PipeOptz: A Framework for Pipeline Optimization

<p align="center">
  <a href="https://pypi.org/project/pipeoptz/"><img alt="PyPI" src="https://img.shields.io/pypi/v/pipeoptz"></a>
  <a href="https://github.com/centralelyon/pipeoptz/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/centralelyon/pipeoptz/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/pipeoptz/"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/pipeoptz"></a>
  <a href="https://github.com/centralelyon/pipeoptz/blob/main/LICENSE"><img alt="PyPI - License" src="https://img.shields.io/pypi/l/pipeoptz"></a>
</p>

**PipeOptz** is a Python library for building, visualizing, and optimizing complex processing pipelines. It allows you to define a series of operations as a graph, manage the flow of data, and then automatically tune the parameters of those operations to achieve a desired outcome.

While it can be used for any sequence of operations, it is particularly powerful for optimizing image processing workflows.

## Core Concepts

The library is built around a few key ideas:

- **`Node`**: A `Node` is the basic building block of a pipeline. It wraps a single Python function and its parameters.

- **`Pipeline`**: The `Pipeline` holds the entire workflow. You add nodes to it and define their dependencies, forming a Directed Acyclic Graph (DAG). The pipeline manages the execution order.

- **`Parameter`**: A `Parameter` defines the search space for a value you want to optimize. The library provides different types, like `IntParameter`, `FloatParameter`, and `ChoiceParameter`.

- **`PipelineOptimizer`**: This is the engine that tunes your pipeline. It takes your pipeline, a set of `Parameter`s to vary, and a `loss_function` to minimize, and uses metaheuristic algorithms (like Genetic Algorithms, Bayesian Optimization, etc.) to find the best parameter values.

## Installation

Install PipeOptz from PyPI:

```bash
pip install pipeoptz
```

For image processing, you will also need OpenCV, which can be installed alongside:

```bash
pip install pipeoptz opencv-python
```

## Usage: A Simple Example

Let's create a basic pipeline with a few arithmetic operations to see how it works.

```python
from pipeoptz import Pipeline, Node

# 1. Define the functions your nodes will execute
def add(x, y):
    return x + y

def multiply(a, b):
    return a * b

# 2. Create a pipeline
pipeline = Pipeline(name="arithmetic_pipeline")

# 3. Create nodes and add them to the pipeline with dependencies
# Node A: 5 + 3 = 8
pipeline.add_node(Node(id="A", func=add, fixed_params={"x": 5, "y": 3}))

# Node B: Takes the output of A as input -> 8 * 10 = 80
pipeline.add_node(Node(id="B", func=multiply, fixed_params={"b": 10}), predecessors={"a": "A"})

# Node C: Takes the output of B as input -> 80 + 1 = 81
pipeline.add_node(Node(id="C", func=add, fixed_params={"y": 1}), predecessors={"x": "B"})


# 4. Run the pipeline
# The result is a tuple: (last_node_id, history_of_all_node_outputs, execution_times)
last_node, history, _ = pipeline.run()

print(f"Pipeline finished at node: {last_node}")
print(f"Result of final node 'C': {history[last_node]}")
print(f"History of all node outputs: {history}")

# 5. Visualize the pipeline
# This creates a .dot file and a .png image of the graph
pipeline.to_dot("pipeline_example.dot", generate_png=True)
```

This script will output:

```
Pipeline finished at node: C
Result of final node 'C': 81
History of all node outputs: {'A': 8, 'B': 80, 'C': 81}
```

And it will generate an image (`pipeline_example.png`) of your pipeline's structure:

<div align="center">
  <img src="examples/basic/pipeline_example.png" alt="Simple Pipeline Graph" width="150"/>
</div>

## Optimizing a Pipeline

The real power of `PipeOptz` comes from optimization. The simple example above uses fixed parameters, but you can easily make them tunable.

To do this, you would:
1.  Create a `PipelineOptimizer`.
2.  Define which parameters to tune using objects like `IntParameter` or `FloatParameter`.
3.  Provide a `loss_function` that calculates how "good" the pipeline's output is.
4.  Run the `optimizer.optimize()` method.

For a complete, runnable optimization example, please see the Jupyter Notebook at: **`examples/advanced/simple.ipynb`**.

## Contributing

Contributions are welcome! If you have suggestions for improvements or find a bug, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.