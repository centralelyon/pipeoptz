# PipeOptz: A Framework for Pipeline Optimization

<p align="center">
  <a href="https://pypi.org/project/pipeoptz/"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/pipeoptz"></a>
  <a href="https://pypi.org/project/pipeoptz/"><img alt="PyPI" src="https://img.shields.io/pypi/v/pipeoptz"></a>
  <a href="https://github.com/centralelyon/pipeoptz/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/centralelyon/pipeoptz/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/centralelyon/pipeoptz/blob/main/LICENSE"><img alt="PyPI - License" src="https://img.shields.io/pypi/l/pipeoptz"></a>
  <a href=""><img alt="pylint Score" src="https://mperlet.github.io/pybadge/badges/9.53.svg"></a>
  <a href="https://joss.theoj.org/papers/a0af364ea2e02b2cd573ce014d436882"><img src="https://joss.theoj.org/papers/a0af364ea2e02b2cd573ce014d436882/status.svg"></a>
</p>

**PipeOptz** is a Python library for building, visualizing, and optimizing complex processing pipelines. It allows you to define a series of operations as a graph, manage the flow of data, and then automatically tune the parameters of those operations to achieve a desired outcome.

The library is built around a few key ideas:

- **`Node`**: A `Node` is the basic building block of a pipeline. It wraps a single Python function and its parameters.

- **`Pipeline`**: The `Pipeline` holds the entire workflow. You add nodes to it and define their dependencies, forming a Directed Acyclic Graph (DAG). The pipeline manages the execution order.

- **`Parameter`**: A `Parameter` defines the search space for a value you want to optimize. The library provides different types, like `IntParameter`, `FloatParameter`, and `ChoiceParameter`.

- **`PipelineOptimizer`**: This is the engine that tunes your pipeline. It takes your pipeline, a set of `Parameter`s to vary, and a `loss_function` to minimize, and uses metaheuristic algorithms (like Genetic Algorithms, Bayesian Optimization, etc.) to find the best parameter values.

The package is provided with a `LICENSE` file which contains the license terms.

## Installation

### Installation from PyPI

The easiest way to install PipeOptz is with `pip`:

```bash
pip install --upgrade --user pipeoptz
```

### Installation from source

If you're reading this `README` from a source distribution, you can install PipeOptz after downloading it with:

```bash
pip install --upgrade --user .
```

You can also install the latest development version directly from GitHub:

```bash
pip install --upgrade --user https://github.com/centralelyon/pipeoptz/archive/main.zip
```

For local development, install PipeOptz in editable mode with its development dependencies:

```bash
pip install --editable ".[dev]"
```

### Checking the installed version

The installed PipeOptz version is available as `pipeoptz.__version__`:

```python
import pipeoptz

print(pipeoptz.__version__)
```

## Quick Start

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
pipeline.add_node(Node(node_id="A", func=add, fixed_params={"x": 5, "y": 3}))

# Node B: Takes the output of A as input -> 8 * 10 = 80
pipeline.add_node(Node(node_id="B", func=multiply, fixed_params={"b": 10}), predecessors={"a": "A"})

# Node C: Takes the output of B as input -> 80 + 1 = 81
pipeline.add_node(Node(node_id="C", func=add, fixed_params={"y": 1}), predecessors={"x": "B"})


# 4. Run the pipeline
# The result is a tuple: (last_node_id, history_of_all_node_outputs, execution_times)
last_node, history, _ = pipeline.run()

print(f"Pipeline finished at node: {last_node}")
print(f"Result of final node 'C': {history[last_node]}")
print(f"History of all node outputs: {history}")

# 5. Visualize the pipeline
# This creates a .dot file and a .png image of the graph
pipeline.to_dot("basic.dot", generate_png=True)
```

This script will output:

```
Pipeline finished at node: C
Result of final node 'C': 81
History of all node outputs: {'A': 8, 'B': 80, 'C': 81}
```

And it will generate an image (`basic.png`) of your pipeline's structure, taken from the `basic.ipynb` example:

<div align="center">
  <img src="https://github.com/centralelyon/pipeoptz/blob/main/examples/basic/basic.png?raw=true" alt="Simple Pipeline Graph" width="120"/>
</div>

## Optimizing a Pipeline

The real power of `PipeOptz` comes from optimization. The simple example above uses fixed parameters, but you can easily make them tunable.

To do this, you would:
1.  Create a `PipelineOptimizer`.
2.  Define which parameters to tune using objects like `IntParameter` or `FloatParameter`.
3.  Provide a `loss_function` that calculates how "good" the pipeline's output is.
4.  Run the `optimizer.optimize()` method.

For a complete, runnable optimization example, please see the Jupyter Notebook at: **`examples/advanced/simple.ipynb`**.

### Monitoring optimization with callbacks

Subclass `Callback` to observe an optimization run. Callbacks receive the optimizer
through `self.optimizer` and lifecycle data through the `logs` dictionary.

```python
from pipeoptz import Callback


class ProgressCallback(Callback):
    def on_optimization_begin(self, logs=None):
        print(f"Starting {logs['method']}")

    def on_iteration_end(self, iteration, logs=None):
        print(f"Iteration {iteration + 1}: loss={logs['best_loss']:.4f}")

    def on_optimization_end(self, logs=None):
        print(f"Optimization {logs['status']}")


best_params, loss_log = optimizer.optimize(
    X,
    y,
    method="GA",
    generations=50,
    callbacks=[ProgressCallback()],
)
```

Available hooks are `on_optimization_begin`, `on_optimization_end`,
`on_iteration_begin`, `on_iteration_end`, `on_evaluation_begin`, and
`on_evaluation_end`. Iteration and evaluation indexes are zero-based.

## Examples
Several example pipelines are provided in the `examples/` directory. These include:
-   `basic/`: A simple pipeline with arithmetic operations.
-   `callback/`: Standalone completion and step-by-step optimization callbacks.
-   `cond/`: A pipeline demonstrating conditional branching.
-   `for/`: A pipeline demonstrating for loops.
-   `while/`: A pipeline demonstrating while loops.
-   `opti/`: A pipeline demonstrating optimization with tunable parameters.

## Docker

You can also use PipeOptz with Docker. This is useful for running the library in a clean, isolated environment.

### Build the Docker image

From the root of the repository, build the Docker image with:

```bash
docker build -t pipeoptz .
```

### Run the Docker container

To start an interactive shell in the container:

```bash
docker run -it --rm pipeoptz
```

You can also mount your local directory to the container to access your files:

```bash
docker run -it --rm -v "$PWD":/workspace pipeoptz
```

This will mount your current directory to `/workspace` inside the container.


## Building Docs

This project uses [MkDocs](https://www.mkdocs.org/) to generate documentation.

To serve the documentation locally, run the following command from the root of the project:

```bash
mkdocs serve
```

This will start a local server, and you can view the documentation by opening your browser to `http://127.0.0.1:8000`.

## Testing
PipeOptz makes use of pytest for its test suite.
```
pip install pytest
pytest
```

## Contributing

Contributions are welcome! Please see the `CONTRIBUTING.md` file for details on how to get started.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

This research is partially funded by ANR, the French National Research Agency with the GLACIS project (grant ANR-21-CE33-0002).
