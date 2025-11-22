---
title: "PipeOptz: A Python Library for Pipeline Optimization"
tags:
  - Python
  - Pipeline
  - Optimization
authors:
  - name: Nicolas Pengov
    orcid: 0009-0003-2615-4016 # TODO: Add your ORCID here
    equal-contrib: false
    affiliation: "1, 2" 
  - name: Théo Jaunet
    orcid: 0000-0003-3081-5123 # TODO: Add your ORCID here
    equal-contrib: false
    affiliation: "1, 2" 
  - name: Romain Vuillemot
    orcid: 0000-0003-1447-6926
    equal-contrib: false
    affiliation: "1, 2" 
affiliations:
 - name: Ecole Centrale de Lyon, France
   index: 1
   ror: 05s6rge65
 - name: LIRIS CNRS UMR 5205, France
   index: 2
date: 18 November 2025
bibliography: paper.bib
---

# Summary

**PipeOptz** is a Python library for the building, visualizing, and fine-tuning of processing pipelines. It enables users to define a series of operations as a DAG (Directed Acyclic Graph) which parameters can then be optimized to achieve a desired outcome. The library is designed to be suitable for a wide range of applications, and is particularly suited for image processing where workflows can be dense and often require parameter tuning.

# Statement of need

In many scientific and engineering domains, complex data processing workflows are common. These workflows, i.e., pipelines, often consist of multiple steps, each with its own set of parameters and outputs. Finding the optimal set of parameters, and their individual influence, for a given task can be a tedious and time-consuming process which is often solved through manual trial and error. This is especially true in fields like image processing, where a sequence of filters and transformations is applied to an image, e.g., to find the best thresholding parameters. As opposed standard deep learning systems, pipelines, and their parameters have also the benefit to be more interpretable, through visualizations, and more easily reproducible.

Existing tools for pipeline management often fall into two categories: heavy-weight workflow orchestration frameworks (e.g., Airflow, Prefect) that are designed for large-scale data engineering tasks, or more specialized machine learning pipeline libraries (e.g., Scikit-learn pipelines) that are focused on linear sequences of operations. From our experience, we found a need for a lightweight, flexible, and Pythonic library that is suited for the easy creation, visualization, and optimization of, non-linear pipelines directly within a Python script.

`PipeOptz` addresses this need by providing an API for defining pipelines as Directed Acyclic Graphs (DAGs), with support for conditional branching and looping. In this graph, each node is a user-defined function in python, to ensure expressivity, and application to various domains. It integrates parameter optimization as a core feature, enabling users to define a search space for their pipeline's parameters and use various baseline optimization algorithms to find the best configuration.

# Functionality

`PipeOptz` is built around the following core concepts:

-  **`Node`**: The basic building block of a pipeline. A `Node` wraps a single Python function and its parameters. We also provide more complex nodes for control flow:
    -   `NodeIf`: for conditional branching (if/else).
    -   `NodeFor`: for 'for' loops.
    -   `NodeWhile`: for 'while' loops.

-   **`Pipeline`**: A `Pipeline` holds the entire workflow. Nodes are added to the pipeline with their dependencies, forming a DAG. The pipeline manages the execution by following a topological order.

-   **`Parameter`**: A `Parameter` defines the type and search space for a value to be optimized. `PipeOptz` provides several types of parameters:
    -   `IntParameter`: for integers within a given range.
    -   `FloatParameter`: for floating-point numbers within a given range.
    -   `ChoiceParameter`: for selecting a value from a list of choices.
    -   `MultiChoiceParameter`: for selecting multiple values from a list of choices.
    -   `BoolParameter`: for boolean values (`True` or `False`).

-   **`PipelineOptimizer`**: The engine that tunes the pipeline. It takes the following as input: a pipeline, a set of parameters to optimize, and a loss function to minimize. It uses various metaheuristic algorithms to find the best parameter values, including:
    -   Grid Search (GS)
    -   Bayesian Optimization (BO)
    -   Ant Colony Optimization (ACO)
    -   Simulated Annealing (SA)
    -   Particle Swarm Optimization (PSO)
    -   Genetic Algorithm (GA)

The library also provides features for:

-   **Visualization**: Pipelines can be visualized as graphs using the `to_dot` and `to_image` methods, which generate Graphviz dot files and PNG images.
-   **Serialization**: Pipelines can be saved to and loaded from JSON files using the `to_json` and `from_json` methods, allowing for easy sharing and reuse of workflows.

# Audience

`PipeOptz` is intended for researchers, data scientists, and engineers who need to build, visualize, and optimize data processing workflows in Python. It is particularly useful for those working in image processing, computer vision, and other scientific domains where pipeline-based workflows are common.

# Example Usage

The following example demonstrates how to use `PipeOptz` to find the minimum of a simple function $f(x, y) = (x - 3)^2 + (y + 1)^2$ using Bayesian Optimization. The pipeline is constructed from multiple nodes to showcase the graph-based approach.

```python
from pipeoptz import Pipeline, Node, FloatParameter, PipelineOptimizer

# Define the functions for the nodes
def squared_error(x, y):
    return (x - y)**2

def add(x, y):
    return x + y

# Create the pipeline
pipe = Pipeline("SimplePipeline")
pipe.add_node(Node("X", squared_error, fixed_params={"x": 0, "y": -3}))
pipe.add_node(Node("Y", squared_error, fixed_params={"x": -1, "y": 0}))
pipe.add_node(Node("Add", add), predecessors={"x": "X", "y": "Y"})

# The loss is the function's output, as we want to minimize it
def loss_func(result, _):
    return result

# Set up the optimizer with tunable parameters
optimizer = PipelineOptimizer(pipe, loss_function=loss_func)
optimizer.add_param(FloatParameter("X", "x", -5.0, 5.0))
optimizer.add_param(FloatParameter("Y", "y", -5.0, 5.0))

# Run the Bayesian Optimization
# We provide a dummy dataset ([{}] and [0]) as this example does not depend on external data.
best_params, loss_log = optimizer.optimize([{}], [0], method="BO", iterations=25, init_points=5)

print("Best parameters found:", best_params)
print(f"Final loss: {loss_log[-1]:.4f}")
```

This script will search for the optimal values for `X.x` and `Y.y` that minimize the final output of the pipeline. The expected output will show the best parameters found, which should be close to `{'X.x': 3.0, 'Y.y': -1.0}`, and a final loss close to 0.

We can visualize the pipeline using Figure \ref{fig:example}:

```python
from PIL import Image
pipe.to_image("pipeline.png")
im = Image.open("pipeline.png")
im.show()
```

![Visualization of the example pipeline.](examples/opti/opti.png?raw=true){#fig:example width=50%}

# Citations

`PipeOptz` complements lightweight helpers designed for algorithm evaluation [@Küderle2023], and its optimization engine builds on Bayesian Optimization techniques [@snoek2012practicalbayesianoptimizationmachine] alongside the broader family of metaheuristics surveyed in @engproc2023059238.

# Acknowledgements

This research is partially funded by ANR, the French National Research Agency with the GLACIS project (grant ANR-21-CE33-0002).

# References
