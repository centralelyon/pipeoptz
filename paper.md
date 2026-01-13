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

In many scientific and engineering domains, complex data processing workflows are common. These workflows, i.e., pipelines, often consist of multiple steps, each with its own set of parameters and outputs. Finding the optimal set of parameters, and their individual influence, for a given task can be a tedious and time-consuming process which is often solved through manual trial and error. This is especially true in fields like image processing [@vanderwalt2014scikitimage], where a sequence of filters and transformations is applied to an image, e.g., to find the best thresholding parameters. As opposed standard deep learning systems, pipelines, and their parameters have also the benefit to be more interpretable, through visualizations, and more easily reproducible.

Existing tools for pipeline management often fall into two categories: heavy-weight workflow orchestration frameworks (e.g., Airflow [@airflow], Prefect [@prefect]) that are designed for large-scale data engineering tasks, or more specialized machine learning pipeline libraries (e.g., Scikit-learn pipelines [@pedregosa2011scikit]) that are focused on linear sequences of operations. From our experience, we found a need for a lightweight, flexible, and Pythonic library that is suited for the easy creation, visualization, and optimization of, non-linear pipelines directly within a Python script.

`PipeOptz` addresses this need by providing an API for defining pipelines as Directed Acyclic Graphs (DAGs), with support for conditional branching and looping. In this graph, each node is a user-defined function in python, to ensure expressivity, and application to various domains. It integrates parameter optimization as a core feature, enabling users to define a search space for their pipeline's parameters and use various baseline optimization algorithms to find the best configuration.

# State of the field

PipeOptz sits at the intersection of workflow orchestration, pipeline representation, and hyperparameter/black-box optimization. Workflow orchestrators such as Apache Airflow [@airflow] and Prefect [@prefect] provide rich operational features (scheduling, monitoring, retries, deployments) and are well-suited for production batch workflows, but they are not designed as lightweight research libraries that expose the pipeline graph as a first-class object for iterative experimentation and optimization inside another tool. On the other end of the spectrum, hyperparameter optimization (HPO) frameworks and Bayesian optimization toolkits typically assume a user-written objective function and leave the internal structure of the computational pipeline implicit in the user’s code, which limits explicit control-flow nodes, graph-level visualization, and step-wise traceability.

PipeOptz was created to support the needs of Descript, where we required (i) an expressive, multi-step pipeline with explicit control-flow (loops and conditionals), (ii) heterogeneous tunable parameters optimized against an application-specific loss function, and (iii) built-in graph visualization and execution traceability for rapid research iteration. In principle, part of this functionality could be implemented by extending an existing optimization toolkit with a custom loss, but the core requirement here is the combination of "pipeline-as-a-graph" modeling, control-flow nodes, and a clean separation between execution and optimization. Because these constraints cut across the fundamental abstractions of existing orchestration/HPO tools, we implemented a dedicated library designed as a reusable backend component for research workflows rather than an operational orchestrator.

To clarify our position with respect to closely related optimization libraries, Bayesian optimization frameworks such as BayesO [@Kim2023_BayesO] and pyGPGO [@Jimenez2017_pyGPGO] focus primarily on sample-efficient search strategies for expensive black-box objectives. In these systems the pipeline is usually encoded inside a single objective function, so the optimizer does not directly represent intermediate steps or control flow. PipeOptz keeps the optimization goal identical (minimize a user-defined loss), but makes the evaluation procedure explicit: the workflow is represented as a graph of nodes with dependencies and control-flow constructs, enabling node-level traceability and visualization while still treating the overall pipeline outcome as the quantity to optimize.

AutoML frameworks such as NiaAML [@Pecnik2021_NiaAML] also address "pipeline + optimization", but they target the automated composition and tuning of machine-learning pipelines within a predefined space of ML components and objectives. PipeOptz is intentionally not ML-specific: it targets research workflows where the pipeline steps are arbitrary Python functions and the loss can encode domain-specific criteria (e.g., balancing geometric accuracy and the number of extracted targets), making it suitable as a backend for alternative approaches beyond conventional ML pipelines.

Finally, some optimization problems are best addressed by algebraic modeling and solver-based approaches. Linopy [@Hofmann2023_Linopy], for example, provides a modeling layer for linear and mixed-integer optimization with labeled n-dimensional variables and solver backends. PipeOptz is complementary: it targets workflows whose objective is evaluated by executing an end-to-end pipeline and cannot be naturally expressed as a linear/mixed-integer model.



# Software Design

`PipeOptz` is designed to make research pipelines explicit and optimizable while keeping them lightweight and fully Python-native. The main design trade-off is to favor expressivity and traceability over an "objective-function-only" interface: instead of hiding the workflow inside a single function, `PipeOptz` represents it as a pipeline graph with explicit dependencies and control-flow nodes. This matters in research workflows where debugging, profiling, and iterating on multi-step processing chains is as important as finding good parameter values.

`PipeOptz` is built around the following core concepts:

-  **`Node`**: The basic building block of a pipeline. A `Node` wraps a single Python function and its parameters. To support non-linear workflows beyond simple DAG composition, we provide dedicated control-flow nodes that embed sub-pipelines:
    -   `NodeIf`: for conditional branching (if/else).
    -   `NodeFor`: for 'for' loops.
    -   `NodeWhile`: for 'while' loops.

-   **`Pipeline`**: A `Pipeline` holds the entire workflow. Nodes are added to the pipeline with their dependencies, forming a DAG. The pipeline manages execution by following a topological order [@kahn1962topsort]. During execution, the library can cache node outputs and record per-node execution time, supporting fine-grained inspection and iterative refinement of complex pipelines.

-   **`Parameter`**: A `Parameter` defines the type and search space for a value to be optimized. `PipeOptz` provides several types of parameters: `IntParameter`, `FloatParameter`, `ChoiceParameter`, `MultiChoiceParameter`, and `BoolParameter`.

-   **`PipelineOptimizer`**: The optimization layer is separated from pipeline execution. It takes a pipeline, a set of parameters to optimize, and a user-defined loss function to minimize, and then evaluates candidate configurations by running the pipeline. `PipeOptz` provides several baseline optimization strategies :
    - Grid Search (GS), 
    - Bayesian Optimization (BO) [@snoek2012practicalbayesianoptimizationmachine; @shahriari2016bayesianoptimization], 
    - Ant Colony Optimization (ACO) [@dorigo1997antcolony], 
    - Simulated Annealing (SA) [@kirkpatrick1983simulatedannealing], 
    - Particle Swarm Optimization (PSO) [@kennedy1995pso]
    - Genetic Algorithm (GA) [@holland1975adaptation].

The library also provides features for:
-   **Visualization**: Pipelines can be visualized as graphs using the `to_dot` and `to_image` methods, which generate Graphviz dot files and PNG images [@gansner2000graphviz].
-   **Serialization**: Pipelines can be saved to and loaded from JSON files using the `to_json` and `from_json` methods, enabling reuse and sharing beyond a single Python script (the `.dot` export is used for visualization and does not capture full pipeline semantics).


# Research Impact Statement

PipeOptz has been used as the optimization backend of Descript-PipeOptz, a research tool aimed at extracting information from hand-drawn or complex visualizations with very limited supervision. In this setting, users provide a small number of manual extractions and the system searches for a pipeline configuration that generalizes the extraction procedure to the remaining targets. This integration demonstrates that PipeOptz supports real research workflows where the objective function is application-specific and the pipeline structure must remain explicit and modifiable.

While current usage is concentrated within a small research team, the contribution is positioned for credible near-term reuse: the project is packaged for Python and distributed via PyPI, released under an MIT license, and includes automated tests, CI, and structured documentation with executable examples. In the Descript-PipeOptz use case, the optimization loop can converge quickly (on the order of tens of seconds to about a minute depending on the chosen pipeline), and the overall workflow enables tasks that would otherwise require extensive manual effort, making the approach practical for rapid iteration in low-data scenarios. Reproducibility is supported by the public repository and the reference implementation in Descript-PipeOptz, which contains all required elements to reproduce the demonstrated experiments.

# AI usage disclosure

We used generative-AI-assisted developer tools during software development and documentation writing, but not for drafting the JOSS manuscript.

Code: GitHub Copilot (Visual Studio Code extension) (last version used : 1.104.1) and Gemini Code Assist (Visual Studio Code extension) (last version used : 2.53) were used primarily for code completion and small refactoring suggestions during implementation. No large, unverified code blocks were accepted as-is; all AI-assisted edits were reviewed, tested, and integrated by the authors, who made the primary architectural and design decisions.

Documentation: Gemini Code Assist was used to accelerate writing of repetitive API documentation (docstrings, README sections). All generated text was manually reviewed and edited for correctness and consistency with the implemented behavior.

Manuscript: No generative AI tools were used to write `paper.md`.

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

`PipeOptz` complements lightweight helpers designed for algorithm evaluation [@Küderle2023], and relies on NumPy [@harris2020numpy], SciPy [@virtanen2020scipy], and scikit-learn [@pedregosa2011scikit] for numerical computing and Gaussian-process-based Bayesian optimization [@rasmussen2006gaussianprocesses]. Its optimization engine builds on Bayesian Optimization techniques [@snoek2012practicalbayesianoptimizationmachine; @shahriari2016bayesianoptimization] alongside the broader family of metaheuristics surveyed in [@engproc2023059238].

# Acknowledgements

This research is partially funded by ANR, the French National Research Agency with the GLACIS project (grant ANR-21-CE33-0002).

# References
