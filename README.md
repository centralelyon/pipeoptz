# PipeOptz: Pipeline Optimization Framework

[![PyPI version](https://badge.fury.io/py/pipeoptz.svg)](https://badge.fury.io/py/pipeoptz)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

**PipeOptz** is a Python library for optimizing the parameters of processing pipelines. It is particularly suited for tasks in image processing, but its generic design allows it to be applied to any domain where a sequence of operations needs to be tuned to achieve an optimal result.

The core idea is to represent a workflow as a Directed Acyclic Graph (DAG) of processing **Nodes**. The library then uses various metaheuristic optimization algorithms to find the best set of parameters for these nodes to minimize a given loss function.

## Core Concepts

The library is built around a few key components:

### `Node`
A `Node` represents a single, executable step in the pipeline. It's a wrapper around a Python function that contains its own set of fixed parameters.

```python
def gaussian_blur(image, k, sigma):
    # Kernel size must be an odd integer
    k = int(k) * 2 + 1
    return cv2.GaussianBlur(image, (k, k), sigmaX=sigma)

# A node that applies a Gaussian blur with a fixed kernel size and sigma
blur_node = Node(id="blur", func=gaussian_blur, fixed_params={'k': 5, 'sigma': 1.0})
```

### `NodeIf`
A special type of node that allows for conditional branching within the pipeline. It executes one of two sub-pipelines (`true_pipeline` or `false_pipeline`) based on the boolean output of a `condition_func`.

### `Pipeline`
The `Pipeline` object manages the collection of nodes and their dependencies, forming a DAG. It determines the correct execution order and handles the flow of data from one node's output to another's input.

### `Parameter`
These objects define the search space for the optimization. They specify which node parameters are tunable and their possible range of values. Subclasses exist for different data types:
- `IntParameter`: An integer within a min/max range.
- `FloatParameter`: A float within a min/max range.
- `ChoiceParameter`: A value from a predefined list of options.
- `BoolParameter`: A `True` or `False` value.
- `MultiChoiceParameter`: A sub-list of choices from a list of options within a min/max length range.

### `PipelineOptimizer`
This is the main engine for the optimization process. It takes the `Pipeline`, a `loss_function` to minimize, and the `Parameter` objects to tune. It provides a simple interface to run various optimization algorithms.

## Installation

Currently, PipeOptz is not yet available on PyPI. You can install it directly from the source code:

```bash
git clone https://github.com/your-username/pipeoptz.git
cd pipeoptz
pip install .
```

## Quick Start: Optimizing an Image Processing Pipeline

Let's walk through a simple example: finding the optimal parameters for a Gaussian blur and a thresholding operation to clean up a noisy, blurry image of a circle.

### 1. Define Node Functions

First, we define the Python functions that will serve as our processing steps.

```python
import cv2
import numpy as np

def gaussian_blur(image, k, sigma):
    # The kernel size needs to be an odd integer
    k = int(k) * 2 + 1
    return cv2.GaussianBlur(image, (k, k), sigmaX=sigma)

def threshold(image, threshold_value):
    _, thresh_img = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    # Create a copy to avoid modifying the original array in place
    im = image.copy()
    im[thresh_img == 0] = 0
    return im
```

### 2. Build the Pipeline

Next, we create a `Pipeline` instance and add our functions as `Node`s. We define the data flow using the `predecessors` argument.

- The `blur` node gets its `image` from the pipeline's runtime input (`run_params:image`).
- The `threshold` node gets its `image` from the output of the `blur` node.

```python
from pipeoptz import Pipeline, Node

pipeline = Pipeline(name="simple_im_pipeline_opz")

# Add a blur node
pipeline.add_node(
    Node(id="blur", func=gaussian_blur, fixed_params={'k': 5, 'sigma': 1.0}), 
    predecessors={'image': 'run_params:image'}
)

# Add a threshold node that takes its input from the 'blur' node
pipeline.add_node(
    Node(id="threshold", func=threshold, fixed_params={'threshold_value': 127}), 
    predecessors={'image': 'blur'}
)

node_id, history, times = pipeline.run({'image':numpy_image})

print("Execution time :", times[0])
plt.imshow(history[node_id])
```

The pipeline output is a tuple of three elements:
- `id`: The output of the last executed node in the pipeline. It is this node whose will be used as the input for the loss function.
- `history`: A dictionary containing the outputs of all nodes in the pipeline, keyed by their `id`. If `optimize_memory=True` (`False` by default) the history will contain the output only of the output nodes.
- `times`: A tuple containing the global pipeline execution time and a dictionary containing the execution time of each node, keyed by their `id`.

You can save the pipeline to re-use it later in a JSON format or visualize the pipeline using GraphViz and generate a .DOT file.

```python
pipeline.to_json("path.json")
pipeline2 = Pipeline.from_json("path.json")

pipeline.to_dot("path.dot", generate_png=True)
plt.imshow("path.png")
```
<center><img src="examples/mmm_example/simple_im_pipeline_opz.png" alt="image" width="auto" height="300px"></center>


### 3. Set up the Optimizer

We define which parameters we want to optimize and their search space. We'll tune `sigma` and `k` for the blur, and `threshold_value` for the threshold.

```python
from pipeoptz import PipelineOptimizer, FloatParameter, IntParameter, mse_loss

# We need some sample data (X) and a target/ground truth (y)
# For this example, assume generate_data() creates a noisy image X and a clean image y
X, y = generate_data() 

# The optimizer needs the pipeline, a loss function, and a timeout
optimizer = PipelineOptimizer(
    pipeline=pipeline,
    loss_function=mse_loss, # Mean Squared Error
    max_time_pipeline=0.01 # If the pipeline take more than this time the pipeline is considered to have failed and its loss will be set to infinity. Set to 0 to disable it. 
)

# Define the search space for each parameter
optimizer.add_param(FloatParameter(node_id='blur', param_name='sigma', min_value=0.1, max_value=20.0))
optimizer.add_param(IntParameter(node_id='blur', param_name='k', min_value=1, max_value=10))
optimizer.add_param(IntParameter(node_id='threshold', param_name='threshold_value', min_value=1, max_value=254))
```

### 4. Run the Optimization

Finally, we can run the optimization using one of the available methods. Let's use the Genetic Algorithm (GA).

```python
# The optimizer takes a list of inputs X and a list of corresponding targets y
X_batch = [{"image": Xi} for Xi in generate_multiple_images()]
y_batch = [yi for yi in generate_multiple_targets()]

best_params, loss_log = optimizer.optimize(
    X_batch, y_batch,
    method="GA", 
    generations=50,
    population_size=20,
    verbose=True
)

print("Best parameters found:")
print(best_params)
```

The optimizer will iterate through generations and find the parameter set that minimizes the `mse_loss` between the pipeline's output and the target image `y`. The loss function must take two arguments: the pipeline's output and the ground truth and return a float.

## Supported Optimization Algorithms

PipeOptz provides a unified interface for several common metaheuristic algorithms. You can select the method via the `method` argument in the `optimize()` call.

- **`GS`**: Grid Search
- **`BO`**: Bayesian Optimization
- **`GA`**: Genetic Algorithm
- **`ACO`**: Ant Colony Optimization
- **`SA`**: Simulated Annealing
- **`PSO`**: Particle Swarm Optimization

Each method has its own set of hyperparameters that can be passed as keyword arguments to the `optimize` function.

---

## A Note on the Example Code

The code snippets in this README are simplified for clarity. For a complete, runnable example, please refer to the Jupyter Notebook in `examples/mmm_example/simple.ipynb` and `examples/mmm_example/complex.ipynb` for a more advanced example.

---