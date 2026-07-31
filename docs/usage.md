# Usage

This page provides examples of how to use PipeOptz to build and optimize pipelines.

## A Simple Example

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
# This creates a Graphviz .dot file without requiring Graphviz to be installed
pipeline.to_dot("pipeline_example.dot")
```

This script will output:

```
Pipeline finished at node: C
Result of final node 'C': 81
History of all node outputs: {'A': 8, 'B': 80, 'C': 81}
```

The generated `pipeline_example.dot` file can be rendered with Graphviz or viewed
with a compatible editor.

## Optimizing a Pipeline

The real power of `PipeOptz` comes from optimization. The simple example above uses fixed parameters, but you can easily make them tunable.

To do this, you would:
1.  Create a `PipelineOptimizer`.
2.  Define which parameters to tune using objects like `IntParameter` or `FloatParameter`.
3.  Provide a `loss_function` that calculates how "good" the pipeline's output is.
4.  Run the `optimizer.optimize()` method.

For a complete, runnable optimization example, see
**`examples/basic_optim/basic_optim.ipynb`** in the
[GitHub repository](https://github.com/centralelyon/pipeoptz).

## Optimization callbacks

Callbacks provide lifecycle hooks for progress reporting, experiment tracking,
checkpointing, or custom monitoring. Subclass `Callback` and pass instances to
`PipelineOptimizer.optimize()`:

```python
from pipeoptz import Callback


class ProgressCallback(Callback):
    def on_iteration_end(self, iteration, logs=None):
        print(iteration, logs["best_loss"], logs["best_params"])


best_params, loss_log = optimizer.optimize(
    X,
    y,
    method="GS",
    max_combinations=100,
    callbacks=[ProgressCallback()],
)
```

The available hooks are:

- `on_optimization_begin(logs)` and `on_optimization_end(logs)`
- `on_iteration_begin(iteration, logs)` and `on_iteration_end(iteration, logs)`
- `on_evaluation_begin(evaluation, logs)` and `on_evaluation_end(evaluation, logs)`

Indexes are zero-based. `self.optimizer` references the active
`PipelineOptimizer`, while `self.params` contains the method and keyword options
for the run. Exceptions raised by callbacks propagate to the caller.

## More Examples

For more advanced examples, please refer to the `examples` directory in the [GitHub repository](https://github.com/centralelyon/pipeoptz/tree/main/examples).
