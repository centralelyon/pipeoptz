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

And it will generate an image (`pipeline_example.png`) of your pipeline's structure.

## Optimizing a Pipeline

The real power of `PipeOptz` comes from optimization. The simple example above uses fixed parameters, but you can easily make them tunable.

To do this, you would:
1.  Create a `PipelineOptimizer`.
2.  Define which parameters to tune using objects like `IntParameter` or `FloatParameter`.
3.  Provide a `loss_function` that calculates how "good" the pipeline's output is.
4.  Run the `optimizer.optimize()` method.

For a complete, runnable optimization example, please see the Jupyter Notebook at: **`examples/advanced/simple.ipynb`** in the [GitHub repository](https://github.com/centralelyon/pipeoptz).

## More Examples

For more advanced examples, please refer to the `examples` directory in the [GitHub repository](https://github.com/centralelyon/pipeoptz/tree/main/examples).
