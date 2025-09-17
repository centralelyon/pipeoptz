# Installation

You can install PipeOptz in two ways: from PyPI or from source.

## Installation from PyPI

It is recommended to install PipeOptz in a virtual environment to avoid conflicts with other packages.

1.  Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2.  To install the latest stable version of PipeOptz from PyPI, run the following command:

    ```bash
    pip install pipeoptz
    ```

## Quick Test

To verify that PipeOptz is installed correctly, you can run the following Python script:

```python
from pipeoptz import Pipeline, Node

# Define simple functions for the pipeline
def add(x, y):
    return x + y

def multiply(a, b):
    return a * b

# Create a pipeline
pipeline = Pipeline("Test Pipeline")

# Create nodes
node_a = Node("A", add, fixed_params={"x": 5, "y": 3})      # A = 5 + 3 = 8
node_b = Node("B", multiply, fixed_params={"b": 10})   # B = A * 10 = 80
node_c = Node("C", add, fixed_params={"y": 2})         # C = B + 2 = 82

# Add nodes to the pipeline with dependencies
pipeline.add_node(node_a)
pipeline.add_node(node_b, predecessors={"a": "A"})
pipeline.add_node(node_c, predecessors={"x": "B"})

# Run the pipeline
last_node, results, _ = pipeline.run()

# Print the result of the last node
print(f"Result of the pipeline: {results[last_node]}")
```

Save the script as `test.py` and run it from your terminal:

```bash
python test.py
```

You should see the following output:

```
Result of the pipeline: 82
```

## Installation from source

If you want to contribute to the development of PipeOptz or if you need the development version, you can install it from source.

1.  Clone the Git repository:

    ```bash
    git clone https://github.com/centralelyon/pipeoptz.git
    ```

2.  Navigate to the project directory:

    ```bash
    cd pipeoptz
    ```

3.  Install the package in editable mode with development dependencies:

    ```bash
    pip install -e .[dev]
    ```

This will install PipeOptz in your Python environment and allow you to modify the source code and see the changes immediately.