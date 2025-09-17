# Welcome to the PipeOptz Documentation

**PipeOptz** is a Python library for building, visualizing, and optimizing complex processing pipelines. It allows you to define a series of operations as a graph, manage the flow of data, and then automatically tune the parameters of those operations to achieve a desired outcome.

While it can be used for any sequence of operations, it is particularly powerful for optimizing image processing workflows.

## Core Concepts

The library is built around a few key ideas:

- **`Node`**: A `Node` is the basic building block of a pipeline. It wraps a single Python function and its parameters.

- **`Pipeline`**: The `Pipeline` holds the entire workflow. You add nodes to it and define their dependencies, forming a Directed Acyclic Graph (DAG). The pipeline manages the execution order.

- **`Parameter`**: A `Parameter` defines the search space for a value you want to optimize. The library provides different types, like `IntParameter`, `FloatParameter`, and `ChoiceParameter`.

- **`PipelineOptimizer`**: This is the engine that tunes your pipeline. It takes your pipeline, a set of `Parameter`s to vary, and a `loss_function` to minimize, and uses metaheuristic algorithms (like Genetic Algorithms, Bayesian Optimization, etc.) to find the best parameter values.

## Getting Started

To get started with PipeOptz, please refer to the following pages:

- **[Installation](installation.md)**: Learn how to install PipeOptz.
- **[Usage](usage.md)**: A guide on how to use the library with examples.
- **[API Reference](api.md)**: The complete API documentation.

## Contributing

Contributions are welcome! If you have suggestions for improvements or find a bug, please feel free to open an issue or submit a pull request on our [GitHub repository](https://github.com/centralelyon/pipeoptz).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
