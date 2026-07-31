"""Run an optimization and report its result from a completion callback."""

from pipeoptz import Callback, IntParameter, Node, Pipeline, PipelineOptimizer


def scale(value, factor):
    """Multiply a value by a tunable factor."""
    return value * factor


def absolute_error(actual, expected):
    """Return the absolute difference between two values."""
    return abs(actual - expected)


class CompletionCallback(Callback):
    """Print the best result after optimization completes."""

    def on_optimization_end(self, logs=None):
        if logs["status"] == "completed":
            print("Optimization completed")
            print(f"Best parameters: {logs['best_params']}")
            print(f"Best loss: {logs['best_loss']:.2f}")


def build_optimizer():
    """Create an optimizer for the scale factor."""
    pipeline = Pipeline("Scale optimization")
    pipeline.add_node(
        Node(node_id="Scale", func=scale, fixed_params={"factor": 1}),
        predecessors={"value": "run_params:value"},
    )

    optimizer = PipelineOptimizer(pipeline, absolute_error)
    optimizer.add_param(
        IntParameter(node_id="Scale", param_name="factor", min_value=0, max_value=5)
    )
    return optimizer


if __name__ == "__main__":
    optimizer = build_optimizer()
    optimizer.optimize(
        X=[{"value": 1}, {"value": 2}, {"value": 3}],
        y=[3, 6, 9],
        method="GS",
        max_combinations=6,
        param_sampling=7,
        callbacks=[CompletionCallback()],
    )
