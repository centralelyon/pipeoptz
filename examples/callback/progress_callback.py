"""Report step-by-step progress while fitting an affine function."""

from pipeoptz import Callback, IntParameter, Node, Pipeline, PipelineOptimizer


def affine(value, slope, intercept):
    """Apply an affine transformation to a value."""
    return slope * value + intercept


def absolute_error(actual, expected):
    """Return the absolute difference between two values."""
    return abs(actual - expected)


class StepProgressCallback(Callback):
    """Print the best loss and parameters after every search step."""

    def on_optimization_begin(self, logs=None):
        print(f"Starting {logs['method']} optimization")

    def on_iteration_end(self, iteration, logs=None):
        step = iteration + 1
        total = logs["total_iterations"]
        print(
            f"Step {step:02d}/{total}: "
            f"best_loss={logs['best_loss']:.2f}, "
            f"best_params={logs['best_params']}"
        )

    def on_optimization_end(self, logs=None):
        print(f"Optimization {logs['status']}")
        if logs["status"] == "completed":
            print(f"Final parameters: {logs['best_params']}")


def build_optimizer():
    """Create an optimizer for the slope and intercept of an affine function."""
    pipeline = Pipeline("Affine fitting")
    pipeline.add_node(
        Node(
            node_id="Affine",
            func=affine,
            fixed_params={"slope": 1, "intercept": 0},
        ),
        predecessors={"value": "run_params:value"},
    )

    optimizer = PipelineOptimizer(pipeline, absolute_error)
    optimizer.add_param(
        IntParameter(node_id="Affine", param_name="slope", min_value=0, max_value=4)
    )
    optimizer.add_param(
        IntParameter(
            node_id="Affine", param_name="intercept", min_value=-1, max_value=3
        )
    )
    return optimizer


if __name__ == "__main__":
    optimizer = build_optimizer()
    optimizer.optimize(
        X=[{"value": 0}, {"value": 1}, {"value": 2}, {"value": 3}],
        y=[1, 3, 5, 7],  # y = 2x + 1
        method="GS",
        max_combinations=25,
        param_sampling=6,
        callbacks=[StepProgressCallback()],
    )
