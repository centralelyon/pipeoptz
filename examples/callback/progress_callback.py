import time
from pipeoptz import Pipeline, Node, PipelineOptimizer, Callback, IntParameter

# ---------------------------------------------------------
# 1. Define the Node Functions (with built-in sleep/wait)
# ---------------------------------------------------------
def add(x, y):
    """Adds two numbers, deliberately slowed down."""
    time.sleep(0.5)  # 0.5 second wait
    return x + y

def multiply(a, b):
    """Multiplies two numbers, deliberately slowed down."""
    time.sleep(0.5)  # 0.5 second wait
    return a * b

# ---------------------------------------------------------
# 2. Build the Pipeline
# ---------------------------------------------------------
pipeline = Pipeline(name="slow_optimization_pipeline")

# Node A will add `x` and `y`. We fix `y=3`, but leave `x` to be optimized.
pipeline.add_node(Node(node_id="A", func=add, fixed_params={"y": 3}))

# Node B takes Node A's output (mapped to `a`) and multiplies it by fixed `b=10`.
pipeline.add_node(Node(node_id="B", func=multiply, fixed_params={"b": 10}), predecessors={"a": "A"})

# ---------------------------------------------------------
# 3. Define the Optimization Search Space & Loss Function
# ---------------------------------------------------------
# We want the optimizer to test values for `x` in Node A between 1 and 20.
# (Note: Syntax for mapping parameters to nodes can vary slightly by version; 
# commonly it uses a dictionary or a specific naming convention like "A__x").
parameters = {
    "A__x": IntParameter(lower_bound=1, upper_bound=20)
}

# We want our pipeline's final output (Node B) to perfectly hit 150.
# If A__x = 12 -> Node A: (12 + 3) = 15 -> Node B: (15 * 10) = 150.
def loss_function(history):
    target_value = 150
    final_output = history["B"]
    # The optimizer will try to minimize this return value (0 is a perfect match)
    return abs(target_value - final_output)

# ---------------------------------------------------------
# 4. Create a Custom "Slow" Callback for Logging
# ---------------------------------------------------------
class SlowProgressCallback(Callback):
    def on_optimization_begin(self, logs=None):
        print(f"\n🚀 Starting Optimization using {logs.get('method', 'Algorithm')}...")
        time.sleep(1) # Wait 1 second before spamming the console

    def on_iteration_end(self, iteration, logs=None):
        print(f"🔄 Iteration {iteration + 1} complete | Best Loss so far: {logs.get('best_loss', 0):.4f}")
        time.sleep(1) # Wait 1 second so you can actually read the step

    def on_optimization_end(self, logs=None):
        print(f"✅ Optimization {logs.get('status', 'Finished')}!\n")

# ---------------------------------------------------------
# 5. Run the Optimizer
# ---------------------------------------------------------
if __name__ == "__main__":
    # Initialize the optimizer engine
    optimizer = PipelineOptimizer(
        pipeline=pipeline,
        parameters=parameters,
        loss_function=loss_function
    )
    
    # X and y are typically used for dataset features/targets in ML pipelines.
    # Since this is a pure math pipeline, we can pass None or dummy data.
    X, y = None, None 
    
    print("Initializing pipeline optimization run...")
    
    # Run the Genetic Algorithm for just 5 generations to keep the test short
    best_params, loss_log = optimizer.optimize(
        X,
        y,
        method="GA",
        generations=5, 
        callbacks=[SlowProgressCallback()]
    )
    
    print(f"🏆 Best parameters found: {best_params}")