class Node:
    """
    Represents a single, executable step (a node) in a processing pipeline.

    A Node encapsulates a function to be executed, along with any fixed parameters
    that function requires. It can also cache its last output to avoid re-computation
    if the inputs haven't changed.

    Attributes:
        id (str): A unique identifier for the node.
        func (callable): The function to be executed by this node.
        fixed_params (dict): A dictionary of parameters that are fixed for this
            node's function and do not change during pipeline execution.
        output: Caches the result of the last execution.
        input_hash_last_exec: Caches the hash of the inputs from the last execution,
            used for memory optimization.
    """
    def __init__(self, id, func, fixed_params=None):
        """
        Initializes a Node.

        Args:
            id (str): The unique identifier for the node.
            func (callable): The function this node will execute.
            fixed_params (dict, optional): A dictionary of keyword arguments that will be
                passed to the function on every execution. Defaults to None.
        """
        self.id = id
        self.func = func
        self.fixed_params = fixed_params if fixed_params is not None else {}
        self.output = None
        self.input_hash_last_exec = None

    def get_id(self):
        """Returns the node's unique identifier."""
        return self.id

    def clear_memory(self):
        """Clears the cached output and input hash."""
        self.output = None
        self.input_hash_last_exec = None

    def execute(self, inputs={}, memory=False):
        """
        Executes the node's function with the given inputs.

        Args:
            inputs (dict, optional): A dictionary of inputs for the node's function.
                These are typically the outputs of predecessor nodes. Defaults to {}.
            memory (bool, optional): If True, the node will cache its output and
                only re-execute if the inputs have changed since the last run.
                Defaults to False.

        Returns:
            The result of the function execution.

        Raises:
            Exception: Propagates any exception that occurs during the function's
                execution, after printing debug information.
        """
        if memory:
            to_hash = list(inputs.values())
            for i, e in enumerate(to_hash):
                # to avoid import numpy only for this test
                if e.__class__.__name__ == "ndarray":
                    to_hash[i] = e.tobytes()
            try:
                current_input_hash = hash(frozenset(to_hash))
            except TypeError:
                current_input_hash = None
        try:
            if not memory:
                # Prioritize runtime inputs over fixed_params in case of a name collision
                return self.func(**{**self.fixed_params, **inputs})
            elif self.output is None or current_input_hash is None or current_input_hash != self.input_hash_last_exec:
                self.output = self.func(**{**self.fixed_params, **inputs})
                self.input_hash_last_exec = current_input_hash
            return self.output
        except Exception as e:
            print(f"Error in executing node {self.id}: {e}")
            print("Node fixed parameters:", self.fixed_params)
            if inputs:
                print("Node inputs:", inputs)
            raise

    def get_fixed_params(self):
        """Returns the dictionary of fixed parameters."""
        return self.fixed_params

    def set_fixed_params(self, fixed_params):
        """
        Sets the fixed parameters for the node.

        Args:
            fixed_params (dict): A dictionary of parameters to set.
        """
        if not isinstance(fixed_params, dict):
            raise ValueError("Fixed parameters must be a dictionary.")
            
        for key, value in fixed_params.items():
            if not isinstance(key, str):
                raise ValueError(f"Key '{key}' is not a string.")
            if key not in self.fixed_params:
                raise ValueError(f"Key '{key}' is not a fixed parameter of node '{self.id}'.")
            self.fixed_params[key] = value
        # Clear memory once after all params are set for efficiency
        self.clear_memory()

    def set_fixed_param(self, key, value):
        """
        Sets a single fixed parameter.

        Args:
            key (str): The name of the parameter.
            value: The value of the parameter.

        Raises:
            ValueError: If the key is not an existing fixed parameter.
        """
        if key not in self.fixed_params:
            raise ValueError(f"Key '{key}' is not a fixed parameter of node '{self.id}'.")
        self.fixed_params[key] = value
        self.clear_memory()

    def is_fixed_param(self, key):
        """Checks if a parameter name is in the fixed parameters."""
        return key in self.fixed_params
        

class NodeIf(Node):
    """
    A conditional node that executes one of two sub-pipelines based on a condition.

    This node allows for branching logic within a pipeline. It evaluates a
    condition function and, based on the boolean result, runs either a
    'true_pipeline' or a 'false_pipeline'.

    Attributes:
        condition_func (callable): A function that returns a boolean value.
        true_pipeline (Pipeline): The pipeline to execute if the condition is True.
        false_pipeline (Pipeline): The pipeline to execute if the condition is False.
    """
    def __init__(self, id, condition_func, true_pipeline, false_pipeline, fixed_params=None):
        super().__init__(id, condition_func, fixed_params=fixed_params)
        self.true_pipeline = true_pipeline
        self.false_pipeline = false_pipeline
        self.skip_failed_images = False
        self.debug = False
    
    def set_run_params(self, skip_failed_images=False, debug=False):
        """
        Sets the run parameters for the sub-pipelines.

        Args:
            skip_failed_images (bool, optional): If True, execution of iterative nodes
                in sub-pipelines will continue even if one iteration fails.
                Defaults to False.
            debug (bool, optional): If True, enables debug printing for sub-pipelines.
                Defaults to False.
        """
        self.skip_failed_images = skip_failed_images
        self.debug = debug

    def execute(self, inputs={}, memory=False):
        """
        Evaluates the condition and executes the corresponding sub-pipeline.

        Inputs for the condition function must be prefixed with "condition_func:"
        in the predecessor mapping. The remaining inputs are passed to the
        chosen sub-pipeline as its `run_params`.

        Args:
            inputs (dict, optional): A dictionary of inputs. Defaults to {}.
            memory (bool, optional): If True, enables caching within the
                sub-pipelines. Defaults to False.

        Returns:
            The output of the final node of the executed sub-pipeline.
        """
        condition_inputs = {}
        for k in inputs:
            if k.startswith("condition_func:"):
                condition_inputs[k[15:]] = inputs[k]
        for k in condition_inputs:
            del inputs["condition_func:"+k]
        if self.func(**self.fixed_params, **condition_inputs):
            id, hist, _ = self.true_pipeline.run(run_params=inputs, optimize_memory= not memory, skip_failed_images=self.skip_failed_images, debug=self.debug)
        else:
            id, hist, _ = self.false_pipeline.run(run_params=inputs, optimize_memory= not memory, skip_failed_images=self.skip_failed_images, debug=self.debug)
        if memory:
            self.output = id, hist
        return hist[id]

    def get_fixed_params(self):
        """
        Gets the fixed parameters of the NodeIf and its sub-pipelines.

        Returns:
            dict: A dictionary containing the node's own fixed parameters and the
                  parameters of the true and false pipelines under the keys
                  "true_pipeline" and "false_pipeline".
        """
        # Returns the fixed parameters of the IF node: fixed_params + those of the pipelines
        true_fixed_params = self.true_pipeline.get_fixed_params()
        false_fixed_params = self.false_pipeline.get_fixed_params()
        return {**self.fixed_params, "true_pipeline": true_fixed_params, "false_pipeline": false_fixed_params}
    
    def set_fixed_params(self, fixed_params):
        """
        Sets the fixed parameters for the NodeIf and its sub-pipelines.
        It expects a dictionary that may contain "true_pipeline" and "false_pipeline" keys.
        """
        for key, value in fixed_params.items():
            if key == "true_pipeline":
                self.true_pipeline.set_fixed_params(value)
            elif key == "false_pipeline":
                self.false_pipeline.set_fixed_params(value)
            else:
                # Reuse the validation logic from set_fixed_param
                if key not in self.fixed_params:
                    raise ValueError(f"Key '{key}' is not a fixed parameter of node '{self.id}'.")
                self.fixed_params[key] = value
        self.clear_memory()
    
    def set_fixed_param(self, key, value):
        """
        Sets a single fixed parameter on the NodeIf or its sub-pipelines.
        For sub-pipelines, the key should be 'true_pipeline' or 'false_pipeline'
        and the value should be a dictionary of parameters.
        """
        if key == "true_pipeline":
            self.true_pipeline.set_fixed_params(value)
        elif key == "false_pipeline":
            self.false_pipeline.set_fixed_params(value)
        else:
            if key not in self.fixed_params:
                raise ValueError(f"Key '{key}' is not a fixed parameter of node '{self.id}'.")
            self.fixed_params[key] = value
        self.clear_memory()

class NodeFor(Node):
    """
    A node that executes a sub-pipeline for a given number of iterations.
    It implements a 'for' loop behavior, where the output of one iteration
    is the input for the next.

    Attributes:
        loop_pipeline (Pipeline): The pipeline to execute at each iteration.
    """
    def __init__(self, id, loop_pipeline, fixed_params=None):
        """
        Initializes a NodeFor.

        Args:
            id (str): The unique identifier for the node.
            loop_pipeline (Pipeline): The pipeline to execute in a loop.
            fixed_params (dict, optional): Fixed parameters for this node.
        """
        super().__init__(id, func=lambda **kwargs: kwargs, fixed_params=fixed_params)
        self.loop_pipeline = loop_pipeline
        self.skip_failed_images = False
        self.debug = False

    def set_run_params(self, skip_failed_images=False, debug=False):
        """
        Sets the run parameters for the sub-pipelines.

        Args:
            skip_failed_images (bool, optional): If True, execution will continue
                even if one iteration fails.
                Defaults to False.
            debug (bool, optional): If True, enables debug printing for sub-pipelines.
                Defaults to False.
        """
        self.skip_failed_images = skip_failed_images
        self.debug = debug

    def execute(self, inputs={}, memory=False):
        """
        Executes the loop. It requires an 'iterations' input for the number of loops,
        and a 'loop_var' input for the initial value that will be passed from one
        iteration to the next.

        Args:
            inputs (dict): Must contain 'iterations' (int) and 'loop_var' (any).
            memory (bool): Enables caching within the sub-pipeline.

        Returns:
            The output of the final iteration.
        """
        iterations = inputs.get('iterations', self.fixed_params.get('iterations'))
        if iterations is None:
            raise ValueError("NodeFor requires an 'iterations' input in 'inputs' or in 'fixed_params'.")
        
        current_loop_var = inputs.get('loop_var')
        if 'loop_var' not in inputs:
            raise ValueError("NodeFor requires a 'loop_var' input for the initial value.")

        for i in range(int(iterations)):
            if self.debug:
                print(f"\rExecuting node: {self.id} iteration {i+1}/{iterations}", end="")
            
            run_params = {'loop_var': current_loop_var, 'loop_index': i, **inputs, **self.fixed_params}

            try:
                last_node_id, hist, _ = self.loop_pipeline.run(
                    run_params=run_params,
                    optimize_memory=not memory,
                    skip_failed_images=self.skip_failed_images,
                    debug=self.debug
                )
                current_loop_var = hist[last_node_id]
            except Exception as e:
                if self.skip_failed_images:
                    print(f"Error in node {self.id} at iteration {i+1}/{iterations}: {e}")
                    continue
                raise e
        
        if self.debug:
            print()
        
        return current_loop_var

    def get_fixed_params(self):
        """
        Gets the fixed parameters of the NodeFor and its sub-pipeline.
        """
        params = {**self.fixed_params, "loop_pipeline": self.loop_pipeline.get_fixed_params()}
        params['iterations'] = self.iterations
        return params
    
    def set_fixed_params(self, fixed_params):
        """
        Sets the fixed parameters for the NodeFor and its sub-pipeline.
        """
        if 'iterations' in fixed_params:
            self.iterations = int(fixed_params.pop('iterations'))

        for key, value in fixed_params.items():
            if key == "loop_pipeline":
                self.loop_pipeline.set_fixed_params(value)
            else:
                if key not in self.fixed_params:
                    raise ValueError(f"Key '{key}' is not a fixed parameter of node '{self.id}'.")
                self.fixed_params[key] = value
        self.clear_memory()
    
    def set_fixed_param(self, key, value):
        """
        Sets a single fixed parameter on the NodeFor or its sub-pipeline.
        """
        if key == 'iterations':
            self.iterations = int(value)
        elif key == "loop_pipeline":
            self.loop_pipeline.set_fixed_params(value)
        else:
            if key not in self.fixed_params:
                raise ValueError(f"Key '{key}' is not a fixed parameter of node '{self.id}'.")
            self.fixed_params[key] = value
        self.clear_memory()
