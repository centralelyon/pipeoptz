from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
import numpy as np
import random as rd
from scipy.stats import norm
from .utils import product
from .parameter import IntParameter, FloatParameter, ChoiceParameter, BoolParameter, MultiChoiceParameter
from .pipeline import Pipeline


class PipelineOptimizer:
    """
    Provides a framework for optimizing pipeline parameters using various algorithms.

    This class defines the interface for different optimization strategies like
    Grid Search, Bayesian Optimization, Genetic Algorithms, etc. The methods
    are placeholders and should be implemented in subclasses or by integrating
    with external optimization libraries.

    Attributes:
        pipeline (Pipeline): The pipeline instance to be optimized.
        params_to_optimize (list): A list of Parameter objects to be tuned.
    """
    def __init__(self, pipeline, loss_function, max_time_pipeline, X, y):
        """
        Initializes a PipelineOptimizer.

        Args:
            pipeline (Pipeline): The pipeline instance to be optimized.
            loss_function (callable): A function that takes the pipeline's output and the
                expected output, and returns a numerical loss value.
            max_time_pipeline (float): The maximum time allowed for a single pipeline run (in seconds).
            X (list): A list of dictionaries, where each dictionary represents the `run_params`
                for a pipeline execution during optimization.
            y (list): A list of expected outputs corresponding to each `run_params` in `X`.
        """
        assert isinstance(pipeline, Pipeline), "pipeline must be an instance of Pipeline"
        assert callable(loss_function), "loss_function must be a callable function"
        assert isinstance(X, list), "X must be a list"
        assert isinstance(y, list), "y must be a list"
        assert len(X) == len(y), "X and y must have the same length"
        assert len(X) > 0, "X must not be empty"
        assert all(isinstance(run_params, dict) for run_params in X), "All items in X must be dictionaries"
        assert isinstance(max_time_pipeline, (int, float)), "max_time_pipeline must be a number"
        assert max_time_pipeline > 0, "max_time_pipeline must be a positive number"

        self.pipeline = pipeline
        self.params_to_optimize = []
        self.max_time_pipeline = max_time_pipeline
        self.X = X
        self.y = y
        self.loss = loss_function

    def add_param(self, param):
        """Adds a parameter to the list of parameters that the optimizer will tune."""
        self.params_to_optimize.append(param)

    def nb_params_possibilities(self):
        """Returns the total number of possible combinations of parameter values."""
        return np.prod([p.get_parametric_space()["range_size"] for p in self.params_to_optimize])

    def set_params(self, values:dict):
        """
        Sets the values of the parameters to optimize.

        Args:
            values (dict): A dictionary where keys are parameter names (node_id.param_name)
                               and values are the values to set.
        """
        for param_str, value in values.items():
            node_id, param_name = param_str.split('.')
            found = False
            for param_obj in self.params_to_optimize:
                if param_obj.node_id == node_id and param_obj.param_name == param_name:
                    param_obj.set_value(value)
                    found = True
                    break
            if not found:
                raise ValueError(f"Parameter {param_str} not found in the parameters to optimize.")

    def set_param(self, node_id, param_name, value):
        """
        Sets the value of a specific parameter in the pipeline.

        Args:
            node_id (str): The ID of the node containing the parameter.
            param_name (str): The name of the parameter to set.
            value: The value to set for the parameter.
        """
        for param in self.params_to_optimize:
            if param.node_id == node_id and param.param_name == param_name:
                param.set_value(value)
                return
        raise ValueError(f"Parameter {node_id}.{param_name} not found in the parameters to optimize.")

    def update_pipeline_params(self):
        """Updates the pipeline with the current parameter values."""
        params = {}
        for param in self.params_to_optimize:
            params[f"{param.node_id}.{param.param_name}"] = param.get_value()
        self.pipeline.set_fixed_params(params)

    def get_params_value(self):
        """
        Returns the current values of the parameters to optimize.

        Returns:
            dict: A dictionary where keys are parameter names (node_id.param_name)
                  and values are their current values.
        """
        values = {}
        for param in self.params_to_optimize:
            values[f"{param.node_id}.{param.param_name}"] = param.get_value()
        return values

    def evaluate(self):
        """
        Evaluates the current parameters of the pipeline.

        This method runs the pipeline with the current parameter values on the `X` and `y` for each run,
        and returns the outputs and the average loss compared to the expected values.

        Returns:
            dict: The outputs of the pipeline after running it with the current parameters.
        """
        self.update_pipeline_params()
        
        # Run the pipeline and return the outputs
        results = []
        loss = 0.
        for i, run_param in enumerate(self.X):
            index, res, t = self.pipeline.run(run_param, optimize_memory=True)
            results.append(res[index])
            loss += self.loss(results[-1], self.y[i])
            if self.max_time_pipeline not in (0, None, -1) and t[0] > self.max_time_pipeline:
                return results+[None]*(len(self.X)-i-1), float("inf")
        loss /= i+1
        return results, loss

    def optimize_ACO(self, iterations=100, ants=20, alpha=1.0, beta=1.0, evaporation_rate=0.3, param_sampling=20, verbose=False):
        """
        Ant Colony Optimization (ACO) with real use of beta for heuristic guidance.

        Args:
            iterations (int): Number of iterations.
            ants (int): Number of ants per iteration.
            alpha (float): Importance of pheromone.
            beta (float): Importance of heuristic (1 / estimated loss).
            evaporation_rate (float): Rate at which pheromones evaporate.
            param_sampling (int): Number of random values to sample for each parameter.

        Returns:
            best_params (dict): Best parameter configuration.
            loss_log (list): Best loss after each iteration.
        """
        # Generate candidate values
        param_names = [f"{p.node_id}.{p.param_name}" for p in self.params_to_optimize]
        domain_values = {name: set() for name in param_names}
        for p, n in zip(self.params_to_optimize, param_names):
            if isinstance(p, IntParameter) and (p.max_value - p.min_value < param_sampling-1):
                domain_values[n] = list(range(p.min_value, p.max_value + 1))
            elif isinstance(p, ChoiceParameter) and len(p.choices) < param_sampling:
                domain_values[n] = p.choices
            elif isinstance(p, BoolParameter):
                domain_values[n] = [True, False]
            else:
                while len(domain_values[n]) < param_sampling:
                    domain_values[n].add(p.get_random_value())
                domain_values[n] = list(domain_values[n])

        # Initialize pheromones and heuristic tables
        pheromones = {
            name: {val: 1.0 for val in domain_values[name]}
            for name in param_names
        }

        # Store previous heuristic (average inverse loss)
        heuristics = {
            name: {val: 1.0 for val in domain_values[name]}
            for name in param_names
        }

        best_params = None
        best_loss = float("inf")
        loss_log = []

        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}", end="\r") if verbose else None
            solutions = []
            losses = []

            for _ in range(ants):
                candidate = {}
                for name in param_names:
                    values = domain_values[name]
                    pher = np.array([pheromones[name][v] for v in values])
                    heur = np.array([heuristics[name][v] for v in values])
                    probs = (pher ** alpha) * (heur ** beta)
                    probs /= probs.sum()
                    selected = np.random.choice(values, p=probs)
                    candidate[name] = selected

                self.set_params(candidate)
                _, loss = self.evaluate()

                solutions.append(candidate)
                losses.append(loss)

                for name in param_names:
                    val = candidate[name]
                    heuristics[name][val] = max(1e-6, 1.0 / (1e-6 + loss))  # avoid the division by zero

                if loss <= best_loss:
                    best_loss = loss
                    best_params = candidate.copy()

            # Evaporation
            for name in pheromones:
                for val in pheromones[name]:
                    pheromones[name][val] *= (1.0 - evaporation_rate)

            # Pheromone deposit by the best ant
            best_idx = int(np.argmin(losses))
            for name, val in solutions[best_idx].items():
                pheromones[name][val] += 1.0 / (1.0 + losses[best_idx])

            loss_log.append(best_loss)

        self.set_params(best_params)
        self.update_pipeline_params()
        return best_params, loss_log

    def optimize_SA(self, iterations=100, initial_temp=1.0, cooling_rate=0.95, verbose=False):
        """
        Optimizes the pipeline using Simulated Annealing (SA).

        Args:
            iterations (int): Number of iterations.
            initial_temp (float): Initial temperature.
            cooling_rate (float): Cooling factor.

        Returns:
            dict: Best parameters found.
            list: Loss log over iterations.
        """
        current_params = {
            f"{p.node_id}.{p.param_name}": p.get_random_value()
            for p in self.params_to_optimize
        }
        self.set_params(current_params)
        _, current_loss = self.evaluate()

        best_params = current_params.copy()
        best_loss = current_loss

        temperature = initial_temp
        loss_log = [best_loss]

        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}", end="\r") if verbose else None
            candidate = current_params.copy()
            # Mutate a random parameter
            name = rd.choice(list(candidate.keys()))
            param = next(p for p in self.params_to_optimize if f"{p.node_id}.{p.param_name}" == name)
            candidate[name] = param.get_random_value()

            self.set_params(candidate)
            _, candidate_loss = self.evaluate()

            delta = candidate_loss - current_loss
            if delta < 0 or np.exp(-delta / temperature) > rd.random():
                current_params = candidate
                current_loss = candidate_loss
                if candidate_loss < best_loss:
                    best_loss = candidate_loss
                    best_params = candidate.copy()

            loss_log.append(best_loss)
            temperature *= cooling_rate

        self.set_params(best_params)
        self.update_pipeline_params()
        return best_params, loss_log

    def optimize_PSO(self, iterations=100, swarm_size=20, inertia=0.5, cognitive=1.5, social=1.5, verbose=False):
        """
        Optimizes the pipeline using Particle Swarm Optimization (PSO).

        Args:
            iterations (int): Number of iterations.
            swarm_size (int): Number of particles.
            inertia (float): Inertia weight.
            cognitive (float): Cognitive parameter.
            social (float): Social parameter.

        Returns:
            dict: Best parameters found.
            list: Loss log over iterations.
        """
        param_names = [f"{p.node_id}.{p.param_name}" for p in self.params_to_optimize]
        particles = []
        velocities = []
        personal_best = []
        personal_best_loss = []
        loss_log = []

        # Initialize swarm
        for _ in range(swarm_size):
            particle = {name: p.get_random_value() for name, p in zip(param_names, self.params_to_optimize)}
            velocity = {name: 0.0 for name in param_names}

            self.set_params(particle)
            _, loss = self.evaluate()

            particles.append(particle)
            velocities.append(velocity)
            personal_best.append(particle.copy())
            personal_best_loss.append(loss)

        best_loss = min(personal_best_loss)
        best_particle = personal_best[np.argmin(personal_best_loss)].copy()

        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}", end="\r") if verbose else None
            for i in range(swarm_size):
                new_particle = {}
                for name in param_names:
                    r1 = rd.random()
                    r2 = rd.random()
                    p_val = particles[i][name]
                    pb_val = personal_best[i][name]
                    gb_val = best_particle[name]

                    # Velocity update (heuristic): not strictly numeric, sample mutation
                    if p_val != pb_val:
                        velocities[i][name] = cognitive * r1
                    elif p_val != gb_val:
                        velocities[i][name] += social * r2
                    else:
                        velocities[i][name] *= inertia

                    if rd.random() < velocities[i][name]:
                        param = next(p for p in self.params_to_optimize if f"{p.node_id}.{p.param_name}" == name)
                        new_particle[name] = param.get_random_value()
                    else:
                        new_particle[name] = particles[i][name]

                self.set_params(new_particle)
                _, loss = self.evaluate()

                if loss < personal_best_loss[i]:
                    personal_best[i] = new_particle.copy()
                    personal_best_loss[i] = loss

                    if loss < best_loss:
                        best_loss = loss
                        best_particle = new_particle.copy()

                particles[i] = new_particle

            loss_log.append(best_loss)

        self.set_params(best_particle)
        self.update_pipeline_params()
        return best_particle, loss_log

    def optimize_GA(self, generations=50, population_size=20, mutation_rate=0.1, crossover_rate=0.7, verbose=False):
        """
        Optimizes the pipeline using Genetic Algorithm (GA).

        Args:
            generations (int): Number of generations.
            population_size (int): Number of individuals in the population.
            mutation_rate (float): Probability of mutation.
            crossover_rate (float): Probability of crossover.

        Returns:
            dict: Best parameters found.
            list: Loss log over generations.
        """
        param_names = [f"{p.node_id}.{p.param_name}" for p in self.params_to_optimize]
        loss_log = []

        def random_individual():
            return {name: p.get_random_value() for name, p in zip(param_names, self.params_to_optimize)}

        def crossover(parent1, parent2):
            return {
                name: parent1[name] if rd.random() < 0.5 else parent2[name]
                for name in param_names
            }

        def mutate(individual):
            for name in param_names:
                if rd.random() < mutation_rate:
                    param = next(p for p in self.params_to_optimize if f"{p.node_id}.{p.param_name}" == name)
                    individual[name] = param.get_random_value()
            return individual

        # Initial population
        population = [random_individual() for _ in range(population_size)]
        evaluated = []
        for ind in population:
            self.set_params(ind)
            _, loss = self.evaluate()
            evaluated.append((ind, loss))

        best_individual = min(evaluated, key=lambda x: x[1])[0].copy()
        best_loss = min(evaluated, key=lambda x: x[1])[1]

        for gen in range(generations):
            print(f"Generation {gen+1}/{generations}", end="\r") if verbose else None
            new_population = []
            evaluated.sort(key=lambda x: x[1])
            parents = [ind for ind, _ in evaluated[:population_size//2]]

            while len(new_population) < population_size:
                if rd.random() < crossover_rate:
                    p1, p2 = rd.sample(parents, 2)
                    child = crossover(p1, p2)
                    child = mutate(child)
                else:
                    child = mutate(rd.choice(parents).copy())
                new_population.append(child)

            evaluated = []
            for ind in new_population:
                self.set_params(ind)
                _, loss = self.evaluate()
                evaluated.append((ind, loss))

                if loss < best_loss:
                    best_loss = loss
                    best_individual = ind.copy()

            loss_log.append(best_loss)

        self.set_params(best_individual)
        self.update_pipeline_params()
        return best_individual, loss_log
    
    def optimize_grid_search(self, max_combinations=100, param_sampling=None, verbose=False):
        """
        Exhaustively searches all possible parameter combinations (within a limited budget).

        Args:
            max_combinations (int): Maximum number of combinations to evaluate.
            param_sampling (int): Number of random values to sample for each parameter.

        Returns:
            dict: Best parameters found.
            list: Loss log over evaluated combinations.
        """
        if param_sampling is None:
            param_sampling = max(10, max_combinations // len(self.params_to_optimize))
        param_names = [f"{p.node_id}.{p.param_name}" for p in self.params_to_optimize]
        param_values = []
        for p in self.params_to_optimize:
            if isinstance(p, IntParameter) and (p.max_value - p.min_value < param_sampling - 1):
                param_values.append(list(range(p.min_value, p.max_value + 1)))
            elif isinstance(p, ChoiceParameter) and len(p.choices) < param_sampling:
                param_values.append(p.choices)
            elif isinstance(p, BoolParameter):
                param_values.append([True, False])
            else:
                values= set()
                while len(values) < param_sampling:
                    values.add(p.get_random_value())
                param_values.append(list(values))

        # We rewrote the product function to handle large search spaces without explicitly listing them all at once.        
        if np.prod([len(v) for v in param_values])>>4 > max_combinations:
            combinations = product(*param_values, random=True, max_combinations=max_combinations, optimize_memory=True)
        else:
            combinations = product(*param_values, random=True, max_combinations=max_combinations)

        best_loss = float("inf")
        best_params = None
        loss_log = []

        for i, combo in enumerate(combinations):
            print(f"Iteration {i+1}/{max_combinations}", end="\r") if verbose else None
            params = dict(zip(param_names, combo))
            self.set_params(params)
            _, loss = self.evaluate()

            if loss <= best_loss:
                best_loss = loss
                best_params = params.copy()

            loss_log.append(best_loss)

        self.set_params(best_params)
        self.update_pipeline_params()
        return best_params, loss_log
   
    @staticmethod
    def _encode(params, param_defs):
        """ 
        Encodes a dictionary of parameters into a numpy array.
        This is used for optimization algorithms that require numerical input like optimize_BO.

        Args:
            params (dict): Dictionary of parameters where keys are parameter names
                           (node_id.param_name) and values are the parameter values.
            param_defs (list): List of tuples (name, Parameter) defining the parameters.

        Returns:
            numpy.ndarray: Encoded array of parameters.
        """
        encoded = []
        for name, p in param_defs:
            val = params[name]
            if isinstance(p, BoolParameter):
                encoded.append(int(val))
            elif isinstance(p, ChoiceParameter):
                encoded.append(p.choices.index(val))
            else:
                encoded.append(val)
        return np.array(encoded)
    
    @staticmethod
    def _decode(x, param_defs):
        """
        Decodes a numpy array into a dictionary of parameters.
        This is used for optimization algorithms that require numerical input like optimize_BO.

        Args:
            x (numpy.ndarray): Encoded array of parameters.
            param_defs (list): List of tuples (name, Parameter) defining the parameters.

        Returns:
            dict: Decoded dictionary of parameters.
        """
        params = {}
        for i, (name, p) in enumerate(param_defs):
            if isinstance(p, BoolParameter):
                params[name] = bool(round(x[i]))
            elif isinstance(p, ChoiceParameter):
                idx = int(round(np.clip(x[i], 0, len(p.choices)-1)))
                params[name] = p.choices[idx]
            elif isinstance(p, IntParameter):
                params[name] = int(round(np.clip(x[i], p.min_value, p.max_value)))
            elif isinstance(p, FloatParameter):
                params[name] = float(np.clip(x[i], p.min_value, p.max_value))
        return params

    def optimize_BO(self, iterations=50, init_points=5, noise_level=0, n_candidates=None, verbose=False):
        """
        Bayesian Optimization using Gaussian Process and Expected Improvement (EI),
        with input normalization and robust handling.

        Args:
            iterations (int): Number of optimization steps.
            init_points (int): Initial random samples before BO starts.
            noise_level (float): Noise level for the Gaussian Process. Lower it is more precise the model but can lead to overfitting.
            n_candidates (int): Number of candidates to evaluate at each step. Suggested to be 100 times the number of parameters.
                                if None, defaults to 100 * number of parameters.

        Returns:
            best_params (dict): Best parameter configuration.
            loss_log (list): Best loss per iteration.
        """
        from sklearn.exceptions import ConvergenceWarning
        import warnings
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        MAXFLOAT = 1e100

        # Filter out MultiChoiceParameter for simplicity in this implementation
        # MultiChoiceParameter is not directly supported by the current encoding/decoding for BO
        assert any(not isinstance(p, MultiChoiceParameter) for p in self.params_to_optimize), "MultiChoiceParameter is not supported in Bayesian Optimization yet."
        param_defs = [(f"{p.node_id}.{p.param_name}", p) for p in self.params_to_optimize]

        n_candidates = 100 * len(param_defs) if n_candidates is None else n_candidates

        # Initial random sampling
        X_raw = []
        Y = []
        for _ in range(init_points):
            sample = {}
            for name, p in param_defs:
                sample[name] = p.get_random_value()
            self.set_params(sample)
            _, loss = self.evaluate()
            X_raw.append(self._encode(sample, param_defs))
            Y.append(min(MAXFLOAT, loss))

        X_raw = np.array(X_raw)
        Y = np.array(Y)

        # Scale X
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        # GP model
        kernel = C(1.0) * Matern(nu=2.5)
        if noise_level > 0:
            kernel += WhiteKernel(noise_level=noise_level)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)

        best_idx = np.argmin(Y)
        best_x = X_raw[best_idx]
        best_loss = Y[best_idx]
        loss_log = [best_loss]

        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}", end="\r") if verbose else None
            gp.fit(X, Y)

            # Generate random candidates
            candidates_raw = []
            for _ in range(n_candidates):
                cand = {name: p.get_random_value() for name, p in param_defs}
                candidates_raw.append(self._encode(cand, param_defs))
            candidates_raw = np.array(candidates_raw)
            candidates = scaler.transform(candidates_raw)

            # EI (Expected Improvement)
            mu, sigma = gp.predict(candidates, return_std=True)
            sigma = np.maximum(sigma, 1e-8)
            improvement = best_loss - mu
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

            x_next = candidates_raw[np.argmax(ei)]
            params = self._decode(x_next, param_defs)
            self.set_params(params)
            _, loss = self.evaluate()

            # update
            X = np.vstack([X, scaler.transform([x_next])])
            Y = np.append(Y, min(MAXFLOAT, loss))

            if loss < best_loss:
                best_loss = loss
                best_x = x_next
            loss_log.append(best_loss)

        self.set_params(self._decode(best_x, param_defs))
        self.update_pipeline_params()
        return self._decode(best_x, param_defs), loss_log
        
    def optimize(self, method, verbose=False, **kwargs):
        """
        Optimizes the pipeline using the specified method.

        Args:
            method (str): The optimization method to use (e.g., "grid_search", "BO", "ACO", "SA", "GA").
            **kwargs: Additional arguments for the chosen optimization method.

        Returns:
            tuple: A tuple containing:
                - dict: The best parameters found.
                - list: A log of the loss values during optimization.
        """
        if method == "grid_search":
            return self.optimize_grid_search(verbose=verbose,**kwargs)
        elif method == "BO":
            return self.optimize_BO(verbose=verbose, **kwargs)
        elif method == "ACO":
            return self.optimize_ACO(verbose=verbose, **kwargs)
        elif method == "SA":
            return self.optimize_SA(verbose=verbose, **kwargs)
        elif method == "GA":
            return self.optimize_GA(verbose=verbose, **kwargs)
        elif method == "PSO":
            return self.optimize_PSO(verbose=verbose, **kwargs)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
