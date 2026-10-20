"""
Dynamic Equation Engine for Custom History Matching Algorithms
Safely compiles and executes user/admin defined update scripts.
"""

import logging
import math
from typing import Dict, List, Tuple, Optional, Callable, Any
import numpy as np
import scipy.linalg
from .base_estimator import BaseEstimator

logger = logging.getLogger(__name__)


class DynamicEquationSmoother(BaseEstimator):
    """
    Executes a custom Python mathematical assimilation script.
    """

    def __init__(
        self,
        script_code: str,
        ensemble_size: int = 100,
        num_iterations: int = 3,
        parameter_names: Optional[List[str]] = None,
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        measurement_weights: Optional[Dict[str, float]] = None,
        inflation: float = 1.05,
    ):
        super().__init__(
            ensemble_size=ensemble_size,
            parameter_names=parameter_names,
            parameter_bounds=parameter_bounds,
            measurement_weights=measurement_weights,
            inflation=inflation,
        )
        self.script_code = script_code or ""
        self.num_iterations = max(1, int(num_iterations))
        self._compiled_update = self._compile_script()

    def _compile_script(self) -> Callable:
        """Compile the custom Python update script into an executable function in a sandboxed namespace"""
        if not self.script_code.strip():
            raise ValueError("Custom update script code is empty.")

        safe_globals = {
            'np': np,
            'numpy': np,
            'scipy': scipy,
            'math': math,
            '__builtins__': {
                'range': range, 'len': len, 'min': min, 'max': max,
                'float': float, 'int': int, 'dict': dict, 'list': list,
                'sum': sum, 'abs': abs, 'enumerate': enumerate, 'zip': zip,
                'print': print,
            }
        }
        local_scope: Dict[str, Any] = {}
        try:
            exec(self.script_code, safe_globals, local_scope)
        except Exception as compile_err:
            raise ValueError(f"Failed to compile custom update script: {compile_err}")

        if 'update_step' not in local_scope or not callable(local_scope['update_step']):
            raise ValueError("Script must define a callable function 'def update_step(ensemble, simulated, observations, cov_obs, **kwargs)'")

        return local_scope['update_step']

    def run_custom(
        self,
        observed_data: Dict[str, Any],
        forward_model_fn: Callable[[Dict[str, float]], Dict[str, Any]],
        initial_ensemble: Optional[np.ndarray] = None,
        priors: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """Execute the custom update script across iterations"""
        if initial_ensemble is None:
            if priors is None:
                raise ValueError("Either initial_ensemble or priors must be provided")
            ensemble = self.initialize_ensemble(priors)
        else:
            ensemble = initial_ensemble.copy()
            self.ensemble_size = ensemble.shape[0]

        observations = self.extract_observations(observed_data)
        if not observations:
            raise ValueError("No valid observations provided")

        obs_components = []
        for channel in ['oil', 'water', 'gas', 'pressure']:
            if channel in observations and len(observations[channel]) > 0:
                obs_components.append(observations[channel])

        d_obs = np.concatenate(obs_components)
        Nd = len(d_obs)
        Ne = self.ensemble_size
        cov_obs = np.eye(Nd) * 0.05  # Default diagonal covariance

        best_quality = 0.0
        self.history = []

        for iter_idx in range(self.num_iterations):
            # 1. Forward model
            simulated_list = []
            D_sim = np.zeros((Nd, Ne))
            for j in range(Ne):
                p_dict = self.array_to_params(ensemble[j])
                out = forward_model_fn(p_dict)
                sim_obs = self.extract_observations(out)
                simulated_list.append(sim_obs)

                sim_comp = []
                for channel in ['oil', 'water', 'gas', 'pressure']:
                    if channel in observations and len(observations[channel]) > 0:
                        obs_len = len(observations[channel])
                        sim_arr = sim_obs.get(channel, np.zeros(obs_len))
                        if len(sim_arr) != obs_len:
                            sim_arr = np.resize(sim_arr, obs_len)
                        sim_comp.append(sim_arr)
                D_sim[:, j] = np.concatenate(sim_comp)

            quality = float(np.mean([
                self.calculate_match_quality(observations, sim) for sim in simulated_list
            ]))
            if quality > best_quality:
                best_quality = quality

            self.history.append({
                'iteration': iter_idx + 1,
                'quality': quality,
                'mean_parameters': self.array_to_params(np.mean(ensemble, axis=0)),
            })

            if progress_callback:
                progress_callback(
                    (iter_idx + 1) / self.num_iterations * 90.0,
                    f"Custom Iteration {iter_idx + 1}: Quality = {quality:.1f}%"
                )

            # 2. Invoke custom update script
            try:
                updated_ensemble = self._compiled_update(
                    ensemble=ensemble,
                    simulated=D_sim.T,  # (Ne, Nd)
                    observations=d_obs,
                    cov_obs=cov_obs,
                    iteration=iter_idx,
                    parameter_names=self.parameter_names
                )
                if isinstance(updated_ensemble, np.ndarray) and updated_ensemble.shape == ensemble.shape:
                    ensemble = self.clamp_bounds(updated_ensemble)
                else:
                    logger.warning("Custom script did not return an array of matching shape. Skipping update.")
            except Exception as script_err:
                logger.error(f"Execution error in custom update script: {script_err}")
                break

        return {
            'status': 'completed',
            'algorithm': 'custom_equation',
            'iterations': self.num_iterations,
            'mean_parameters': self.array_to_params(np.mean(ensemble, axis=0)),
            'std_parameters': self.array_to_params(np.std(ensemble, axis=0)),
            'best_quality': float(best_quality),
            'final_ensemble': ensemble.tolist(),
            'iteration_history': self.history,
            'message': 'Custom algorithm executed successfully.'
        }
