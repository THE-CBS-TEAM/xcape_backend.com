"""
Ensemble Kalman Filter (EnKF) Algorithm
Advanced automated calibration using Ensemble Kalman Filter
"""

import logging
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import numpy as np
from datetime import datetime
from scipy import stats
from .base_estimator import BaseEstimator, DEFAULT_PARAMETER_BOUNDS

logger = logging.getLogger(__name__)


class EnKFFilter(BaseEstimator):
    """
    Implements Ensemble Kalman Filter for parameter estimation.
    Supports dynamic parameter vectors, dynamic prior bounds, measurement weighting,
    and multiplicative inflation.
    """

    def __init__(
        self,
        ensemble_size: int = 50,
        inflation: float = 1.05,
        measurement_weights: Optional[Dict] = None,
        parameter_names: Optional[List[str]] = None,
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        super().__init__(
            ensemble_size=ensemble_size,
            parameter_names=parameter_names or ['initial_pressure', 'porosity', 'permeability', 'water_saturation'],
            parameter_bounds=parameter_bounds,
            measurement_weights=measurement_weights,
            inflation=inflation,
        )
        self.ensemble_history: List[Dict[str, Any]] = []
        self.parameter_history: List[Dict[str, Any]] = []
    
    def run_enkf(
        self,
        observed_data: Dict,
        forward_model_fn: Callable,
        initial_ensemble: np.ndarray,
        measurement_error: Optional[Dict] = None,
        measurement_weights: Optional[Dict] = None,
        num_iterations: int = 10,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Run Ensemble Kalman Filter
        """
        logger.info(f"Starting EnKF with {self.ensemble_size} ensemble members")
        
        ensemble = initial_ensemble.copy()
        observations = self.extract_observations(observed_data)
        
        if measurement_error is None:
            measurement_error = {k: max(np.max(v) * 0.02, 1e-3) 
                                for k, v in observations.items() if len(v) > 0}

        if measurement_weights is not None:
            self.measurement_weights = measurement_weights
        
        best_quality = 0.0
        best_params = None
        
        # Main EnKF loop
        for iteration in range(num_iterations):
            logger.info(f"EnKF iteration {iteration + 1}/{num_iterations}")
            
            # Run forward model for all ensemble members
            simulated_data = []
            for i, params in enumerate(ensemble):
                params_dict = self.array_to_params(params)
                sim_output = forward_model_fn(params_dict)
                simulated_data.append(self.extract_observations(sim_output))
            
            # Calculate match quality
            quality = self._calculate_ensemble_quality(observations, simulated_data)
            
            if quality > best_quality:
                best_quality = quality
                best_params = np.mean(ensemble, axis=0)
            
            if progress_callback:
                progress_callback(
                    (iteration + 1) / num_iterations * 100,
                    f"Iteration {iteration + 1}: Quality = {quality:.2f}%"
                )
            
            # Perform update step
            if iteration < num_iterations - 1:
                ensemble = self._update_ensemble(
                    ensemble,
                    observations,
                    simulated_data,
                    measurement_error
                )
            
            self.ensemble_history.append({
                'iteration': iteration,
                'ensemble': ensemble.copy(),
                'quality': quality
            })
        
        mean_params = np.mean(ensemble, axis=0)
        std_params = np.std(ensemble, axis=0)
        
        results = {
            'status': 'completed',
            'algorithm': 'enkf',
            'iterations': num_iterations,
            'ensemble_size': self.ensemble_size,
            'mean_parameters': self.array_to_params(mean_params),
            'std_parameters': self.array_to_params(std_params),
            'best_quality': float(best_quality),
            'final_ensemble': ensemble.tolist(),
            'covariance_matrix': np.cov(ensemble.T).tolist() if len(self.parameter_names) > 1 else [[float(np.var(ensemble))]],
            'message': 'EnKF calibration completed'
        }
        
        logger.info(f"EnKF completed. Best quality: {best_quality:.2f}%")
        return results
    
    def _update_ensemble(
        self,
        ensemble: np.ndarray,
        observations: Dict,
        simulated_data: List[Dict],
        measurement_error: Dict
    ) -> np.ndarray:
        """
        Update ensemble using Kalman update equations
        """
        N = ensemble.shape[0]
        obs_keys = [k for k in ['oil', 'water', 'gas', 'pressure'] if k in observations and len(observations[k]) > 0]
        if not obs_keys:
            return ensemble

        obs_vec = np.concatenate([observations[k] for k in obs_keys])
        
        H_matrix = []
        for sim_dict in simulated_data:
            sim_components = []
            for k in obs_keys:
                obs_arr = observations[k]
                sim_arr = np.array(sim_dict.get(k, np.zeros(len(obs_arr))), dtype=float)
                if len(sim_arr) != len(obs_arr) and len(sim_arr) > 0 and len(obs_arr) > 0:
                    sim_arr = np.interp(
                        np.linspace(0, 1, len(obs_arr)),
                        np.linspace(0, 1, len(sim_arr)),
                        sim_arr
                    )
                elif len(sim_arr) != len(obs_arr):
                    sim_arr = np.resize(sim_arr, len(obs_arr))
                sim_components.append(sim_arr)
            sim_vec = np.concatenate(sim_components)
            H_matrix.append(sim_vec)
        H_matrix = np.array(H_matrix).T  # H x N
        
        h_mean = np.mean(H_matrix, axis=1, keepdims=True)
        Y_anom = H_matrix - h_mean
        X_anom = (ensemble - np.mean(ensemble, axis=0, keepdims=True)).T
        
        error_values = []
        for k in obs_keys:
            n_obs_type = len(observations[k])
            error_std = measurement_error.get(k, 0.1)
            error_values.extend([error_std] * n_obs_type)
        
        error_variances = np.array(error_values, dtype=float) ** 2

        try:
            weighted_y_anom = Y_anom / error_variances[:, np.newaxis]
            ensemble_system = np.eye(N) + Y_anom.T @ weighted_y_anom
            gain_projection = np.linalg.solve(
                ensemble_system,
                Y_anom.T / error_variances[np.newaxis, :]
            )
            K_gain = X_anom @ gain_projection
            
            updated_ensemble = ensemble.copy()
            for i in range(N):
                innovation = obs_vec - H_matrix[:, i]
                updated_ensemble[i, :] += K_gain @ innovation

            # Multiplicative inflation
            if self.inflation > 1.0:
                mean_vec = np.mean(updated_ensemble, axis=0, keepdims=True)
                updated_ensemble = mean_vec + self.inflation * (updated_ensemble - mean_vec)

            # Strict dynamic box clamping using BaseEstimator bounds
            return self.clamp_bounds(updated_ensemble)
            
        except np.linalg.LinAlgError as e:
            logger.warning(f"Singular matrix in Kalman update: {e}. Using ensemble mean.")
            return ensemble
    
    def _calculate_ensemble_quality(self, observations: Dict, simulated_data: List[Dict]) -> float:
        """Calculate match quality for entire ensemble"""
        qualities = [self.calculate_match_quality(observations, sim) for sim in simulated_data]
        return float(np.mean(qualities)) if qualities else 0.0

    def _array_to_params(self, param_array: np.ndarray) -> Dict:
        """Backward-compatibility alias"""
        return self.array_to_params(param_array)

    def _extract_observations(self, data: Dict) -> Dict:
        """Backward-compatibility alias"""
        return self.extract_observations(data)
    
    def get_history(self) -> List[Dict]:
        """Get EnKF iteration history"""
        return self.ensemble_history
