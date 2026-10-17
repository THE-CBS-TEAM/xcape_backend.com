"""
Ensemble Kalman Filter (EnKF) Algorithm
Advanced automated calibration using Ensemble Kalman Filter
"""

import logging
from typing import Dict, List, Tuple, Optional, Callable, Union
import numpy as np
from datetime import datetime
from scipy import stats

logger = logging.getLogger(__name__)


class EnKFFilter:
    """Implements Ensemble Kalman Filter for parameter estimation

    Enhancements:
    - Supports measurement weighting (oil/water/gas/pressure)
    - Applies a simple multiplicative inflation after each update
    - (Hook for localization can be added later)
    """

    def __init__(self, ensemble_size: int = 50, inflation: float = 1.05, measurement_weights: Optional[Dict] = None):
        """
        Initialize EnKF Filter

        Args:
            ensemble_size: Number of ensemble members
            inflation: Multiplicative inflation factor applied to ensemble anomalies after update
            measurement_weights: Optional dict of weights for observation types (e.g. {'oil':0.5,'water':0.2,...})
        """
        self.ensemble_size = ensemble_size
        self.ensemble_history = []
        self.parameter_history = []
        self.inflation = float(inflation)
        self.measurement_weights = measurement_weights or None
    
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
        
        Args:
            observed_data: Observed measurements
            forward_model_fn: Forward model function
            initial_ensemble: Initial ensemble of parameters (N x M)
                             N = ensemble size, M = number of parameters
            measurement_error: Measurement error std dev for each observation
            num_iterations: Number of EnKF iterations
            progress_callback: Function for progress updates
            
        Returns:
            EnKF results with updated parameters and uncertainty
        """
        logger.info(f"Starting EnKF with {self.ensemble_size} ensemble members")
        
        # Initialize
        ensemble = initial_ensemble.copy()
        observations = self._extract_observations(observed_data)
        
        if measurement_error is None:
            # Default: use tighter 2% measurement error but ensure a small floor
            measurement_error = {k: max(np.max(v) * 0.02, 1e-3) 
                                for k, v in observations.items()}

        # Use measurement weights passed to run or default from the filter
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
                params_dict = self._array_to_params(params)
                sim_output = forward_model_fn(params_dict)
                simulated_data.append(self._extract_observations(sim_output))
            
            # Calculate match quality
            quality = self._calculate_ensemble_quality(observations, simulated_data)
            
            if quality > best_quality:
                best_quality = quality
                best_params = np.mean(ensemble, axis=0)
            
            # Update progress
            if progress_callback:
                progress_callback(
                    (iteration + 1) / num_iterations * 100,
                    f"Iteration {iteration + 1}: Quality = {quality:.2f}%"
                )
            
            # Perform update step
            if iteration < num_iterations - 1:  # No update on last iteration
                ensemble = self._update_ensemble(
                    ensemble,
                    observations,
                    simulated_data,
                    measurement_error
                )
            
            # Store history
            self.ensemble_history.append({
                'iteration': iteration,
                'ensemble': ensemble.copy(),
                'quality': quality
            })
        
        # Calculate statistics
        mean_params = np.mean(ensemble, axis=0)
        std_params = np.std(ensemble, axis=0)
        
        results = {
            'status': 'completed',
            'iterations': num_iterations,
            'ensemble_size': self.ensemble_size,
            'mean_parameters': mean_params.tolist(),
            'std_parameters': std_params.tolist(),
            'best_quality': float(best_quality),
            'final_ensemble': ensemble.tolist(),
            'covariance_matrix': np.cov(ensemble.T).tolist(),
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
        
        Args:
            ensemble: Current ensemble (N x M)
            observations: Dictionary of observed data
            simulated_data: List of simulated dictionaries
            measurement_error: Measurement error dict
            
        Returns:
            Updated ensemble
        """
        N = ensemble.shape[0]
        M = ensemble.shape[1]   # Number of parameters
        
        # Concatenate all observations into single vector
        obs_vec = np.concatenate([observations[k] for k in observations.keys()])
        
        # Build simulated observation matrix (H x N)
        H_matrix = []
        for sim_dict in simulated_data:
            sim_components = []
            for k in observations.keys():
                obs_arr = observations[k]
                # get simulated array or zeros if missing
                sim_arr = np.array(sim_dict.get(k, np.zeros(len(obs_arr))), dtype=float)
                # If lengths differ, resample/interpolate simulated to match observed length
                if len(sim_arr) != len(obs_arr) and len(sim_arr) > 0 and len(obs_arr) > 0:
                    try:
                        sim_arr = np.interp(
                            np.linspace(0, 1, len(obs_arr)),
                            np.linspace(0, 1, len(sim_arr)),
                            sim_arr
                        )
                    except Exception:
                        # fallback to truncation or padding
                        if len(sim_arr) > len(obs_arr):
                            sim_arr = sim_arr[:len(obs_arr)]
                        else:
                            pad = np.zeros(len(obs_arr) - len(sim_arr))
                            sim_arr = np.concatenate([sim_arr, pad])
                # ensure correct length
                if len(sim_arr) != len(obs_arr):
                    sim_arr = np.resize(sim_arr, len(obs_arr))
                sim_components.append(sim_arr)
            sim_vec = np.concatenate(sim_components)
            H_matrix.append(sim_vec)
        H_matrix = np.array(H_matrix).T  # H x N
        
        # Calculate mean simulated observations
        h_mean = np.mean(H_matrix, axis=1, keepdims=True)
        
        # Anomaly matrices
        Y_anom = H_matrix - h_mean  # H x N (observation space anomalies)
        X_anom = (ensemble - np.mean(ensemble, axis=0, keepdims=True)).T  # M x N (parameter space anomalies)
        
        # Measurement error covariance matrix - one error per observation (not per type)
        # Create error array for each observation point
        error_values = []
        for k in observations.keys():
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
            
            # Update each ensemble member
            updated_ensemble = ensemble.copy()
            for i in range(N):
                # Innovation: real observations - simulated for this member
                innovation = obs_vec - H_matrix[:, i]
                updated_ensemble[i, :] += K_gain @ innovation
            # Apply simple multiplicative inflation to maintain ensemble spread
            try:
                # Basic sanity clamps per-parameter to avoid extreme/unphysical values
                # Parameter order: ['initial_pressure', 'porosity', 'permeability', 'water_saturation']
                # Apply Kalman update first, then inflation, then clamp
                mean_vec = np.mean(updated_ensemble, axis=0, keepdims=True)
                anomalies = updated_ensemble - mean_vec
                updated_ensemble = mean_vec + self.inflation * anomalies

                # Clamp physically meaningful bounds to avoid collapse to zeros or NaNs
                # initial_pressure: [100, 10000]
                updated_ensemble[:, 0] = np.clip(updated_ensemble[:, 0], 100.0, 10000.0)
                # porosity: expect fraction (0.01 - 0.5) but allow percent-style values up to 50
                updated_ensemble[:, 1] = np.where(updated_ensemble[:, 1] > 5.0,
                                                  np.clip(updated_ensemble[:, 1] / 100.0, 0.01, 0.5),
                                                  np.clip(updated_ensemble[:, 1], 0.01, 0.5))
                # permeability: [0.1, 1e6]
                updated_ensemble[:, 2] = np.clip(updated_ensemble[:, 2], 0.1, 1e6)
                # water_saturation: expect fraction (0-1) possibly expressed as percent
                updated_ensemble[:, 3] = np.where(updated_ensemble[:, 3] > 5.0,
                                                  np.clip(updated_ensemble[:, 3] / 100.0, 0.0, 1.0),
                                                  np.clip(updated_ensemble[:, 3], 0.0, 1.0))

            except Exception:
                # If anything goes wrong with inflation/clamping, return uninflated ensemble
                logger.warning("Inflation/clamping step failed; returning uninflated ensemble")

            return updated_ensemble
            
        except np.linalg.LinAlgError as e:
            logger.warning(f"Singular matrix in Kalman update: {e}. Using ensemble mean.")
            return ensemble
    
    def _calculate_ensemble_quality(self, observations: Dict, 
                                   simulated_data: List[Dict]) -> float:
        """Calculate match quality for entire ensemble"""
        qualities = []
        
        for sim_dict in simulated_data:
            quality = self._calculate_match_quality(observations, sim_dict)
            qualities.append(quality)
        
        return float(np.mean(qualities))
    
    def _calculate_match_quality(self, observed: Dict, simulated: Dict) -> float:
        """Calculate match quality between observed and simulated data"""
        try:
            errors = []
            
            # Determine weights: prefer filter-level weights if set, otherwise equal weights
            keys = [k for k in observed.keys() if k in simulated]
            if not keys:
                return 0.0

            if self.measurement_weights:
                weights = {k: float(self.measurement_weights.get(k, 1.0)) for k in keys}
            else:
                weights = {k: 1.0 for k in keys}

            # Normalize weights
            total_w = sum(weights.values()) if sum(weights.values()) > 0 else len(weights)

            weighted_score = 0.0
            for key in keys:
                obs = np.array(observed[key])
                sim = np.array(simulated[key])

                # Interpolate if lengths differ
                if len(obs) != len(sim):
                    sim = np.interp(np.linspace(0, 1, len(obs)),
                                   np.linspace(0, 1, len(sim)), sim)

                # Normalized RMSE
                rmse = np.sqrt(np.mean((obs - sim) ** 2))
                nrmse = rmse / (np.max(obs) - np.min(obs)) if np.ptp(obs) > 0 else 1.0
                score = max(0, 100 * (1 - nrmse))
                weighted_score += score * (weights[key] / total_w)

            return float(weighted_score)
            
        except Exception as e:
            logger.error(f"Error calculating quality: {e}")
            return 0.0
    
    def _extract_observations(self, data: Dict) -> Dict:
        """Extract observations from data"""
        try:
            # Check if data already has oil/water/gas/pressure keys (format from run_enkf_with_forecasts)
            if all(key in data for key in ['oil', 'water', 'gas', 'pressure']):
                return {
                    'oil': np.array(data['oil'], dtype=float),
                    'water': np.array(data['water'], dtype=float),
                    'gas': np.array(data['gas'], dtype=float),
                    'pressure': np.array(data['pressure'], dtype=float)
                }
            
            # Check if data has production_data sub-dict
            if 'production_data' in data:
                prod_data = data['production_data']
                return {
                    'oil': np.array(prod_data.get('oil', prod_data.get('Oil_bbl', [])), dtype=float),
                    'water': np.array(prod_data.get('water', prod_data.get('Water_bbl', [])), dtype=float),
                    'gas': np.array(prod_data.get('gas', prod_data.get('Gas_scf', [])), dtype=float),
                    'pressure': np.array(prod_data.get('pressure', prod_data.get('Pressure_psi', [])), dtype=float)
                }
            
            # Last resort fallback if data is completely empty
            logger.warning("No observation data found in extract_observations. Using empty dict.")
            return {}
        except Exception as e:
            logger.warning(f"Could not extract observations: {e}")
            return {}
    
    def _array_to_params(self, param_array: np.ndarray) -> Dict:
        """Convert parameter array to dictionary"""
        param_names = ['initial_pressure', 'porosity', 'permeability', 'water_saturation']
        return {name: float(value) for name, value in zip(param_names, param_array)}
    
    def initialize_ensemble(
        self,
        mean_params: Dict,
        std_params: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Initialize ensemble from mean and std of parameters
        
        Args:
            mean_params: Mean parameter values
            std_params: Standard deviation of parameters (default 10%)
            
        Returns:
            Ensemble matrix (ensemble_size x num_params)
        """
        if std_params is None:
            std_params = {k: v * 0.1 for k, v in mean_params.items()}
        
        param_names = list(mean_params.keys())
        means = np.array([mean_params[k] for k in param_names])
        stds = np.array([std_params.get(k, means[i] * 0.1) 
                        for i, k in enumerate(param_names)])
        
        # Generate ensemble from normal distribution
        ensemble = np.zeros((self.ensemble_size, len(param_names)))
        for i in range(len(param_names)):
            ensemble[:, i] = np.random.normal(means[i], stds[i], self.ensemble_size)
        
        return ensemble
    
    def get_history(self) -> List[Dict]:
        """Get EnKF iteration history"""
        return self.ensemble_history
