"""
Ensemble Smoother with Multiple Data Assimilation (ES-MDA)
Reference: Emerick, A. A., & Reynolds, A. C. (2013).
Ensemble smoother with multiple data assimilation. Computers & Geosciences, 55, 3-15.
"""

import logging
from typing import Dict, List, Tuple, Optional, Callable, Any
import numpy as np
from .base_estimator import BaseEstimator, DEFAULT_PARAMETER_BOUNDS

logger = logging.getLogger(__name__)


class ESMDASmoother(BaseEstimator):
    """
    Implements Ensemble Smoother with Multiple Data Assimilation (ES-MDA)
    Assimilates complete historical production data over Na iterations with covariance inflation.
    """

    def __init__(
        self,
        ensemble_size: int = 100,
        num_iterations: int = 4,
        alpha_schedule: str = "geometric",
        parameter_names: Optional[List[str]] = None,
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        measurement_weights: Optional[Dict[str, float]] = None,
        measurement_errors: Optional[Dict[str, float]] = None,
        inflation: float = 1.05,
    ):
        super().__init__(
            ensemble_size=ensemble_size,
            parameter_names=parameter_names,
            parameter_bounds=parameter_bounds,
            measurement_weights=measurement_weights,
            inflation=inflation,
        )
        self.num_iterations = max(1, int(num_iterations))
        self.alpha_schedule_type = alpha_schedule.lower() if alpha_schedule else "geometric"
        self.measurement_errors = measurement_errors or {
            'oil': 0.05,      # 5% relative error
            'water': 0.07,    # 7% relative error
            'gas': 0.10,      # 10% relative error
            'pressure': 0.03  # 3% relative error
        }
        self.alphas = self._compute_alpha_schedule()

    def _compute_alpha_schedule(self) -> List[float]:
        """
        Compute inflation coefficients alpha_l satisfying sum(1/alpha_l) = 1.
        """
        Na = self.num_iterations
        if Na == 1:
            return [1.0]

        if self.alpha_schedule_type == "uniform":
            # alpha_l = Na for all l
            return [float(Na)] * Na

        # Geometric decreasing schedule: larger steps initially, smaller later
        # alpha_l = alpha_0 * r^(l-1)
        r = 0.5
        factors = [r ** i for i in range(Na)]
        # sum(1 / (C * factors[i])) = 1 => (1/C) * sum(1 / factors[i]) = 1 => C = sum(1/factors[i])
        inv_sum = sum(1.0 / f for f in factors)
        alphas = [inv_sum * f for f in factors]
        logger.info(f"Computed ES-MDA geometric alpha schedule (Na={Na}): {[round(a, 3) for a in alphas]}")
        return alphas

    def run_esmda(
        self,
        observed_data: Dict[str, Any],
        forward_model_fn: Callable[[Dict[str, float]], Dict[str, Any]],
        initial_ensemble: Optional[np.ndarray] = None,
        priors: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute full ES-MDA history matching routine.
        """
        logger.info(f"Starting ES-MDA with Ne={self.ensemble_size}, Na={self.num_iterations}")

        # Initialize ensemble if not provided
        if initial_ensemble is None:
            if priors is None:
                raise ValueError("Either initial_ensemble or priors must be provided to ESMDASmoother")
            ensemble = self.initialize_ensemble(priors)
        else:
            ensemble = initial_ensemble.copy()
            self.ensemble_size = ensemble.shape[0]

        # Extract observations
        observations = self.extract_observations(observed_data)
        if not observations:
            raise ValueError("No observation channels (oil, water, gas, pressure) found in observed_data")

        # Flatten observations into single 1D vector d_obs
        obs_components = []
        error_components = []
        for channel in ['oil', 'water', 'gas', 'pressure']:
            if channel in observations and len(observations[channel]) > 0:
                arr = observations[channel]
                obs_components.append(arr)
                rel_err = float(self.measurement_errors.get(channel, 0.05))
                # Absolute std dev per observation point with a small minimum floor
                channel_err = np.maximum(arr * rel_err, 1e-3)
                error_components.append(channel_err)

        d_obs = np.concatenate(obs_components)
        d_err = np.concatenate(error_components)
        cov_D_diag = d_err ** 2  # Diagonal elements of CD
        Nd = len(d_obs)
        Ne = self.ensemble_size
        Np = len(self.parameter_names)

        self.history = []
        best_quality = 0.0
        best_ensemble = ensemble.copy()

        # Initial simulated output for prior ensemble
        prior_simulated = []
        for j in range(Ne):
            p_dict = self.array_to_params(ensemble[j])
            out = forward_model_fn(p_dict)
            prior_simulated.append(self.extract_observations(out))

        prior_quality = np.mean([
            self.calculate_match_quality(observations, sim) for sim in prior_simulated
        ])
        best_quality = float(prior_quality)
        self.history.append({
            'iteration': 0,
            'quality': float(prior_quality),
            'mean_parameters': self.array_to_params(np.mean(ensemble, axis=0)),
            'std_parameters': self.array_to_params(np.std(ensemble, axis=0)),
        })

        if progress_callback:
            progress_callback(10.0, f"Prior Ensemble Initialized: Match Quality = {prior_quality:.1f}%")

        # ES-MDA Iteration Loop
        current_simulated = prior_simulated
        for iter_idx in range(self.num_iterations):
            alpha_l = self.alphas[iter_idx]
            step_progress = 10.0 + (iter_idx + 1) / self.num_iterations * 85.0
            logger.info(f"ES-MDA Iteration {iter_idx + 1}/{self.num_iterations} (alpha = {alpha_l:.2f})")

            # 1. Build simulated observation matrix D_sim of shape (Nd, Ne)
            D_sim = np.zeros((Nd, Ne))
            for j in range(Ne):
                sim_dict = current_simulated[j]
                sim_comp = []
                for channel in ['oil', 'water', 'gas', 'pressure']:
                    if channel in observations and len(observations[channel]) > 0:
                        obs_len = len(observations[channel])
                        sim_arr = sim_dict.get(channel, np.zeros(obs_len))
                        if len(sim_arr) != obs_len and len(sim_arr) > 0:
                            sim_arr = np.interp(
                                np.linspace(0, 1, obs_len),
                                np.linspace(0, 1, len(sim_arr)),
                                sim_arr
                            )
                        elif len(sim_arr) != obs_len:
                            sim_arr = np.resize(sim_arr, obs_len)
                        sim_comp.append(sim_arr)
                D_sim[:, j] = np.concatenate(sim_comp)

            # 2. Parameter anomalies X (Np x Ne) and simulated observation anomalies Y (Nd x Ne)
            m_mean = np.mean(ensemble, axis=0, keepdims=True)  # (1, Np)
            d_mean = np.mean(D_sim, axis=1, keepdims=True)     # (Nd, 1)

            X = (ensemble - m_mean).T  # (Np, Ne)
            Y = D_sim - d_mean         # (Nd, Ne)

            # 3. Covariances
            # CMD = (X @ Y.T) / (Ne - 1)
            # CDD = (Y @ Y.T) / (Ne - 1)
            # Regularized Kalman gain in subspace:
            # We solve: (Y @ Y.T + (Ne - 1) * alpha_l * diag(cov_D_diag)) @ Gamma = Y
            # Then K = X @ Gamma^T
            # For numerical stability and large Nd, use SVD / solve formulation:
            try:
                # Diagonal scaling
                inv_std = 1.0 / np.sqrt(cov_D_diag * alpha_l)
                Y_scaled = Y * inv_std[:, np.newaxis]  # (Nd, Ne)

                # Low-rank subspace inversion (Ne x Ne)
                ensemble_system = np.eye(Ne) * (Ne - 1) + Y_scaled.T @ Y_scaled
                inv_system = np.linalg.pinv(ensemble_system)

                # Perturbed observations for all ensemble members: d_uc,j = d_obs + sqrt(alpha_l) * CD^(1/2) * z_j
                perturbations = np.random.normal(0.0, 1.0, (Nd, Ne)) * np.sqrt(cov_D_diag * alpha_l)[:, np.newaxis]
                D_uc = d_obs[:, np.newaxis] + perturbations

                # Innovation matrix (Nd x Ne)
                Innovation = D_uc - D_sim

                # Scaled innovation: (Nd x Ne)
                scaled_innov = Innovation * inv_std[:, np.newaxis]

                # Update step: delta_M = X @ inv_system @ Y_scaled.T @ scaled_innov
                delta_M = (X @ inv_system) @ (Y_scaled.T @ scaled_innov)  # (Np, Ne)

                # Update ensemble realizations: M_new = M + delta_M.T
                updated_ensemble = ensemble + delta_M.T

                # Spread preservation via multiplicative inflation
                if self.inflation > 1.0:
                    new_mean = np.mean(updated_ensemble, axis=0, keepdims=True)
                    updated_ensemble = new_mean + self.inflation * (updated_ensemble - new_mean)

                # Strict projection onto physical parameter box constraints
                ensemble = self.clamp_bounds(updated_ensemble)

            except Exception as update_err:
                logger.error(f"ES-MDA matrix update failed at iteration {iter_idx + 1}: {update_err}")

            # Re-run forward model for all members with updated parameters
            current_simulated = []
            for j in range(Ne):
                p_dict = self.array_to_params(ensemble[j])
                out = forward_model_fn(p_dict)
                current_simulated.append(self.extract_observations(out))

            iter_quality = np.mean([
                self.calculate_match_quality(observations, sim) for sim in current_simulated
            ])

            if iter_quality > best_quality:
                best_quality = float(iter_quality)
                best_ensemble = ensemble.copy()

            self.history.append({
                'iteration': iter_idx + 1,
                'alpha': float(alpha_l),
                'quality': float(iter_quality),
                'mean_parameters': self.array_to_params(np.mean(ensemble, axis=0)),
                'std_parameters': self.array_to_params(np.std(ensemble, axis=0)),
            })

            if progress_callback:
                progress_callback(
                    step_progress,
                    f"ES-MDA Step {iter_idx + 1}/{self.num_iterations}: Match Quality = {iter_quality:.1f}%"
                )

        final_mean = np.mean(ensemble, axis=0)
        final_std = np.std(ensemble, axis=0)

        results = {
            'status': 'completed',
            'algorithm': 'esmda',
            'iterations': self.num_iterations,
            'ensemble_size': self.ensemble_size,
            'mean_parameters': self.array_to_params(final_mean),
            'std_parameters': self.array_to_params(final_std),
            'best_quality': float(best_quality),
            'prior_quality': float(prior_quality),
            'final_ensemble': ensemble.tolist(),
            'covariance_matrix': np.cov(ensemble.T).tolist() if Np > 1 else [[float(np.var(ensemble))]],
            'iteration_history': self.history,
            'message': f"ES-MDA completed with {self.num_iterations} data assimilation iterations."
        }
        logger.info(f"ES-MDA finished successfully. Best Quality: {best_quality:.2f}%")
        return results
