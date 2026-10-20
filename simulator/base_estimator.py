"""
Base Parameter Estimator Interface for Reservoir History Matching
Provides unified ensemble initialization, parameter clamping, observation extraction,
and metric calculation across EnKF, ES-MDA, ES, and custom equation engines.
"""

import logging
from typing import Dict, List, Tuple, Optional, Callable, Any
import numpy as np

logger = logging.getLogger(__name__)


# Standard physical bounds across all common reservoir parameters
DEFAULT_PARAMETER_BOUNDS = {
    'permeability': (0.1, 10000.0),       # mD
    'porosity': (0.01, 0.45),              # fraction (0.01 - 0.45)
    'initial_pressure': (500.0, 15000.0),  # psi
    'water_saturation': (0.02, 0.95),      # fraction (0.02 - 0.95)
    'thickness': (2.0, 1000.0),            # ft (net pay)
    'area': (5.0, 10000.0),                # acres (drainage area)
    'rock_compressibility': (1e-7, 1e-4),  # 1/psi
    'oil_viscosity': (0.1, 200.0),         # cp
    'well_productivity_index': (0.05, 100.0), # STB/d/psi
}


class BaseEstimator:
    """Base class for history matching and data assimilation estimators"""

    def __init__(
        self,
        ensemble_size: int = 100,
        parameter_names: Optional[List[str]] = None,
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        measurement_weights: Optional[Dict[str, float]] = None,
        inflation: float = 1.05,
    ):
        self.ensemble_size = int(ensemble_size)
        self.parameter_names = parameter_names or [
            'initial_pressure', 'porosity', 'permeability', 'water_saturation'
        ]
        self.parameter_bounds = parameter_bounds or {}
        # Ensure default bounds for all specified parameters
        for p in self.parameter_names:
            if p not in self.parameter_bounds:
                self.parameter_bounds[p] = DEFAULT_PARAMETER_BOUNDS.get(p, (0.001, 1e6))

        self.measurement_weights = measurement_weights or {
            'oil': 1.0, 'water': 1.0, 'gas': 1.0, 'pressure': 1.0
        }
        self.inflation = float(inflation)
        self.history: List[Dict[str, Any]] = []

    def initialize_ensemble(
        self,
        priors: Dict[str, Any],
        std_fractions: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Generate initial prior parameter realizations.
        priors: dict mapping param_name -> prior_mean (or dict with 'mean' and 'std')
        """
        std_fractions = std_fractions or {}
        n_params = len(self.parameter_names)
        ensemble = np.zeros((self.ensemble_size, n_params))

        for idx, param in enumerate(self.parameter_names):
            val = priors.get(param, 1.0)
            if isinstance(val, dict):
                mean = float(val.get('mean', val.get('prior_mean', 1.0)))
                std = float(val.get('std', mean * 0.15))
            else:
                mean = float(val) if val is not None else 1.0
                std_frac = std_fractions.get(param, 0.15)
                std = max(abs(mean * std_frac), 1e-4)

            # Normal distribution with standard deviation
            samples = np.random.normal(mean, std, self.ensemble_size)
            ensemble[:, idx] = samples

        # Enforce physical box bounds on initial prior
        return self.clamp_bounds(ensemble)

    def clamp_bounds(self, ensemble: np.ndarray) -> np.ndarray:
        """Strictly project ensemble realizations onto specified parameter box constraints"""
        clamped = ensemble.copy()
        for idx, param in enumerate(self.parameter_names):
            min_b, max_b = self.parameter_bounds.get(param, DEFAULT_PARAMETER_BOUNDS.get(param, (0.001, 1e6)))

            # Normalize percent values if needed (porosity and water_saturation)
            if param in ('porosity', 'water_saturation') and min_b < 1.0 and max_b <= 1.0:
                # If someone entered 20 instead of 0.2, normalize values > 1
                clamped[:, idx] = np.where(clamped[:, idx] > 1.0, clamped[:, idx] / 100.0, clamped[:, idx])

            clamped[:, idx] = np.clip(clamped[:, idx], min_b, max_b)
        return clamped

    def array_to_params(self, row: np.ndarray) -> Dict[str, float]:
        """Convert a 1D parameter array to a named dictionary"""
        return {name: float(row[i]) for i, name in enumerate(self.parameter_names)}

    def extract_observations(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Unified observation array extractor from dictionary or production_data subdict"""
        if not data:
            return {}

        obs = {}
        source = data.get('production_data', data)

        key_maps = {
            'oil': ['oil', 'Oil_bbl', 'oil_rate'],
            'water': ['water', 'Water_bbl', 'water_rate'],
            'gas': ['gas', 'Gas_scf', 'gas_rate'],
            'pressure': ['pressure', 'Pressure_psi', 'reservoir_pressure', 'bhp']
        }

        for channel, aliases in key_maps.items():
            for alias in aliases:
                if alias in source and len(source[alias]) > 0:
                    try:
                        obs[channel] = np.array(source[alias], dtype=float)
                        break
                    except (ValueError, TypeError):
                        pass

        return obs

    def calculate_match_quality(
        self,
        observed: Dict[str, np.ndarray],
        simulated: Dict[str, np.ndarray]
    ) -> float:
        """
        Calculate weighted NRMSE match quality percentage (0-100%).
        """
        try:
            keys = [k for k in observed.keys() if k in simulated and len(observed[k]) > 0]
            if not keys:
                return 0.0

            weights = {k: float(self.measurement_weights.get(k, 1.0)) for k in keys}
            total_w = sum(weights.values()) if sum(weights.values()) > 0 else len(keys)

            weighted_score = 0.0
            for key in keys:
                obs = np.array(observed[key], dtype=float)
                sim = np.array(simulated[key], dtype=float)

                if len(obs) != len(sim) and len(obs) > 0 and len(sim) > 0:
                    sim = np.interp(
                        np.linspace(0, 1, len(obs)),
                        np.linspace(0, 1, len(sim)),
                        sim
                    )

                rmse = np.sqrt(np.mean((obs - sim) ** 2))
                denom = np.ptp(obs)
                nrmse = rmse / denom if denom > 0 else (1.0 if rmse > 0 else 0.0)
                score = max(0.0, 100.0 * (1.0 - nrmse))
                weighted_score += score * (weights[key] / total_w)

            return float(np.clip(weighted_score, 0.0, 100.0))
        except Exception as e:
            logger.error(f"Error calculating match quality: {e}")
            return 0.0
