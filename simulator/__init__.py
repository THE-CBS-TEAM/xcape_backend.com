"""
Simulator module for XCAPE
Handles OPM Flow integration, baseline matching, and dynamic multi-algorithm history matching:
- Ensemble Smoother with Multiple Data Assimilation (ES-MDA)
- Ensemble Kalman Filter (EnKF)
- Single-Step Ensemble Smoother (ES)
- Dynamic Equation Smoother
"""

from .engine import SimulationEngine
from .baseline_matcher import BaselineMatcher
from .base_estimator import BaseEstimator, DEFAULT_PARAMETER_BOUNDS
from .enkf_filter import EnKFFilter
from .esmda_smoother import ESMDASmoother
from .equation_engine import DynamicEquationSmoother
from .forecast_generator import ForecastGenerator

__all__ = [
    'SimulationEngine',
    'BaselineMatcher',
    'BaseEstimator',
    'DEFAULT_PARAMETER_BOUNDS',
    'EnKFFilter',
    'ESMDASmoother',
    'DynamicEquationSmoother',
    'ForecastGenerator',
]
