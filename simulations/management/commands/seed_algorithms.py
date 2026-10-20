import json
from django.core.management.base import BaseCommand
from simulations.models import AlgorithmDefinition


class Command(BaseCommand):
    help = "Seed initial history matching algorithms (ES-MDA, EnKF, Single-step ES, Custom Equation Template)"

    def handle(self, *args, **options):
        algorithms = [
            {
                "slug": "esmda",
                "name": "Ensemble Smoother with Multiple Data Assimilation (ES-MDA)",
                "category": "smoother",
                "execution_mode": "builtin_esmda",
                "is_default": True,
                "is_active": True,
                "equation_latex": (
                    r"\mathbf{K}^{(l)} = \mathbf{C}_{MD}^{(l)} \left( \mathbf{C}_{DD}^{(l)} + \alpha_l \mathbf{C}_D \right)^{-1}" "\n"
                    r"\mathbf{m}_j^{(l+1)} = \mathbf{m}_j^{(l)} + \mathbf{K}^{(l)} \left( \mathbf{d}_{uc, j}^{(l)} - \mathbf{d}_j^{(l)} \right)" "\n"
                    r"\text{where } \mathbf{d}_{uc, j}^{(l)} = \mathbf{d}_{obs} + \sqrt{\alpha_l} \, \mathbf{C}_D^{1/2} \mathbf{z}_j, \quad \sum_{l=1}^{N_a} \frac{1}{\alpha_l} = 1"
                ),
                "description": (
                    "Industry gold-standard for automated reservoir history matching. Assimilates the complete historical production profile "
                    "at once over multiple iterations (Na) using covariance inflation. Avoids sequential model restarts, preserves geological "
                    "plausibility, and prevents ensemble collapse."
                ),
                "references": "Emerick, A. A., & Reynolds, A. C. (2013). Ensemble smoother with multiple data assimilation. Computers & Geosciences, 55, 3-15.",
                "default_hyperparameters": {
                    "ensemble_size": 100,
                    "num_iterations": 4,
                    "alpha_schedule": "geometric",
                    "target_rmse": 0.03,
                    "inflation_factor": 1.05
                },
                "hyperparameter_schema": {
                    "fields": [
                        {"name": "ensemble_size", "label": "Ensemble Size (Ne)", "type": "number", "min": 20, "max": 500, "default": 100},
                        {"name": "num_iterations", "label": "Assimilation Steps (Na)", "type": "number", "min": 1, "max": 10, "default": 4},
                        {"name": "alpha_schedule", "label": "Alpha Schedule", "type": "select", "options": ["geometric", "uniform"], "default": "geometric"},
                        {"name": "target_rmse", "label": "Target RMSE", "type": "number", "step": 0.005, "default": 0.03},
                        {"name": "inflation_factor", "label": "Inflation Factor (β)", "type": "number", "step": 0.01, "default": 1.05}
                    ]
                }
            },
            {
                "slug": "enkf",
                "name": "Ensemble Kalman Filter (EnKF)",
                "category": "filter",
                "execution_mode": "builtin_enkf",
                "is_default": False,
                "is_active": True,
                "equation_latex": (
                    r"\mathbf{K}_k = \mathbf{C}_{xy} \left( \mathbf{C}_{yy} + \mathbf{R} \right)^{-1}" "\n"
                    r"\mathbf{x}_{k, j}^a = \mathbf{x}_{k, j}^f + \mathbf{K}_k \left( \mathbf{y}_k + \mathbf{v}_{k, j} - \mathbf{H} \mathbf{x}_{k, j}^f \right)"
                ),
                "description": (
                    "Sequential Kalman filter updating reservoir parameter states across sequential observation timesteps. "
                    "Appropriate for real-time well surveillance data streaming and dynamic reservoir monitoring."
                ),
                "references": "Evensen, G. (1994). Sequential data assimilation with a nonlinear quasi-geostrophic model using Monte Carlo methods.",
                "default_hyperparameters": {
                    "ensemble_size": 100,
                    "num_iterations": 10,
                    "target_rmse": 0.05,
                    "localization_radius": 5
                },
                "hyperparameter_schema": {
                    "fields": [
                        {"name": "ensemble_size", "label": "Ensemble Size (Ne)", "type": "number", "min": 20, "max": 500, "default": 100},
                        {"name": "num_iterations", "label": "Max Filter Iterations", "type": "number", "min": 1, "max": 50, "default": 10},
                        {"name": "target_rmse", "label": "Target RMSE", "type": "number", "step": 0.01, "default": 0.05},
                        {"name": "localization_radius", "label": "Localization Radius (Grid Blocks)", "type": "number", "default": 5}
                    ]
                }
            },
            {
                "slug": "es",
                "name": "Single-Step Ensemble Smoother (ES)",
                "category": "smoother",
                "execution_mode": "builtin_es",
                "is_default": False,
                "is_active": True,
                "equation_latex": (
                    r"\mathbf{K} = \mathbf{C}_{MD} \left( \mathbf{C}_{DD} + \mathbf{C}_D \right)^{-1}" "\n"
                    r"\mathbf{m}_j^a = \mathbf{m}_j^f + \mathbf{K} \left( \mathbf{d}_{obs} + \mathbf{\epsilon}_j - \mathbf{d}_j^f \right)"
                ),
                "description": (
                    "Single-step global smoother updating prior realizations in one single matrix projection without iteration. "
                    "Provides an ultra-fast initial screening to test parameter sensitivities and baseline prior uncertainty reduction."
                ),
                "references": "van Leeuwen, P. J., & Evensen, G. (1996). Data assimilation and inverse modeling in ocean models.",
                "default_hyperparameters": {
                    "ensemble_size": 100,
                    "target_rmse": 0.05
                },
                "hyperparameter_schema": {
                    "fields": [
                        {"name": "ensemble_size", "label": "Ensemble Size (Ne)", "type": "number", "min": 20, "max": 500, "default": 100},
                        {"name": "target_rmse", "label": "Target RMSE", "type": "number", "step": 0.01, "default": 0.05}
                    ]
                }
            },
            {
                "slug": "custom_template",
                "name": "Custom Equation Smoother (Admin Template)",
                "category": "custom_equation",
                "execution_mode": "custom_script",
                "is_default": False,
                "is_active": False,
                "equation_latex": (
                    r"\mathbf{m}_{new} = \mathbf{m} + \mathbf{K}_{custom} (\mathbf{d}_{obs} - \mathbf{d}_{sim})"
                ),
                "description": (
                    "Dynamic user/admin-defined mathematical assimilation equation executed via secure script sandbox. "
                    "Enables custom regularization, non-Gaussian transformations, or experimental weighting."
                ),
                "references": "XCAPE Dynamic Algorithm Engine Documentation",
                "default_hyperparameters": {
                    "ensemble_size": 100,
                    "num_iterations": 3
                },
                "hyperparameter_schema": {
                    "fields": [
                        {"name": "ensemble_size", "label": "Ensemble Size (Ne)", "type": "number", "default": 100},
                        {"name": "num_iterations", "label": "Iterations", "type": "number", "default": 3}
                    ]
                },
                "custom_update_script": (
                    "# Define the update function for your custom algorithm\n"
                    "def update_step(ensemble, simulated, observations, cov_obs, **kwargs):\n"
                    "    '''\n"
                    "    ensemble: np.ndarray of shape (Ne, N_params)\n"
                    "    simulated: np.ndarray of shape (Ne, N_obs)\n"
                    "    observations: np.ndarray of shape (N_obs,)\n"
                    "    cov_obs: np.ndarray of shape (N_obs, N_obs)\n"
                    "    Returns: updated_ensemble of shape (Ne, N_params)\n"
                    "    '''\n"
                    "    import numpy as np\n"
                    "    Ne = ensemble.shape[0]\n"
                    "    m_mean = np.mean(ensemble, axis=0, keepdims=True)\n"
                    "    d_mean = np.mean(simulated, axis=0, keepdims=True)\n"
                    "    X = (ensemble - m_mean).T\n"
                    "    Y = (simulated - d_mean).T\n"
                    "    CMD = (X @ Y.T) / (Ne - 1)\n"
                    "    CDD = (Y @ Y.T) / (Ne - 1)\n"
                    "    inv_term = np.linalg.pinv(CDD + cov_obs)\n"
                    "    K = CMD @ inv_term\n"
                    "    updated = ensemble.copy()\n"
                    "    for j in range(Ne):\n"
                    "        innov = observations - simulated[j]\n"
                    "        updated[j] += K @ innov\n"
                    "    return updated\n"
                )
            }
        ]

        for alg_data in algorithms:
            slug = alg_data.pop("slug")
            obj, created = AlgorithmDefinition.objects.update_or_create(
                slug=slug,
                defaults=alg_data
            )
            action = "Created" if created else "Updated"
            self.stdout.write(self.style.SUCCESS(f"[OK] {action} algorithm: {obj.name} [{obj.slug}]"))
