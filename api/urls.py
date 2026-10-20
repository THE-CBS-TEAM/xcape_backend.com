from django.urls import path, include
from rest_framework.routers import DefaultRouter
from api.views import (
    AuthenticationViewSet, UserViewSet, DashboardViewSet,
    DatasetViewSet, ForecastViewSet, SimulationRunViewSet,
    AlgorithmDefinitionViewSet,
    HealthCheckView, SensitivityHealthView, SensitivityDatasetPreviewView, SensitivitySimulationView,
)

router = DefaultRouter()
router.register(r'auth', AuthenticationViewSet, basename='auth')
router.register(r'users', UserViewSet, basename='user')
router.register(r'dashboard', DashboardViewSet, basename='dashboard')
router.register(r'datasets', DatasetViewSet, basename='dataset')
router.register(r'algorithms', AlgorithmDefinitionViewSet, basename='algorithm')
router.register(r'simulations', SimulationRunViewSet, basename='simulation')
router.register(r'forecasts', ForecastViewSet, basename='forecast')

urlpatterns = [
    path('health/', HealthCheckView.as_view(), name='health-check'),
    path('health', HealthCheckView.as_view(), name='health-check-no-slash'),
    path('', include(router.urls)),
    # Sensitivity compatibility endpoints for embedded app
    path('sensitivity/health/', SensitivityHealthView.as_view(), name='sensitivity-health'),
    path('sensitivity/health', SensitivityHealthView.as_view(), name='sensitivity-health-no-slash'),
    path('sensitivity/dataset-preview/', SensitivityDatasetPreviewView.as_view(), name='sensitivity-dataset-preview'),
    path('sensitivity/dataset-preview', SensitivityDatasetPreviewView.as_view(), name='sensitivity-dataset-preview-no-slash'),
    path('sensitivity/simulate/', SensitivitySimulationView.as_view(), name='sensitivity-simulate'),
    path('sensitivity/simulate', SensitivitySimulationView.as_view(), name='sensitivity-simulate-no-slash'),
]
