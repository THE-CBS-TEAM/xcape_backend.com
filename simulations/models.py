from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone

User = get_user_model()


class Dataset(models.Model):
    """User uploaded datasets for simulation"""
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='datasets')
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    file = models.FileField(upload_to='datasets/')
    filename = models.CharField(max_length=255)
    file_size = models.BigIntegerField()  # in bytes
    
    # Production data metadata
    production_data = models.JSONField(default=dict, blank=True)
    
    # Tracking
    uploaded_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'dataset'
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.user.username} - {self.name}"


class SimulationRun(models.Model):
    """Track all simulation runs performed by users"""
    
    MATCHING_TYPE_CHOICES = [
        ('baseline', 'Baseline Matching'),
        ('enkf', 'Ensemble Kalman Filter (EnKF)'),
        ('sensitivity', 'Sensitivity Analysis'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]
    
    # User and dataset association
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='simulation_runs')
    dataset = models.ForeignKey(Dataset, on_delete=models.SET_NULL, null=True, blank=True,
                                related_name='simulations')
    
    # Simulation configuration
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    matching_type = models.CharField(max_length=20, choices=MATCHING_TYPE_CHOICES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    # Parameters (oilfield units)
    initial_pressure = models.FloatField(help_text="psi")
    porosity = models.FloatField(help_text="%")
    permeability = models.FloatField(help_text="mD")
    water_saturation = models.FloatField(help_text="%")
    
    # Extended reservoir parameters (optional — used for OOIP)
    oil_viscosity = models.FloatField(null=True, blank=True, help_text="cp")
    rock_compressibility = models.FloatField(null=True, blank=True, help_text="1/psi")
    thickness = models.FloatField(null=True, blank=True, help_text="ft - net pay thickness")
    area = models.FloatField(null=True, blank=True, help_text="acres - drainage area")
    oil_formation_volume_factor = models.FloatField(null=True, blank=True, help_text="RB/STB")
    oil_initial_saturation = models.FloatField(null=True, blank=True, help_text="fraction 0-1")
    temperature = models.FloatField(null=True, blank=True, help_text="°F")
    depth = models.FloatField(null=True, blank=True, help_text="ft")
    
    # Progress tracking
    progress = models.IntegerField(default=0, help_text="0-100 percentage")
    total_steps = models.IntegerField(default=100)
    current_step = models.IntegerField(default=0)
    
    # Results
    match_quality = models.FloatField(null=True, blank=True, help_text="0-100 matching percentage")
    error_message = models.TextField(blank=True, null=True)
    results_data = models.JSONField(default=dict, blank=True, 
                                    help_text="Simulation output and results")
    
    # Tracking
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    duration_seconds = models.IntegerField(null=True, blank=True, 
                                           help_text="Execution duration in seconds")
    
    class Meta:
        db_table = 'simulation_run'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', '-created_at']),
            models.Index(fields=['status']),
        ]
    
    def __str__(self):
        return f"{self.user.username} - {self.name} ({self.matching_type})"
    
    def calculate_duration(self):
        """Calculate simulation duration if completed"""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds())
        return None


class Forecast(models.Model):
    """Generated forecasts from simulation runs"""
    
    simulation = models.ForeignKey(SimulationRun, on_delete=models.CASCADE, 
                                   related_name='forecasts')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='forecasts')
    
    # Forecast parameters
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    forecast_type = models.CharField(max_length=50, choices=[
        ('prior', 'Prior Forecast'),
        ('posterior', 'Posterior Forecast'),
    ])
    
    # Forecast data
    forecast_date = models.DateField()
    forecast_period_days = models.IntegerField(default=365, help_text="Forecast period in days")
    
    # Results and predictions
    predicted_parameters = models.JSONField(default=dict, 
                                           help_text="Updated reservoir parameters")
    predictions = models.JSONField(default=dict, 
                                   help_text="Predicted production data")
    uncertainty_bounds = models.JSONField(default=dict,
                                         help_text="Upper and lower uncertainty bounds")
    
    # Tracking
    generated_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'forecast'
        ordering = ['-generated_at']
        indexes = [
            models.Index(fields=['user', '-generated_at']),
            models.Index(fields=['simulation']),
        ]
    
    def __str__(self):
        return f"{self.user.username} - {self.name} ({self.forecast_type})"


class SimulationStatistics(models.Model):
    """Aggregated statistics for user dashboard"""
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='statistics')
    
    # Counters
    total_simulations = models.IntegerField(default=0)
    completed_simulations = models.IntegerField(default=0)
    failed_simulations = models.IntegerField(default=0)
    
    # By type
    baseline_simulations = models.IntegerField(default=0)
    enkf_simulations = models.IntegerField(default=0)
    
    # File statistics
    total_datasets_uploaded = models.IntegerField(default=0)
    total_forecasts_generated = models.IntegerField(default=0)
    
    # Performance
    avg_match_quality = models.FloatField(default=0.0)
    avg_simulation_duration = models.IntegerField(default=0, help_text="in seconds")
    best_match_quality = models.FloatField(default=0.0)
    
    # Last activity
    last_simulation_date = models.DateTimeField(null=True, blank=True)
    
    # Tracking
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'simulation_statistics'
        verbose_name_plural = 'Simulation Statistics'
    
    def __str__(self):
        return f"Statistics for {self.user.username}"
