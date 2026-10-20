from django.db import models
from django.contrib.auth.models import AbstractUser
from django.core.validators import URLValidator


class CustomUser(AbstractUser):
    """Extended user model with additional fields for researcher profiles"""
    
    # Basic info
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=20, blank=True, null=True)
    profile_image = models.ImageField(upload_to='profile_images/', blank=True, null=True)
    bio = models.TextField(blank=True, null=True)
    
    # Institution & Research info
    institution = models.CharField(max_length=255, blank=True, null=True, 
                                   help_text="e.g., University of Lagos")
    department = models.CharField(max_length=255, blank=True, null=True,
                                  help_text="e.g., Petroleum Engineering")
    research_area = models.CharField(max_length=255, blank=True, null=True,
                                    help_text="e.g., Reservoir Simulation")
    years_experience = models.IntegerField(blank=True, null=True,
                                          help_text="Years in the field")
    
    # Account management
    is_active = models.BooleanField(default=True)
    date_joined = models.DateTimeField(auto_now_add=True)
    last_login = models.DateTimeField(null=True, blank=True)
    
    # Preferences
    notifications_enabled = models.BooleanField(default=True)
    allow_data_tracking = models.BooleanField(default=True)
    
    class Meta:
        db_table = 'custom_user'
        verbose_name = 'User'
        verbose_name_plural = 'Users'
    
    def __str__(self):
        full_name = self.get_full_name().strip()
        name = full_name or self.username or self.email or f"User #{self.pk}"
        return f"{name} ({self.email})" if self.email and name != self.email else str(name)


class UserActivityLog(models.Model):
    """Track user activities for analytics and monitoring"""
    
    ACTIVITY_CHOICES = [
        ('login', 'Login'),
        ('logout', 'Logout'),
        ('simulation_start', 'Simulation Started'),
        ('simulation_complete', 'Simulation Completed'),
        ('simulation_failed', 'Simulation Failed'),
        ('simulation_interpreted', 'Simulation Interpreted'),
        ('dataset_upload', 'Dataset Uploaded'),
        ('forecast_generated', 'Forecast Generated'),
        ('settings_update', 'Settings Updated'),
    ]
    
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='activity_logs', null=True, blank=True)
    activity_type = models.CharField(max_length=50, choices=ACTIVITY_CHOICES)
    description = models.TextField(blank=True, null=True)
    ip_address = models.GenericIPAddressField(blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'user_activity_log'
        ordering = ['-timestamp']
        verbose_name = 'User Activity Log'
        verbose_name_plural = 'User Activity Logs'
    
    def __str__(self):
        user_name = (
            self.user.username if (self.user and self.user.username)
            else (self.user.email if self.user else "Anonymous / System")
        )
        act = dict(self.ACTIVITY_CHOICES).get(self.activity_type, self.activity_type)
        time_str = self.timestamp.strftime('%Y-%m-%d %H:%M') if self.timestamp else ''
        return f"{user_name} - {act} ({time_str})"
