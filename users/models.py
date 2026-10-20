import uuid
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.core.validators import RegexValidator
from django.utils import timezone

username_validator = RegexValidator(
    regex=r'^[\w.@+ -]+$',
    message='Enter a valid username. Letters, digits, spaces, and @/./+/-/_ characters are allowed.',
)


class CustomUser(AbstractUser):
    """Extended user model with additional fields for researcher profiles"""
    username = models.CharField(
        'username',
        max_length=150,
        unique=True,
        help_text='Required. 150 characters or fewer. Letters, digits, spaces and @/./+/-/_ only.',
        validators=[username_validator],
        error_messages={
            'unique': 'A user with that username already exists.',
        },
    )
    
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


class Workspace(models.Model):
    """Collaborative team workspace grouping reservoir simulations, datasets, and members"""
    name = models.CharField(max_length=255)
    owner = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='owned_workspaces')
    description = models.TextField(blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'workspace'
        ordering = ['-created_at']
        verbose_name = 'Workspace'
        verbose_name_plural = 'Workspaces'

    def __str__(self):
        owner_name = self.owner.get_full_name() or self.owner.username
        return f"{self.name} (Owner: {owner_name})"


class WorkspaceMember(models.Model):
    """Membership link between users and workspaces with role-based access"""
    ROLE_CHOICES = [
        ('owner', 'Owner'),
        ('admin', 'Admin'),
        ('engineer', 'Reservoir Engineer'),
        ('viewer', 'Viewer / Analyst'),
    ]

    workspace = models.ForeignKey(Workspace, on_delete=models.CASCADE, related_name='members')
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='workspace_memberships')
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='engineer')
    joined_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'workspace_member'
        unique_together = ('workspace', 'user')
        ordering = ['joined_at']
        verbose_name = 'Workspace Member'
        verbose_name_plural = 'Workspace Members'

    def __str__(self):
        return f"{self.user.username} in {self.workspace.name} ({self.role})"


class WorkspaceInvitation(models.Model):
    """One-time access invitation token for collaborative team onboarding"""
    ROLE_CHOICES = [
        ('admin', 'Admin'),
        ('engineer', 'Reservoir Engineer'),
        ('viewer', 'Viewer / Analyst'),
    ]

    workspace = models.ForeignKey(Workspace, on_delete=models.CASCADE, related_name='invitations')
    invited_by = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='sent_invitations')
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='engineer')
    token = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    is_used = models.BooleanField(default=False)
    used_by = models.ForeignKey(CustomUser, on_delete=models.SET_NULL, null=True, blank=True, related_name='accepted_invitations')
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = 'workspace_invitation'
        ordering = ['-created_at']
        verbose_name = 'Workspace Invitation'
        verbose_name_plural = 'Workspace Invitations'

    def is_valid(self):
        if self.is_used:
            return False
        if self.expires_at and timezone.now() > self.expires_at:
            return False
        return True

    def __str__(self):
        return f"Invite to {self.workspace.name} as {self.role} ({'Used' if self.is_used else 'Active'})"


def get_user_workspaces(user):
    """Return all workspaces where user is owner or member, creating a default one if none exist"""
    memberships = WorkspaceMember.objects.filter(user=user).select_related('workspace', 'workspace__owner')
    if not memberships.exists():
        display_name = user.get_full_name().strip() or user.username
        ws_name = f"{display_name}'s Workspace"
        ws = Workspace.objects.create(name=ws_name, owner=user, description=f"Primary collaborative workspace for {display_name}")
        WorkspaceMember.objects.create(workspace=ws, user=user, role='owner')
        memberships = WorkspaceMember.objects.filter(user=user).select_related('workspace', 'workspace__owner')
    return memberships


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
        ('workspace_invite_created', 'Workspace Invite Created'),
        ('workspace_joined', 'Workspace Joined'),
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
