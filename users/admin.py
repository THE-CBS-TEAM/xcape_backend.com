import uuid
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.utils.html import format_html
from django.urls import reverse
from users.models import CustomUser, UserActivityLog, Workspace, WorkspaceMember, WorkspaceInvitation
from simulations.models import SimulationRun, Dataset


class UserActivityLogInline(admin.TabularInline):
    model = UserActivityLog
    extra = 0
    can_delete = False
    readonly_fields = ['activity_badge', 'description', 'ip_address', 'timestamp']
    fields = ['activity_badge', 'description', 'ip_address', 'timestamp']
    ordering = ['-timestamp']
    max_num = 15

    def activity_badge(self, obj):
        colors = {
            'login': '#2563EB',
            'logout': '#64748B',
            'simulation_start': '#7C3AED',
            'simulation_complete': '#16A34A',
            'simulation_failed': '#DC2626',
            'simulation_interpreted': '#0D9488',
            'dataset_upload': '#D97706',
            'forecast_generated': '#EA580C',
            'workspace_invite_created': '#8B5CF6',
            'workspace_joined': '#059669',
        }
        color = colors.get(obj.activity_type, '#475569')
        label = dict(UserActivityLog.ACTIVITY_CHOICES).get(obj.activity_type, obj.activity_type)
        return format_html(
            '<span style="background-color: {}; color: white; padding: 2px 8px; border-radius: 9999px; font-size: 11px; font-weight: 600;">{}</span>',
            color, label
        )
    activity_badge.short_description = 'Activity'

    def has_add_permission(self, request, obj=None):
        return False


class SimulationRunInline(admin.TabularInline):
    model = SimulationRun
    extra = 0
    show_change_link = True
    fields = ['name', 'matching_type', 'status_badge', 'match_quality_display', 'progress', 'created_at']
    readonly_fields = ['status_badge', 'match_quality_display', 'created_at']
    ordering = ['-created_at']
    max_num = 10

    def status_badge(self, obj):
        colors = {
            'completed': '#16A34A',
            'running': '#2563EB',
            'failed': '#DC2626',
            'pending': '#D97706',
            'cancelled': '#6B7280',
        }
        color = colors.get(obj.status, '#6B7280')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 2px 8px; border-radius: 9999px; font-size: 11px; font-weight: 600;">{}</span>',
            color, obj.get_status_display()
        )
    status_badge.short_description = 'Status'

    def match_quality_display(self, obj):
        if obj.match_quality is not None:
            color = '#16A34A' if obj.match_quality >= 80 else '#D97706' if obj.match_quality >= 60 else '#DC2626'
            return format_html(
                '<strong style="color: {};">{:.1f}%</strong>',
                color, obj.match_quality
            )
        return '-'
    match_quality_display.short_description = 'Match Quality'

    def has_add_permission(self, request, obj=None):
        return False


class DatasetInline(admin.TabularInline):
    model = Dataset
    extra = 0
    show_change_link = True
    fields = ['name', 'filename', 'file_size_display', 'uploaded_at']
    readonly_fields = ['name', 'filename', 'file_size_display', 'uploaded_at']
    ordering = ['-uploaded_at']
    max_num = 10

    def file_size_display(self, obj):
        size = obj.file_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    file_size_display.short_description = 'File Size'

    def has_add_permission(self, request, obj=None):
        return False


class WorkspaceMemberInline(admin.TabularInline):
    model = WorkspaceMember
    extra = 1
    autocomplete_fields = ['user']
    fields = ['user', 'role', 'joined_at']
    readonly_fields = ['joined_at']


class WorkspaceInvitationInline(admin.TabularInline):
    model = WorkspaceInvitation
    extra = 0
    readonly_fields = ['token', 'invited_by', 'role', 'is_used', 'used_by', 'created_at', 'expires_at']
    fields = ['token', 'invited_by', 'role', 'is_used', 'used_by', 'created_at', 'expires_at']


@admin.register(Workspace)
class WorkspaceAdmin(admin.ModelAdmin):
    list_display = ['name', 'owner_link', 'member_count_display', 'created_at']
    search_fields = ['name', 'owner__username', 'owner__email', 'owner__first_name', 'owner__last_name']
    list_filter = ['created_at']
    inlines = [WorkspaceMemberInline, WorkspaceInvitationInline]

    def owner_link(self, obj):
        url = reverse('admin:users_customuser_change', args=[obj.owner.id])
        return format_html('<a href="{}" style="font-weight: 600;">{}</a>', url, obj.owner.get_full_name() or obj.owner.username)
    owner_link.short_description = 'Owner'

    def member_count_display(self, obj):
        count = obj.members.count()
        return format_html('<span style="font-weight: 600; color: #2563EB;">{} member(s)</span>', count)
    member_count_display.short_description = 'Members'


@admin.register(WorkspaceMember)
class WorkspaceMemberAdmin(admin.ModelAdmin):
    list_display = ['workspace', 'user_link', 'role_badge', 'joined_at']
    list_filter = ['role', 'joined_at']
    search_fields = ['workspace__name', 'user__username', 'user__email']

    def user_link(self, obj):
        url = reverse('admin:users_customuser_change', args=[obj.user.id])
        return format_html('<a href="{}">{}</a>', url, obj.user.get_full_name() or obj.user.username)
    user_link.short_description = 'Member'

    def role_badge(self, obj):
        colors = {
            'owner': '#7C3AED',
            'admin': '#2563EB',
            'engineer': '#16A34A',
            'viewer': '#64748B',
        }
        color = colors.get(obj.role, '#64748B')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 2px 8px; border-radius: 9999px; font-size: 11px; font-weight: 600;">{}</span>',
            color, obj.get_role_display()
        )
    role_badge.short_description = 'Role'


@admin.register(WorkspaceInvitation)
class WorkspaceInvitationAdmin(admin.ModelAdmin):
    list_display = ['workspace', 'invited_by', 'role', 'status_badge', 'token', 'created_at', 'expires_at']
    list_filter = ['role', 'is_used', 'created_at']
    search_fields = ['workspace__name', 'invited_by__username', 'token']

    def status_badge(self, obj):
        if obj.is_used:
            return format_html('<span style="background-color: #64748B; color: white; padding: 2px 8px; border-radius: 9999px; font-size: 11px;">Used</span>')
        if not obj.is_valid():
            return format_html('<span style="background-color: #DC2626; color: white; padding: 2px 8px; border-radius: 9999px; font-size: 11px;">Expired</span>')
        return format_html('<span style="background-color: #16A34A; color: white; padding: 2px 8px; border-radius: 9999px; font-size: 11px;">Active / Ready</span>')
    status_badge.short_description = 'Status'


@admin.register(CustomUser)
class CustomUserAdmin(BaseUserAdmin):
    """Full-featured administrator interface for CustomUser management"""

    list_display = [
        'username', 'email', 'full_name_display', 'institution',
        'simulations_badge', 'datasets_badge', 'is_staff', 'is_active', 'date_joined'
    ]
    list_filter = ['is_staff', 'is_superuser', 'is_active', 'institution', 'date_joined']
    search_fields = ['username', 'email', 'first_name', 'last_name', 'institution', 'department', 'phone']
    ordering = ['-date_joined']
    actions = ['activate_users', 'deactivate_users', 'make_staff', 'revoke_staff']
    inlines = [SimulationRunInline, DatasetInline, UserActivityLogInline]

    fieldsets = (
        ('Account Credentials', {
            'fields': ('username', 'email', 'password')
        }),
        ('Personal & Contact Info', {
            'fields': ('first_name', 'last_name', 'phone', 'profile_image', 'bio')
        }),
        ('Academic & Research Affiliation', {
            'fields': ('institution', 'department', 'research_area', 'years_experience')
        }),
        ('Permissions & Roles', {
            'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')
        }),
        ('Notification Preferences', {
            'fields': ('notifications_enabled', 'allow_data_tracking'),
            'classes': ('collapse',)
        }),
        ('Important Dates', {
            'fields': ('last_login', 'date_joined'),
            'classes': ('collapse',)
        }),
    )

    def full_name_display(self, obj):
        name = obj.get_full_name().strip()
        return name if name else '-'
    full_name_display.short_description = 'Full Name'

    def simulations_badge(self, obj):
        count = obj.simulation_runs.count()
        if count == 0:
            return format_html('<span style="color: #9CA3AF;">0</span>')
        return format_html(
            '<span style="background-color: #EEF2FF; color: #4F46E5; padding: 2px 8px; border-radius: 9999px; font-size: 12px; font-weight: 600;">{} runs</span>',
            count
        )
    simulations_badge.short_description = 'Simulations'

    def datasets_badge(self, obj):
        count = obj.datasets.count()
        if count == 0:
            return format_html('<span style="color: #9CA3AF;">0</span>')
        return format_html(
            '<span style="background-color: #F0FDF4; color: #16A34A; padding: 2px 8px; border-radius: 9999px; font-size: 12px; font-weight: 600;">{} files</span>',
            count
        )
    datasets_badge.short_description = 'Datasets'

    # Bulk actions
    def activate_users(self, request, queryset):
        updated = queryset.update(is_active=True)
        self.message_user(request, f"{updated} user(s) successfully activated.")
    activate_users.short_description = "Activate selected users"

    def deactivate_users(self, request, queryset):
        updated = queryset.update(is_active=False)
        self.message_user(request, f"{updated} user(s) successfully deactivated.")
    deactivate_users.short_description = "Deactivate selected users"

    def make_staff(self, request, queryset):
        updated = queryset.update(is_staff=True)
        self.message_user(request, f"{updated} user(s) granted staff permissions.")
    make_staff.short_description = "Grant staff access to selected users"

    def revoke_staff(self, request, queryset):
        updated = queryset.update(is_staff=False)
        self.message_user(request, f"{updated} user(s) staff permissions revoked.")
    revoke_staff.short_description = "Revoke staff access from selected users"


@admin.register(UserActivityLog)
class UserActivityLogAdmin(admin.ModelAdmin):
    """Activity and audit log admin with rich filtering, search, and user links"""

    list_display = ['id', 'user_link', 'activity_badge', 'description_snippet', 'ip_address', 'timestamp']
    list_filter = ['activity_type', ('timestamp', admin.DateFieldListFilter)]
    search_fields = ['user__username', 'user__email', 'description', 'ip_address']
    readonly_fields = ['user', 'activity_type', 'description', 'ip_address', 'timestamp']
    list_select_related = ['user']
    date_hierarchy = 'timestamp'
    list_per_page = 50
    ordering = ['-timestamp']

    def user_link(self, obj):
        if not obj.user:
            return format_html('<em>Anonymous / System</em>')
        url = reverse('admin:users_customuser_change', args=[obj.user.id])
        name = obj.user.get_full_name() or obj.user.username or obj.user.email
        return format_html('<a href="{}" style="font-weight: 600;">{}</a>', url, name)
    user_link.short_description = 'User'

    def activity_badge(self, obj):
        colors = {
            'login': '#2563EB',
            'logout': '#64748B',
            'simulation_start': '#7C3AED',
            'simulation_complete': '#16A34A',
            'simulation_failed': '#DC2626',
            'simulation_interpreted': '#0D9488',
            'dataset_upload': '#D97706',
            'forecast_generated': '#EA580C',
            'workspace_invite_created': '#8B5CF6',
            'workspace_joined': '#059669',
        }
        color = colors.get(obj.activity_type, '#475569')
        label = dict(UserActivityLog.ACTIVITY_CHOICES).get(obj.activity_type, obj.activity_type)
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 10px; border-radius: 9999px; font-size: 11px; font-weight: 600;">{}</span>',
            color, label
        )
    activity_badge.short_description = 'Activity'

    def description_snippet(self, obj):
        if not obj.description:
            return '-'
        return obj.description[:80] + ('...' if len(obj.description) > 80 else '')
    description_snippet.short_description = 'Description'

    def has_add_permission(self, request):
        return False
