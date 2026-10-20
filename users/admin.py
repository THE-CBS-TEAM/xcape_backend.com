from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.utils.html import format_html
from django.urls import reverse
from users.models import CustomUser, UserActivityLog
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
            'cancelled': '#64748B',
        }
        color = colors.get(obj.status, '#475569')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 2px 8px; border-radius: 9999px; font-size: 11px; font-weight: 600;">{}</span>',
            color, obj.get_status_display()
        )
    status_badge.short_description = 'Status'

    def match_quality_display(self, obj):
        if obj.match_quality is not None:
            color = '#16A34A' if obj.match_quality >= 80 else ('#D97706' if obj.match_quality >= 60 else '#DC2626')
            return format_html('<strong style="color: {};">{}%</strong>', color, f"{obj.match_quality:.1f}")
        return '-'
    match_quality_display.short_description = 'Match Quality'

    def has_add_permission(self, request, obj=None):
        return False


class DatasetInline(admin.TabularInline):
    model = Dataset
    extra = 0
    show_change_link = True
    fields = ['name', 'filename', 'file_size_display', 'uploaded_at']
    readonly_fields = ['file_size_display', 'uploaded_at']
    ordering = ['-uploaded_at']
    max_num = 5

    def file_size_display(self, obj):
        if not obj.file_size:
            return '0 B (Manual)'
        if obj.file_size > 1024 * 1024:
            return f"{obj.file_size / (1024 * 1024):.2f} MB"
        return f"{obj.file_size / 1024:.1f} KB"
    file_size_display.short_description = 'Size'

    def has_add_permission(self, request, obj=None):
        return False


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
        ('Permissions & Staff Roles', {
            'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')
        }),
        ('Preferences & Audit', {
            'fields': ('notifications_enabled', 'allow_data_tracking', 'last_login', 'date_joined'),
            'classes': ('collapse',)
        }),
    )

    def full_name_display(self, obj):
        return obj.get_full_name() or '-'
    full_name_display.short_description = 'Full Name'

    def simulations_badge(self, obj):
        count = obj.simulation_runs.count()
        url = reverse('admin:simulations_simulationrun_changelist') + f'?user__id__exact={obj.id}'
        return format_html(
            '<a href="{}" style="text-decoration:none;"><span style="background-color: #2563EB; color: white; padding: 2px 8px; border-radius: 9999px; font-size: 11px; font-weight: 600;">{} runs</span></a>',
            url, count
        )
    simulations_badge.short_description = 'Simulations'

    def datasets_badge(self, obj):
        count = obj.datasets.count()
        url = reverse('admin:simulations_dataset_changelist') + f'?user__id__exact={obj.id}'
        return format_html(
            '<a href="{}" style="text-decoration:none;"><span style="background-color: #0D9488; color: white; padding: 2px 8px; border-radius: 9999px; font-size: 11px; font-weight: 600;">{} datasets</span></a>',
            url, count
        )
    datasets_badge.short_description = 'Datasets'

    # Admin actions
    @admin.action(description="Activate selected users")
    def activate_users(self, request, queryset):
        updated = queryset.update(is_active=True)
        self.message_user(request, f"{updated} user(s) successfully activated.")

    @admin.action(description="Deactivate selected users")
    def deactivate_users(self, request, queryset):
        updated = queryset.update(is_active=False)
        self.message_user(request, f"{updated} user(s) successfully deactivated.")

    @admin.action(description="Grant Staff (Admin) status to selected users")
    def make_staff(self, request, queryset):
        updated = queryset.update(is_staff=True)
        self.message_user(request, f"{updated} user(s) granted staff permissions.")

    @admin.action(description="Revoke Staff status from selected users")
    def revoke_staff(self, request, queryset):
        updated = queryset.update(is_staff=False)
        self.message_user(request, f"{updated} user(s) staff permissions revoked.")


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
