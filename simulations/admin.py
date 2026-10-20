import json
from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils import timezone
from simulations.models import Dataset, SimulationRun, Forecast, SimulationStatistics


class ForecastInline(admin.TabularInline):
    model = Forecast
    extra = 0
    show_change_link = True
    fields = ['name', 'forecast_type', 'forecast_period_days', 'generated_at']
    readonly_fields = ['generated_at']
    max_num = 5

    def has_add_permission(self, request, obj=None):
        return False


class SimulationRunDatasetInline(admin.TabularInline):
    model = SimulationRun
    extra = 0
    show_change_link = True
    fields = ['name', 'matching_type', 'status_badge', 'match_quality_display', 'created_at']
    readonly_fields = ['status_badge', 'match_quality_display', 'created_at']
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
            return f"{obj.match_quality:.1f}%"
        return '-'
    match_quality_display.short_description = 'Match Quality'

    def has_add_permission(self, request, obj=None):
        return False


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    """Dataset admin with file size display, production data metadata inspection, and user links"""

    list_display = ['name', 'user_link', 'filename', 'file_size_formatted', 'has_production_data', 'simulations_count', 'uploaded_at']
    list_filter = ['uploaded_at', 'user']
    search_fields = ['name', 'description', 'filename', 'user__username', 'user__email']
    readonly_fields = ['uploaded_at', 'updated_at', 'file_size_formatted', 'production_data_summary']
    list_select_related = ['user']
    date_hierarchy = 'uploaded_at'
    list_per_page = 25
    inlines = [SimulationRunDatasetInline]

    fieldsets = (
        ('Basic Information', {
            'fields': ('user', 'name', 'description')
        }),
        ('File Information', {
            'fields': ('file', 'filename', 'file_size', 'file_size_formatted')
        }),
        ('Production & Reservoir Data', {
            'fields': ('production_data_summary', 'production_data')
        }),
        ('Timestamps', {
            'fields': ('uploaded_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

    def user_link(self, obj):
        if not obj.user:
            return '-'
        url = reverse('admin:users_customuser_change', args=[obj.user.id])
        name = obj.user.get_full_name() or obj.user.username or obj.user.email
        return format_html('<a href="{}" style="font-weight: 600;">{}</a>', url, name)
    user_link.short_description = 'Owner'

    def file_size_formatted(self, obj):
        if not obj.file_size:
            return '0 B (Manual Entry)'
        if obj.file_size > 1024 * 1024:
            return f"{obj.file_size / (1024 * 1024):.2f} MB"
        return f"{obj.file_size / 1024:.1f} KB"
    file_size_formatted.short_description = 'File Size'

    def has_production_data(self, obj):
        if not obj.production_data or not isinstance(obj.production_data, dict):
            return format_html('<span style="color: #94A3B8;">None</span>')
        keys = list(obj.production_data.keys())
        has_rm = 'reservoir_model' in keys
        pts = len(obj.production_data.get('Days', []))
        desc = f"{pts} pts" if pts > 0 else "Custom"
        if has_rm:
            desc += " + Model"
        return format_html('<span style="color: #16A34A; font-weight: 600;">✓ {}</span>', desc)
    has_production_data.short_description = 'Data Structure'

    def simulations_count(self, obj):
        count = obj.simulations.count()
        url = reverse('admin:simulations_simulationrun_changelist') + f'?dataset__id__exact={obj.id}'
        return format_html('<a href="{}">{} run(s)</a>', url, count)
    simulations_count.short_description = 'Used In'

    def production_data_summary(self, obj):
        if not obj.production_data or not isinstance(obj.production_data, dict):
            return "No production data present."
        keys = list(obj.production_data.keys())
        summary = {
            'top_level_keys': keys,
            'data_points_days': len(obj.production_data.get('Days', [])),
            'has_reservoir_model': 'reservoir_model' in obj.production_data,
        }
        if 'reservoir_model' in obj.production_data:
            summary['reservoir_model_sections'] = list(obj.production_data['reservoir_model'].keys()) if isinstance(obj.production_data['reservoir_model'], dict) else 'non-dict'
        return format_html('<pre style="background: #0f172a; color: #38bdf8; padding: 10px; border-radius: 6px;">{}</pre>', json.dumps(summary, indent=2))
    production_data_summary.short_description = 'Data Summary'


@admin.register(SimulationRun)
class SimulationRunAdmin(admin.ModelAdmin):
    """Full-featured SimulationRun management with progress bars, status badges, and administrative actions"""

    list_display = [
        'id', 'name', 'user_link', 'matching_type_badge', 'status_badge',
        'progress_bar', 'match_quality_formatted', 'duration_display', 'created_at'
    ]
    list_filter = ['status', 'matching_type', 'created_at', 'user']
    search_fields = ['name', 'description', 'user__username', 'user__email', 'error_message']
    readonly_fields = [
        'created_at', 'started_at', 'completed_at', 'duration_seconds', 'duration_display',
        'progress_bar', 'results_summary'
    ]
    list_select_related = ['user', 'dataset']
    date_hierarchy = 'created_at'
    list_per_page = 25
    actions = ['mark_as_completed', 'mark_as_failed', 'reset_to_pending']
    inlines = [ForecastInline]

    fieldsets = (
        ('Simulation Identification', {
            'fields': ('user', 'name', 'description', 'dataset')
        }),
        ('Status & Progress Tracking', {
            'fields': ('status', 'matching_type', 'progress_bar', 'progress', 'current_step', 'total_steps', 'error_message')
        }),
        ('Reservoir Model Core Parameters', {
            'fields': ('initial_pressure', 'porosity', 'permeability', 'water_saturation')
        }),
        ('Extended Volumetric Parameters (OOIP)', {
            'fields': ('area', 'thickness', 'oil_viscosity', 'rock_compressibility', 'oil_formation_volume_factor', 'oil_initial_saturation', 'temperature', 'depth'),
            'classes': ('collapse',)
        }),
        ('Matching Results & Outputs', {
            'fields': ('match_quality', 'results_summary', 'results_data'),
            'classes': ('collapse',)
        }),
        ('Execution Timeline', {
            'fields': ('created_at', 'started_at', 'completed_at', 'duration_display', 'duration_seconds'),
            'classes': ('collapse',)
        }),
    )

    def user_link(self, obj):
        if not obj.user:
            return '-'
        url = reverse('admin:users_customuser_change', args=[obj.user.id])
        name = obj.user.get_full_name() or obj.user.username or obj.user.email
        return format_html('<a href="{}" style="font-weight: 600;">{}</a>', url, name)
    user_link.short_description = 'User'

    def matching_type_badge(self, obj):
        colors = {
            'baseline': '#2563EB',
            'enkf': '#7C3AED',
            'sensitivity': '#0D9488',
        }
        color = colors.get(obj.matching_type, '#475569')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 2px 8px; border-radius: 9999px; font-size: 11px; font-weight: 600;">{}</span>',
            color, obj.get_matching_type_display()
        )
    matching_type_badge.short_description = 'Type'

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
            '<span style="background-color: {}; color: white; padding: 3px 10px; border-radius: 9999px; font-size: 11px; font-weight: 700;">{}</span>',
            color, obj.get_status_display()
        )
    status_badge.short_description = 'Status'

    def progress_bar(self, obj):
        val = max(0, min(100, obj.progress or 0))
        color = '#16A34A' if obj.status == 'completed' else ('#DC2626' if obj.status == 'failed' else '#2563EB')
        return format_html(
            '<div style="width: 100px; background-color: #E2E8F0; border-radius: 4px; overflow: hidden; height: 14px; position: relative;">'
            '<div style="width: {}%; background-color: {}; height: 100%;"></div>'
            '<span style="position: absolute; top: 0; left: 0; width: 100%; text-align: center; font-size: 9px; font-weight: 700; line-height: 14px; color: #0F172A;">{}%</span>'
            '</div>',
            val, color, val
        )
    progress_bar.short_description = 'Progress'

    def match_quality_formatted(self, obj):
        if obj.match_quality is not None:
            color = '#16A34A' if obj.match_quality >= 80 else ('#D97706' if obj.match_quality >= 60 else '#DC2626')
            return format_html('<strong style="color: {}; font-size: 13px;">{}%</strong>', color, f"{obj.match_quality:.1f}")
        return '-'
    match_quality_formatted.short_description = 'Match Quality'

    def duration_display(self, obj):
        secs = obj.duration_seconds
        if secs is None and obj.started_at and obj.completed_at:
            secs = int((obj.completed_at - obj.started_at).total_seconds())
        if secs is not None:
            mins, s = divmod(secs, 60)
            return f"{mins}m {s}s" if mins > 0 else f"{s}s"
        return '-'
    duration_display.short_description = 'Duration'

    def results_summary(self, obj):
        if not obj.results_data or not isinstance(obj.results_data, dict):
            return "No output results recorded."
        rd = obj.results_data
        summary = {
            'match_quality': rd.get('match_quality'),
            'ooip': rd.get('ooip'),
            'iterations': rd.get('iterations'),
            'has_prior_ensemble': 'prior_ensemble' in rd,
            'has_posterior_ensemble': 'posterior_ensemble' in rd,
            'oil_predicted_len': len(rd.get('oil_predicted', [])) if isinstance(rd.get('oil_predicted'), list) else None,
        }
        return format_html('<pre style="background: #0f172a; color: #38bdf8; padding: 10px; border-radius: 6px; max-height: 250px; overflow-y: auto;">{}</pre>', json.dumps(summary, indent=2))
    results_summary.short_description = 'Results Summary'

    # Admin actions
    @admin.action(description="Mark selected simulations as Completed")
    def mark_as_completed(self, request, queryset):
        updated = queryset.update(status='completed', progress=100, completed_at=timezone.now())
        self.message_user(request, f"{updated} simulation(s) marked as completed.")

    @admin.action(description="Mark selected simulations as Failed")
    def mark_as_failed(self, request, queryset):
        updated = queryset.update(status='failed', completed_at=timezone.now())
        self.message_user(request, f"{updated} simulation(s) marked as failed.")

    @admin.action(description="Reset selected simulations to Pending")
    def reset_to_pending(self, request, queryset):
        updated = queryset.update(status='pending', progress=0, started_at=None, completed_at=None, error_message=None)
        self.message_user(request, f"{updated} simulation(s) reset to pending.")


@admin.register(Forecast)
class ForecastAdmin(admin.ModelAdmin):
    """Forecast admin interface"""

    list_display = ['name', 'user_link', 'simulation_link', 'forecast_type_badge', 'forecast_period_days', 'generated_at']
    list_filter = ['forecast_type', 'generated_at', 'user']
    search_fields = ['name', 'description', 'user__username', 'simulation__name']
    readonly_fields = ['generated_at', 'updated_at', 'predictions_summary']
    list_select_related = ['user', 'simulation']
    date_hierarchy = 'generated_at'
    list_per_page = 25

    fieldsets = (
        ('Basic Information', {
            'fields': ('user', 'simulation', 'name', 'description')
        }),
        ('Forecast Configuration', {
            'fields': ('forecast_type', 'forecast_date', 'forecast_period_days')
        }),
        ('Predictions & Uncertainty', {
            'fields': ('predictions_summary', 'predicted_parameters', 'predictions', 'uncertainty_bounds'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('generated_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

    def user_link(self, obj):
        if not obj.user:
            return '-'
        url = reverse('admin:users_customuser_change', args=[obj.user.id])
        name = obj.user.get_full_name() or obj.user.username or obj.user.email
        return format_html('<a href="{}">{}</a>', url, name)
    user_link.short_description = 'User'

    def simulation_link(self, obj):
        if not obj.simulation:
            return '-'
        url = reverse('admin:simulations_simulationrun_change', args=[obj.simulation.id])
        return format_html('<a href="{}">#{} {}</a>', url, obj.simulation.id, obj.simulation.name)
    simulation_link.short_description = 'Parent Simulation'

    def forecast_type_badge(self, obj):
        return format_html(
            '<span style="background-color: #0D9488; color: white; padding: 2px 8px; border-radius: 9999px; font-size: 11px; font-weight: 600;">{}</span>',
            obj.forecast_type
        )
    forecast_type_badge.short_description = 'Type'

    def predictions_summary(self, obj):
        if not obj.predictions or not isinstance(obj.predictions, dict):
            return "No predictions data."
        summary = {
            'keys': list(obj.predictions.keys()),
            'sample_keys': {k: len(v) if isinstance(v, list) else type(v).__name__ for k, v in list(obj.predictions.items())[:5]}
        }
        return format_html('<pre style="background: #0f172a; color: #38bdf8; padding: 10px; border-radius: 6px;">{}</pre>', json.dumps(summary, indent=2))
    predictions_summary.short_description = 'Predictions Summary'


@admin.register(SimulationStatistics)
class SimulationStatisticsAdmin(admin.ModelAdmin):
    """Aggregate simulation statistics admin"""

    list_display = [
        'user_link', 'total_simulations', 'completed_simulations', 'failed_simulations',
        'baseline_simulations', 'enkf_simulations', 'total_datasets_uploaded',
        'avg_match_quality_display', 'best_match_quality_display', 'updated_at'
    ]
    search_fields = ['user__username', 'user__email']
    readonly_fields = [
        'user', 'total_simulations', 'completed_simulations', 'failed_simulations',
        'baseline_simulations', 'enkf_simulations', 'total_datasets_uploaded',
        'total_forecasts_generated', 'avg_match_quality', 'avg_simulation_duration',
        'best_match_quality', 'last_simulation_date', 'updated_at'
    ]
    list_select_related = ['user']

    def user_link(self, obj):
        if not obj.user:
            return '-'
        url = reverse('admin:users_customuser_change', args=[obj.user.id])
        name = obj.user.get_full_name() or obj.user.username or obj.user.email
        return format_html('<a href="{}" style="font-weight: 600;">{}</a>', url, name)
    user_link.short_description = 'User'

    def avg_match_quality_display(self, obj):
        if obj.avg_match_quality is not None:
            return f"{obj.avg_match_quality:.1f}%"
        return '-'
    avg_match_quality_display.short_description = 'Avg Quality'

    def best_match_quality_display(self, obj):
        if obj.best_match_quality is not None:
            return f"{obj.best_match_quality:.1f}%"
        return '-'
    best_match_quality_display.short_description = 'Best Quality'

    def has_add_permission(self, request):
        return False
