import json
from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils import timezone
from simulations.models import Dataset, SimulationRun, Forecast, SimulationStatistics, AlgorithmDefinition


@admin.register(AlgorithmDefinition)
class AlgorithmDefinitionAdmin(admin.ModelAdmin):
    """Dynamic algorithm definition admin allowing admins to register algorithms and update equations"""

    list_display = ['name', 'slug', 'category_badge', 'execution_mode', 'is_active', 'is_default', 'created_at']
    list_filter = ['category', 'execution_mode', 'is_active', 'is_default']
    search_fields = ['name', 'slug', 'description', 'equation_latex']
    prepopulated_fields = {'slug': ('name',)}
    list_editable = ['is_active', 'is_default']
    readonly_fields = ['created_at', 'updated_at', 'equation_preview']

    fieldsets = (
        ('Core Identification', {
            'fields': ('name', 'slug', 'category', 'execution_mode', 'is_active', 'is_default')
        }),
        ('Mathematical Formulation & Documentation', {
            'fields': ('description', 'equation_latex', 'equation_preview', 'references')
        }),
        ('Hyperparameter Configuration Schema', {
            'fields': ('hyperparameter_schema', 'default_hyperparameters'),
            'classes': ('collapse',),
            'description': 'JSON schema defining the configuration fields dynamically presented to users in the simulator UI.'
        }),
        ('Dynamic Update Equation Script (Custom Mode)', {
            'fields': ('custom_update_script',),
            'classes': ('collapse',),
            'description': 'Python script executed for custom equations. Must define: def update_step(ensemble, simulated, observations, cov_obs, **kwargs).'
        }),
        ('Audit Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )

    def category_badge(self, obj):
        colors = {
            'smoother': '#0D9488',
            'filter': '#2563EB',
            'optimization': '#7C3AED',
            'custom_equation': '#D97706',
        }
        color = colors.get(obj.category, '#475569')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 10px; border-radius: 9999px; font-size: 11px; font-weight: 600;">{}</span>',
            color, obj.get_category_display()
        )
    category_badge.short_description = 'Category'

    def equation_preview(self, obj):
        if not obj.equation_latex:
            return format_html('<em>No equation specified</em>')
        return format_html(
            '<div style="background: #1E293B; color: #38BDF8; padding: 12px; border-radius: 6px; font-family: monospace; font-size: 13px; white-space: pre-wrap; line-height: 1.5;">{}</div>',
            obj.equation_latex
        )
    equation_preview.short_description = 'LaTeX Equation Preview'


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
        'id', 'name', 'user_link', 'matching_type_badge', 'algorithm_link', 'status_badge',
        'progress_bar', 'match_quality_formatted', 'duration_display', 'created_at'
    ]
    list_filter = ['status', 'matching_type', 'algorithm_definition', 'created_at', 'user']
    search_fields = ['name', 'description', 'user__username', 'user__email', 'error_message']
    readonly_fields = [
        'created_at', 'started_at', 'completed_at', 'duration_seconds', 'duration_display',
        'progress_bar', 'results_summary', 'algorithm_config_display'
    ]
    list_select_related = ['user', 'dataset', 'algorithm_definition']
    date_hierarchy = 'created_at'
    list_per_page = 25
    actions = ['mark_as_completed', 'mark_as_failed', 'reset_to_pending']
    inlines = [ForecastInline]

    fieldsets = (
        ('Simulation Identification', {
            'fields': ('user', 'name', 'description', 'dataset', 'algorithm_definition')
        }),
        ('Status & Progress Tracking', {
            'fields': ('status', 'matching_type', 'progress_bar', 'progress', 'current_step', 'total_steps', 'error_message')
        }),
        ('Dynamic Algorithm Configuration', {
            'fields': ('algorithm_config_display', 'algorithm_config'),
            'classes': ('collapse',)
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

    def algorithm_link(self, obj):
        if not obj.algorithm_definition:
            return format_html('<span style="color: #64748B;">Standard</span>')
        url = reverse('admin:simulations_algorithmdefinition_change', args=[obj.algorithm_definition.id])
        return format_html('<a href="{}" style="font-weight: 600; color: #0284C7;">{}</a>', url, obj.algorithm_definition.name)
    algorithm_link.short_description = 'Algorithm'

    def matching_type_badge(self, obj):
        colors = {
            'baseline': '#2563EB',
            'enkf': '#7C3AED',
            'esmda': '#0D9488',
            'es': '#0284C7',
            'custom_equation': '#D97706',
            'sensitivity': '#475569',
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
        if obj.match_quality is None:
            return '-'
        val = obj.match_quality
        color = '#16A34A' if val >= 75 else ('#D97706' if val >= 50 else '#DC2626')
        return format_html(
            '<span style="color: {}; font-weight: 700;">{:.1f}%</span>',
            color, val
        )
    match_quality_formatted.short_description = 'Match'

    def duration_display(self, obj):
        dur = obj.calculate_duration() or obj.duration_seconds
        if dur is None:
            return '-'
        mins, secs = divmod(dur, 60)
        return f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
    duration_display.short_description = 'Duration'

    def algorithm_config_display(self, obj):
        if not obj.algorithm_config or not isinstance(obj.algorithm_config, dict):
            return "No dynamic configuration attached (default parameters used)."
        return format_html(
            '<pre style="background: #0f172a; color: #38bdf8; padding: 10px; border-radius: 6px; max-height: 300px; overflow-y: auto;">{}</pre>',
            json.dumps(obj.algorithm_config, indent=2)
        )
    algorithm_config_display.short_description = 'Dynamic Config Overview'

    def results_summary(self, obj):
        if not obj.results_data or not isinstance(obj.results_data, dict):
            return "No results data."
        summary = {
            'keys': list(obj.results_data.keys()),
            'has_enkf_result': 'enkf_result' in obj.results_data,
            'has_esmda_result': 'esmda_result' in obj.results_data,
            'has_ooip': 'ooip' in obj.results_data,
            'match_quality': obj.results_data.get('match_quality'),
        }
        if 'ooip' in obj.results_data and isinstance(obj.results_data['ooip'], dict):
            summary['ooip_metrics'] = {
                'ooip_mmbbl': obj.results_data['ooip'].get('ooip_mmbbl'),
                'rf_percent': obj.results_data['ooip'].get('recovery_factor_percent'),
                'status': obj.results_data['ooip'].get('status'),
            }
        return format_html('<pre style="background: #0f172a; color: #38bdf8; padding: 10px; border-radius: 6px;">{}</pre>', json.dumps(summary, indent=2))
    results_summary.short_description = 'Results Overview'

    # Admin actions
    def mark_as_completed(self, request, queryset):
        rows = queryset.update(status='completed', progress=100, completed_at=timezone.now())
        self.message_user(request, f"{rows} simulation(s) marked as completed.")
    mark_as_completed.short_description = "Mark selected simulations as Completed"

    def mark_as_failed(self, request, queryset):
        rows = queryset.update(status='failed', completed_at=timezone.now(), error_message='Marked as failed by administrator')
        self.message_user(request, f"{rows} simulation(s) marked as failed.")
    mark_as_failed.short_description = "Mark selected simulations as Failed"

    def reset_to_pending(self, request, queryset):
        rows = queryset.update(status='pending', progress=0, current_step=0, started_at=None, completed_at=None, error_message=None)
        self.message_user(request, f"{rows} simulation(s) reset to Pending.")
    reset_to_pending.short_description = "Reset selected simulations to Pending"


@admin.register(Forecast)
class ForecastAdmin(admin.ModelAdmin):
    """Forecast admin with parameter comparison and prediction inspection"""

    list_display = ['name', 'simulation_link', 'user_link', 'forecast_type_badge', 'forecast_period_days', 'generated_at']
    list_filter = ['forecast_type', 'forecast_date', 'generated_at']
    search_fields = ['name', 'description', 'user__username', 'simulation__name']
    readonly_fields = ['generated_at', 'updated_at', 'predictions_summary']
    list_select_related = ['user', 'simulation']
    date_hierarchy = 'generated_at'
    list_per_page = 25

    def user_link(self, obj):
        if not obj.user:
            return '-'
        url = reverse('admin:users_customuser_change', args=[obj.user.id])
        name = obj.user.get_full_name() or obj.user.username or obj.user.email
        return format_html('<a href="{}" style="font-weight: 600;">{}</a>', url, name)
    user_link.short_description = 'User'

    def simulation_link(self, obj):
        if not obj.simulation:
            return '-'
        url = reverse('admin:simulations_simulationrun_change', args=[obj.simulation.id])
        return format_html('<a href="{}" style="font-weight: 600;">#{} {}</a>', url, obj.simulation.id, obj.simulation.name)
    simulation_link.short_description = 'Simulation'

    def forecast_type_badge(self, obj):
        color = '#2563EB' if obj.forecast_type == 'prior' else '#16A34A'
        return format_html(
            '<span style="background-color: {}; color: white; padding: 2px 8px; border-radius: 9999px; font-size: 11px; font-weight: 600;">{}</span>',
            color, obj.get_forecast_type_display() if hasattr(obj, 'get_forecast_type_display') else obj.forecast_type
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
