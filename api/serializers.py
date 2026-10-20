from rest_framework import serializers
from django.contrib.auth import get_user_model
from users.models import UserActivityLog, CustomUser, Workspace, WorkspaceMember, WorkspaceInvitation, get_user_workspaces
from simulations.models import Dataset, SimulationRun, Forecast, SimulationStatistics, AlgorithmDefinition
import sys

User = get_user_model()


# ============ User Serializers ============

class UserRegistrationSerializer(serializers.ModelSerializer):
    username = serializers.CharField(max_length=150)
    password = serializers.CharField(write_only=True, min_length=8)
    password2 = serializers.CharField(write_only=True, min_length=8)
    
    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'first_name', 'last_name', 'password', 'password2',
                  'institution', 'department']
    
    def validate_username(self, value):
        val = value.strip()
        if len(val) < 3:
            raise serializers.ValidationError("Username must be at least 3 characters long.")
        return val

    def validate(self, data):
        if data['password'] != data.pop('password2'):
            raise serializers.ValidationError({"password": "Passwords do not match."})
        return data
    
    def create(self, validated_data):
        raw_username = validated_data['username'].strip()
        first_name = validated_data.get('first_name', '').strip()
        last_name = validated_data.get('last_name', '').strip()
        if not first_name and ' ' in raw_username:
            parts = raw_username.split(' ', 1)
            first_name = parts[0]
            last_name = parts[1] if len(parts) > 1 else ''

        user = CustomUser.objects.create_user(
            username=raw_username,
            email=validated_data['email'],
            first_name=first_name,
            last_name=last_name,
            password=validated_data['password'],
            institution=validated_data.get('institution', ''),
            department=validated_data.get('department', ''),
        )
        # Automatically create personal workspace for user
        ws_name = f"{user.first_name.strip() or user.username}'s Workspace"
        ws = Workspace.objects.create(
            name=ws_name,
            owner=user,
            description=f"Primary collaborative workspace for {user.username}"
        )
        WorkspaceMember.objects.create(workspace=ws, user=user, role='owner')
        return user


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'phone',
                  'profile_image', 'bio', 'institution', 'department', 'research_area',
                  'years_experience', 'date_joined']
        read_only_fields = ['id', 'date_joined']


class UserDetailSerializer(serializers.ModelSerializer):
    statistics = serializers.SerializerMethodField()
    recent_simulations = serializers.SerializerMethodField()
    
    class Meta:
        model = CustomUser
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'phone',
                  'profile_image', 'bio', 'institution', 'department', 'research_area',
                  'years_experience', 'date_joined', 'last_login', 'statistics',
                  'recent_simulations']
        read_only_fields = ['id', 'date_joined', 'last_login']
    
    def get_statistics(self, obj):
        try:
            stats = obj.statistics
            return SimulationStatisticsSerializer(stats).data
        except SimulationStatistics.DoesNotExist:
            return None
    
    def get_recent_simulations(self, obj):
        recent = obj.simulation_runs.all()[:5]
        return SimulationRunListSerializer(recent, many=True).data


class WorkspaceMemberSerializer(serializers.ModelSerializer):
    user_id = serializers.IntegerField(source='user.id', read_only=True)
    username = serializers.CharField(source='user.username', read_only=True)
    full_name = serializers.SerializerMethodField()
    email = serializers.EmailField(source='user.email', read_only=True)
    role_display = serializers.CharField(source='get_role_display', read_only=True)

    class Meta:
        model = WorkspaceMember
        fields = ['id', 'user_id', 'username', 'full_name', 'email', 'role', 'role_display', 'joined_at']
        read_only_fields = ['id', 'joined_at']

    def get_full_name(self, obj):
        return obj.user.get_full_name().strip() or obj.user.username


class WorkspaceInvitationSerializer(serializers.ModelSerializer):
    role_display = serializers.CharField(source='get_role_display', read_only=True)
    invited_by_name = serializers.SerializerMethodField()
    is_valid = serializers.SerializerMethodField()

    class Meta:
        model = WorkspaceInvitation
        fields = ['id', 'workspace', 'invited_by', 'invited_by_name', 'role', 'role_display', 'token', 'is_used', 'is_valid', 'created_at', 'expires_at']
        read_only_fields = ['id', 'token', 'is_used', 'created_at']

    def get_invited_by_name(self, obj):
        return obj.invited_by.get_full_name().strip() or obj.invited_by.username

    def get_is_valid(self, obj):
        return obj.is_valid()


class WorkspaceSerializer(serializers.ModelSerializer):
    owner_name = serializers.SerializerMethodField()
    owner_email = serializers.EmailField(source='owner.email', read_only=True)
    current_user_role = serializers.SerializerMethodField()
    members = WorkspaceMemberSerializer(many=True, read_only=True)
    member_count = serializers.SerializerMethodField()
    invitations = serializers.SerializerMethodField()

    class Meta:
        model = Workspace
        fields = ['id', 'name', 'description', 'owner', 'owner_name', 'owner_email',
                  'current_user_role', 'members', 'member_count', 'invitations', 'created_at', 'updated_at']
        read_only_fields = ['id', 'owner', 'created_at', 'updated_at']

    def get_owner_name(self, obj):
        return obj.owner.get_full_name().strip() or obj.owner.username

    def get_current_user_role(self, obj):
        request = self.context.get('request')
        if not request or not request.user or not request.user.is_authenticated:
            return None
        membership = obj.members.filter(user=request.user).first()
        return membership.role if membership else ('owner' if obj.owner == request.user else None)

    def get_member_count(self, obj):
        return obj.members.count()

    def get_invitations(self, obj):
        request = self.context.get('request')
        if not request or not request.user or not request.user.is_authenticated:
            return []
        membership = obj.members.filter(user=request.user).first()
        if obj.owner == request.user or (membership and membership.role in ('owner', 'admin')):
            invs = obj.invitations.filter(is_used=False).order_by('-created_at')[:20]
            return WorkspaceInvitationSerializer(invs, many=True).data
        return []


class UserActivityLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserActivityLog
        fields = ['id', 'activity_type', 'description', 'timestamp']
        read_only_fields = fields


# ============ Simulation Data Serializers ============

class DatasetSerializer(serializers.ModelSerializer):
    file_url = serializers.SerializerMethodField()
    file = serializers.FileField(write_only=True, required=False, allow_null=True)
    production_data = serializers.JSONField(write_only=False, required=False, allow_null=True)
    
    class Meta:
        model = Dataset
        fields = ['id', 'name', 'description', 'file', 'filename', 'file_size', 'uploaded_at',
                  'updated_at', 'file_url', 'production_data']
        read_only_fields = ['id', 'filename', 'file_size', 'uploaded_at', 'updated_at']
    
    def validate(self, data):
        """Ensure either file or production_data is provided"""
        file = data.get('file')
        production_data = data.get('production_data')
        
        if not file and not production_data:
            raise serializers.ValidationError(
                "Either 'file' or 'production_data' must be provided."
            )
        
        # If production_data is provided, validate its structure
        if production_data and not file:
            self._validate_production_data_structure(production_data)
        
        return data
    
    def _validate_production_data_structure(self, production_data):
        """Validate production data has required fields for direct entry"""
        if not isinstance(production_data, dict):
            raise serializers.ValidationError(
                "production_data must be a dictionary."
            )
        
        # Check for production data (Days, Oil_bbl, Water_bbl, Gas_scf, Pressure_psi)
        required_fields = ['Days', 'Oil_bbl', 'Water_bbl', 'Gas_scf', 'Pressure_psi']
        has_production_fields = all(field in production_data for field in required_fields)
        
        # Or check for reservoir model
        has_reservoir_model = 'reservoir_model' in production_data
        
        if not (has_production_fields or has_reservoir_model):
            raise serializers.ValidationError(
                "production_data must contain either production fields "
                "(Days, Oil_bbl, Water_bbl, Gas_scf, Pressure_psi) or a 'reservoir_model' object."
            )
        
        # If production fields exist, validate they're all lists/arrays of same length
        if has_production_fields:
            lengths = {field: len(production_data[field]) 
                      for field in required_fields 
                      if isinstance(production_data[field], (list, tuple))}
            
            if lengths and len(set(lengths.values())) > 1:
                raise serializers.ValidationError(
                    f"All production data arrays must have the same length. Got: {lengths}"
                )
            
            if not lengths or min(lengths.values()) < 1:
                raise serializers.ValidationError(
                    "Production data must contain at least one data point."
                )
    
    def get_file_url(self, obj):
        request = self.context.get('request')
        if obj.file and request:
            return request.build_absolute_uri(obj.file.url)
        return None
    
    def create(self, validated_data):
        file = validated_data.pop('file', None)
        production_data = validated_data.pop('production_data', None)
        
        print(f"[DATASET CREATE] file={file}, production_data keys={list(production_data.keys()) if production_data else None}", file=sys.stderr, flush=True)
        
        if file:
            # File-based upload (existing behavior)
            dataset = Dataset.objects.create(
                file=file,
                filename=file.name,
                file_size=file.size,
                **validated_data
            )
        else:
            # Direct production_data entry (new behavior)
            dataset = Dataset.objects.create(
                file=None,
                filename=f"manual_entry_{validated_data.get('name', 'dataset')}",
                file_size=0,
                production_data=production_data,
                **validated_data
            )
        
        print(f"[DATASET CREATE] SUCCESS - dataset pk={dataset.pk}", file=sys.stderr, flush=True)
        return dataset


class AlgorithmDefinitionSerializer(serializers.ModelSerializer):
    """Serializer for dynamically registered history matching algorithms"""

    class Meta:
        model = AlgorithmDefinition
        fields = [
            'id', 'name', 'slug', 'category', 'execution_mode',
            'equation_latex', 'description', 'references',
            'hyperparameter_schema', 'default_hyperparameters',
            'is_active', 'is_default'
        ]


class SimulationRunListSerializer(serializers.ModelSerializer):
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)
    workspace_name = serializers.CharField(source='workspace.name', read_only=True, default='')
    
    class Meta:
        model = SimulationRun
        fields = ['id', 'name', 'matching_type', 'status', 'progress', 'match_quality',
                  'created_at', 'completed_at', 'dataset_name', 'workspace_name', 'duration_seconds']
        read_only_fields = ['id', 'created_at', 'completed_at']


class SimulationRunDetailSerializer(serializers.ModelSerializer):
    dataset = DatasetSerializer(read_only=True)
    algorithm_definition = AlgorithmDefinitionSerializer(read_only=True)
    forecasts = serializers.SerializerMethodField()
    
    class Meta:
        model = SimulationRun
        fields = ['id', 'name', 'description', 'matching_type', 'status', 'dataset', 'algorithm_definition', 'algorithm_config',
                  'initial_pressure', 'porosity', 'permeability', 'water_saturation',
                  'oil_viscosity', 'rock_compressibility', 'thickness', 'area',
                  'oil_formation_volume_factor', 'oil_initial_saturation',
                  'temperature', 'depth',
                  'progress', 'match_quality', 'error_message', 'results_data',
                  'created_at', 'started_at', 'completed_at', 'duration_seconds',
                  'forecasts', 'workspace']
        read_only_fields = ['id', 'progress', 'match_quality', 'error_message',
                           'results_data', 'created_at', 'started_at', 'completed_at',
                           'duration_seconds']
    
    def get_forecasts(self, obj):
        forecasts = obj.forecasts.all()
        return ForecastSerializer(forecasts, many=True).data


class TolerantAlgorithmField(serializers.Field):
    """Tolerantly accept integer PK or slug or None without 400 failure"""
    def to_internal_value(self, data):
        if not data:
            return None
        try:
            if isinstance(data, int) or (isinstance(data, str) and str(data).isdigit()):
                algo = AlgorithmDefinition.objects.filter(pk=int(data)).first()
                if algo:
                    return algo
            algo = AlgorithmDefinition.objects.filter(slug=str(data)).first()
            if algo:
                return algo
            return None
        except Exception:
            return None

    def to_representation(self, value):
        return value.pk if value else None

class SimulationRunCreateSerializer(serializers.ModelSerializer):
    initial_pressure = serializers.FloatField(required=False, allow_null=True)
    porosity = serializers.FloatField(required=False, allow_null=True)
    permeability = serializers.FloatField(required=False, allow_null=True)
    water_saturation = serializers.FloatField(required=False, allow_null=True)
    algorithm_definition = TolerantAlgorithmField(required=False, allow_null=True)
    workspace = serializers.PrimaryKeyRelatedField(queryset=Workspace.objects.all(), required=False, allow_null=True)

    class Meta:
        model = SimulationRun
        fields = ['id', 'name', 'description', 'matching_type', 'dataset', 'algorithm_definition', 'algorithm_config',
                  'initial_pressure', 'porosity', 'permeability', 'water_saturation',
                  'oil_viscosity', 'rock_compressibility', 'thickness', 'area',
                  'oil_formation_volume_factor', 'oil_initial_saturation',
                  'temperature', 'depth', 'workspace', 'created_at']
        read_only_fields = ['id', 'created_at']
    
    def create(self, validated_data):
        request = self.context['request']
        validated_data['user'] = request.user

        # Intelligent algorithm resolution fallback
        if not validated_data.get('algorithm_definition'):
            m_type = validated_data.get('matching_type')
            if m_type:
                validated_data['algorithm_definition'] = AlgorithmDefinition.objects.filter(slug=m_type).first()

        # Workspace association fallback
        if not validated_data.get('workspace'):
            ws_id = request.data.get('workspace_id') or request.headers.get('X-Workspace-Id')
            if ws_id:
                validated_data['workspace'] = Workspace.objects.filter(id=ws_id).first()
            if not validated_data.get('workspace'):
                mships = get_user_workspaces(request.user)
                if mships.exists():
                    validated_data['workspace'] = mships.first().workspace

        # Try to infer sensible defaults from the uploaded model JSON
        # for any fields that are missing in the user's explicit input.
        dataset = validated_data.get('dataset')

        if dataset:
            try:
                prod = getattr(dataset, 'production_data', {}) or {}
                # Prefer explicitly stored reservoir_model, otherwise fall back
                rm = prod.get('reservoir_model') if isinstance(prod, dict) else None
                if not rm:
                    # Could be raw_json or other structure
                    if isinstance(prod, dict) and prod.get('raw_json'):
                        rm = prod.get('raw_json')
                    else:
                        rm = prod if isinstance(prod, dict) else None

                if isinstance(rm, dict):
                    # Extract candidates from common locations
                    ip = rm.get('initial_pressure') or rm.get('pressure')
                    por = rm.get('porosity')
                    perm = rm.get('permeability')
                    wsat = rm.get('water_saturation') or rm.get('waterSat') or rm.get('sw')
                    temp = rm.get('temperature')
                    depth = rm.get('depth')
                    area = rm.get('area')
                    thick = rm.get('thickness') or rm.get('net_pay_thickness')
                    fvf = rm.get('formation_volume_factor') or rm.get('oil_formation_volume_factor')
                    comp = rm.get('rock_compressibility')
                    visc = rm.get('oil_viscosity')
                    isat = rm.get('oil_initial_sat') or rm.get('oil_initial_saturation')

                    # Check nested sections
                    if 'fluid_properties' in rm and isinstance(rm['fluid_properties'], dict):
                        fp = rm['fluid_properties']
                        ip = ip or fp.get('initial_pressure') or fp.get('pressure')
                        wsat = wsat or fp.get('water_saturation') or fp.get('waterSat')
                        fvf = fvf or fp.get('formation_volume_factor')
                        visc = visc or fp.get('oil_viscosity')
                        isat = isat or fp.get('oil_initial_sat')

                    if 'rock_properties' in rm and isinstance(rm['rock_properties'], dict):
                        rp = rm['rock_properties']
                        por = por or rp.get('porosity') or rp.get('poro')
                        perm = perm or rp.get('permeability') or rp.get('perm')
                        temp = temp or rp.get('temperature')
                        depth = depth or rp.get('depth')

                    if 'rock_mechanics' in rm and isinstance(rm['rock_mechanics'], dict):
                        rmec = rm['rock_mechanics']
                        comp = comp or rmec.get('rock_compressibility')

                    if 'reservoir_geometry' in rm and isinstance(rm['reservoir_geometry'], dict):
                        rg = rm['reservoir_geometry']
                        area = area or rg.get('area') or rg.get('drainage_area')
                        thick = thick or rg.get('thickness') or rg.get('net_pay_thickness')

                    # If porosity/permeability are arrays (per-cell), take the mean
                    try:
                        if isinstance(por, (list, tuple)) and por:
                            por = sum(float(x) for x in por) / len(por)
                    except Exception:
                        por = None

                    try:
                        if isinstance(perm, (list, tuple)) and perm:
                            perm = sum(float(x) for x in perm) / len(perm)
                    except Exception:
                        perm = None

                    # Normalize porosity/water saturation if provided as percentage
                    try:
                        if isinstance(por, (int, float)) and por > 1:
                            por = float(por) / 100.0
                    except Exception:
                        pass

                    try:
                        if isinstance(wsat, (int, float)) and wsat > 1:
                            wsat = float(wsat) / 100.0
                    except Exception:
                        pass

                    # Set values only when missing and when we have a valid numeric value
                    def set_if_missing(key, value):
                        if value is None or value == '':
                            return
                        if validated_data.get(key) in (None, '', 0, 0.0):
                            try:
                                v = float(value)
                                if v > 0:
                                    validated_data[key] = v
                            except Exception:
                                pass

                    set_if_missing('initial_pressure', ip)
                    set_if_missing('porosity', por)
                    set_if_missing('permeability', perm)
                    set_if_missing('water_saturation', wsat)
                    set_if_missing('temperature', temp)
                    set_if_missing('depth', depth)
                    set_if_missing('area', area)
                    set_if_missing('thickness', thick)
                    set_if_missing('oil_formation_volume_factor', fvf)
                    set_if_missing('rock_compressibility', comp)
                    set_if_missing('oil_viscosity', visc)
                    set_if_missing('oil_initial_saturation', isat)
            except Exception:
                # Fail silently: fallback to whatever the user provided or defaults
                pass

        # Fallback values from algorithm_config['parameters_to_estimate'] if available
        algo_config = validated_data.get('algorithm_config') or {}
        params_to_est = algo_config.get('parameters_to_estimate') or []
        for p in params_to_est:
            name = p.get('name')
            init_val = p.get('initial_guess')
            if name and init_val is not None:
                if validated_data.get(name) in (None, '', 0, 0.0):
                    try:
                        validated_data[name] = float(init_val)
                    except Exception:
                        pass

        # Physical defaults if still missing (satisfies non-null database fields)
        if validated_data.get('initial_pressure') in (None, ''):
            validated_data['initial_pressure'] = 3000.0
        if validated_data.get('porosity') in (None, ''):
            validated_data['porosity'] = 20.0
        if validated_data.get('permeability') in (None, ''):
            validated_data['permeability'] = 100.0
        if validated_data.get('water_saturation') in (None, ''):
            validated_data['water_saturation'] = 25.0

        return super().create(validated_data)

class ForecastSerializer(serializers.ModelSerializer):
    class Meta:
        model = Forecast
        fields = ['id', 'name', 'description', 'forecast_type', 'forecast_date',
                  'forecast_period_days', 'predicted_parameters', 'predictions',
                  'uncertainty_bounds', 'generated_at']
        read_only_fields = ['id', 'generated_at']


class SimulationStatisticsSerializer(serializers.ModelSerializer):
    class Meta:
        model = SimulationStatistics
        fields = ['total_simulations', 'completed_simulations', 'failed_simulations',
                  'baseline_simulations', 'enkf_simulations', 'total_datasets_uploaded',
                  'total_forecasts_generated', 'avg_match_quality', 'avg_simulation_duration',
                  'best_match_quality', 'last_simulation_date']
        read_only_fields = fields


# ============ Dashboard Serializers ============

class DashboardSummarySerializer(serializers.Serializer):
    """Combines user info with statistics for dashboard view"""
    user = UserSerializer()
    statistics = SimulationStatisticsSerializer()
    recent_simulations = SimulationRunListSerializer(many=True)
    recent_forecasts = ForecastSerializer(many=True)
    recent_datasets = DatasetSerializer(many=True)
    
    def to_representation(self, instance):
        """Custom representation to handle dict/object mixing"""
        return instance


# ============ Sensitivity Analysis Request Serializers (embedded from external tool)
class WellDatasetRowSerializer(serializers.Serializer):
    well_name = serializers.CharField()
    true_vertical_depth_ft = serializers.FloatField(min_value=1000)
    tubing_id_in = serializers.FloatField(min_value=1.0)
    reservoir_pressure_psia = serializers.FloatField(min_value=100)
    bubble_point_pressure_psia = serializers.FloatField(min_value=50)
    productivity_index_bpd_psi = serializers.FloatField(min_value=0.01)
    wellhead_pressure_psig = serializers.FloatField(min_value=0)
    water_cut_fraction = serializers.FloatField(min_value=0, max_value=0.99)
    oil_api = serializers.FloatField(min_value=10, max_value=60)
    gas_specific_gravity = serializers.FloatField(min_value=0.55, max_value=1.5)
    oil_specific_gravity = serializers.FloatField(min_value=0.6, max_value=1.1)
    gas_oil_ratio_scf_stb = serializers.FloatField(min_value=0)
    temperature_f = serializers.FloatField(min_value=60, max_value=350)


class SensitivityRequestSerializer(serializers.Serializer):
    dataset_row = WellDatasetRowSerializer()
    gas_injection_min_mmscfpd = serializers.FloatField(min_value=0)
    gas_injection_max_mmscfpd = serializers.FloatField(min_value=0)
    gas_injection_step_mmscfpd = serializers.FloatField(min_value=0.0)
    wellhead_pressure_psig_values = serializers.ListField(
        child=serializers.FloatField(min_value=0), allow_empty=False
    )
    water_cut_fraction_values = serializers.ListField(
        child=serializers.FloatField(min_value=0, max_value=0.99), allow_empty=False
    )

    def validate(self, attrs: dict):
        if attrs["gas_injection_max_mmscfpd"] < attrs["gas_injection_min_mmscfpd"]:
            raise serializers.ValidationError(
                "gas_injection_max_mmscfpd must be greater than or equal to gas_injection_min_mmscfpd."
            )
        return attrs
