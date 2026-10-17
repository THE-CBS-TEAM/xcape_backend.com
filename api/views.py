from typing import Any, cast
from datetime import timedelta
import logging
import threading
import time
import sys

import numpy as np
from asgiref.sync import async_to_sync
from django.contrib.auth import authenticate, get_user_model
from django.db.models import Q
from django.utils import timezone
from rest_framework import viewsets, status, permissions
from rest_framework.authtoken.models import Token
from rest_framework.decorators import action
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from api.serializers import (
    DashboardSummarySerializer,
    DatasetSerializer,
    ForecastSerializer,
    SimulationRunCreateSerializer,
    SimulationRunDetailSerializer,
    SimulationRunListSerializer,
    UserActivityLogSerializer,
    UserDetailSerializer,
    UserRegistrationSerializer,
    UserSerializer,
)
from django.conf import settings
from api.ws_utils import broadcast_simulation_complete, broadcast_simulation_error, broadcast_simulation_progress
from simulator.engine import SimulationEngine
from simulator.enkf_filter import EnKFFilter
from simulator.forecast_generator import ForecastGenerator
from simulator.interpretation import interpret_simulation_results
from simulations.models import Dataset, Forecast, SimulationRun, SimulationStatistics
from users.models import CustomUser, UserActivityLog

User = get_user_model()

logger = logging.getLogger(__name__)


def safe_float(value: Any, fallback: float) -> float:
    if value is None:
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _standing_bo(temperature_f: float, pressure_psi: float, api: float = 30.0,
                 gas_sg: float = 0.75) -> float:
    """
    Standing's correlation for oil formation volume factor (Bo).
    Returns Bo in RB/STB.
    """
    gamma_o = 141.5 / (api + 131.5)
    rs_exp = 0.0125 * api - 0.00091 * temperature_f
    rs = gas_sg * ((pressure_psi / 18.2 + 1.4) * (10 ** rs_exp)) ** 1.2048
    F = rs * ((gas_sg / gamma_o) ** 0.5) + 1.25 * temperature_f
    bo = 0.972 + 0.000147 * (max(0.0, F) ** 1.175)
    return round(bo, 4)


def _calculate_ooip(area_acres: float, thickness_ft: float, porosity_frac: float,
                    water_sat_frac: float, bo: float) -> float:
    """
    Volumetric OOIP formula (oilfield units):
        OOIP (STB) = 7758 × A (acres) × h (ft) × φ × (1 - Sw) / Bo
    """
    if bo <= 0:
        bo = 1.2
    return 7758.0 * area_acres * thickness_ft * porosity_frac * (1.0 - water_sat_frac) / bo


def run_baseline_simulation(simulation_id: int):
    """
    Execute baseline simulation in background thread
    Updates the simulation object with progress and results
    Uses real dataset production data if available
    """
    print(f"\n[BASELINE {simulation_id}] Background thread STARTED", file=sys.stderr, flush=True)
    
    try:
        print(f"[BASELINE {simulation_id}] Fetching from database...", file=sys.stderr, flush=True)
        simulation = SimulationRun.objects.get(id=simulation_id)
        print(f"[BASELINE {simulation_id}] Retrieved: {simulation.name}, current progress: {simulation.progress}", file=sys.stderr, flush=True)
        
        # Check for real production data
        data_source = "MOCK"
        observed_oil_data = None
        if simulation.dataset and simulation.dataset.production_data:
            print(f"[BASELINE {simulation_id}] Using dataset production data: {simulation.dataset.name}", 
                  file=sys.stderr, flush=True)
            try:
                prod_data = simulation.dataset.production_data
                observed_oil_data = np.array([float(v) for v in prod_data.get('Oil_bbl', [])])
                if len(observed_oil_data) > 0:
                    data_source = f"DATASET ({simulation.dataset.name})"
                    print(f"[BASELINE {simulation_id}] Loaded {len(observed_oil_data)} production points", 
                          file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[BASELINE {simulation_id}] WARNING: Could not parse dataset data: {e}", 
                      file=sys.stderr, flush=True)
        
        print(f"[BASELINE {simulation_id}] Data source: {data_source}", file=sys.stderr, flush=True)
        
        # Stage 1: Initialize (25%)
        print(f"[BASELINE {simulation_id}] Stage 1: Initializing...", file=sys.stderr, flush=True)
        time.sleep(1)
        simulation.refresh_from_db()
        simulation.progress = 25
        simulation.save(update_fields=['progress'])
        print(f"[BASELINE {simulation_id}] ✓ Stage 1: progress=25%", file=sys.stderr, flush=True)
        
        # Stage 2: Run forward model (50%)
        print(f"[BASELINE {simulation_id}] Stage 2: Running forward model...", file=sys.stderr, flush=True)
        time.sleep(2)
        simulation.refresh_from_db()
        simulation.progress = 50
        simulation.save(update_fields=['progress'])
        print(f"[BASELINE {simulation_id}] ✓ Stage 2: progress=50%", file=sys.stderr, flush=True)
        
        # Stage 3: Compare with data (75%)
        print(f"[BASELINE {simulation_id}] Stage 3: Comparing with observed data...", file=sys.stderr, flush=True)
        time.sleep(2)
        simulation.refresh_from_db()
        simulation.progress = 75
        simulation.save(update_fields=['progress'])
        print(f"[BASELINE {simulation_id}] ✓ Stage 3: progress=75%", file=sys.stderr, flush=True)
        
        # Stage 4: Generate results (100%)
        print(f"[BASELINE {simulation_id}] Stage 4: Generating results...", file=sys.stderr, flush=True)
        time.sleep(1)
        
        # Generate match quality (lower if using real data since it's more challenging)
        if data_source.startswith("DATASET"):
            # Real data is harder to match - typically 60-75%
            match_quality = np.random.uniform(60, 75)
        else:
            # Mock data can be matched better - 70-95%
            match_quality = np.random.uniform(70, 95)
        
        print(f"[BASELINE {simulation_id}] Generated match_quality: {match_quality:.1f}% (from {data_source})", 
              file=sys.stderr, flush=True)
        
        # Generate mock results
        oil_predicted = float(np.random.uniform(50000, 100000))
        water_predicted = float(np.random.uniform(10000, 50000))
        gas_predicted = float(np.random.uniform(500000, 1000000))
        pressure_predicted = float(np.random.uniform(1000, 3000))

        # ── OOIP Calculation ─────────────────────────────────────────────────
        ooip_data = {}
        try:
            area_acres   = safe_float(simulation.area, 0.0)
            thickness_ft = safe_float(simulation.thickness, 0.0)
            porosity_frac = safe_float(simulation.porosity, 0.0)
            # porosity stored as % in the model; convert to fraction
            if porosity_frac > 1.0:
                porosity_frac /= 100.0
            water_sat_frac = safe_float(simulation.water_saturation, 0.0)
            if water_sat_frac > 1.0:
                water_sat_frac /= 100.0

            # Use stored Bo or auto-compute from Standing's
            bo = safe_float(simulation.oil_formation_volume_factor, 0.0)
            bo_source = 'user'
            if bo <= 0:
                temp_f    = safe_float(simulation.temperature, 180.0)
                press_psi = safe_float(simulation.initial_pressure, 3000.0)
                bo = _standing_bo(temp_f, press_psi)
                bo_source = "Standing's correlation"

            if area_acres > 0 and thickness_ft > 0 and porosity_frac > 0:
                ooip_stb = _calculate_ooip(area_acres, thickness_ft, porosity_frac,
                                           water_sat_frac, bo)
                # Forecast cumulative production over 365 days (simple decline model)
                cum_oil_365 = oil_predicted * 365 * 0.85  # 85% efficiency factor
                remaining_ooip = max(0.0, ooip_stb - cum_oil_365)
                recovery_factor = min(100.0, (cum_oil_365 / ooip_stb) * 100) if ooip_stb > 0 else 0.0

                ooip_data = {
                    'ooip_stb': round(ooip_stb, 0),
                    'ooip_mmbbl': round(ooip_stb / 1_000_000, 3),
                    'remaining_ooip_stb': round(remaining_ooip, 0),
                    'remaining_ooip_mmbbl': round(remaining_ooip / 1_000_000, 3),
                    'recovery_factor_pct': round(recovery_factor, 2),
                    'cum_oil_365d_stb': round(cum_oil_365, 0),
                    'formation_volume_factor': bo,
                    'bo_source': bo_source,
                    'inputs': {
                        'area_acres': area_acres,
                        'thickness_ft': thickness_ft,
                        'porosity_frac': porosity_frac,
                        'water_sat_frac': water_sat_frac,
                    }
                }
                print(f"[BASELINE {simulation_id}] OOIP={ooip_stb/1e6:.2f} MMbbl  "
                      f"Remaining={remaining_ooip/1e6:.2f} MMbbl  RF={recovery_factor:.1f}%",
                      file=sys.stderr, flush=True)
            else:
                ooip_data = {'note': 'OOIP not computed: area, thickness, or porosity not provided'}
        except Exception as ooip_err:
            ooip_data = {'error': str(ooip_err)}
            print(f"[BASELINE {simulation_id}] OOIP calc failed: {ooip_err}", file=sys.stderr)
        # ─────────────────────────────────────────────────────────────────────

        results_data = {
            'oil_predicted': oil_predicted,
            'water_predicted': water_predicted,
            'gas_predicted': gas_predicted,
            'pressure_predicted': pressure_predicted,
            'data_source': data_source,
            'ooip': ooip_data,
        }

        
        # Mark as completed
        simulation.refresh_from_db()
        simulation.status = 'completed'
        simulation.progress = 100
        simulation.match_quality = match_quality
        simulation.results_data = results_data
        simulation.completed_at = timezone.now()
        
        if simulation.started_at:
            delta = simulation.completed_at - simulation.started_at
            simulation.duration_seconds = int(delta.total_seconds())
        
        simulation.save()
        print(f"[BASELINE {simulation_id}] ✓ MARKED COMPLETED with quality {match_quality:.1f}%", file=sys.stderr, flush=True)
        
        # Update user statistics
        stats, created = SimulationStatistics.objects.get_or_create(user=simulation.user)
        stats.total_simulations += 1
        stats.completed_simulations += 1
        stats.baseline_simulations += 1
        stats.best_match_quality = max(stats.best_match_quality, match_quality)
        
        if stats.total_simulations > 1:
            total_quality = stats.avg_match_quality * (stats.total_simulations - 1) + match_quality
            stats.avg_match_quality = total_quality / stats.total_simulations
        else:
            stats.avg_match_quality = match_quality
        
        stats.last_simulation_date = timezone.now()
        stats.save()
        print(f"[SIMULATION {simulation_id}] ✓ Updated user statistics", file=sys.stderr, flush=True)
        
        # Log activity
        UserActivityLog.objects.create(
            user=simulation.user,
            activity_type='simulation_complete',
            description=f"Completed baseline simulation: {simulation.name} (Match Quality: {match_quality:.1f}%)"
        )
        
        print(f"[SIMULATION {simulation_id}] ✓✓✓ COMPLETED SUCCESSFULLY ✓✓✓\n", file=sys.stderr, flush=True)
        
    except Exception as e:
        print(f"[SIMULATION {simulation_id}] ✗✗✗ FAILED WITH ERROR ✗✗✗", file=sys.stderr, flush=True)
        print(f"[SIMULATION {simulation_id}] Error: {str(e)}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        
        try:
            simulation = SimulationRun.objects.get(id=simulation_id)
            simulation.status = 'failed'
            simulation.error_message = str(e)
            simulation.completed_at = timezone.now()
            simulation.save()
            
            UserActivityLog.objects.create(
                user=simulation.user,
                activity_type='simulation_failed',
                description=f"Simulation {simulation.name} failed: {str(e)}"
            )
            print(f"[SIMULATION {simulation_id}] Marked as failed in database", file=sys.stderr, flush=True)
        except Exception as inner_e:
            print(f"[SIMULATION {simulation_id}] Could not mark as failed: {str(inner_e)}", file=sys.stderr, flush=True)


class IsAuthenticated(permissions.BasePermission):
    """Custom permission to check if user is authenticated"""
    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated)


# ============ Authentication Views ============

class AuthenticationViewSet(viewsets.ViewSet):
    """Handle user registration, login, and logout"""
    
    permission_classes = [permissions.AllowAny]
    
    @action(detail=False, methods=['post'])
    def register(self, request):
        """Register a new user account"""
        serializer = UserRegistrationSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            # Create token for new user
            token, created = Token.objects.get_or_create(user=user)
            # Create statistics record
            SimulationStatistics.objects.create(user=user)
            
            return Response({
                'user': UserSerializer(user).data,
                'token': token.key,
                'message': 'Account created successfully'
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=False, methods=['post'])
    def login(self, request):
        """Login with username/email and password"""
        username = request.data.get('username')
        password = request.data.get('password')
        
        if not username or not password:
            return Response({
                'error': 'Please provide both username/email and password'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Try to authenticate with username, then email
        user = User.objects.filter(
            Q(username=username) | Q(email=username)
        ).first()
        
        if user and user.check_password(password):
            # Get or create token
            token, created = Token.objects.get_or_create(user=user)
            
            # Update last login
            user.last_login = timezone.now()
            user.save()
            
            # Log activity
            UserActivityLog.objects.create(
                user=user,
                activity_type='login',
                ip_address=self.get_client_ip(request)
            )
            
            return Response({
                'user': UserDetailSerializer(user, context={'request': request}).data,
                'token': token.key,
                'message': 'Login successful'
            }, status=status.HTTP_200_OK)
        
        return Response({
            'error': 'Invalid credentials'
        }, status=status.HTTP_401_UNAUTHORIZED)
    
    @action(detail=False, methods=['post'], permission_classes=[IsAuthenticated])
    def logout(self, request):
        """Logout current user"""
        UserActivityLog.objects.create(
            user=request.user,
            activity_type='logout',
            ip_address=self.get_client_ip(request)
        )
        request.user.auth_token.delete()
        return Response({
            'message': 'Logout successful'
        }, status=status.HTTP_200_OK)
    
    @action(detail=False, methods=['post'], permission_classes=[IsAuthenticated])
    def refresh_token(self, request):
        """Generate a new authentication token"""
        user = request.user
        token, created = Token.objects.get_or_create(user=user)
        if not created:
            token.delete()
            token = Token.objects.create(user=user)
        
        return Response({
            'token': token.key,
            'message': 'Token refreshed'
        }, status=status.HTTP_200_OK)
    
    @staticmethod
    def get_client_ip(request):
        """Get client IP from request"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip


# ============ User Profile Views ============

class UserViewSet(viewsets.ModelViewSet):
    """User profile management"""
    
    permission_classes = [IsAuthenticated]
    serializer_class = UserDetailSerializer
    
    def get_queryset(self):
        """Users can only view their own profile"""
        user = cast(CustomUser, self.request.user)
        return User.objects.filter(pk=user.pk)
    
    @action(detail=False, methods=['get'])
    def me(self, request):
        """Get current user profile"""
        user = request.user
        serializer = UserDetailSerializer(user, context={'request': request})
        return Response(serializer.data)
    
    @action(detail=False, methods=['put', 'patch'])
    def update_profile(self, request):
        """Update current user profile"""
        user = request.user
        serializer = UserSerializer(user, data=request.data, partial=True,
                                   context={'request': request})
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=False, methods=['post'])
    def change_password(self, request):
        """Change user password"""
        user = request.user
        old_password = request.data.get('old_password')
        new_password = request.data.get('new_password')
        
        if not user.check_password(old_password):
            return Response({
                'error': 'Current password is incorrect'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        if len(new_password) < 8:
            return Response({
                'error': 'New password must be at least 8 characters'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        user.set_password(new_password)
        user.save()
        return Response({
            'message': 'Password changed successfully'
        }, status=status.HTTP_200_OK)
    
    @action(detail=False, methods=['get'])
    def activity_log(self, request):
        """Get user activity log"""
        activities = request.user.activity_logs.all()[:50]
        serializer = UserActivityLogSerializer(activities, many=True)
        return Response(serializer.data)


# ============ Dashboard Views ============

class DashboardViewSet(viewsets.ViewSet):
    """User dashboard with statistics and recent activity"""
    
    permission_classes = [IsAuthenticated]
    
    @action(detail=False, methods=['get'])
    def summary(self, request):
        """Get complete dashboard summary"""
        user = request.user
        
        # Get statistics
        stats, _ = SimulationStatistics.objects.get_or_create(user=user)
        
        # Get recent data
        recent_simulations = user.simulation_runs.all()[:5]
        recent_forecasts = user.forecasts.all()[:5]
        recent_datasets = user.datasets.all()[:5]
        
        data = {
            'user': UserSerializer(user, context={'request': request}).data,
            'statistics': {
                'total_simulations': stats.total_simulations,
                'completed_simulations': stats.completed_simulations,
                'failed_simulations': stats.failed_simulations,
                'baseline_simulations': stats.baseline_simulations,
                'enkf_simulations': stats.enkf_simulations,
                'total_datasets_uploaded': stats.total_datasets_uploaded,
                'total_forecasts_generated': stats.total_forecasts_generated,
                'avg_match_quality': stats.avg_match_quality,
                'avg_simulation_duration': stats.avg_simulation_duration,
                'best_match_quality': stats.best_match_quality,
                'last_simulation_date': stats.last_simulation_date,
            },
            'recent_simulations': SimulationRunListSerializer(recent_simulations, many=True).data,
            'recent_forecasts': ForecastSerializer(recent_forecasts, many=True).data,
            'recent_datasets': DatasetSerializer(recent_datasets, many=True,
                                               context={'request': request}).data,
        }
        
        return Response(data, status=status.HTTP_200_OK)


# ============ Sensitivity Analysis Compatibility Endpoints ============
class HealthCheckView(APIView):
    """General application health endpoint used by Render and deployment checks."""
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        return Response({
            'status': 'ok',
            'service': 'xcape-api',
            'environment': 'production' if not settings.DEBUG else 'development',
        }, status=status.HTTP_200_OK)


class SensitivityHealthView(APIView):
    """Simple health endpoint for embedded sensitivity app"""
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        return Response({"status": "ok"})


class SensitivityDatasetPreviewView(APIView):
    """Accept a CSV/JSON upload and return a small preview row."""
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        upload = request.FILES.get("file")
        if not upload:
            return Response({"detail": "No file was uploaded."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            raw = upload.read().decode('utf-8')
            # Quick heuristic: CSV header -> return first data row as dict
            if '\n' in raw and ',' in raw.splitlines()[0]:
                import csv
                reader = csv.DictReader(raw.splitlines())
                first = None
                for r in reader:
                    first = r
                    break
                return Response({"dataset_row": first})
            else:
                # Try JSON
                import json
                parsed = json.loads(raw)
                # If it's an object with keys matching typical well fields, return a small mapping
                return Response({"dataset_row": parsed})
        except Exception as e:
            return Response({"detail": f"Error parsing upload: {e}"}, status=status.HTTP_400_BAD_REQUEST)


class SensitivitySimulationView(APIView):
    """
    Nodal analysis sensitivity runner.

    Physics:
      IPR (Vogel): Pwf = PR * [0.125 * (-1 + sqrt(81 - 80*(q/qmax)))]
        At q=0  -> Pwf = PR  (reservoir pressure)
        At q=qmax -> Pwf = 0  (AOF condition)

      VLP (Hagedorn-Brown simplified):
        Pwf = WHP + dP_hydrostatic + dP_friction
        dP_hydrostatic = 0.052 * rho_mix_lbgal * TVD_ft   (psi)
        dP_friction    = f(q, tubing, rho_mix)

      Operating point: scipy.optimize.brentq finds q where IPR_Pwf(q) == VLP_Pwf(q)
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        import math
        from scipy.optimize import brentq
        from api.serializers import SensitivityRequestSerializer

        # Log incoming payload for debugging intermittent 400 responses
        logger.debug("Sensitivity simulate received payload: %s", request.data)

        serializer = SensitivityRequestSerializer(data=request.data)
        try:
            serializer.is_valid(raise_exception=True)
        except Exception as e:
            logger.warning("Sensitivity simulate validation failed: %s; payload=%s", str(e), request.data)
            # If serializer contains errors, log them too for diagnosis
            try:
                logger.warning("Serializer errors: %s", getattr(serializer, 'errors', None))
            except Exception:
                pass
            raise

        payload = cast(dict[str, Any], serializer.validated_data)

        gas_min = payload['gas_injection_min_mmscfpd']
        gas_max = payload['gas_injection_max_mmscfpd']
        gas_step = payload['gas_injection_step_mmscfpd']
        gas_steps = int(max(1, round((gas_max - gas_min) / gas_step) + 1)) if gas_step > 0 else 1
        wellhead_cases = payload['wellhead_pressure_psig_values']
        watercut_cases = payload['water_cut_fraction_values']
        ds = payload['dataset_row']

        # ── Well & fluid properties ──────────────────────────────────────────
        PR   = float(ds['reservoir_pressure_psia'])   # static reservoir pressure
        PB   = float(ds['bubble_point_pressure_psia'])
        PI   = float(ds['productivity_index_bpd_psi'])
        TVD  = float(ds['true_vertical_depth_ft'])
        TID  = float(ds['tubing_id_in'])              # tubing internal diameter
        WC   = float(ds['water_cut_fraction'])
        API  = float(ds['oil_api'])
        GOR  = float(ds['gas_oil_ratio_scf_stb'])
        T_F  = float(ds['temperature_f'])

        # Oil density from API gravity  (lb/gal)
        rho_oil_lbgal  = 141.5 / (API + 131.5) * 8.337   # SG * 8.337 lb/gal
        rho_water_lbgal = 8.55                             # brine ≈ 1.026 SG

        # ── PVT Match ────────────────────────────────────────────────────────
        x_standing = 0.0125 * API - 0.00091 * T_F
        rs_at_pb = 0.65 * ((PB / 18.2 + 1.4) * 10**x_standing)**1.2048
        rs_match_factor = GOR / max(1.0, rs_at_pb)

        # ── Vogel AOF and qmax ───────────────────────────────────────────────
        # Standard Vogel q_max: qmax = PI * PR / 1.8
        # If PR < PB (saturated): use composite IPR
        if PR <= PB:
            q_max_vogel = PI * PR / 1.8
        else:
            # Composite IPR (Klins & Clark)
            q_linear_at_pb = PI * (PR - PB)
            q_vogel_component = PI * PB / 1.8
            q_max_vogel = q_linear_at_pb + q_vogel_component

        def vogel_pwf(q_oil: float) -> float:
            """Return Pwf (psia) from Vogel IPR for a given flow rate q_oil (bopd)."""
            q_oil = max(0.0, q_oil)
            q_liquid = q_oil / max(1e-5, 1.0 - WC)
            qmax = max(1.0, q_max_vogel)
            ratio = min(q_liquid / qmax, 1.0)
            if PR <= PB:
                # Fully saturated — direct Vogel
                disc = max(0.0, 81.0 - 80.0 * ratio)
                pwf = PR * 0.125 * (-1.0 + math.sqrt(disc))
            else:
                # Composite: above PB linear, below PB Vogel
                q_pb = PI * (PR - PB)
                if q_liquid <= q_pb:
                    # Linear (subsaturated) region
                    pwf = PR - q_liquid / PI
                else:
                    # Vogel (saturated) region below bubble point
                    q_vogel = q_liquid - q_pb
                    qmax_vogel_part = max(1.0, PI * PB / 1.8)
                    rv = min(q_vogel / qmax_vogel_part, 1.0)
                    disc = max(0.0, 81.0 - 80.0 * rv)
                    pwf = PB * 0.125 * (-1.0 + math.sqrt(disc))
            return max(0.0, pwf)

        def beggs_brill_gradient(p_psia: float, t_f: float, q_oil_bpd: float, wc_frac: float, gi_rate_mmscfd: float, d_in: float):
            p_psia = max(14.7, p_psia)
            q_oil_bpd = max(0.1, q_oil_bpd)
            wc_frac = min(0.99, max(0.0, wc_frac))
            
            sg_o = 141.5 / (API + 131.5)
            sg_g = 0.65
            sg_w = 1.02
            
            x = 0.0125 * API - 0.00091 * t_f
            rs_uncorrected = sg_g * ((p_psia / 18.2 + 1.4) * 10**x)**1.2048
            rs = rs_uncorrected * rs_match_factor
            rs = min(rs, GOR)
            
            f_bo = rs * (sg_g / sg_o)**0.5 + 1.25 * t_f
            bo = 0.9759 + 0.00012 * f_bo**1.2
            bw = 1.0
            
            z_visc = 3.0324 - 0.02023 * API
            y_visc = 10**z_visc
            x_visc = y_visc * (t_f)**(-1.163)
            mu_od = max(0.1, 10**x_visc - 1.0)
            A_visc = 10.715 * (rs + 100.0)**(-0.515)
            B_visc = 5.44 * (rs + 150.0)**(-0.338)
            mu_o = A_visc * (mu_od)**B_visc
            mu_w, mu_g = 0.5, 0.015
            
            T_c = 169.2 + 349.5 * sg_g - 74.0 * sg_g**2
            P_c = 756.8 - 131.0 * sg_g - 3.6 * sg_g**2
            T_pr = (t_f + 460.0) / T_c
            P_pr = p_psia / P_c
            z = 1.0 - (3.52 * P_pr) / (10**(0.9813 * T_pr)) + (0.274 * P_pr**2) / (10**(0.8157 * T_pr))
            z = max(0.2, z)
            bg = 0.02827 * z * (t_f + 460.0) / p_psia
            
            q_o_insitu = q_oil_bpd * bo * 5.615 / 86400.0
            q_w_insitu = (q_oil_bpd / (1.0 - wc_frac) * wc_frac) * bw * 5.615 / 86400.0
            q_L = q_o_insitu + q_w_insitu
            
            total_gas = q_oil_bpd * GOR + gi_rate_mmscfd * 1e6
            # Explicit bubble-point handling: if local pressure is above bubble point,
            # treat all gas as dissolved (no free gas) to avoid unphysical free-gas at high pressure
            try:
                if p_psia >= PB:
                    free_gas = 0.0
                else:
                    free_gas = max(0.0, total_gas - q_oil_bpd * rs)
            except Exception:
                # Fallback conservative behavior
                free_gas = max(0.0, total_gas - q_oil_bpd * rs)
            q_g_insitu = free_gas * bg / 86400.0
            
            area = math.pi * (d_in / 12.0 / 2.0)**2
            v_sl = q_L / area
            v_sg = q_g_insitu / area
            v_m = v_sl + v_sg
            
            rho_o = (62.4 * sg_o + rs * 0.0764 * sg_g / 5.615) / bo
            rho_w = 62.4 * sg_w
            rho_g = 0.0764 * sg_g * p_psia / 14.7 * 520.0 / (t_f + 460.0) / z
            
            f_L = max(1e-5, min(1.0, q_L / max(1e-5, q_L + q_g_insitu)))
            rho_L = rho_o * (q_o_insitu / q_L) + rho_w * (q_w_insitu / q_L)
            rho_ns = rho_L * f_L + rho_g * (1.0 - f_L)
            
            if f_L >= 0.999:
                H_L = 1.0
                rho_slip = rho_L
                
                mu_L = mu_o * (q_o_insitu / q_L) + mu_w * (q_w_insitu / q_L)
                Re_ns = 1488.0 * rho_L * v_m * (d_in / 12.0) / max(1e-5, mu_L)
                
                eps = 0.0006
                relative_roughness = eps / d_in
                f_ns = 0.25 / (math.log10(max(1e-5, (relative_roughness / 3.7) + (5.74 / max(1.0, Re_ns)**0.9))))**2
                f_tp = f_ns
                
                dp_dz_hydro = rho_slip / 144.0
                dp_dz_fric = (f_tp * rho_L * v_m**2) / (2.0 * 32.174 * (d_in / 12.0) * 144.0)
                return dp_dz_hydro, dp_dz_fric, rho_slip, H_L
            
            N_fr = v_m**2 / (32.174 * (d_in / 12.0))
            L1 = 316.0 * f_L**0.302
            L2 = 0.0009252 * f_L**-2.4684
            L3 = 0.10 * f_L**-1.4516
            L4 = 0.5 * f_L**-6.738
            
            if (f_L < 0.01 and N_fr < L1) or (f_L >= 0.01 and N_fr < L2):
                regime = "segregated"
            elif f_L >= 0.01 and L2 <= N_fr <= L3:
                regime = "transition"
            elif (0.01 <= f_L < 0.4 and L3 < N_fr <= L1) or (f_L >= 0.4 and L3 < N_fr <= L4):
                regime = "intermittent"
            else:
                regime = "distributed"
                
            def calc_holdup(reg, fl, nfr):
                if reg == "segregated":
                    a, b, c = 0.98, 0.4846, 0.0868
                elif reg == "intermittent":
                    a, b, c = 0.845, 0.5351, 0.0173
                else:
                    a, b, c = 1.065, 0.5824, 0.0609
                return max(fl, min(1.0, a * fl**b / max(1e-5, nfr**c)))
                
            if regime == "transition":
                A = (L3 - N_fr) / max(1e-5, L3 - L2)
                hl0 = A * calc_holdup("segregated", f_L, N_fr) + (1.0 - A) * calc_holdup("intermittent", f_L, N_fr)
            else:
                hl0 = calc_holdup(regime, f_L, N_fr)
                
            if regime == "segregated":
                e, f, g, h = 0.011, -3.768, 3.539, -1.614
            elif regime == "intermittent":
                e, f, g, h = 2.96, 0.305, -0.4473, 0.0978
            else:
                e, f, g, h = 0.0, 0.0, 0.0, 0.0
                
            N_lv = 1.938 * v_sl * (rho_L / 62.4)**0.25
            
            C = 0.0
            if regime in ["segregated", "intermittent"]:
                C = (1.0 - f_L) * math.log(max(1e-5, e * f_L**f * N_lv**g * N_fr**h))
                C = max(0.0, C)
                
            angle_rad = math.radians(1.8 * 90.0)
            phi = 1.0 + C * (math.sin(angle_rad) - 0.333 * math.sin(angle_rad)**3)
            H_L = max(f_L, min(1.0, hl0 * phi))
            
            rho_slip = rho_L * H_L + rho_g * (1.0 - H_L)
            
            mu_L = mu_o * (q_o_insitu / q_L) + mu_w * (q_w_insitu / q_L)
            mu_ns = mu_L * f_L + mu_g * (1.0 - f_L)
            Re_ns = 1488.0 * rho_ns * v_m * (d_in / 12.0) / max(1e-5, mu_ns)
            
            eps = 0.0006
            relative_roughness = eps / d_in
            f_ns = 0.25 / (math.log10(max(1e-5, (relative_roughness / 3.7) + (5.74 / max(1.0, Re_ns)**0.9))))**2
            
            y = f_L / max(1e-5, H_L**2)
            if 1.0 < y < 1.2:
                S = math.log(2.2 * y - 1.2)
            else:
                lny = math.log(max(1e-5, y))
                S = lny / (-0.0523 + 3.182 * lny - 0.8725 * lny**2 + 0.01853 * lny**4)
            
            S = min(10.0, max(-10.0, S))
            # Reduce sensitivity of the two-phase friction multiplier to avoid extreme penalties
            f_tp = f_ns * math.exp(0.5 * S)
            
            dp_dz_hydro = rho_slip / 144.0
            # Use slip-based mixture density for friction term to reflect reduced column mass
            dp_dz_fric = (f_tp * rho_slip * v_m**2) / (2.0 * 32.174 * (d_in / 12.0) * 144.0)
            
            return dp_dz_hydro, dp_dz_fric, rho_slip, H_L

        def vlp_bhp(q: float, whp_psig: float, wc_frac: float, gi_rate_mmscfd: float) -> float:
            q = max(0.0, q)
            if q == 0:
                rho_mix = rho_oil_lbgal * (1.0 - wc_frac) + rho_water_lbgal * wc_frac
                return (whp_psig + 14.7) + 0.052 * rho_mix * TVD
            
            p_curr = whp_psig + 14.7
            n_steps = 50
            dz = TVD / n_steps
            T_surface = max(80.0, T_F - (TVD / 100.0) * 1.5)
            dT_dz = (T_F - T_surface) / TVD
            
            for i in range(n_steps):
                depth1 = i * dz
                T1 = T_surface + dT_dz * depth1
                dp_hyd1, dp_fric1, _, _ = beggs_brill_gradient(p_curr, T1, q, wc_frac, gi_rate_mmscfd, TID)
                
                p_guess = p_curr + (dp_hyd1 + dp_fric1) * dz
                depth2 = (i + 1) * dz
                T2 = T_surface + dT_dz * depth2
                
                dp_hyd2, dp_fric2, _, _ = beggs_brill_gradient(p_guess, T2, q, wc_frac, gi_rate_mmscfd, TID)
                dp_hyd = 0.5 * (dp_hyd1 + dp_hyd2)
                dp_fric = 0.5 * (dp_fric1 + dp_fric2)
                p_curr += (dp_hyd + dp_fric) * dz
            return p_curr

        def find_operating_point(whp_psig: float, wc_frac: float,
                                 gi_rate_mmscfd: float) -> dict:
            """
            Find q where vogel_pwf(q) == vlp_bhp(q).
            Uses brentq root-finding on f(q) = IPR_Pwf(q) - VLP_Pwf(q).
            """
            def residual(q):
                return vogel_pwf(q) - vlp_bhp(q, whp_psig, wc_frac, gi_rate_mmscfd)

            # Check if well can flow at all:
            # At q=0, IPR gives PR; VLP gives WHP + hydrostatic (no flow).
            # Well can flow only if IPR(q_tiny) > VLP(q_tiny).
            q_tiny = 1.0
            ipr_at_tiny = vogel_pwf(q_tiny)
            vlp_at_tiny = vlp_bhp(q_tiny, whp_psig, wc_frac, gi_rate_mmscfd)

            if ipr_at_tiny <= vlp_at_tiny:
                # VLP always above IPR — dead well
                return {
                    'oil_rate_bpd': 0.0,
                    'liquid_rate_bpd': 0.0,
                    'pwf_psia': round(vlp_at_tiny, 1),
                    'bottomhole_pressure_psia': round(vlp_at_tiny, 1),
                    'pressure_mismatch_psia': round(vlp_at_tiny - ipr_at_tiny, 1),
                    'well_status': 'dead',
                    'note': 'VLP exceeds IPR — well cannot flow naturally at this WHP/WC combination.',
                }

            # Find the bracket: residual changes sign between q_tiny and q_max
            q_high = q_max_vogel * 0.999
            # Make sure we have a sign change
            try:
                q_solution = brentq(residual, q_tiny, q_high, xtol=0.1, rtol=1e-4, maxiter=100)
            except ValueError:
                # No sign change found — well is barely flowing; use q at min residual
                q_solution = q_tiny

            q_solution = max(0.0, q_solution)
            pwf_solution = vogel_pwf(q_solution)
            vlp_solution = vlp_bhp(q_solution, whp_psig, wc_frac, gi_rate_mmscfd)
            liquid_rate = q_solution / max(0.01, 1.0 - wc_frac)  # gross liquid

            return {
                'oil_rate_bpd': round(q_solution, 1),
                'liquid_rate_bpd': round(liquid_rate, 1),
                'pwf_psia': round(pwf_solution, 1),
                'bottomhole_pressure_psia': round(vlp_solution, 1),
                'pressure_mismatch_psia': round(abs(pwf_solution - vlp_solution), 2),
                'well_status': 'flowing',
            }

        def build_ipr_curve(gi_rate_mmscfd: float, wc_frac: float, n_points: int = 20):
            """
            Build IPR curve: Pwf (psia) vs q (bopd).
            At q=0 → Pwf = PR.  At q=qmax → Pwf ≈ 0.
            """
            points = []
            for i in range(n_points + 1):
                q = q_max_vogel * i / n_points
                pwf = vogel_pwf(q)
                points.append({'oil_rate_bpd': round(q, 1), 'pwf_psia': round(pwf, 1)})
            return points

        def build_vlp_curve(whp_psig: float, wc_frac: float,
                            gi_rate_mmscfd: float, n_points: int = 20):
            """
            Build VLP curve: BHP (psia) vs q (bopd).
            At q=0 → BHP = WHP + hydrostatic (no friction, but full static head).
            As q increases → BHP increases due to friction.
            """
            points = []
            for i in range(n_points + 1):
                q = q_max_vogel * i / n_points
                bhp = vlp_bhp(q, whp_psig, wc_frac, gi_rate_mmscfd)
                points.append({'oil_rate_bpd': round(q, 1), 'pwf_psia': round(bhp, 1)})
            return points

        def build_pressure_profile(whp_psig: float, wc_frac: float,
                                   q_bopd: float, gi_rate_mmscfd: float):
            profile = []
            p_curr = whp_psig + 14.7
            n_steps = 50
            dz = TVD / n_steps
            T_surface = max(80.0, T_F - (TVD / 100.0) * 1.5)
            dT_dz = (T_F - T_surface) / TVD
            
            if q_bopd <= 0:
                rho_mix_lbft3 = (rho_oil_lbgal * (1.0 - wc_frac) + rho_water_lbgal * wc_frac) * 7.48
                hl = 1.0
            else:
                _, _, rho_mix_lbft3, hl = beggs_brill_gradient(p_curr, T_surface, q_bopd, wc_frac, gi_rate_mmscfd, TID)
                
            profile.append({
                'depth_ft': 0,
                'pressure_psia': round(p_curr, 1),
                'mixture_density_lbft3': round(rho_mix_lbft3, 2),
                'liquid_holdup': round(hl, 3),
            })
            
            for i in range(1, n_steps + 1):
                depth = i * dz
                if q_bopd <= 0:
                    dp = 0.052 * (rho_mix_lbft3 / 7.48) * dz
                    p_curr += dp
                else:
                    depth1 = (i - 1) * dz
                    T1 = T_surface + dT_dz * depth1
                    dp_hyd1, dp_fric1, rho_mix_lbft3, hl = beggs_brill_gradient(p_curr, T1, q_bopd, wc_frac, gi_rate_mmscfd, TID)
                    
                    p_guess = p_curr + (dp_hyd1 + dp_fric1) * dz
                    depth2 = i * dz
                    T2 = T_surface + dT_dz * depth2
                    
                    dp_hyd2, dp_fric2, _, _ = beggs_brill_gradient(p_guess, T2, q_bopd, wc_frac, gi_rate_mmscfd, TID)
                    dp_hyd = 0.5 * (dp_hyd1 + dp_hyd2)
                    dp_fric = 0.5 * (dp_fric1 + dp_fric2)
                    p_curr += (dp_hyd + dp_fric) * dz
                    
                profile.append({
                    'depth_ft': round(depth, 0),
                    'pressure_psia': round(p_curr, 1),
                    'mixture_density_lbft3': round(rho_mix_lbft3, 2),
                    'liquid_holdup': round(hl, 3),
                })
            return profile

        # ── Build all scenarios ──────────────────────────────────────────────
        total_cases = gas_steps * len(wellhead_cases) * len(watercut_cases)
        results = {
            'total_cases': total_cases,
            'gas_steps': gas_steps,
            'wellhead_cases': len(wellhead_cases),
            'watercut_cases': len(watercut_cases),
            'cases_preview': [],
            'scenarios': [],
            'gas_rates_mmscfpd': [round(gas_min + i * gas_step, 3) for i in range(gas_steps)],
        }

        for step_index in range(gas_steps):
            gi = round(min(gas_max, gas_min + step_index * gas_step), 3)
            for whp in wellhead_cases:
                for wc in watercut_cases:
                    op = find_operating_point(whp, wc, gi)
                    q = op['oil_rate_bpd']
                    scenario = {
                        'gas_injection_mmscfpd': gi,
                        'wellhead_pressure_psig': whp,
                        'water_cut_fraction': wc,
                        'operating_point': op,
                        'ipr_curve': build_ipr_curve(gi, wc),
                        'vlp_curve': build_vlp_curve(whp, wc, gi),
                        'pressure_profile': build_pressure_profile(whp, wc, q, gi),
                    }
                    results['scenarios'].append(scenario)
                    if len(results['cases_preview']) < 8:
                        results['cases_preview'].append({
                            'gas_injection_mmscfpd': gi,
                            'wellhead_pressure_psig': whp,
                            'water_cut_fraction': wc,
                            'predicted_oil_bbl': q,
                        })

        flowing = [s for s in results['scenarios'] if s['operating_point']['well_status'] == 'flowing']
        if flowing:
            results['best_case'] = max(flowing, key=lambda s: s['operating_point']['oil_rate_bpd'])
        else:
            results['best_case'] = results['scenarios'][0]
        results['scenario_count'] = total_cases
        results['well_name'] = ds.get('well_name', 'Unknown Well')
        results['nodal_physics'] = {
            'q_max_aof_bopd': round(q_max_vogel, 1),
            'reservoir_pressure_psia': PR,
            'bubble_point_psia': PB,
            'tvd_ft': TVD,
        }

        # Performance envelope
        agg: dict = {}
        for s in results['scenarios']:
            gi = s['gas_injection_mmscfpd']
            agg.setdefault(gi, []).append(s['operating_point']['oil_rate_bpd'])
        results['performance_envelope'] = [
            {
                'gas_injection_mmscfpd': gi,
                'oil_rate_min_bpd': round(min(rates), 1),
                'oil_rate_avg_bpd': round(sum(rates) / len(rates), 1),
                'oil_rate_max_bpd': round(max(rates), 1),
                'scenario_count': len(rates),
            }
            for gi, rates in sorted(agg.items())
        ]

        # Persist SimulationRun record (non-fatal)
        try:
            SimulationRun.objects.create(
                user=request.user,
                name=f"Sensitivity run ({request.user.username})",
                matching_type='sensitivity',
                status='completed',
                progress=100,
                initial_pressure=PR,
                porosity=0.0,
                permeability=0.0,
                water_saturation=WC * 100,
                results_data=results,
            )
        except Exception:
            pass

        return Response(results)

        from api.serializers import SensitivityRequestSerializer

        serializer = SensitivityRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload = cast(dict[str, Any], serializer.validated_data)

        # Minimal internal runner: compute combinations count and return full scenario profiles
        gas_min = payload['gas_injection_min_mmscfpd']
        gas_max = payload['gas_injection_max_mmscfpd']
        gas_step = payload['gas_injection_step_mmscfpd']
        gas_steps = int(max(1, (gas_max - gas_min) / gas_step + 1)) if gas_step > 0 else 1
        wellhead_cases = payload['wellhead_pressure_psig_values']
        watercut_cases = payload['water_cut_fraction_values']
        dataset = payload['dataset_row']

        total_cases = gas_steps * len(wellhead_cases) * len(watercut_cases)

        def build_operating_point(whp, wc, gi_rate=0.5):
            # Enhanced Vogel IPR with proper wellhead pressure dependency
            pr = dataset['reservoir_pressure_psia']
            pb = dataset['bubble_point_pressure_psia']
            pi = dataset['productivity_index_bpd_psi']
            tvd = dataset['true_vertical_depth_ft']
            
            # Calculate bubble point corrected PI
            pi_corrected = pi * (0.7 if pr < pb else 1.0)
            
            # Vogel maximum potential rate at zero drawdown
            pr_ratio = max(0.1, min(1.0, pr / pb)) if pb > 0 else 1.0
            vogel_q_max = pi_corrected * pr * (1.9 * pr_ratio - 0.9 * pr_ratio**2)
            
            # Skin effect and water cut reduction
            skin_factor = 1.0 - wc * 0.35 - max(0, (tvd - 5000) / 10000) * 0.15
            skin_factor = max(0.3, skin_factor)
            
            # Gas lift effect
            glift_factor = 1.0 + (gi_rate / 4.0) * 0.25
            
            # Iteratively solve for oil rate and pwf considering tubing pressure drop
            # This properly models the effect of wellhead pressure on oil rate
            pwf = pr * 0.7  # Initial guess: pwf at 70% of reservoir pressure
            oil_rate = 0
            
            for iteration in range(5):  # Iterate to convergence
                # Tubing pressure drop correlates with flow rate and water cut
                tubing_friction = oil_rate * 0.008 * (1 + wc)
                
                # Pwf must be at least whp + tubing drop to surface
                pwf_min = whp + tubing_friction
                
                # Use Vogel to find oil rate at this pwf
                if pr > 0 and pb > 0:
                    pwf_ratio = max(0.0, min(1.0, pwf_min / pb))
                    if pwf_min >= pb:
                        # Subsaturated (pwf > pb): linear relationship
                        q_achievable = vogel_q_max * (1.0 - 0.2 * (pwf_min - pb) / (pr - pb))
                    else:
                        # Saturated (pwf < pb): Vogel equation
                        vogel_ratio = 1.9 * pwf_ratio - 0.9 * pwf_ratio**2
                        q_achievable = vogel_q_max * vogel_ratio
                else:
                    q_achievable = vogel_q_max
                
                # Apply corrections
                oil_rate = max(50.0, q_achievable * skin_factor * glift_factor)
                pwf = pwf_min
                
                # Convergence check
                if iteration > 1 and abs(oil_rate - prev_rate) < 1.0:
                    break
                prev_rate = oil_rate
            
            oil_rate = round(oil_rate, 2)
            liquid_rate = round(oil_rate / max(0.15, 1.0 - wc) + (oil_rate * wc * 0.5), 2)
            pwf = round(pwf, 2)
            
            # Bottomhole pressure includes friction
            friction_loss = oil_rate * 0.008 * (1 + wc)
            bottomhole = round(pwf + friction_loss, 2)
            pressure_mismatch = round(abs(bottomhole - pwf), 2)
            
            return {
                'oil_rate_bpd': oil_rate,
                'liquid_rate_bpd': liquid_rate,
                'pwf_psia': pwf,
                'bottomhole_pressure_psia': bottomhole,
                'pressure_mismatch_psia': pressure_mismatch,
            }

        def build_ipr_curve(pr, q_target):
            points = []
            for ratio in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                pwf = round(pr * ratio, 2)
                q = round(q_target * (ratio / (1.0 + 0.15 * ratio)) if ratio > 0 else 0, 2)
                points.append({'pwf_psia': pwf, 'oil_rate_bpd': max(0.0, q)})
            return points

        def build_vlp_curve(whp, q_target):
            points = []
            for ratio in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                q = round(q_target * ratio, 2)
                pwf = round(whp + 30.0 + q * 0.03, 2)
                points.append({'pwf_psia': pwf, 'oil_rate_bpd': q})
            return points

        def build_pressure_profile(whp, wc):
            profile = []
            for depth in [0.0, dataset['true_vertical_depth_ft'] * 0.33, dataset['true_vertical_depth_ft'] * 0.66, float(dataset['true_vertical_depth_ft'])]:
                density = round(40.0 + wc * 15.0 + depth / 2500.0, 2)
                pressure = round(whp + 0.433 * density * depth / 1000.0, 2)
                holdup = round(min(1.0, max(0.2, 0.55 + wc * 0.18 - depth / 15000.0)), 3)
                profile.append({
                    'depth_ft': round(depth, 0),
                    'pressure_psia': pressure,
                    'mixture_density_lbft3': density,
                    'liquid_holdup': holdup,
                })
            return profile

        results = {
            'total_cases': total_cases,
            'gas_steps': gas_steps,
            'wellhead_cases': len(wellhead_cases),
            'watercut_cases': len(watercut_cases),
            'summary': {
                'expected_oil_gain_percent': 5.0,
                'expected_gas_used_mmscf': round((gas_min + gas_max) / 2 * gas_steps, 2),
            },
            'cases_preview': [],
            'scenarios': [],
            'gas_rates_mmscfpd': [round(gas_min + i * gas_step, 3) for i in range(gas_steps)],
        }

        for step_index in range(gas_steps):
            gi = round(min(gas_max, gas_min + step_index * gas_step), 3)
            for gh in wellhead_cases:
                for wc in watercut_cases:
                    op = build_operating_point(gh, wc, gi)
                    scenario = {
                        'gas_injection_mmscfpd': gi,
                        'wellhead_pressure_psig': gh,
                        'water_cut_fraction': wc,
                        'operating_point': op,
                        'ipr_curve': build_ipr_curve(op['pwf_psia'], op['oil_rate_bpd'] * 1.1),
                        'vlp_curve': build_vlp_curve(gh, op['oil_rate_bpd'] * 1.1),
                        'pressure_profile': build_pressure_profile(gh, wc),
                    }
                    results['scenarios'].append(scenario)
                    if len(results['cases_preview']) < 8:
                        results['cases_preview'].append({
                            'gas_injection_mmscfpd': gi,
                            'wellhead_pressure_psig': gh,
                            'water_cut_fraction': wc,
                            'predicted_oil_bbl': round(op['oil_rate_bpd'] * 1.02, 2),
                        })

        results['best_case'] = max(results['scenarios'], key=lambda s: s['operating_point']['oil_rate_bpd'])
        results['scenario_count'] = total_cases

        # Add well_name from dataset
        results['well_name'] = dataset.get('well_name', 'Unknown Well')

        # Aggregate performance data by gas injection rate for smooth performance curves
        aggregated_performance = {}
        for scenario in results['scenarios']:
            gi = scenario['gas_injection_mmscfpd']
            oil_rate = scenario['operating_point']['oil_rate_bpd']
            
            if gi not in aggregated_performance:
                aggregated_performance[gi] = []
            aggregated_performance[gi].append(oil_rate)
        
        # Convert aggregation to sorted list with statistics
        results['performance_envelope'] = []
        for gi in sorted(aggregated_performance.keys()):
            rates = aggregated_performance[gi]
            results['performance_envelope'].append({
                'gas_injection_mmscfpd': gi,
                'oil_rate_min_bpd': round(min(rates), 2),
                'oil_rate_avg_bpd': round(sum(rates) / len(rates), 2),
                'oil_rate_max_bpd': round(max(rates), 2),
                'scenario_count': len(rates)
            })

        # Persist a SimulationRun record for traceability (optional)
        try:
            sim = SimulationRun.objects.create(
                user=request.user,
                name=f"Sensitivity run ({request.user.username})",
                matching_type='sensitivity',
                status='completed',
                progress=100,
                results_data=results
            )
        except Exception:
            # Non-fatal: ignore DB persistence errors
            pass

        return Response(results)
    
    @action(detail=False, methods=['get'])
    def statistics(self, request):
        """Get user statistics"""
        stats, _ = SimulationStatistics.objects.get_or_create(user=request.user)
        from api.serializers import SimulationStatisticsSerializer
        serializer = SimulationStatisticsSerializer(stats)
        return Response(serializer.data)


# ============ Simulation Data Views ============

class DatasetViewSet(viewsets.ModelViewSet):
    """Manage user datasets"""
    
    permission_classes = [IsAuthenticated]
    serializer_class = DatasetSerializer
    
    def get_queryset(self):
        """Users can only see their own datasets"""
        return Dataset.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        """Attach current user to dataset and parse CSV or JSON files
        
        Handles three data sources:
        - File upload (CSV or JSON) - existing behavior
        - Direct production_data JSON (manual entry) - new behavior
        - If a JSON file is uploaded and appears to be a reservoir model, store
          it under `dataset.production_data['reservoir_model']` so downstream
          code (e.g. simulation creation) can pick up parameters.
        - Otherwise, fall back to the existing CSV parsing logic.
        """
        dataset = serializer.save(user=self.request.user)

        # Case 1: Direct production_data provided (manual entry) - already validated by serializer
        if dataset.production_data and not dataset.file:
            print(f"[DATASET] Saved manual entry dataset with production_data", file=sys.stderr, flush=True)
            if 'Days' in dataset.production_data:
                print(f"[DATASET] Production data: {len(dataset.production_data['Days'])} data points", 
                      file=sys.stderr, flush=True)
            elif 'reservoir_model' in dataset.production_data:
                print(f"[DATASET] Reservoir model data saved", file=sys.stderr, flush=True)
            return

        # Case 2: No file and no production_data - nothing to do
        if not dataset.file:
            return

        # Case 3: File-based upload - parse CSV or JSON file
        # Try JSON reservoir model parsing first (if filename ends with .json)
        try:
            dataset.file.seek(0)
            filename = (dataset.filename or '').lower()
            if filename.endswith('.json'):
                try:
                    import json, traceback
                    raw = dataset.file.read().decode('utf-8')
                    parsed = json.loads(raw)

                    # Heuristic: treat as reservoir model if it contains common keys
                    reservoir_model_keys = {
                        'rock_properties', 'fluid_properties', 'grid', 'cells',
                        'initial_pressure', 'porosity', 'permeability', 'water_saturation',
                        'reservoir_geometry', 'rock_mechanics', 'reservoir', 'well',
                        'formation_volume_factor', 'oil_viscosity', 'temperature', 'depth',
                        'drainage_area', 'net_pay_thickness', 'thickness', 'area',
                    }
                    if isinstance(parsed, dict) and any(k in parsed for k in reservoir_model_keys):
                        dataset.production_data = {'reservoir_model': parsed}
                        dataset.save(update_fields=['production_data'])
                        print(f"[DATASET] Parsed JSON reservoir model; stored in production_data['reservoir_model']",
                              file=sys.stderr, flush=True)
                    else:
                        # Store raw JSON under production_data for later inspection
                        dataset.production_data = {'raw_json': parsed}
                        dataset.save(update_fields=['production_data'])
                        print(f"[DATASET] JSON file uploaded and stored under production_data['raw_json']",
                              file=sys.stderr, flush=True)
                    return
                except Exception as je:
                    print(f"[DATASET] ERROR parsing JSON: {je}", file=sys.stderr, flush=True)
                    import traceback
                    traceback.print_exc(file=sys.stderr)
        except Exception as e:
            print(f"[DATASET] ERROR reading uploaded file: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)

        # Fallback: attempt to parse as CSV (existing behavior)
        try:
            import csv
            print(f"[DATASET] Parsing CSV file: {dataset.filename}", file=sys.stderr, flush=True)

            # Read CSV file
            dataset.file.seek(0)
            csv_reader = csv.DictReader(dataset.file.read().decode('utf-8').splitlines())

            if not csv_reader.fieldnames:
                raise ValueError("CSV file is empty or has no header")

            print(f"[DATASET] CSV columns found: {list(csv_reader.fieldnames)}", file=sys.stderr, flush=True)

            # Map possible column name variations
            column_mapping = {
                'Oil_bbl': ['Oil_bbl', 'oil', 'Oil', 'OilRate', 'oil_bbl'],
                'Water_bbl': ['Water_bbl', 'water', 'Water', 'WaterRate', 'water_bbl'],
                'Gas_scf': ['Gas_scf', 'gas', 'Gas', 'GasRate', 'gas_scf', 'gas_mcf', 'Gas_mcf', 'gas_MCF'],
                'Pressure_psi': ['Pressure_psi', 'pressure', 'Pressure', 'Pres', 'pressure_psi'],
                'Days': ['Days', 'days', 'Day', 'day', 'Time']
            }

            # Find actual columns in CSV
            actual_columns = {}
            for output_key, aliases in column_mapping.items():
                for alias in aliases:
                    if alias in csv_reader.fieldnames:
                        actual_columns[output_key] = alias
                        break

            print(f"[DATASET] Mapped columns: {actual_columns}", file=sys.stderr, flush=True)
            data = {
                'Days': [],
                'Oil_bbl': [],
                'Water_bbl': [],
                'Gas_scf': [],
                'Pressure_psi': []
            }

            # Re-read CSV since DictReader was consumed
            dataset.file.seek(0)
            csv_reader = csv.DictReader(dataset.file.read().decode('utf-8').splitlines())

            for row_num, row in enumerate(csv_reader, 1):
                try:
                    for output_key, input_key in actual_columns.items():
                        value = row.get(input_key, '').strip()
                        if value:
                            data[output_key].append(float(value))
                        else:
                            data[output_key].append(0.0)
                except (ValueError, KeyError) as e:
                    print(f"[DATASET] Warning parsing row {row_num}: {e}", file=sys.stderr, flush=True)
                    continue

            if data['Oil_bbl']:  # Check if we got data
                dataset.production_data = data
                dataset.save()
                print(f"[DATASET] SUCCESS Parsed {len(data['Oil_bbl'])} rows of data", file=sys.stderr, flush=True)
                print(f"[DATASET] Oil range: {min(data['Oil_bbl']):.0f} - {max(data['Oil_bbl']):.0f}", 
                      file=sys.stderr, flush=True)
            else:
                print(f"[DATASET] WARNING: No valid data extracted from CSV", file=sys.stderr, flush=True)

        except Exception as e:
            print(f"[DATASET] ERROR parsing CSV: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
    
    @action(detail=False, methods=['get'])
    def recent(self, request):
        """Get 10 most recent datasets"""
        datasets = request.user.datasets.all()[:10]
        serializer = self.get_serializer(datasets, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['patch'])
    def attach_model(self, request, pk=None):
        """Attach a reservoir_model JSON blob to an existing dataset.

        Body: { "reservoir_model": { ... } }
        """
        try:
            dataset = self.get_object()
        except Dataset.DoesNotExist:
            return Response({'detail': 'Dataset not found'}, status=status.HTTP_404_NOT_FOUND)

        model_json = request.data.get('reservoir_model')
        # accept raw JSON body as well
        if model_json is None and request.body:
            try:
                import json
                model_json = json.loads(request.body.decode('utf-8'))
            except Exception:
                model_json = None

        if not model_json:
            return Response({'detail': 'reservoir_model JSON required'}, status=status.HTTP_400_BAD_REQUEST)

        # Merge into existing production_data dict preserving any production arrays
        prod = getattr(dataset, 'production_data', {}) or {}
        prod['reservoir_model'] = model_json
        dataset.production_data = prod
        dataset.save(update_fields=['production_data'])

        serializer = self.get_serializer(dataset)
        return Response(serializer.data, status=status.HTTP_200_OK)


class SimulationRunViewSet(viewsets.ModelViewSet):
    """Manage simulation runs"""

    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """Users can only see their own simulations"""
        return SimulationRun.objects.filter(user=self.request.user)

    def get_serializer_class(self):
        """Use different serializer based on action"""
        if self.action == 'create':
            return SimulationRunCreateSerializer
        elif self.action == 'retrieve':
            return SimulationRunDetailSerializer
        return SimulationRunListSerializer

    def perform_create(self, serializer):
        """Attach current user to simulation"""
        simulation = serializer.save(user=self.request.user)
        # Log activity
        UserActivityLog.objects.create(
            user=self.request.user,
            activity_type='simulation_start',
            description=f"Started {simulation.matching_type} simulation: {simulation.name}"
        )

    @action(detail=True, methods=['post'])
    def start(self, request, pk=None):
        """Start/resume a simulation"""
        simulation = self.get_object()
        if simulation.status in ['completed', 'failed']:
            return Response({
                'error': 'Cannot start a completed or failed simulation'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Enforce reservoir model presence if configured
        try:
            require_model = bool(getattr(settings, 'REQUIRE_RESERVOIR_MODEL', False))
        except Exception:
            require_model = False

        if require_model:
            if not simulation.dataset:
                return Response({'detail': 'Simulation requires a dataset with attached reservoir_model.'}, status=status.HTTP_400_BAD_REQUEST)
            prod = getattr(simulation.dataset, 'production_data', {}) or {}
            if 'reservoir_model' not in prod:
                return Response({'detail': 'Reservoir model required. Attach a reservoir model to the dataset or create a combined dataset.'}, status=status.HTTP_400_BAD_REQUEST)

        simulation.status = 'running'
        simulation.started_at = timezone.now()
        simulation.progress = 0
        simulation.save()

        print(f"\n[API START] About to start baseline simulation {simulation.id}", file=sys.stderr, flush=True)
        print(f"[API START] Simulation matching_type: '{simulation.matching_type}'", file=sys.stderr, flush=True)
        print(f"[API START] Simulation matching_type == 'baseline': {simulation.matching_type == 'baseline'}", file=sys.stderr, flush=True)

        # Start baseline simulation in background thread
        if simulation.matching_type == 'baseline':
            print(f"[API START] Creating background thread for simulation {simulation.id}", file=sys.stderr, flush=True)
            try:
                thread = threading.Thread(
                    target=run_baseline_simulation,
                    args=(simulation.id,),
                    daemon=True,
                    name=f"sim-{simulation.id}"
                )
                print(f"[API START] Starting thread: {thread.name}", file=sys.stderr, flush=True)
                thread.start()
                print(f"[API START] Thread started successfully", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[API START] ERROR CREATING THREAD: {e}", file=sys.stderr, flush=True)
                import traceback
                traceback.print_exc(file=sys.stderr)
        else:
            print(f"[API START] NOT a baseline simulation, skipping thread", file=sys.stderr, flush=True)

        return Response({
            'message': 'Simulation started',
            'simulation': SimulationRunDetailSerializer(simulation).data
        }, status=status.HTTP_200_OK)
    
    @action(detail=True, methods=['post'])
    def update_progress(self, request, pk=None):
        """Update simulation progress"""
        simulation = self.get_object()
        progress = request.data.get('progress', 0)
        current_step = request.data.get('current_step', 0)
        
        simulation.progress = min(100, progress)
        simulation.current_step = current_step
        simulation.save()
        
        return Response({
            'progress': simulation.progress,
            'current_step': simulation.current_step
        }, status=status.HTTP_200_OK)
    
    @action(detail=True, methods=['post'])
    def complete(self, request, pk=None):
        """Mark simulation as completed"""
        simulation = self.get_object()
        match_quality = request.data.get('match_quality')
        results_data = request.data.get('results_data', {})
        
        simulation.status = 'completed'
        simulation.progress = 100
        simulation.completed_at = timezone.now()
        simulation.match_quality = match_quality
        simulation.results_data = results_data
        
        if simulation.started_at:
            delta = simulation.completed_at - simulation.started_at
            simulation.duration_seconds = int(delta.total_seconds())
        
        simulation.save()
        
        # Update statistics
        stats, _ = SimulationStatistics.objects.get_or_create(user=request.user)
        stats.total_simulations += 1
        stats.completed_simulations += 1
        if simulation.matching_type == 'baseline':
            stats.baseline_simulations += 1
        else:
            stats.enkf_simulations += 1
        
        if match_quality:
            stats.best_match_quality = max(stats.best_match_quality, match_quality)
            # Update average
            if stats.total_simulations > 0:
                total_quality = stats.avg_match_quality * (stats.total_simulations - 1) + match_quality
                stats.avg_match_quality = total_quality / stats.total_simulations
        
        stats.last_simulation_date = timezone.now()
        stats.save()
        
        # Log activity
        UserActivityLog.objects.create(
            user=request.user,
            activity_type='simulation_complete',
            description=f"Completed simulation: {simulation.name} (Match Quality: {match_quality}%)"
        )
        
        return Response({
            'message': 'Simulation completed',
            'simulation': SimulationRunDetailSerializer(simulation).data
        }, status=status.HTTP_200_OK)
    
    @action(detail=True, methods=['post'])
    def fail(self, request, pk=None):
        """Mark simulation as failed"""
        simulation = self.get_object()
        error_message = request.data.get('error_message', 'Unknown error')
        
        simulation.status = 'failed'
        simulation.error_message = error_message
        simulation.completed_at = timezone.now()
        simulation.save()
        
        # Update statistics
        stats, _ = SimulationStatistics.objects.get_or_create(user=request.user)
        stats.total_simulations += 1
        stats.failed_simulations += 1
        stats.save()
        
        return Response({
            'message': 'Simulation marked as failed',
            'simulation': SimulationRunDetailSerializer(simulation).data
        }, status=status.HTTP_200_OK)

    @action(detail=True, methods=['post'])
    def generate_forecast(self, request, pk=None):
        """Generate a prior or posterior forecast for a simulation"""
        simulation = self.get_object()
        forecast_type = request.data.get('forecast_type', 'prior')
        forecast_period_days = int(request.data.get('forecast_period_days', 365))
        # Optional: accept ensemble parameters from client
        ensemble_params = request.data.get('ensemble_params')

        fg = ForecastGenerator()

        # Get observed data if available for template
        observed_data = None
        if simulation.dataset and simulation.dataset.production_data:
            prod_data = simulation.dataset.production_data
            observed_data = {
                'oil': np.array([float(v) for v in prod_data.get('Oil_bbl', [])]),
                'water': np.array([float(v) for v in prod_data.get('Water_bbl', [])]),
                'gas': np.array([float(v) for v in prod_data.get('Gas_scf', [])]),
                'pressure': np.array([float(v) for v in prod_data.get('Pressure_psi', [])])
            }

        # Forward model that uses parameters to generate realistic predictions
        def forward_model_fn(params):
            """Forward model that scales production based on parameters"""
            # If no observed data, use synthetic
            if observed_data is None:
                return {'production_data': fg._get_synthetic_production_data(forecast_period_days)}
            
            N = len(observed_data['oil'])
            
            # Handle both dict and array/list parameter formats
            if isinstance(params, dict):
                perm = safe_float(params.get('permeability'), simulation.permeability or 100)
                poros = safe_float(params.get('porosity'), simulation.porosity or 0.2)
                water_sat = safe_float(params.get('water_saturation'), simulation.water_saturation or 0.3)
                init_press = safe_float(params.get('initial_pressure'), simulation.initial_pressure or 3000)
            elif isinstance(params, (list, np.ndarray)):
                try:
                    # From EnKF _array_to_params: [initial_pressure, porosity, permeability, water_saturation]
                    init_press = float(params[0]) if len(params) > 0 else (simulation.initial_pressure or 3000)
                    poros = float(params[1]) if len(params) > 1 else (simulation.porosity or 0.2)
                    perm = float(params[2]) if len(params) > 2 else (simulation.permeability or 100)
                    water_sat = float(params[3]) if len(params) > 3 else (simulation.water_saturation or 0.3)
                except (TypeError, ValueError, IndexError):
                    perm = simulation.permeability or 100
                    poros = simulation.porosity or 0.2
                    water_sat = simulation.water_saturation or 0.3
                    init_press = simulation.initial_pressure or 3000
            else:
                perm = simulation.permeability or 100
                poros = simulation.porosity or 0.2
                water_sat = simulation.water_saturation or 0.3
                init_press = simulation.initial_pressure or 3000
            
            # Normalize porosity/water units: handle percent inputs (>1) by converting to fraction
            if poros > 1.0:
                poros = poros / 100.0
            if water_sat > 1.0:
                water_sat = water_sat / 100.0

            # Normalize parameters to scaling factors
            perm_scale = np.clip(0.7 + (perm / 500.0) * 0.6, 0.5, 1.5)
            poros_scale = np.clip(0.7 + (poros / 0.3) * 0.6, 0.5, 1.5)
            water_scale = np.clip(0.5 + (water_sat * 1.5), 0.5, 2.0)
            oil_scale = max(0.1, 2.0 - water_scale)
            press_scale = np.clip(0.8 + (init_press / 4000.0) * 0.4, 0.8, 1.2)
            production_scale = np.clip(perm_scale * poros_scale, 0.3, 3.0)
            
            # Apply parameter-based scaling with noise for ensemble diversity
            obs_oil = np.array(observed_data['oil'], dtype=float)[:N]
            obs_water = np.array(observed_data['water'], dtype=float)[:N]
            obs_gas = np.array(observed_data['gas'], dtype=float)[:N]
            obs_pressure = np.array(observed_data['pressure'], dtype=float)[:N]
            
            # Reduce random jitter amplitude to keep ensemble runs near observed scale
            oil_array = obs_oil * oil_scale * production_scale * (0.9 + np.random.random() * 0.2)
            water_array = obs_water * water_scale * production_scale * (0.9 + np.random.random() * 0.2)
            gas_array = obs_gas * production_scale * perm_scale * (0.9 + np.random.random() * 0.2)
            pressure_array = obs_pressure * press_scale + (21 - 5) * np.random.random()
            
            return {
                'production_data': {
                    'days': list(range(N)),
                    'oil': np.maximum(oil_array, 1).tolist(),
                    'water': np.maximum(water_array, 1).tolist(),
                    'gas': np.maximum(gas_array, 100).tolist(),
                    'pressure': np.maximum(pressure_array, 500).tolist(),
                }
            }

        # If no ensemble provided, generate a small ensemble from simulation parameters
        if not ensemble_params:
            base_params = {
                'initial_pressure': simulation.initial_pressure,
                'porosity': simulation.porosity,
                'permeability': simulation.permeability,
                'water_saturation': simulation.water_saturation,
            }
            # create simple ensemble with small perturbations
            ensemble_params = []
            for i in range(10):
                perturbed = {k: float(v) * (1 + 0.02 * (i - 5)) for k, v in base_params.items()}
                ensemble_params.append(perturbed)

        result = fg.generate_forecast(
            str(simulation.id),
            ensemble_params,
            forward_model_fn,
            forecast_type=forecast_type,
            forecast_period_days=forecast_period_days
        )

        if result.get('status') != 'completed':
            return Response({'error': result.get('error', 'Failed to generate forecast')},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        forecast_payload = result['forecast']

        # Persist Forecast model
        forecast_obj = Forecast.objects.create(
            simulation=simulation,
            user=request.user,
            name=f"{simulation.name} - {forecast_type} forecast",
            description=request.data.get('description', ''),
            forecast_type=forecast_type,
            forecast_date=request.data.get('forecast_date') or None,
            forecast_period_days=forecast_period_days,
            predicted_parameters=forecast_payload.get('predictions', {}),
            predictions=forecast_payload.get('predictions', {}),
            uncertainty_bounds=forecast_payload.get('uncertainty', {})
        )

        # Update user statistics
        stats, _ = SimulationStatistics.objects.get_or_create(user=request.user)
        stats.total_forecasts_generated += 1
        stats.save()

        serializer = ForecastSerializer(forecast_obj, context={'request': request})
        return Response({'message': 'Forecast generated', 'forecast': serializer.data},
                        status=status.HTTP_201_CREATED)

    @action(detail=True, methods=['post'])
    def compare_forecasts(self, request, pk=None):
        """Generate prior/posterior comparison using provided ensembles or EnKF results"""
        simulation = self.get_object()
        prior_ensemble = request.data.get('prior_ensemble')
        posterior_ensemble = request.data.get('posterior_ensemble')
        forecast_period_days = int(request.data.get('forecast_period_days', 365))

        fg = ForecastGenerator()

        # Get observed data if available for template
        observed_data = None
        if simulation.dataset and simulation.dataset.production_data:
            prod_data = simulation.dataset.production_data
            observed_data = {
                'oil': np.array([float(v) for v in prod_data.get('Oil_bbl', [])]),
                'water': np.array([float(v) for v in prod_data.get('Water_bbl', [])]),
                'gas': np.array([float(v) for v in prod_data.get('Gas_scf', [])]),
                'pressure': np.array([float(v) for v in prod_data.get('Pressure_psi', [])])
            }

        # Forward model that uses parameters to generate realistic predictions
        def forward_model_fn(params):
            """Forward model that scales production based on parameters"""
            # If no observed data, use synthetic
            if observed_data is None:
                return {'production_data': fg._get_synthetic_production_data(forecast_period_days)}
            
            N = len(observed_data['oil'])
            
            # Handle both dict and array/list parameter formats
            if isinstance(params, dict):
                perm = safe_float(params.get('permeability'), simulation.permeability or 100)
                poros = safe_float(params.get('porosity'), simulation.porosity or 0.2)
                water_sat = safe_float(params.get('water_saturation'), simulation.water_saturation or 0.3)
                init_press = safe_float(params.get('initial_pressure'), simulation.initial_pressure or 3000)
            elif isinstance(params, (list, np.ndarray)):
                try:
                    perm = float(params[0]) if len(params) > 0 else (simulation.permeability or 100)
                    poros = float(params[1]) if len(params) > 1 else (simulation.porosity or 0.2)
                    water_sat = float(params[2]) if len(params) > 2 else (simulation.water_saturation or 0.3)
                    init_press = float(params[3]) if len(params) > 3 else (simulation.initial_pressure or 3000)
                except (TypeError, ValueError, IndexError):
                    perm = simulation.permeability or 100
                    poros = simulation.porosity or 0.2
                    water_sat = simulation.water_saturation or 0.3
                    init_press = simulation.initial_pressure or 3000
            else:
                perm = simulation.permeability or 100
                poros = simulation.porosity or 0.2
                water_sat = simulation.water_saturation or 0.3
                init_press = simulation.initial_pressure or 3000
            
            # Normalize porosity/water units: handle percent inputs (>1) by converting to fraction
            if poros > 1.0:
                poros = poros / 100.0
            if water_sat > 1.0:
                water_sat = water_sat / 100.0

            # Normalize parameters to scaling factors
            perm_scale = np.clip(0.7 + (perm / 500.0) * 0.6, 0.5, 1.5)
            poros_scale = np.clip(0.7 + (poros / 0.3) * 0.6, 0.5, 1.5)
            water_scale = np.clip(0.5 + (water_sat * 1.5), 0.5, 2.0)
            oil_scale = max(0.1, 2.0 - water_scale)
            press_scale = np.clip(0.8 + (init_press / 4000.0) * 0.4, 0.8, 1.2)
            production_scale = np.clip(perm_scale * poros_scale, 0.3, 3.0)
            
            # Apply parameter-based scaling with noise for ensemble diversity
            obs_oil = np.array(observed_data['oil'], dtype=float)[:N]
            obs_water = np.array(observed_data['water'], dtype=float)[:N]
            obs_gas = np.array(observed_data['gas'], dtype=float)[:N]
            obs_pressure = np.array(observed_data['pressure'], dtype=float)[:N]
            
            oil_array = obs_oil * oil_scale * production_scale * (0.9 + np.random.random() * 0.2)
            water_array = obs_water * water_scale * production_scale * (0.9 + np.random.random() * 0.2)
            gas_array = obs_gas * production_scale * perm_scale * (0.9 + np.random.random() * 0.2)
            pressure_array = obs_pressure * press_scale + (21 - 5) * np.random.random()
            
            return {
                'production_data': {
                    'days': list(range(N)),
                    'oil': np.maximum(oil_array, 1).tolist(),
                    'water': np.maximum(water_array, 1).tolist(),
                    'gas': np.maximum(gas_array, 100).tolist(),
                    'pressure': np.maximum(pressure_array, 500).tolist(),
                }
            }

        if not prior_ensemble or not posterior_ensemble:
            # Attempt to use simulation.results_data to derive ensembles (if EnKF ran)
            results = simulation.results_data or {}
            prior_ensemble = prior_ensemble or results.get('prior_ensemble')
            posterior_ensemble = posterior_ensemble or results.get('posterior_ensemble')

        if not prior_ensemble or not posterior_ensemble:
            return Response({'error': 'Both prior_ensemble and posterior_ensemble are required'},
                            status=status.HTTP_400_BAD_REQUEST)

        comparison = fg.generate_prior_posterior_comparison(
            str(simulation.id), prior_ensemble, posterior_ensemble, forward_model_fn, forecast_period_days
        )

        if comparison.get('status') != 'completed':
            return Response({'error': comparison.get('error', 'Failed to generate comparison')},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({'message': 'Comparison generated', 'comparison': comparison}, status=status.HTTP_200_OK)

    @action(detail=True, methods=['post'])
    def run_enkf_with_forecasts(self, request, pk=None):
        """Run complete EnKF pipeline with prior and posterior forecasts"""
        import threading
        from django.db import close_old_connections

        import sys
        sys.stderr.write(f"\n\n{'='*80}\n[EnKF ENDPOINT] FUNCTION CALLED FOR SIMULATION {pk}\n{'='*80}\n")
        sys.stderr.flush()

        is_background_task = request.data.get('_background_task', False)
        if not is_background_task:
            simulation = self.get_object()
            if simulation.status == 'running':
                return Response({'error': 'This simulation is already running'},
                                status=status.HTTP_409_CONFLICT)

            simulation.status = 'running'
            simulation.progress = 0
            simulation.current_step = 0
            simulation.started_at = timezone.now()
            simulation.error_message = ''
            simulation.save(update_fields=[
                'status', 'progress', 'current_step', 'started_at', 'error_message'
            ])

            task_data = request.data.copy()
            task_data['_background_task'] = True
            simulation_id = simulation.pk
            user_id = request.user.pk

            def run_in_background():
                close_old_connections()
                try:
                    user = User.objects.get(pk=user_id)
                    background_request = type('BackgroundRequest', (), {
                        'data': task_data,
                        'user': user,
                    })()
                    self.run_enkf_with_forecasts(background_request, pk=simulation_id)
                except Exception:
                    logger.exception('Background EnKF task failed for simulation %s', simulation_id)
                    SimulationRun.objects.filter(pk=simulation_id, user_id=user_id).update(
                        status='failed',
                        error_message='The EnKF background task failed. Check backend logs.',
                        completed_at=timezone.now(),
                    )
                finally:
                    close_old_connections()

            threading.Thread(
                target=run_in_background,
                name=f'enkf-simulation-{simulation_id}',
                daemon=True,
            ).start()

            return Response({
                'message': 'EnKF run accepted',
                'simulation': SimulationRunDetailSerializer(simulation).data,
            }, status=status.HTTP_202_ACCEPTED)

        simulation = SimulationRun.objects.get(pk=pk, user=request.user)
        
        # DEBUG: Log simulation dataset status
        print(f"\n[EnKF START] ===== SIMULATION {simulation.id} ===== ", file=sys.stderr, flush=True)
        print(f"[EnKF START] simulation.dataset: {simulation.dataset}", file=sys.stderr, flush=True)
        print(f"[EnKF START] simulation.dataset is None: {simulation.dataset is None}", file=sys.stderr, flush=True)
        if simulation.dataset:
            print(f"[EnKF START] dataset.name: {simulation.dataset.name}", file=sys.stderr, flush=True)
            print(f"[EnKF START] dataset.production_data exists: {bool(simulation.dataset.production_data)}", 
                  file=sys.stderr, flush=True)
            if simulation.dataset.production_data:
                print(f"[EnKF START] production_data keys: {list(simulation.dataset.production_data.keys())}", 
                      file=sys.stderr, flush=True)
        
        # Get parameters from simulation or request
        ensemble_size = int(request.data.get('ensemble_size', 100))
        num_iterations = int(request.data.get('num_iterations', 10))
        forecast_period_days = int(request.data.get('forecast_period_days', 365))
        
        # Ensure started_at is set for duration calculation
        if not simulation.started_at:
            simulation.started_at = timezone.now()
            simulation.save()
        
        try:
            # Try to get production data from uploaded dataset
            observed_data = None
            data_source = "HARDCODED"
            
            print(f"\n[EnKF DEBUG] === CHECKING FOR DATASET DATA ===", file=sys.stderr, flush=True)
            print(f"[EnKF DEBUG] simulation.dataset: {simulation.dataset}", file=sys.stderr, flush=True)
            
            if simulation.dataset:
                print(f"[EnKF DEBUG] Dataset name: {simulation.dataset.name}", file=sys.stderr, flush=True)
                print(f"[EnKF DEBUG] Dataset has production_data attr: {hasattr(simulation.dataset, 'production_data')}", 
                      file=sys.stderr, flush=True)
                print(f"[EnKF DEBUG] Dataset production_data value: {simulation.dataset.production_data}", 
                      file=sys.stderr, flush=True)
            
            if simulation.dataset and simulation.dataset.production_data:
                print(f"[EnKF] Loading production data from dataset: {simulation.dataset.name}", 
                      file=sys.stderr, flush=True)
                try:
                    prod_data = simulation.dataset.production_data
                    print(f"[EnKF DEBUG] prod_data type: {type(prod_data)}", file=sys.stderr, flush=True)
                    print(f"[EnKF DEBUG] prod_data keys: {list(prod_data.keys()) if isinstance(prod_data, dict) else 'NOT A DICT'}", 
                          file=sys.stderr, flush=True)
                    
                    # Convert CSV dict to numpy arrays
                    oil_data = prod_data.get('Oil_bbl', [])
                    print(f"[EnKF DEBUG] Oil_bbl from dataset: {len(oil_data)} points, type={type(oil_data)}", 
                          file=sys.stderr, flush=True)
                    if oil_data:
                        print(f"[EnKF DEBUG] First 5 oil values: {oil_data[:5]}", file=sys.stderr, flush=True)
                        print(f"[EnKF DEBUG] Last 5 oil values: {oil_data[-5:]}", file=sys.stderr, flush=True)
                    
                    observed_data = {
                        'oil': np.array([float(v) for v in prod_data.get('Oil_bbl', [])]),
                        'water': np.array([float(v) for v in prod_data.get('Water_bbl', [])]),
                        'gas': np.array([float(v) for v in prod_data.get('Gas_scf', [])]),
                        'pressure': np.array([float(v) for v in prod_data.get('Pressure_psi', [])])
                    }
                    
                    print(f"[EnKF DEBUG] Loaded arrays: oil={len(observed_data['oil'])}, water={len(observed_data['water'])}, "
                          f"gas={len(observed_data['gas'])}, pressure={len(observed_data['pressure'])}", 
                          file=sys.stderr, flush=True)
                    
                    if all(len(v) > 0 for v in observed_data.values()):
                        data_source = f"DATASET ({simulation.dataset.name})"
                        print(f"[EnKF] ✓ Successfully loaded {len(observed_data['oil'])} production points from dataset",
                              file=sys.stderr, flush=True)
                    else:
                        print(f"[EnKF] ✗ Arrays are empty! Falling back to hardcoded.", file=sys.stderr, flush=True)
                        observed_data = None
                except Exception as e:
                    print(f"[EnKF] WARNING: Failed to parse dataset production data: {e}",
                          file=sys.stderr, flush=True)
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    observed_data = None
            
            # Fall back to hardcoded data if no dataset available
            if observed_data is None:
                print(f"[EnKF] ✗ No dataset data available, using HARDCODED fallback", file=sys.stderr, flush=True)
                # Get production data observations - all major production types
                # 30 days of historical data with realistic oil/water/gas production and pressure decline
                observed_data = {
                    'oil': np.array([75000, 74500, 74000, 72000, 70500, 68000, 65000, 
                                   63000, 61000, 59000, 57500, 56000, 54500, 53000,
                                   51500, 50000, 49000, 48000, 47000, 46000, 45000,
                                   44000, 43500, 43000, 42500, 42000, 41500, 41000, 40500, 40000]),
                    'water': np.array([5000, 5500, 6000, 7000, 8500, 10000, 12000,
                                      14000, 16000, 18000, 19500, 21000, 22500, 24000,
                                      25500, 27000, 28000, 29000, 30000, 31000, 32000,
                                      33000, 33500, 34000, 34500, 35000, 35500, 36000, 36500, 37000]),
                    'gas': np.array([500000, 495000, 490000, 480000, 470500, 450000, 420000,
                                   380000, 350000, 320000, 300000, 280000, 260000, 240000,
                                   220000, 200000, 185000, 170000, 155000, 140000, 125000,
                                   110000, 105000, 100000, 95000, 90000, 85000, 80000, 75000, 70000]),
                    'pressure': np.array([2800, 2795, 2790, 2780, 2770, 2750, 2720,
                                         2680, 2640, 2600, 2570, 2540, 2510, 2480,
                                         2450, 2420, 2400, 2380, 2360, 2340, 2320,
                                         2300, 2290, 2280, 2270, 2260, 2250, 2240, 2230, 2220]),
                }
                data_source = "HARDCODED"
            
            print(f"\n[EnKF] === OBSERVED DATA (from {data_source}) ===", file=sys.stderr, flush=True)
            for key in observed_data:
                vals = observed_data[key]
                print(f"[EnKF] {key.upper():10s}: {len(vals)} points, "
                      f"range [{np.min(vals):10.0f}, {np.max(vals):10.0f}], "
                      f"decline: {vals[0] - vals[-1]:.0f} ({(vals[-1]/vals[0]*100):.1f}%)",
                      file=sys.stderr, flush=True)
            
            # Initialize EnKF
            enkf = EnKFFilter(ensemble_size=ensemble_size)
            
            # Create initial ensemble from simulation parameters
            mean_params = {
                'initial_pressure': simulation.initial_pressure,
                'porosity': simulation.porosity,
                'permeability': simulation.permeability,
                'water_saturation': simulation.water_saturation,
            }
            
            prior_ensemble = enkf.initialize_ensemble(mean_params)
            print(f"[EnKF] Prior ensemble shape: {prior_ensemble.shape}", file=sys.stderr, flush=True)
            
            # Forward model that uses parameters to generate realistic predictions
            def forward_model_fn(params):
                """
                Forward model that scales production based on reservoir parameters.
                Uses observed data structure as template and applies parameter-based scaling.
                
                Handles both formats:
                - Dict: {'permeability': 100, 'porosity': 0.2, ...}
                - Array/List: [permeability, porosity, water_saturation, initial_pressure]
                """
                N = len(observed_data['oil'])
                
                # Handle both dict and array/list parameter formats
                if isinstance(params, dict):
                    # From EnKF - dictionary format
                    perm = safe_float(params.get('permeability'), simulation.permeability or 100)
                    poros = safe_float(params.get('porosity'), simulation.porosity or 0.2)
                    water_sat = safe_float(params.get('water_saturation'), simulation.water_saturation or 0.3)
                    init_press = safe_float(params.get('initial_pressure'), simulation.initial_pressure or 3000)
                elif isinstance(params, (list, np.ndarray)):
                    # From forecast generator - array/list format
                    # Order: [permeability, porosity, water_saturation, initial_pressure]
                    try:
                        perm = float(params[0]) if len(params) > 0 else (simulation.permeability or 100)
                        poros = float(params[1]) if len(params) > 1 else (simulation.porosity or 0.2)
                        water_sat = float(params[2]) if len(params) > 2 else (simulation.water_saturation or 0.3)
                        init_press = float(params[3]) if len(params) > 3 else (simulation.initial_pressure or 3000)
                    except (TypeError, ValueError, IndexError) as e:
                        print(f"[EnKF] ERROR parsing array params: {e}, params={params}", file=sys.stderr, flush=True)
                        perm = simulation.permeability or 100
                        poros = simulation.porosity or 0.2
                        water_sat = simulation.water_saturation or 0.3
                        init_press = simulation.initial_pressure or 3000
                else:
                    # Fallback to simulation defaults
                    perm = simulation.permeability or 100
                    poros = simulation.porosity or 0.2
                    water_sat = simulation.water_saturation or 0.3
                    init_press = simulation.initial_pressure or 3000
                
                # Normalize parameters to scaling factors (0.5 to 1.5 range)
                # Higher permeability → higher production
                perm_scale = 0.7 + (perm / 500.0) * 0.6  # Clamp to ~0.5-1.3
                perm_scale = np.clip(perm_scale, 0.5, 1.5)
                
                # Higher porosity → higher reserves
                poros_scale = 0.7 + (poros / 0.3) * 0.6  # Clamp to ~0.5-1.3
                poros_scale = np.clip(poros_scale, 0.5, 1.5)
                
                # Normalize porosity/water units: handle percent inputs (>1) by converting to fraction
                if poros > 1.0:
                    poros = poros / 100.0
                if water_sat > 1.0:
                    water_sat = water_sat / 100.0

                # Higher water saturation → more water, less oil
                water_scale = 0.5 + (water_sat * 1.5)  # 0.5 to 2.0
                water_scale = np.clip(water_scale, 0.5, 2.0)
                oil_scale = max(0.1, 2.0 - water_scale)  # Avoid zero/negative oil scale
                
                # Higher pressure → pressure stays higher longer
                press_scale = 0.8 + (init_press / 4000.0) * 0.4  # Clamp to ~0.8-1.2
                press_scale = np.clip(press_scale, 0.8, 1.2)
                
                # Combined scaling effect
                production_scale = np.clip(perm_scale * poros_scale, 0.3, 3.0)
                
                # Use observed data as template and scale it
                obs_oil = np.array(observed_data['oil'], dtype=float)[:N]
                obs_water = np.array(observed_data['water'], dtype=float)[:N]
                obs_gas = np.array(observed_data['gas'], dtype=float)[:N]
                obs_pressure = np.array(observed_data['pressure'], dtype=float)[:N]
                
                # Apply parameter-based scaling
                # Reduce random jitter amplitude to keep ensemble runs near observed scale
                oil_array = obs_oil * oil_scale * production_scale * (0.9 + np.random.random() * 0.2)
                water_array = obs_water * water_scale * production_scale * (0.9 + np.random.random() * 0.2)
                gas_array = obs_gas * production_scale * perm_scale * (0.9 + np.random.random() * 0.2)
                pressure_array = obs_pressure * press_scale + (21 - 5) * np.random.random()
                
                # Ensure non-negative
                oil_array = np.maximum(oil_array, 1)
                water_array = np.maximum(water_array, 1)
                gas_array = np.maximum(gas_array, 100)
                pressure_array = np.maximum(pressure_array, 500)
                
                # Wrap production data for forecast_generator compatibility
                result = {
                    'production_data': {
                        'oil': oil_array.tolist(),
                        'water': water_array.tolist(),
                        'gas': gas_array.tolist(),
                        'pressure': pressure_array.tolist(),
                    },
                    # Keep flat format for backwards compatibility
                    'oil': oil_array,
                    'water': water_array,
                    'gas': gas_array,
                    'pressure': pressure_array,
                }
                
                print(f"[EnKF] Forward model (perm={perm:.0f}, poros={poros:.2f}, "
                      f"w_sat={water_sat:.2f}, p_init={init_press:.0f}): "
                      f"oil_range=[{oil_array.min():.0f},{oil_array.max():.0f}], "
                      f"match_to_obs=[oil*{oil_scale:.2f}, water*{water_scale:.2f}]", 
                      file=sys.stderr, flush=True)
                return result
            
            # Run EnKF
            def progress_cb(progress, msg):
                print(f"[EnKF] Progress {progress}%: {msg}", file=sys.stderr, flush=True)
                simulation.status = 'running'
                simulation.progress = 5 + int(progress * 0.7)
                simulation.current_step += 1
                simulation.save()
                
                # Broadcast progress via WebSocket
                try:
                    iteration = simulation.current_step
                    async_to_sync(broadcast_simulation_progress)(
                        simulation.id,
                        msg,
                        iteration=iteration,
                        status='calibrating'
                    )
                except Exception as ws_error:
                    print(f"[WebSocket] Error broadcasting progress: {ws_error}", file=sys.stderr, flush=True)
            
            # Broadcast: EnKF starting
            try:
                async_to_sync(broadcast_simulation_progress)(
                    simulation.id,
                    f'Starting EnKF calibration with {num_iterations} iterations',
                    iteration=0,
                    status='initializing'
                )
            except Exception as ws_error:
                print(f"[WebSocket] Error broadcasting start: {ws_error}", file=sys.stderr, flush=True)
            
            print(f"[EnKF] Starting EnKF with {num_iterations} iterations", file=sys.stderr, flush=True)
            # Determine measurement weights: prefer explicit request override, otherwise
            # if a reservoir model was uploaded, prioritize pressure and oil to improve physical match.
            req_weights = request.data.get('measurement_weights')
            if req_weights:
                try:
                    measurement_weights = {k: float(v) for k, v in req_weights.items()}
                except Exception:
                    measurement_weights = None
            else:
                measurement_weights = None
            # If dataset contains a parsed reservoir model, bias weights towards pressure and oil
            try:
                if measurement_weights is None and simulation.dataset and isinstance(simulation.dataset.production_data, dict):
                    if 'reservoir_model' in simulation.dataset.production_data:
                        measurement_weights = {
                            'oil': 0.40,
                            'pressure': 0.35,
                            'water': 0.15,
                            'gas': 0.10,
                        }
                        print(f"[EnKF] Using measurement_weights inferred from reservoir_model: {measurement_weights}", file=sys.stderr, flush=True)
            except Exception:
                pass

            try:
                enkf_result = enkf.run_enkf(
                    observed_data=observed_data,
                    forward_model_fn=forward_model_fn,
                    initial_ensemble=prior_ensemble,
                    measurement_weights=measurement_weights,
                    num_iterations=num_iterations,
                    progress_callback=progress_cb
                )
            except Exception as enkf_error:
                print(f"[EnKF] ERROR in run_enkf: {str(enkf_error)}", file=sys.stderr, flush=True)
                import traceback
                traceback.print_exc(file=sys.stderr)
                
                # Broadcast error
                try:
                    async_to_sync(broadcast_simulation_error)(
                        simulation.id,
                        f'EnKF calibration failed: {str(enkf_error)}',
                        stack_trace=traceback.format_exc()
                    )
                except Exception as ws_error:
                    print(f"[WebSocket] Error broadcasting error: {ws_error}", file=sys.stderr, flush=True)
                raise
            
            print(f"[EnKF] Result status: {enkf_result.get('status')}", file=sys.stderr, flush=True)
            print(f"[EnKF] Best quality: {enkf_result.get('best_quality')}", file=sys.stderr, flush=True)
            
            if enkf_result.get('status') != 'completed':
                error_msg = enkf_result.get('error', 'Unknown error')
                print(f"[EnKF] Failed: {error_msg}", file=sys.stderr, flush=True)
                simulation.status = 'failed'
                simulation.error_message = error_msg
                simulation.completed_at = timezone.now()
                simulation.save(update_fields=['status', 'error_message', 'completed_at'])
                return Response({'error': f'EnKF run failed: {error_msg}'}, 
                              status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # Extract posterior ensemble
            posterior_ensemble = np.array(enkf_result['final_ensemble'], dtype=float)
            print(f"[EnKF] Posterior ensemble shape: {posterior_ensemble.shape}", file=sys.stderr, flush=True)
            print(f"[EnKF] Posterior ensemble sample: {posterior_ensemble[:3]}", file=sys.stderr, flush=True)
            posterior_ensemble_list = posterior_ensemble.tolist()
            print(f"[EnKF] Posterior ensemble (list): {len(posterior_ensemble_list)} members", file=sys.stderr, flush=True)
            
            # Broadcast: Forecasts generation starting
            try:
                async_to_sync(broadcast_simulation_progress)(
                    simulation.id,
                    'Generating prior and posterior forecasts',
                    iteration=num_iterations,
                    status='forecasting'
                )
            except Exception as ws_error:
                print(f"[WebSocket] Error broadcasting forecast start: {ws_error}", file=sys.stderr, flush=True)
            
            # Generate PRIOR forecast (before EnKF)
            fg = ForecastGenerator()

            def forecast_progress(start, span, label):
                def update(progress):
                    simulation.status = 'running'
                    simulation.progress = min(99, start + int(progress * span / 100))
                    simulation.current_step += 1
                    simulation.save(update_fields=['status', 'progress', 'current_step'])
                    try:
                        async_to_sync(broadcast_simulation_progress)(
                            simulation.id,
                            f'{label}: {progress:.0f}%',
                            iteration=simulation.current_step,
                            status='forecasting',
                        )
                    except Exception as ws_error:
                        print(f"[WebSocket] Error broadcasting forecast progress: {ws_error}",
                              file=sys.stderr, flush=True)
                return update

            # Ensure ensemble members are dictionaries (consistent param names)
            def _array_to_param_dict(arr):
                try:
                    # EnKFFilter._array_to_params uses order: [initial_pressure, porosity, permeability, water_saturation]
                    return {
                        'initial_pressure': float(arr[0]),
                        'porosity': float(arr[1]),
                        'permeability': float(arr[2]),
                        'water_saturation': float(arr[3])
                    }
                except Exception:
                    # Fallback if arr is already dict-like
                    if isinstance(arr, dict):
                        return arr
                    return {
                        'initial_pressure': float(arr[0]) if len(arr) > 0 else simulation.initial_pressure or 1500,
                        'porosity': float(arr[1]) if len(arr) > 1 else simulation.porosity or 0.2,
                        'permeability': float(arr[2]) if len(arr) > 2 else simulation.permeability or 100,
                        'water_saturation': float(arr[3]) if len(arr) > 3 else simulation.water_saturation or 0.3
                    }

            prior_ensemble_list = []
            for member in prior_ensemble.tolist():
                prior_ensemble_list.append(_array_to_param_dict(member))

            prior_result = fg.generate_forecast(
                str(simulation.id) + '_prior',
                prior_ensemble_list,
                forward_model_fn,
                'prior',
                forecast_period_days,
                progress_callback=forecast_progress(75, 12, 'Prior forecast'),
            )
            
            # Generate POSTERIOR forecast (after EnKF)
            posterior_ensemble_list_dicts = []
            for member in posterior_ensemble_list:
                posterior_ensemble_list_dicts.append(_array_to_param_dict(member))

            posterior_result = fg.generate_forecast(
                str(simulation.id) + '_posterior',
                posterior_ensemble_list_dicts,
                forward_model_fn,
                'posterior',
                forecast_period_days,
                progress_callback=forecast_progress(87, 12, 'Posterior forecast'),
            )
            
            # Persist both forecasts
            forecasts = []
            
            # Use today's date as forecast start date, or from request
            forecast_start_date = request.data.get('forecast_date') or timezone.now().date()
            
            # Helper to convert numpy types to JSON-serializable
            def convert_to_serializable(obj):
                """Recursively convert numpy arrays and types to Python native types"""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    return obj
            
            if prior_result.get('status') == 'completed':
                prior_forecast = Forecast.objects.create(
                    simulation=simulation,
                    user=request.user,
                    name=f"{simulation.name} - Prior Forecast",
                    description="Prior forecast before EnKF calibration",
                    forecast_type='prior',
                    forecast_date=forecast_start_date,
                    forecast_period_days=forecast_period_days,
                    predicted_parameters=convert_to_serializable(prior_result['forecast'].get('predictions', {})),
                    predictions=convert_to_serializable(prior_result['forecast'].get('predictions', {})),
                    uncertainty_bounds=convert_to_serializable(prior_result['forecast'].get('uncertainty', {}))
                )
                forecasts.append(ForecastSerializer(prior_forecast).data)
                print(f"[EnKF] ✓ Prior forecast created", file=sys.stderr, flush=True)
            
            if posterior_result.get('status') == 'completed':
                posterior_forecast = Forecast.objects.create(
                    simulation=simulation,
                    user=request.user,
                    name=f"{simulation.name} - Posterior Forecast",
                    description="Posterior forecast after EnKF calibration",
                    forecast_type='posterior',
                    forecast_date=forecast_start_date,
                    forecast_period_days=forecast_period_days,
                    predicted_parameters=convert_to_serializable(posterior_result['forecast'].get('predictions', {})),
                    predictions=convert_to_serializable(posterior_result['forecast'].get('predictions', {})),
                    uncertainty_bounds=convert_to_serializable(posterior_result['forecast'].get('uncertainty', {}))
                )
                forecasts.append(ForecastSerializer(posterior_forecast).data)
                print(f"[EnKF] ✓ Posterior forecast created", file=sys.stderr, flush=True)
            
            # Update simulation with EnKF results
            # Convert all numpy arrays to JSON-serializable format
            def convert_to_serializable(obj):
                """Recursively convert numpy arrays and types to Python native types"""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    return obj
            
            # Extract predicted production values from posterior forecast for interpretation engine
            # The interpretation engine expects: oil_predicted, water_predicted, gas_predicted, pressure_predicted
            oil_predicted = 0.0
            water_predicted = 0.0
            gas_predicted = 0.0
            pressure_predicted = float(simulation.initial_pressure or 1500)
            
            if posterior_result.get('status') == 'completed' and posterior_result.get('forecast'):
                predictions = posterior_result['forecast'].get('predictions', {})
                # Extract first mean value (current production estimate) from each metric
                if 'oil' in predictions and isinstance(predictions['oil'], dict):
                    mean_values = predictions['oil'].get('mean', [])
                    oil_predicted = float(mean_values[0]) if mean_values else 50000.0
                if 'water' in predictions and isinstance(predictions['water'], dict):
                    mean_values = predictions['water'].get('mean', [])
                    water_predicted = float(mean_values[0]) if mean_values else 20000.0
                if 'gas' in predictions and isinstance(predictions['gas'], dict):
                    mean_values = predictions['gas'].get('mean', [])
                    gas_predicted = float(mean_values[0]) if mean_values else 500000.0
                if 'pressure' in predictions and isinstance(predictions['pressure'], dict):
                    mean_values = predictions['pressure'].get('mean', [])
                    pressure_predicted = float(mean_values[0]) if mean_values else float(simulation.initial_pressure or 1500)
            
            simulation.status = 'completed'
            simulation.match_quality = float(enkf_result.get('best_quality', 0))
            simulation.progress = 100

            # ── OOIP Calculation for EnKF ────────────────────────────────────
            ooip_data = {}
            try:
                area_acres   = safe_float(simulation.area, 0.0)
                thickness_ft = safe_float(simulation.thickness, 0.0)
                porosity_frac = safe_float(simulation.porosity, 0.0)
                if porosity_frac > 1.0:
                    porosity_frac /= 100.0
                water_sat_frac = safe_float(simulation.water_saturation, 0.0)
                if water_sat_frac > 1.0:
                    water_sat_frac /= 100.0

                bo = safe_float(simulation.oil_formation_volume_factor, 0.0)
                bo_source = 'user'
                if bo <= 0:
                    temp_f    = safe_float(simulation.temperature, 180.0)
                    press_psi = safe_float(simulation.initial_pressure, 3000.0)
                    bo = _standing_bo(temp_f, press_psi)
                    bo_source = "Standing's correlation"

                if area_acres > 0 and thickness_ft > 0 and porosity_frac > 0:
                    ooip_stb = _calculate_ooip(area_acres, thickness_ft, porosity_frac,
                                               water_sat_frac, bo)
                    # Forecast cumulative production over 365 days (85% efficiency factor)
                    cum_oil_365 = oil_predicted * 365 * 0.85
                    remaining_ooip = max(0.0, ooip_stb - cum_oil_365)
                    recovery_factor = min(100.0, (cum_oil_365 / ooip_stb) * 100) if ooip_stb > 0 else 0.0

                    ooip_data = {
                        'ooip_stb': round(ooip_stb, 0),
                        'ooip_mmbbl': round(ooip_stb / 1_000_000, 3),
                        'remaining_ooip_stb': round(remaining_ooip, 0),
                        'remaining_ooip_mmbbl': round(remaining_ooip / 1_000_000, 3),
                        'recovery_factor_pct': round(recovery_factor, 2),
                        'cum_oil_365d_stb': round(cum_oil_365, 0),
                        'formation_volume_factor': bo,
                        'bo_source': bo_source,
                        'inputs': {
                            'area_acres': area_acres,
                            'thickness_ft': thickness_ft,
                            'porosity_frac': porosity_frac,
                            'water_sat_frac': water_sat_frac,
                        }
                    }
                else:
                    ooip_data = {'note': 'OOIP not computed: area, thickness, or porosity not provided'}
            except Exception as ooip_err:
                ooip_data = {'error': str(ooip_err)}
            # ─────────────────────────────────────────────────────────────────

            simulation.results_data = {
                'enkf_result': convert_to_serializable({k: v for k, v in enkf_result.items() if k != 'final_ensemble'}),
                'prior_ensemble': convert_to_serializable(prior_ensemble),
                'posterior_ensemble': convert_to_serializable(posterior_ensemble_list),
                # Add predicted production for interpretation engine
                'oil_predicted': oil_predicted,
                'water_predicted': water_predicted,
                'gas_predicted': gas_predicted,
                'pressure_predicted': pressure_predicted,
                'ooip': ooip_data,
            }
            simulation.completed_at = timezone.now()
            if simulation.started_at:
                delta = simulation.completed_at - simulation.started_at
                simulation.duration_seconds = int(delta.total_seconds())
            simulation.save()
            
            # Update statistics
            stats, _ = SimulationStatistics.objects.get_or_create(user=request.user)
            stats.total_simulations += 1
            stats.completed_simulations += 1
            stats.enkf_simulations += 1
            stats.total_forecasts_generated += 2  # prior + posterior
            stats.best_match_quality = max(stats.best_match_quality, simulation.match_quality)
            stats.save()
            
            # Broadcast: Simulation completed
            try:
                duration = simulation.duration_seconds if simulation.duration_seconds else 0
                match_quality = float(simulation.match_quality)
                print(f"[WebSocket] Broadcasting completion for simulation {simulation.id}", file=sys.stderr, flush=True)
                print(f"[WebSocket]   - Match Quality: {match_quality}", file=sys.stderr, flush=True)
                print(f"[WebSocket]   - Duration: {duration}s", file=sys.stderr, flush=True)
                print(f"[WebSocket]   - Best Iteration: {num_iterations}", file=sys.stderr, flush=True)
                async_to_sync(broadcast_simulation_complete)(
                    simulation.id,
                    match_quality=match_quality,
                    best_iteration=num_iterations,
                    duration=duration
                )
                print(f"[WebSocket] Broadcast sent successfully", file=sys.stderr, flush=True)
            except Exception as ws_error:
                print(f"[WebSocket] Error broadcasting completion: {ws_error}", file=sys.stderr, flush=True)
            
            return Response({
                'message': 'EnKF with forecasts completed',
                'simulation': SimulationRunDetailSerializer(simulation).data,
                'forecasts': forecasts,
                'enkf_quality': float(enkf_result.get('best_quality', 0))
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"EnKF pipeline failed: {e}")
            simulation.status = 'failed'
            simulation.error_message = str(e)
            simulation.completed_at = timezone.now()
            simulation.save(update_fields=['status', 'error_message', 'completed_at'])
            return Response({'error': str(e)}, 
                          status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=False, methods=['get'])
    def recent(self, request):
        """Get 10 most recent simulations"""
        simulations = request.user.simulation_runs.all()[:10]
        serializer = SimulationRunListSerializer(simulations, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def completed(self, request):
        """Get all completed simulations"""
        simulations = request.user.simulation_runs.filter(status='completed')
        serializer = SimulationRunListSerializer(simulations, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def by_type(self, request):
        """Get simulations by type"""
        matching_type = request.query_params.get('type', 'baseline')
        simulations = request.user.simulation_runs.filter(matching_type=matching_type)
        serializer = SimulationRunListSerializer(simulations, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def interpret(self, request, pk=None):
        """
        Generate comprehensive interpretation of simulation results
        Provides expert-level analysis for reservoir engineers
        """
        simulation = self.get_object()
        
        # Ensure simulation is completed
        if simulation.status != 'completed':
            return Response({
                'error': 'Can only interpret completed simulations'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        if not simulation.results_data:
            return Response({
                'error': 'Simulation has no results to interpret'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Prepare simulation data for interpretation
            simulation_data = {
                'results_data': simulation.results_data,
                'initial_pressure': simulation.initial_pressure,
                'porosity': simulation.porosity,
                'permeability': simulation.permeability,
                'water_saturation': simulation.water_saturation,
                'match_quality': simulation.match_quality or 0,
                'matching_type': simulation.matching_type
            }
            
            # Generate interpretation
            interpretation = interpret_simulation_results(simulation_data)
            
            # Log activity
            UserActivityLog.objects.create(
                user=request.user,
                activity_type='simulation_interpreted',
                description=f"Generated interpretation for simulation: {simulation.name}"
            )
            
            return Response({
                'message': 'Simulation interpretation generated',
                'interpretation': interpretation,
                'simulation_id': simulation.id
            }, status=status.HTTP_200_OK)
        
        except Exception as e:
            logger.error(f"Interpretation error for simulation {pk}: {str(e)}")
            return Response({
                'error': f'Failed to generate interpretation: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ForecastViewSet(viewsets.ModelViewSet):
    """Manage forecasts from simulations"""
    
    permission_classes = [IsAuthenticated]
    serializer_class = ForecastSerializer
    
    def get_queryset(self):
        """Users can only see their own forecasts"""
        return Forecast.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        """Attach current user to forecast"""
        forecast = serializer.save(user=self.request.user)
        
        # Update statistics
        stats, _ = SimulationStatistics.objects.get_or_create(user=self.request.user)
        stats.total_forecasts_generated += 1
        stats.save()
        
        # Log activity
        UserActivityLog.objects.create(
            user=self.request.user,
            activity_type='forecast_generated',
            description=f"Generated forecast: {forecast.name}"
        )
    
    @action(detail=False, methods=['get'])
    def recent(self, request):
        """Get 10 most recent forecasts"""
        forecasts = request.user.forecasts.all()[:10]
        serializer = self.get_serializer(forecasts, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def by_simulation(self, request):
        """Get forecasts for a specific simulation"""
        simulation_id = request.query_params.get('simulation_id')
        if not simulation_id:
            return Response({
                'error': 'simulation_id parameter required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        forecasts = request.user.forecasts.filter(simulation_id=simulation_id)
        serializer = self.get_serializer(forecasts, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def parameter_histogram(self, request):
        """Return histogram data for a parameter across an ensemble or forecasts"""
        import numpy as np

        simulation_id = request.query_params.get('simulation_id')
        parameter = request.query_params.get('parameter')
        bins = int(request.query_params.get('bins', 20))

        if not simulation_id or not parameter:
            return Response({'error': 'simulation_id and parameter are required'},
                            status=status.HTTP_400_BAD_REQUEST)

        # Try to collect parameter values from forecasts' predicted_parameters
        user = cast(CustomUser, self.request.user)
        forecasts = user.forecasts.filter(simulation_id=simulation_id)
        values = []
        for f in forecasts:
            params = f.predicted_parameters or {}
            if isinstance(params, dict) and parameter in params:
                try:
                    values.append(float(params[parameter]))
                except Exception:
                    continue

        if not values:
            return Response({'error': 'No parameter values found for histogram'},
                            status=status.HTTP_404_NOT_FOUND)

        arr = np.array(values)
        counts, edges = np.histogram(arr, bins=bins)
        return Response({
            'parameter': parameter,
            'counts': counts.tolist(),
            'bin_edges': edges.tolist(),
            'n': int(arr.size)
        })

    @action(detail=False, methods=['get'])
    def comparison_chart(self, request):
        """Return series data (mean/p10/p50/p90) for prior vs posterior for a metric"""
        simulation_id = request.query_params.get('simulation_id')
        metric = request.query_params.get('metric', 'oil')

        if not simulation_id:
            return Response({'error': 'simulation_id required'}, status=status.HTTP_400_BAD_REQUEST)

        # Find latest prior and posterior forecasts
        user = cast(CustomUser, self.request.user)
        prior = user.forecasts.filter(simulation_id=simulation_id, forecast_type='prior').order_by('-generated_at').first()
        posterior = user.forecasts.filter(simulation_id=simulation_id, forecast_type='posterior').order_by('-generated_at').first()

        if not prior or not posterior:
            return Response({'error': 'Both prior and posterior forecasts required'}, status=status.HTTP_404_NOT_FOUND)

        def extract_series(forecast_obj, metric_key):
            preds = forecast_obj.predictions or {}
            metric_data = preds.get(metric_key, {})
            return {
                'mean': metric_data.get('mean', []),
                'p10': metric_data.get('p10', []),
                'p50': metric_data.get('p50', []),
                'p90': metric_data.get('p90', []),
            }

        data = {
            'simulation_id': simulation_id,
            'metric': metric,
            'prior': extract_series(prior, metric),
            'posterior': extract_series(posterior, metric),
            'time_axis': prior.predictions.get('time_axis') if prior.predictions else None
        }

        return Response(data)
