from unittest.mock import patch

from django.test import TestCase
from rest_framework.test import APIClient

from simulations.models import SimulationRun
from users.models import CustomUser


class EnKfDispatchTests(TestCase):
    def setUp(self):
        self.user = CustomUser.objects.create_user(
            username="enkf-user",
            email="enkf-user@example.com",
            password="test-password",
        )
        self.simulation = SimulationRun.objects.create(
            user=self.user,
            name="EnKF dispatch test",
            matching_type="enkf",
            initial_pressure=2800,
            porosity=20,
            permeability=100,
            water_saturation=20,
        )
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

    @patch("threading.Thread")
    def test_enkf_returns_accepted_and_marks_simulation_running(self, thread_class):
        response = self.client.post(
            f"/api/simulations/{self.simulation.pk}/run_enkf_with_forecasts/",
            {"ensemble_size": 100, "num_iterations": 10},
            format="json",
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.data["simulation"]["status"], "running")
        self.assertEqual(response.data["simulation"]["progress"], 0)
        thread_class.return_value.start.assert_called_once_with()

        self.simulation.refresh_from_db()
        self.assertEqual(self.simulation.status, "running")
        self.assertIsNotNone(self.simulation.started_at)