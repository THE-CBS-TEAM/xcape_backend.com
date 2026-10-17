from django.test import TestCase
from django.urls import reverse


class DeploymentReadinessTests(TestCase):
    def test_health_endpoint_ok(self):
        response = self.client.get(reverse('health-check'))
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload['status'], 'ok')
        self.assertEqual(payload['service'], 'xcape-api')

    def test_health_endpoint_without_slash_ok(self):
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'ok')

    def test_sensitivity_health_ok(self):
        response = self.client.get(reverse('sensitivity-health'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'ok')

    def test_auth_routes_exist(self):
        self.assertTrue(reverse('auth-login'))
