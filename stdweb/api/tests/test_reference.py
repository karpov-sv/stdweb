"""
Tests for reference data endpoints.
"""

import pytest
from rest_framework import status


class TestReferenceData:
    """Tests for GET /api/reference/ and /api/reference/{type}/"""

    def test_get_all_reference_data(self, api_client):
        """Can get all reference data (no auth required)."""
        response = api_client.get('/api/reference/')
        assert response.status_code == status.HTTP_200_OK
        assert 'filters' in response.data
        assert 'catalogs' in response.data
        assert 'templates' in response.data
        # Should be lists of keys
        assert isinstance(response.data['filters'], list)
        assert isinstance(response.data['catalogs'], list)
        assert isinstance(response.data['templates'], list)

    def test_get_filters(self, api_client):
        """Can get filters reference data."""
        response = api_client.get('/api/reference/filters/')
        assert response.status_code == status.HTTP_200_OK
        # Should be a dict with filter details
        assert isinstance(response.data, dict)
        # Check some common filters exist
        assert 'R' in response.data or 'V' in response.data or 'B' in response.data

    def test_get_catalogs(self, api_client):
        """Can get catalogs reference data."""
        response = api_client.get('/api/reference/catalogs/')
        assert response.status_code == status.HTTP_200_OK
        assert isinstance(response.data, dict)

    def test_get_templates(self, api_client):
        """Can get templates reference data."""
        response = api_client.get('/api/reference/templates/')
        assert response.status_code == status.HTTP_200_OK
        assert isinstance(response.data, dict)

    def test_invalid_type_returns_all(self, api_client):
        """Invalid type returns all reference data."""
        response = api_client.get('/api/reference/invalid_type/')
        assert response.status_code == status.HTTP_200_OK
        # Should fall through to default (all data)
        assert 'filters' in response.data
        assert 'catalogs' in response.data
        assert 'templates' in response.data

    def test_no_authentication_required(self, api_client):
        """Reference endpoints don't require authentication."""
        # All reference endpoints should be accessible without auth
        endpoints = [
            '/api/reference/',
            '/api/reference/filters/',
            '/api/reference/catalogs/',
            '/api/reference/templates/',
        ]
        for endpoint in endpoints:
            response = api_client.get(endpoint)
            assert response.status_code == status.HTTP_200_OK, f"Failed for {endpoint}"
