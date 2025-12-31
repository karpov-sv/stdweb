"""
Tests for preset endpoints.
"""

import pytest
from rest_framework import status

from stdweb.models import Preset


class TestPresetList:
    """Tests for GET /api/presets/"""

    def test_list_presets(self, authenticated_client, preset):
        """Can list presets."""
        response = authenticated_client.get('/api/presets/')
        assert response.status_code == status.HTTP_200_OK
        assert len(response.data) == 1
        assert response.data[0]['name'] == 'Test Preset'

    def test_list_multiple_presets(self, authenticated_client, db):
        """Can list multiple presets."""
        Preset.objects.create(name='Preset 1', config={'filter': 'R'})
        Preset.objects.create(name='Preset 2', config={'filter': 'V'})
        Preset.objects.create(name='Preset 3', config={'filter': 'B'})

        response = authenticated_client.get('/api/presets/')
        assert response.status_code == status.HTTP_200_OK
        assert len(response.data) == 3

    def test_list_empty(self, authenticated_client, db):
        """Empty preset list returns empty array."""
        response = authenticated_client.get('/api/presets/')
        assert response.status_code == status.HTTP_200_OK
        assert response.data == []

    def test_unauthenticated_denied(self, api_client):
        """Unauthenticated access denied."""
        response = api_client.get('/api/presets/')
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestPresetDetail:
    """Tests for GET /api/presets/{id}/"""

    def test_get_preset(self, authenticated_client, preset):
        """Can get preset details."""
        response = authenticated_client.get(f'/api/presets/{preset.id}/')
        assert response.status_code == status.HTTP_200_OK
        assert response.data['id'] == preset.id
        assert response.data['name'] == 'Test Preset'
        assert 'config' in response.data
        assert response.data['config']['filter'] == 'V'

    def test_get_nonexistent_preset(self, authenticated_client):
        """Nonexistent preset returns 404."""
        response = authenticated_client.get('/api/presets/99999/')
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_unauthenticated_denied(self, api_client, preset):
        """Unauthenticated access denied."""
        response = api_client.get(f'/api/presets/{preset.id}/')
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
