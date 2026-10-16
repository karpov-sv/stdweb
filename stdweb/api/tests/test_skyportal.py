"""
Tests for POST /api/tasks/{id}/skyportal/
SkyPortal itself is mocked.
"""

import os
import pytest
import requests
from astropy.table import Table
from django.contrib.auth.models import Permission
from rest_framework import status

from stdweb.models import ActionLog


@pytest.fixture
def skyportal_client(api_client, user, settings):
    """Authenticated client whose user may upload to SkyPortal."""
    user.user_permissions.add(Permission.objects.get(codename='skyportal_upload'))
    settings.SKYPORTAL_TOKEN = 'test-token'
    api_client.force_authenticate(user=user)
    return api_client


@pytest.fixture
def phot_task(task):
    """Task with image time and direct target measurement."""
    task.config['time'] = '2024-01-01T12:00:00'
    task.save()
    Table({
        'mag_filter_name': ['BPmag'],
        'mag_calib': [19.8],
        'mag_calib_err': [0.12],
        'mag_limit': [21.5],
    }).write(os.path.join(task.path(), 'target.vot'), format='votable')
    return task


@pytest.fixture
def mock_skyportal(mocker):
    """Mock SkyPortal instrument lookup and photometry upload."""
    from stdweb import views_skyportal
    lookup = mocker.patch.object(views_skyportal, 'skyportal_get_instrument_id_by_name', return_value=7)
    upload = mocker.patch.object(views_skyportal, 'skyportal_upload_photometry',
                                 return_value={'status': 'success', 'data': {'ids': [456]}})
    return lookup, upload


def post(client, task, **kwargs):
    data = {'source_id': 'ZTF21aaqjmps', 'instrument_name': 'TAROT-TCA'}
    data.update(kwargs)
    return client.post(f'/api/tasks/{task.id}/skyportal/', data, format='json')


class TestSkyPortalUpload:

    def test_upload_success(self, skyportal_client, phot_task, mock_skyportal):
        _, upload = mock_skyportal
        response = post(skyportal_client, phot_task)
        assert response.status_code == status.HTTP_200_OK
        assert response.data['skyportal_photometry_ids'] == [456]
        assert response.data['filter'] == 'gaia::gbp'
        assert response.data['magsys'] == 'ab'
        assert response.data['mjd'] == pytest.approx(60310.5)
        assert response.data['mag'] == pytest.approx(19.8)
        assert response.data['is_diff'] is False

        upload.assert_called_once()
        assert upload.call_args.args[0] == 'ZTF21aaqjmps'
        assert upload.call_args.kwargs['instrument'] == 7

        assert ActionLog.objects.filter(action='task_skyportal_upload', task=phot_task).exists()

    def test_limit_only(self, skyportal_client, phot_task, mock_skyportal):
        response = post(skyportal_client, phot_task, limit_only=True)
        assert response.status_code == status.HTTP_200_OK
        assert response.data['mag'] is None
        assert response.data['limiting_mag'] == pytest.approx(21.5)

    def test_requires_permission(self, authenticated_client, phot_task, settings, mock_skyportal):
        settings.SKYPORTAL_TOKEN = 'test-token'
        response = post(authenticated_client, phot_task)
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_requires_token(self, skyportal_client, phot_task, settings, mock_skyportal):
        settings.SKYPORTAL_TOKEN = None
        response = post(skyportal_client, phot_task)
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_missing_fields(self, skyportal_client, phot_task, mock_skyportal):
        response = skyportal_client.post(f'/api/tasks/{phot_task.id}/skyportal/', {}, format='json')
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert 'source_id' in response.data
        assert 'instrument_name' in response.data

    def test_unknown_instrument(self, skyportal_client, phot_task, mock_skyportal):
        lookup, upload = mock_skyportal
        lookup.return_value = None
        response = post(skyportal_client, phot_task)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        upload.assert_not_called()

    def test_skyportal_unreachable(self, skyportal_client, phot_task, mock_skyportal):
        lookup, _ = mock_skyportal
        lookup.side_effect = requests.ConnectionError('down')
        response = post(skyportal_client, phot_task)
        assert response.status_code == status.HTTP_502_BAD_GATEWAY

    def test_no_measurement(self, skyportal_client, task, mock_skyportal):
        task.config['time'] = '2024-01-01T12:00:00'
        task.save()
        response = post(skyportal_client, task)
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_corrupt_measurement(self, skyportal_client, phot_task, mock_skyportal):
        with open(os.path.join(phot_task.path(), 'target.vot'), 'w') as f:
            f.write('garbage')
        response = post(skyportal_client, phot_task)
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_subtracted_missing(self, skyportal_client, phot_task, mock_skyportal):
        response = post(skyportal_client, phot_task, types='subtracted')
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_skyportal_rejects(self, skyportal_client, phot_task, mock_skyportal):
        _, upload = mock_skyportal
        upload.return_value = {'status': 'error', 'message': 'Invalid obj_id'}
        response = post(skyportal_client, phot_task)
        assert response.status_code == status.HTTP_502_BAD_GATEWAY
        assert response.data['error'] == 'Invalid obj_id'
