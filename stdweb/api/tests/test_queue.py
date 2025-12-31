"""
Tests for queue management endpoints.
These tests mock Celery inspection API.
"""

import pytest
from unittest.mock import MagicMock, patch
from rest_framework import status


class TestQueueList:
    """Tests for GET /api/queue/"""

    def test_empty_queue(self, authenticated_client, mock_celery_inspect):
        """Empty queue returns empty list."""
        response = authenticated_client.get('/api/queue/')
        assert response.status_code == status.HTTP_200_OK
        assert response.data == []

    def test_queue_with_active_tasks(self, authenticated_client, mock_celery_inspect, task):
        """Active tasks are listed."""
        mock_celery_inspect.active.return_value = {
            'worker1': [{
                'id': 'celery-task-id-1',
                'name': 'stdweb.celery_tasks.task_photometry',
                'time_start': 1234567890.0,
            }]
        }

        # Link celery task to our task
        task.celery_id = 'celery-task-id-1'
        task.save()

        response = authenticated_client.get('/api/queue/')
        assert response.status_code == status.HTTP_200_OK
        assert len(response.data) == 1
        assert response.data[0]['id'] == 'celery-task-id-1'
        assert response.data[0]['state'] == 'active'
        assert response.data[0]['name'] == 'task_photometry'
        assert response.data[0]['task_id'] == task.id

    def test_queue_with_pending_tasks(self, authenticated_client, mock_celery_inspect):
        """Pending (reserved) tasks are listed."""
        mock_celery_inspect.reserved.return_value = {
            'worker1': [{
                'id': 'celery-task-id-2',
                'name': 'stdweb.celery_tasks.task_inspect',
            }]
        }

        response = authenticated_client.get('/api/queue/')
        assert response.status_code == status.HTTP_200_OK
        assert len(response.data) == 1
        assert response.data[0]['state'] == 'pending'

    def test_queue_with_scheduled_tasks(self, authenticated_client, mock_celery_inspect):
        """Scheduled tasks are listed."""
        mock_celery_inspect.scheduled.return_value = {
            'worker1': [{
                'id': 'celery-task-id-3',
                'name': 'stdweb.celery_tasks.task_subtraction',
            }]
        }

        response = authenticated_client.get('/api/queue/')
        assert response.status_code == status.HTTP_200_OK
        assert len(response.data) == 1
        assert response.data[0]['state'] == 'scheduled'

    def test_queue_shows_chain_position(self, authenticated_client, mock_celery_inspect, task):
        """Chain position is shown for tasks in a chain."""
        mock_celery_inspect.active.return_value = {
            'worker1': [{
                'id': 'celery-task-id-2',
                'name': 'stdweb.celery_tasks.task_photometry',
            }]
        }

        task.celery_id = 'celery-task-id-1'
        task.celery_chain_ids = ['celery-task-id-1', 'celery-task-id-2', 'celery-task-id-3']
        task.save()

        response = authenticated_client.get('/api/queue/')
        assert response.status_code == status.HTTP_200_OK
        assert len(response.data) == 1
        assert response.data[0]['chain_position'] == 2
        assert response.data[0]['chain_total'] == 3

    def test_unauthenticated_denied(self, api_client, mock_celery_inspect):
        """Unauthenticated access denied."""
        response = api_client.get('/api/queue/')
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestQueueDetail:
    """Tests for GET /api/queue/{id}/"""

    def test_get_task_status(self, authenticated_client, task):
        """Can get Celery task status."""
        task.celery_id = 'test-celery-id'
        task.save()

        with patch('stdweb.api.views.celery_app.app') as mock_app:
            mock_result = MagicMock()
            mock_result.id = 'test-celery-id'
            mock_result.state = 'STARTED'
            mock_result.ready.return_value = False
            mock_app.AsyncResult.return_value = mock_result

            response = authenticated_client.get('/api/queue/test-celery-id/')

        assert response.status_code == status.HTTP_200_OK
        assert response.data['id'] == 'test-celery-id'
        assert response.data['state'] == 'STARTED'
        assert response.data['ready'] is False
        assert response.data['task_id'] == task.id

    def test_get_completed_task(self, authenticated_client):
        """Can get completed Celery task status."""
        with patch('stdweb.api.views.celery_app.app') as mock_app:
            mock_result = MagicMock()
            mock_result.id = 'completed-task-id'
            mock_result.state = 'SUCCESS'
            mock_result.ready.return_value = True
            mock_result.successful.return_value = True
            mock_result.failed.return_value = False
            mock_app.AsyncResult.return_value = mock_result

            response = authenticated_client.get('/api/queue/completed-task-id/')

        assert response.status_code == status.HTTP_200_OK
        assert response.data['ready'] is True
        assert response.data['successful'] is True
        assert response.data['failed'] is False

    def test_get_failed_task(self, authenticated_client):
        """Can get failed Celery task status."""
        with patch('stdweb.api.views.celery_app.app') as mock_app:
            mock_result = MagicMock()
            mock_result.id = 'failed-task-id'
            mock_result.state = 'FAILURE'
            mock_result.ready.return_value = True
            mock_result.successful.return_value = False
            mock_result.failed.return_value = True
            mock_app.AsyncResult.return_value = mock_result

            response = authenticated_client.get('/api/queue/failed-task-id/')

        assert response.status_code == status.HTTP_200_OK
        assert response.data['ready'] is True
        assert response.data['successful'] is False
        assert response.data['failed'] is True

    def test_unauthenticated_denied(self, api_client):
        """Unauthenticated access denied."""
        response = api_client.get('/api/queue/some-id/')
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestQueueTerminate:
    """Tests for POST /api/queue/{id}/terminate/"""

    def test_staff_can_terminate(self, staff_client, task, mock_revoke_task_chain):
        """Staff can terminate Celery task."""
        task.celery_id = 'test-celery-id'
        task.save()

        response = staff_client.post('/api/queue/test-celery-id/terminate/')
        assert response.status_code == status.HTTP_200_OK
        assert response.data['id'] == 'test-celery-id'
        assert response.data['task_id'] == task.id
        assert response.data['revoked_count'] == 3
        mock_revoke_task_chain.assert_called_once()

    def test_non_staff_denied(self, authenticated_client, task):
        """Non-staff users cannot terminate."""
        task.celery_id = 'test-celery-id'
        task.save()

        response = authenticated_client.post('/api/queue/test-celery-id/terminate/')
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_terminate_unknown_task(self, staff_client):
        """Can terminate task not linked to Django task."""
        with patch('stdweb.api.views.celery_app.app') as mock_app:
            response = staff_client.post('/api/queue/unknown-celery-id/terminate/')

        assert response.status_code == status.HTTP_200_OK
        assert response.data['id'] == 'unknown-celery-id'
        assert response.data['revoked_count'] == 1
        assert response.data['state'] == 'terminated'

    def test_terminate_by_chain_id(self, staff_client, task, mock_revoke_task_chain):
        """Can terminate by chain task ID."""
        task.celery_id = 'first-task-id'
        task.celery_chain_ids = ['first-task-id', 'second-task-id', 'third-task-id']
        task.save()

        response = staff_client.post('/api/queue/second-task-id/terminate/')
        assert response.status_code == status.HTTP_200_OK
        assert response.data['task_id'] == task.id

    def test_unauthenticated_denied(self, api_client):
        """Unauthenticated access denied."""
        response = api_client.post('/api/queue/some-id/terminate/')
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
