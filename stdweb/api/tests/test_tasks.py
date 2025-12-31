"""
Tests for task CRUD endpoints.
"""

import pytest
from django.urls import reverse
from rest_framework import status

from stdweb.models import Task


class TestTaskList:
    """Tests for GET /api/tasks/"""

    def test_unauthenticated_denied(self, api_client):
        """Unauthenticated requests should be denied."""
        response = api_client.get('/api/tasks/')
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_authenticated_returns_own_tasks(self, authenticated_client, task, other_user_task):
        """Regular user should only see their own tasks."""
        response = authenticated_client.get('/api/tasks/')
        assert response.status_code == status.HTTP_200_OK
        assert len(response.data) == 1
        assert response.data[0]['id'] == task.id

    def test_staff_sees_all_tasks(self, staff_client, task, other_user_task):
        """Staff user should see all tasks."""
        response = staff_client.get('/api/tasks/')
        assert response.status_code == status.HTTP_200_OK
        assert len(response.data) == 2

    def test_view_all_permission_sees_all(self, api_client, user_with_view_all, task, other_user_task):
        """User with view_all_tasks should see all tasks."""
        # Refresh user to load permissions
        from django.contrib.auth.models import User
        user = User.objects.get(pk=user_with_view_all.pk)
        api_client.force_authenticate(user=user)

        response = api_client.get('/api/tasks/')
        assert response.status_code == status.HTTP_200_OK
        assert len(response.data) == 2

    def test_returns_correct_fields(self, authenticated_client, task):
        """Response should contain expected fields."""
        response = authenticated_client.get('/api/tasks/')
        assert response.status_code == status.HTTP_200_OK

        task_data = response.data[0]
        assert 'id' in task_data
        assert 'original_name' in task_data
        assert 'title' in task_data
        assert 'state' in task_data
        assert 'user' in task_data
        assert 'created' in task_data
        # List serializer should NOT include config
        assert 'config' not in task_data


class TestTaskCreate:
    """Tests for POST /api/tasks/"""

    def test_create_minimal_task(self, authenticated_client, temp_tasks_path):
        """Task can be created with minimal data."""
        response = authenticated_client.post('/api/tasks/', {
            'original_name': 'new_image.fits'
        })
        assert response.status_code == status.HTTP_201_CREATED
        assert response.data['original_name'] == 'new_image.fits'
        assert Task.objects.filter(original_name='new_image.fits').exists()

    def test_create_with_config(self, authenticated_client, temp_tasks_path):
        """Task can be created with config."""
        response = authenticated_client.post('/api/tasks/', {
            'original_name': 'new_image.fits',
            'config': {'filter': 'R', 'cat_name': 'ps1'}
        }, format='json')
        assert response.status_code == status.HTTP_201_CREATED

        task = Task.objects.get(id=response.data['id'])
        assert task.config['filter'] == 'R'

    def test_create_with_preset(self, authenticated_client, preset, temp_tasks_path):
        """Task can be created with preset."""
        response = authenticated_client.post('/api/tasks/', {
            'original_name': 'new_image.fits',
            'preset': preset.id
        })
        assert response.status_code == status.HTTP_201_CREATED

        task = Task.objects.get(id=response.data['id'])
        assert task.config['filter'] == preset.config['filter']

    def test_unauthenticated_denied(self, api_client):
        """Unauthenticated requests should be denied."""
        response = api_client.post('/api/tasks/', {
            'original_name': 'new_image.fits'
        })
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestTaskDetail:
    """Tests for GET /api/tasks/{id}/"""

    def test_get_own_task(self, authenticated_client, task):
        """User can get their own task."""
        response = authenticated_client.get(f'/api/tasks/{task.id}/')
        assert response.status_code == status.HTTP_200_OK
        assert response.data['id'] == task.id
        assert 'config' in response.data

    def test_get_other_user_task_denied(self, authenticated_client, other_user_task):
        """User cannot get another user's task."""
        response = authenticated_client.get(f'/api/tasks/{other_user_task.id}/')
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_staff_can_get_any_task(self, staff_client, other_user_task):
        """Staff can get any task."""
        response = staff_client.get(f'/api/tasks/{other_user_task.id}/')
        assert response.status_code == status.HTTP_200_OK

    def test_nonexistent_task(self, authenticated_client):
        """Nonexistent task returns 404."""
        response = authenticated_client.get('/api/tasks/99999/')
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_includes_path(self, authenticated_client, task):
        """Response should include task path."""
        response = authenticated_client.get(f'/api/tasks/{task.id}/')
        assert 'path' in response.data


class TestTaskUpdate:
    """Tests for PATCH /api/tasks/{id}/"""

    def test_update_title(self, authenticated_client, task):
        """User can update task title."""
        response = authenticated_client.patch(f'/api/tasks/{task.id}/', {
            'title': 'Updated Title'
        })
        assert response.status_code == status.HTTP_200_OK

        task.refresh_from_db()
        assert task.title == 'Updated Title'

    def test_update_title_post(self, authenticated_client, task):
        """User can update task title."""
        response = authenticated_client.post(f'/api/tasks/{task.id}/', {
            'title': 'Updated Title'
        })
        assert response.status_code == status.HTTP_200_OK

        task.refresh_from_db()
        assert task.title == 'Updated Title'

    def test_update_config(self, authenticated_client, task):
        """User can update task config."""
        response = authenticated_client.patch(f'/api/tasks/{task.id}/', {
            'config': {'filter': 'V', 'new_param': 'value'}
        }, format='json')
        assert response.status_code == status.HTTP_200_OK

        task.refresh_from_db()
        assert task.config['filter'] == 'V'
        assert task.config['new_param'] == 'value'

    def test_update_other_user_task_denied(self, authenticated_client, other_user_task):
        """User cannot update another user's task."""
        response = authenticated_client.patch(f'/api/tasks/{other_user_task.id}/', {
            'title': 'Hacked'
        })
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_staff_can_update_any_task(self, staff_client, other_user_task):
        """Staff can update any task."""
        response = staff_client.patch(f'/api/tasks/{other_user_task.id}/', {
            'title': 'Staff Updated'
        })
        assert response.status_code == status.HTTP_200_OK

    def test_cannot_update_read_only_fields(self, authenticated_client, task):
        """Read-only fields should be ignored."""
        original_id = task.id
        response = authenticated_client.patch(f'/api/tasks/{task.id}/', {
            'id': 99999,
            'celery_id': 'fake-celery-id'
        }, format='json')
        assert response.status_code == status.HTTP_200_OK

        task.refresh_from_db()
        assert task.id == original_id


class TestTaskDelete:
    """Tests for DELETE /api/tasks/{id}/"""

    def test_delete_own_task(self, authenticated_client, task):
        """User can delete their own task."""
        task_id = task.id
        response = authenticated_client.delete(f'/api/tasks/{task_id}/')
        assert response.status_code == status.HTTP_204_NO_CONTENT
        assert not Task.objects.filter(id=task_id).exists()

    def test_delete_other_user_task_denied(self, authenticated_client, other_user_task):
        """User cannot delete another user's task."""
        response = authenticated_client.delete(f'/api/tasks/{other_user_task.id}/')
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert Task.objects.filter(id=other_user_task.id).exists()

    def test_staff_can_delete_any_task(self, staff_client, other_user_task):
        """Staff can delete any task."""
        task_id = other_user_task.id
        response = staff_client.delete(f'/api/tasks/{task_id}/')
        assert response.status_code == status.HTTP_204_NO_CONTENT
        assert not Task.objects.filter(id=task_id).exists()


class TestTaskDuplicate:
    """Tests for POST /api/tasks/{id}/duplicate/"""

    def test_duplicate_own_task(self, authenticated_client, task):
        """User can duplicate their own task."""
        response = authenticated_client.post(f'/api/tasks/{task.id}/duplicate/')
        assert response.status_code == status.HTTP_201_CREATED
        assert response.data['id'] != task.id
        assert response.data['original_name'] == task.original_name
        assert '(copy)' in response.data['title']

    def test_duplicate_preserves_config(self, authenticated_client, task):
        """Duplicate should preserve config."""
        response = authenticated_client.post(f'/api/tasks/{task.id}/duplicate/')
        assert response.status_code == status.HTTP_201_CREATED

        new_task = Task.objects.get(id=response.data['id'])
        assert new_task.config == task.config

    def test_duplicate_other_user_task_denied(self, authenticated_client, other_user_task):
        """User cannot duplicate another user's task."""
        response = authenticated_client.post(f'/api/tasks/{other_user_task.id}/duplicate/')
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_duplicate_creates_new_directory(self, authenticated_client, task_with_file):
        """Duplicate should create new task directory with files."""
        import os

        response = authenticated_client.post(f'/api/tasks/{task_with_file.id}/duplicate/')
        assert response.status_code == status.HTTP_201_CREATED

        new_task = Task.objects.get(id=response.data['id'])
        assert os.path.exists(new_task.path())
        assert os.path.exists(os.path.join(new_task.path(), 'image.fits'))
