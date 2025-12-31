"""
Tests for task processing endpoints.
These tests use mocking for Celery and processing functions.
"""

import os
import pytest
from rest_framework import status


class TestTaskProcess:
    """Tests for POST /api/tasks/{id}/process/"""

    def test_process_single_step(self, authenticated_client, task, mock_celery_tasks):
        """Can process with single step."""
        response = authenticated_client.post(
            f'/api/tasks/{task.id}/process/',
            {'steps': ['inspect']},
            format='json'
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.data['id'] == task.id
        assert response.data['steps'] == ['inspect']
        mock_celery_tasks.run_task_steps.assert_called_once()

    def test_process_multiple_steps(self, authenticated_client, task, mock_celery_tasks):
        """Can process with multiple steps."""
        response = authenticated_client.post(
            f'/api/tasks/{task.id}/process/',
            {'steps': ['inspect', 'photometry', 'subtraction']},
            format='json'
        )
        assert response.status_code == status.HTTP_200_OK
        assert len(response.data['steps']) == 3

    def test_process_with_config_override(self, authenticated_client, task, mock_celery_tasks):
        """Config override should be applied."""
        response = authenticated_client.post(
            f'/api/tasks/{task.id}/process/',
            {
                'steps': ['photometry'],
                'config': {'filter': 'V', 'sn': 10}
            },
            format='json'
        )
        assert response.status_code == status.HTTP_200_OK

        task.refresh_from_db()
        assert task.config['filter'] == 'V'
        assert task.config['sn'] == 10

    def test_process_empty_steps_rejected(self, authenticated_client, task):
        """Empty steps should be rejected."""
        response = authenticated_client.post(
            f'/api/tasks/{task.id}/process/',
            {'steps': []},
            format='json'
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_process_invalid_step_rejected(self, authenticated_client, task):
        """Invalid step name should be rejected."""
        response = authenticated_client.post(
            f'/api/tasks/{task.id}/process/',
            {'steps': ['invalid_step']},
            format='json'
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert 'Invalid step' in response.data.get('error', '')

    def test_process_other_user_task_denied(self, authenticated_client, other_user_task, mock_celery_tasks):
        """Cannot process another user's task."""
        response = authenticated_client.post(
            f'/api/tasks/{other_user_task.id}/process/',
            {'steps': ['inspect']},
            format='json'
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN
        mock_celery_tasks.run_task_steps.assert_not_called()

    def test_step_mapping(self, authenticated_client, task, mock_celery_tasks):
        """Steps should be mapped to celery task names."""
        response = authenticated_client.post(
            f'/api/tasks/{task.id}/process/',
            {'steps': ['simple_transients']},
            format='json'
        )
        assert response.status_code == status.HTTP_200_OK

        # Check that the step was mapped correctly
        call_args = mock_celery_tasks.run_task_steps.call_args
        assert 'simple_transients' in call_args[0][1]


class TestTaskCancel:
    """Tests for POST /api/tasks/{id}/cancel/"""

    def test_cancel_running_task(self, authenticated_client, task, mock_revoke_task_chain):
        """Can cancel running task."""
        task.celery_id = 'some-celery-id'
        task.save()

        response = authenticated_client.post(f'/api/tasks/{task.id}/cancel/')
        assert response.status_code == status.HTTP_200_OK
        assert response.data['id'] == task.id
        assert response.data['revoked_count'] == 3
        mock_revoke_task_chain.assert_called_once_with(task)

    def test_cancel_not_running_task(self, authenticated_client, task):
        """Cannot cancel task that is not running."""
        response = authenticated_client.post(f'/api/tasks/{task.id}/cancel/')
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert 'not running' in response.data.get('error', '')

    def test_cancel_other_user_task_denied(self, authenticated_client, other_user_task):
        """Cannot cancel another user's task."""
        other_user_task.celery_id = 'some-celery-id'
        other_user_task.save()

        response = authenticated_client.post(f'/api/tasks/{other_user_task.id}/cancel/')
        assert response.status_code == status.HTTP_403_FORBIDDEN


class TestTaskFix:
    """Tests for POST /api/tasks/{id}/fix/"""

    def test_fix_task(self, authenticated_client, task_with_file, mock_processing_functions):
        """Can fix task with image file."""
        response = authenticated_client.post(f'/api/tasks/{task_with_file.id}/fix/')
        assert response.status_code == status.HTTP_200_OK
        assert response.data['status'] == 'fixed'
        mock_processing_functions['fix_image'].assert_called_once()

    def test_fix_missing_image(self, authenticated_client, task, mock_processing_functions):
        """Fix without image file returns error."""
        response = authenticated_client.post(f'/api/tasks/{task.id}/fix/')
        assert response.status_code == status.HTTP_404_NOT_FOUND
        mock_processing_functions['fix_image'].assert_not_called()

    def test_fix_other_user_task_denied(self, authenticated_client, other_user_task):
        """Cannot fix another user's task."""
        response = authenticated_client.post(f'/api/tasks/{other_user_task.id}/fix/')
        assert response.status_code == status.HTTP_403_FORBIDDEN


class TestTaskCrop:
    """Tests for POST /api/tasks/{id}/crop/"""

    def test_crop_task(self, authenticated_client, task_with_file, mock_processing_functions):
        """Can crop task image."""
        response = authenticated_client.post(
            f'/api/tasks/{task_with_file.id}/crop/',
            {'x1': 100, 'y1': 100, 'x2': 900, 'y2': 900},
            format='json'
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.data['status'] == 'cropped'
        mock_processing_functions['crop_image'].assert_called_once()

    def test_crop_with_negative_values(self, authenticated_client, task_with_file, mock_processing_functions):
        """Can crop with negative values (offset from edge)."""
        response = authenticated_client.post(
            f'/api/tasks/{task_with_file.id}/crop/',
            {'x1': 10, 'y1': 10, 'x2': -10, 'y2': -10},
            format='json'
        )
        assert response.status_code == status.HTTP_200_OK

    def test_crop_missing_image(self, authenticated_client, task, mock_processing_functions):
        """Crop without image file returns error."""
        response = authenticated_client.post(
            f'/api/tasks/{task.id}/crop/',
            {'x1': 100, 'y1': 100, 'x2': 900, 'y2': 900},
            format='json'
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_crop_other_user_task_denied(self, authenticated_client, other_user_task):
        """Cannot crop another user's task."""
        response = authenticated_client.post(
            f'/api/tasks/{other_user_task.id}/crop/',
            {'x1': 100, 'y1': 100, 'x2': 900, 'y2': 900},
            format='json'
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN


class TestTaskDestripe:
    """Tests for POST /api/tasks/{id}/destripe/"""

    def test_destripe_horizontal(self, authenticated_client, task_with_file, mock_processing_functions):
        """Can destripe horizontally."""
        response = authenticated_client.post(
            f'/api/tasks/{task_with_file.id}/destripe/',
            {'direction': 'horizontal'},
            format='json'
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.data['status'] == 'destriped'

        # Check correct parameters passed
        call_kwargs = mock_processing_functions['preprocess_image'].call_args[1]
        assert call_kwargs['destripe_horizontal'] is True
        assert call_kwargs['destripe_vertical'] is False

    def test_destripe_vertical(self, authenticated_client, task_with_file, mock_processing_functions):
        """Can destripe vertically."""
        response = authenticated_client.post(
            f'/api/tasks/{task_with_file.id}/destripe/',
            {'direction': 'vertical'},
            format='json'
        )
        assert response.status_code == status.HTTP_200_OK

        call_kwargs = mock_processing_functions['preprocess_image'].call_args[1]
        assert call_kwargs['destripe_horizontal'] is False
        assert call_kwargs['destripe_vertical'] is True

    def test_destripe_both(self, authenticated_client, task_with_file, mock_processing_functions):
        """Can destripe both directions."""
        response = authenticated_client.post(
            f'/api/tasks/{task_with_file.id}/destripe/',
            {'direction': 'both'},
            format='json'
        )
        assert response.status_code == status.HTTP_200_OK

        call_kwargs = mock_processing_functions['preprocess_image'].call_args[1]
        assert call_kwargs['destripe_horizontal'] is True
        assert call_kwargs['destripe_vertical'] is True

    def test_destripe_default_direction(self, authenticated_client, task_with_file, mock_processing_functions):
        """Default direction should be 'both'."""
        response = authenticated_client.post(
            f'/api/tasks/{task_with_file.id}/destripe/',
            {},
            format='json'
        )
        assert response.status_code == status.HTTP_200_OK

        call_kwargs = mock_processing_functions['preprocess_image'].call_args[1]
        assert call_kwargs['destripe_horizontal'] is True
        assert call_kwargs['destripe_vertical'] is True

    def test_destripe_missing_image(self, authenticated_client, task, mock_processing_functions):
        """Destripe without image file returns error."""
        response = authenticated_client.post(
            f'/api/tasks/{task.id}/destripe/',
            {'direction': 'both'},
            format='json'
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_destripe_other_user_task_denied(self, authenticated_client, other_user_task):
        """Cannot destripe another user's task."""
        response = authenticated_client.post(
            f'/api/tasks/{other_user_task.id}/destripe/',
            {'direction': 'both'},
            format='json'
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN
