"""
Tests for permission helper functions.
"""

import pytest
from django.http import Http404
from unittest.mock import Mock

from stdweb.api.views import (
    check_task_permission,
    validate_task_path,
    sanitize_data_path,
)


class TestCheckTaskPermission:
    """Tests for check_task_permission helper function."""

    def test_staff_user_has_permission(self, staff_user, task):
        """Staff users should have permission to any task."""
        request = Mock()
        request.user = staff_user
        request.method = 'GET'

        assert check_task_permission(request, task) is True

    def test_staff_user_can_modify(self, staff_user, task):
        """Staff users should be able to modify any task."""
        request = Mock()
        request.user = staff_user
        request.method = 'PATCH'

        assert check_task_permission(request, task) is True

    def test_owner_has_permission(self, user, task):
        """Task owner should have full permission."""
        request = Mock()
        request.user = user
        request.method = 'GET'

        assert check_task_permission(request, task) is True

    def test_owner_can_modify(self, user, task):
        """Task owner should be able to modify their task."""
        request = Mock()
        request.user = user
        request.method = 'PATCH'

        assert check_task_permission(request, task) is True

    def test_other_user_denied(self, other_user, task):
        """Other users should be denied access."""
        request = Mock()
        request.user = other_user
        request.method = 'GET'

        assert check_task_permission(request, task) is False

    def test_view_all_permission_allows_read(self, user_with_view_all, task):
        """User with view_all_tasks can read any task."""
        # Need to refresh user to get permissions
        user_with_view_all = type(user_with_view_all).objects.get(pk=user_with_view_all.pk)
        request = Mock()
        request.user = user_with_view_all
        request.method = 'GET'

        assert check_task_permission(request, task) is True

    def test_view_all_permission_denies_write(self, user_with_view_all, task):
        """User with view_all_tasks cannot modify tasks."""
        user_with_view_all = type(user_with_view_all).objects.get(pk=user_with_view_all.pk)
        request = Mock()
        request.user = user_with_view_all
        request.method = 'PATCH'

        assert check_task_permission(request, task) is False

    def test_edit_all_permission_allows_write(self, user_with_edit_all, task):
        """User with edit_all_tasks can modify any task."""
        user_with_edit_all = type(user_with_edit_all).objects.get(pk=user_with_edit_all.pk)
        request = Mock()
        request.user = user_with_edit_all
        request.method = 'PATCH'

        assert check_task_permission(request, task) is True

    def test_head_request_allowed_for_view_all(self, user_with_view_all, task):
        """HEAD requests should be treated as read-only."""
        user_with_view_all = type(user_with_view_all).objects.get(pk=user_with_view_all.pk)
        request = Mock()
        request.user = user_with_view_all
        request.method = 'HEAD'

        assert check_task_permission(request, task) is True

    def test_options_request_allowed_for_view_all(self, user_with_view_all, task):
        """OPTIONS requests should be treated as read-only."""
        user_with_view_all = type(user_with_view_all).objects.get(pk=user_with_view_all.pk)
        request = Mock()
        request.user = user_with_view_all
        request.method = 'OPTIONS'

        assert check_task_permission(request, task) is True


class TestValidateTaskPath:
    """Tests for validate_task_path helper function."""

    def test_valid_path(self, task):
        """Valid paths should be accepted."""
        result = validate_task_path(task, 'image.fits')
        assert result.endswith('image.fits')
        assert task.path() in result

    def test_valid_nested_path(self, task):
        """Valid nested paths should be accepted."""
        result = validate_task_path(task, 'subdir/file.txt')
        assert 'subdir/file.txt' in result

    def test_directory_traversal_rejected(self, task):
        """Directory traversal attempts should raise Http404."""
        with pytest.raises(Http404):
            validate_task_path(task, '../../../etc/passwd')

    def test_double_dot_in_middle_rejected(self, task):
        """Paths with .. in middle should be rejected if they escape."""
        with pytest.raises(Http404):
            validate_task_path(task, 'subdir/../../etc/passwd')

    def test_absolute_path_rejected(self, task):
        """Absolute paths that escape task directory should be rejected."""
        with pytest.raises(Http404):
            validate_task_path(task, '/etc/passwd')


class TestSanitizeDataPath:
    """Tests for sanitize_data_path helper function."""

    def test_valid_path(self):
        """Valid paths should be returned unchanged."""
        result = sanitize_data_path('subdir/file.fits')
        assert result == 'subdir/file.fits'

    def test_leading_slash_stripped(self):
        """Leading slashes should be stripped."""
        result = sanitize_data_path('/subdir/file.fits')
        assert result == 'subdir/file.fits'

    def test_double_dot_start_rejected(self):
        """Paths starting with .. should be rejected."""
        with pytest.raises(Http404):
            sanitize_data_path('../etc/passwd')

    def test_double_dot_in_middle_rejected(self):
        """Paths with /../ should be rejected."""
        with pytest.raises(Http404):
            sanitize_data_path('subdir/../../../etc/passwd')

    def test_normalized_path(self):
        """Paths should be normalized."""
        result = sanitize_data_path('subdir//file.fits')
        assert '//' not in result
