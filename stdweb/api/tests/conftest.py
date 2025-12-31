"""
Pytest fixtures for STDWeb REST API tests.
"""

import os
import shutil
import tempfile

import pytest
from django.conf import settings
from django.contrib.auth.models import User, Permission
from rest_framework.test import APIClient

from stdweb.models import Task, Preset


@pytest.fixture
def api_client():
    """Return an API client."""
    return APIClient()


@pytest.fixture
def user(db):
    """Create a regular user."""
    return User.objects.create_user(
        username='testuser',
        email='test@example.com',
        password='testpass123'
    )


@pytest.fixture
def other_user(db):
    """Create another regular user."""
    return User.objects.create_user(
        username='otheruser',
        email='other@example.com',
        password='testpass123'
    )


@pytest.fixture
def staff_user(db):
    """Create a staff user."""
    return User.objects.create_user(
        username='staffuser',
        email='staff@example.com',
        password='testpass123',
        is_staff=True
    )


@pytest.fixture
def user_with_view_all(db):
    """Create a user with view_all_tasks permission."""
    from django.contrib.contenttypes.models import ContentType
    from stdweb.models import Task

    user = User.objects.create_user(
        username='viewalluser',
        email='viewall@example.com',
        password='testpass123'
    )
    content_type = ContentType.objects.get_for_model(Task)
    perm, _ = Permission.objects.get_or_create(
        codename='view_all_tasks',
        defaults={
            'name': 'Can view tasks from all users',
            'content_type': content_type
        }
    )
    user.user_permissions.add(perm)
    return user


@pytest.fixture
def user_with_edit_all(db):
    """Create a user with edit_all_tasks permission."""
    from django.contrib.contenttypes.models import ContentType
    from stdweb.models import Task

    user = User.objects.create_user(
        username='editalluser',
        email='editall@example.com',
        password='testpass123'
    )
    content_type = ContentType.objects.get_for_model(Task)
    perm, _ = Permission.objects.get_or_create(
        codename='edit_all_tasks',
        defaults={
            'name': 'Can modify tasks from all users',
            'content_type': content_type
        }
    )
    user.user_permissions.add(perm)
    return user


@pytest.fixture
def authenticated_client(api_client, user):
    """Return an authenticated API client."""
    api_client.force_authenticate(user=user)
    return api_client


@pytest.fixture
def staff_client(api_client, staff_user):
    """Return a staff-authenticated API client."""
    api_client.force_authenticate(user=staff_user)
    return api_client


@pytest.fixture
def temp_tasks_path(tmp_path, settings):
    """Set up a temporary tasks directory."""
    original_path = settings.TASKS_PATH
    settings.TASKS_PATH = str(tmp_path / 'tasks')
    os.makedirs(settings.TASKS_PATH, exist_ok=True)
    yield settings.TASKS_PATH
    settings.TASKS_PATH = original_path


@pytest.fixture
def temp_data_path(tmp_path, settings):
    """Set up a temporary data directory."""
    original_path = settings.DATA_PATH
    settings.DATA_PATH = str(tmp_path / 'data')
    os.makedirs(settings.DATA_PATH, exist_ok=True)
    yield settings.DATA_PATH
    settings.DATA_PATH = original_path


@pytest.fixture
def task(db, user, temp_tasks_path):
    """Create a task for testing."""
    task = Task.objects.create(
        original_name='test_image.fits',
        title='Test Task',
        state='uploaded',
        user=user,
        config={'filter': 'R', 'cat_name': 'ps1'}
    )
    # Create task directory
    os.makedirs(task.path(), exist_ok=True)
    return task


@pytest.fixture
def task_with_file(task):
    """Create a task with a test file."""
    filepath = os.path.join(task.path(), 'image.fits')
    with open(filepath, 'wb') as f:
        f.write(b'SIMPLE  = T' + b' ' * 2869 + b'END' + b' ' * 2877)  # Minimal FITS header
    return task


@pytest.fixture
def other_user_task(db, other_user, temp_tasks_path):
    """Create a task belonging to another user."""
    task = Task.objects.create(
        original_name='other_image.fits',
        title='Other Task',
        state='uploaded',
        user=other_user,
        config={}
    )
    os.makedirs(task.path(), exist_ok=True)
    return task


@pytest.fixture
def preset(db):
    """Create a preset for testing."""
    return Preset.objects.create(
        name='Test Preset',
        config={'filter': 'V', 'cat_name': 'gaiaedr3', 'sn': 5}
    )


@pytest.fixture
def mock_celery_tasks(mocker):
    """Mock celery_tasks module."""
    mock = mocker.patch('stdweb.api.views.celery_tasks')
    return mock


@pytest.fixture
def mock_revoke_task_chain(mocker):
    """Mock revoke_task_chain function."""
    mock = mocker.patch('stdweb.api.views.revoke_task_chain')
    mock.return_value = 3  # Simulate 3 tasks revoked
    return mock


@pytest.fixture
def mock_celery_inspect(mocker):
    """Mock Celery inspection API."""
    mock_app = mocker.patch('stdweb.api.views.celery_app.app')
    mock_inspect = mocker.MagicMock()
    mock_app.control.inspect.return_value = mock_inspect
    mock_inspect.active.return_value = None
    mock_inspect.reserved.return_value = None
    mock_inspect.scheduled.return_value = None
    return mock_inspect


@pytest.fixture
def mock_processing_functions(mocker):
    """Mock image processing functions."""
    return {
        'fix_image': mocker.patch('stdweb.api.views.fix_image'),
        'crop_image': mocker.patch('stdweb.api.views.crop_image'),
        'preprocess_image': mocker.patch('stdweb.api.views.preprocess_image'),
    }
