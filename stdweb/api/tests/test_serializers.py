"""
Tests for API serializers.
"""

import pytest

from stdweb.api.serializers import (
    TaskSerializer,
    TaskCreateSerializer,
    TaskListSerializer,
    PresetSerializer,
    ProcessRequestSerializer,
    CropRequestSerializer,
    DestripeRequestSerializer,
)


class TestProcessRequestSerializer:
    """Tests for ProcessRequestSerializer."""

    def test_valid_single_step(self):
        """Valid single step should pass."""
        serializer = ProcessRequestSerializer(data={'steps': ['inspect']})
        assert serializer.is_valid()
        assert serializer.validated_data['steps'] == ['inspect']

    def test_valid_multiple_steps(self):
        """Valid multiple steps should pass."""
        serializer = ProcessRequestSerializer(data={
            'steps': ['inspect', 'photometry', 'subtraction']
        })
        assert serializer.is_valid()
        assert len(serializer.validated_data['steps']) == 3

    def test_all_valid_steps(self):
        """All valid step names should pass."""
        all_steps = ['inspect', 'photometry', 'transients', 'subtraction', 'cleanup']
        serializer = ProcessRequestSerializer(data={'steps': all_steps})
        assert serializer.is_valid()

    def test_invalid_step_rejected(self):
        """Invalid step names should be rejected."""
        serializer = ProcessRequestSerializer(data={'steps': ['invalid_step']})
        assert not serializer.is_valid()
        assert 'steps' in serializer.errors

    def test_empty_steps_rejected(self):
        """Empty steps list should be rejected."""
        serializer = ProcessRequestSerializer(data={'steps': []})
        assert not serializer.is_valid()

    def test_missing_steps_rejected(self):
        """Missing steps field should be rejected."""
        serializer = ProcessRequestSerializer(data={})
        assert not serializer.is_valid()

    def test_config_optional(self):
        """Config should be optional."""
        serializer = ProcessRequestSerializer(data={'steps': ['inspect']})
        assert serializer.is_valid()
        assert serializer.validated_data.get('config') == {}

    def test_config_accepted(self):
        """Config should be accepted when provided."""
        serializer = ProcessRequestSerializer(data={
            'steps': ['photometry'],
            'config': {'filter': 'R', 'cat_name': 'ps1'}
        })
        assert serializer.is_valid()
        assert serializer.validated_data['config'] == {'filter': 'R', 'cat_name': 'ps1'}


class TestCropRequestSerializer:
    """Tests for CropRequestSerializer."""

    def test_all_coordinates(self):
        """All coordinates should be accepted."""
        serializer = CropRequestSerializer(data={
            'x1': 100, 'y1': 100, 'x2': 900, 'y2': 900
        })
        assert serializer.is_valid()

    def test_partial_coordinates(self):
        """Partial coordinates should be accepted (all optional)."""
        serializer = CropRequestSerializer(data={'x1': 100, 'x2': 900})
        assert serializer.is_valid()

    def test_empty_data(self):
        """Empty data should be valid (all fields optional)."""
        serializer = CropRequestSerializer(data={})
        assert serializer.is_valid()

    def test_negative_values_allowed(self):
        """Negative values should be allowed (offset from edge)."""
        serializer = CropRequestSerializer(data={
            'x1': 10, 'y1': 10, 'x2': -10, 'y2': -10
        })
        assert serializer.is_valid()

    def test_non_integer_rejected(self):
        """Non-integer values should be rejected."""
        serializer = CropRequestSerializer(data={'x1': 'not_an_int'})
        assert not serializer.is_valid()


class TestDestripeRequestSerializer:
    """Tests for DestripeRequestSerializer."""

    def test_horizontal(self):
        """Horizontal direction should be accepted."""
        serializer = DestripeRequestSerializer(data={'direction': 'horizontal'})
        assert serializer.is_valid()
        assert serializer.validated_data['direction'] == 'horizontal'

    def test_vertical(self):
        """Vertical direction should be accepted."""
        serializer = DestripeRequestSerializer(data={'direction': 'vertical'})
        assert serializer.is_valid()
        assert serializer.validated_data['direction'] == 'vertical'

    def test_both(self):
        """Both direction should be accepted."""
        serializer = DestripeRequestSerializer(data={'direction': 'both'})
        assert serializer.is_valid()
        assert serializer.validated_data['direction'] == 'both'

    def test_default_is_both(self):
        """Default direction should be 'both'."""
        serializer = DestripeRequestSerializer(data={})
        assert serializer.is_valid()
        assert serializer.validated_data['direction'] == 'both'

    def test_invalid_direction_rejected(self):
        """Invalid direction should be rejected."""
        serializer = DestripeRequestSerializer(data={'direction': 'diagonal'})
        assert not serializer.is_valid()


class TestTaskSerializer:
    """Tests for TaskSerializer."""

    def test_serializes_task(self, task):
        """Task should be serialized correctly."""
        serializer = TaskSerializer(task)
        data = serializer.data

        assert data['id'] == task.id
        assert data['original_name'] == 'test_image.fits'
        assert data['title'] == 'Test Task'
        assert data['state'] == 'uploaded'
        assert data['user'] == task.user.username
        assert 'config' in data
        assert 'path' in data

    def test_read_only_fields(self, task):
        """Read-only fields should not be updateable."""
        serializer = TaskSerializer(task, data={
            'id': 9999,
            'user': 'hacker',
            'celery_id': 'fake-id'
        }, partial=True)
        assert serializer.is_valid()
        # Read-only fields should be ignored
        instance = serializer.save()
        assert instance.id == task.id

    def test_config_updateable(self, task):
        """Config should be updateable via PATCH."""
        serializer = TaskSerializer(task, data={
            'config': {'filter': 'V', 'new_key': 'new_value'}
        }, partial=True)
        assert serializer.is_valid()
        instance = serializer.save()
        assert instance.config['filter'] == 'V'
        assert instance.config['new_key'] == 'new_value'


class TestTaskCreateSerializer:
    """Tests for TaskCreateSerializer."""

    def test_minimal_creation(self, db, user):
        """Task can be created with minimal data."""
        serializer = TaskCreateSerializer(data={
            'original_name': 'new_image.fits'
        })
        assert serializer.is_valid(), serializer.errors
        task = serializer.save(user=user)
        assert task.original_name == 'new_image.fits'
        assert task.user == user

    def test_with_config(self, db, user):
        """Task can be created with config."""
        serializer = TaskCreateSerializer(data={
            'original_name': 'new_image.fits',
            'config': {'filter': 'R'}
        })
        assert serializer.is_valid()
        task = serializer.save(user=user)
        assert task.config['filter'] == 'R'

    def test_preset_applies_config(self, db, user, preset):
        """Preset config should be applied to new task."""
        serializer = TaskCreateSerializer(data={
            'original_name': 'new_image.fits',
            'preset': preset.id
        })
        assert serializer.is_valid()
        task = serializer.save(user=user)
        assert task.config['filter'] == 'V'
        assert task.config['cat_name'] == 'gaiaedr3'

    def test_inline_config_overrides_preset(self, db, user, preset):
        """Inline config should override preset values."""
        serializer = TaskCreateSerializer(data={
            'original_name': 'new_image.fits',
            'preset': preset.id,
            'config': {'filter': 'B'}  # Override preset's 'V'
        })
        assert serializer.is_valid()
        task = serializer.save(user=user)
        assert task.config['filter'] == 'B'  # Overridden
        assert task.config['cat_name'] == 'gaiaedr3'  # From preset


class TestTaskListSerializer:
    """Tests for TaskListSerializer."""

    def test_excludes_config(self, task):
        """Config should not be included in list serialization."""
        serializer = TaskListSerializer(task)
        data = serializer.data

        assert 'config' not in data
        assert 'id' in data
        assert 'original_name' in data
        assert 'state' in data


class TestPresetSerializer:
    """Tests for PresetSerializer."""

    def test_serializes_preset(self, preset):
        """Preset should be serialized correctly."""
        serializer = PresetSerializer(preset)
        data = serializer.data

        assert data['id'] == preset.id
        assert data['name'] == 'Test Preset'
        assert 'config' in data
        assert data['config']['filter'] == 'V'
