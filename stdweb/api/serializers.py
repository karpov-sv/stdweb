"""
Serializers for the STDWeb REST API.
"""

import os
import shutil

from rest_framework import serializers

from django.conf import settings
from django.contrib.auth.models import Group

from stdweb.models import Task, Preset, accessible_groups


class GroupSharingMixin:
    """Validate that a task is only shared with groups the requester may use.

    Mirrors the web UI, which restricts sharing to the user's own groups
    (staff may share with any group). Requires ``request`` in the serializer
    context.
    """

    def validate_groups(self, groups):
        request = self.context.get('request')
        if request is None:
            return groups
        allowed = set(accessible_groups(request.user))
        invalid = [g.name for g in groups if g not in allowed]
        if invalid:
            raise serializers.ValidationError(
                "You cannot share with group(s): " + ", ".join(invalid)
            )
        return groups


class TaskSerializer(GroupSharingMixin, serializers.ModelSerializer):
    """
    Serializer for Task model.
    Includes nested config as a JSON object.
    """
    user = serializers.ReadOnlyField(source='user.username')
    path = serializers.SerializerMethodField()
    groups = serializers.SlugRelatedField(
        many=True,
        slug_field='name',
        queryset=Group.objects.all(),
        required=False,
    )

    class Meta:
        model = Task
        fields = [
            'id',
            'original_name',
            'title',
            'state',
            'user',
            'groups',
            'created',
            'modified',
            'completed',
            'config',
            'celery_id',
            'path',
        ]
        read_only_fields = [
            'id',
            'user',
            'created',
            'modified',
            'completed',
            'celery_id',
            'path',
        ]

    def get_path(self, obj):
        """Return the task directory path, relative to the tasks base directory."""
        return os.path.relpath(obj.path(), settings.TASKS_PATH)


class TaskCreateSerializer(GroupSharingMixin, serializers.ModelSerializer):
    """
    Serializer for creating new tasks.
    Handles file upload and initial configuration.
    """
    file = serializers.FileField(write_only=True, required=False)
    preset = serializers.PrimaryKeyRelatedField(
        queryset=Preset.objects.all(),
        required=False,
        write_only=True
    )
    groups = serializers.SlugRelatedField(
        many=True,
        slug_field='name',
        queryset=Group.objects.all(),
        required=False,
    )

    class Meta:
        model = Task
        fields = [
            'id',
            'original_name',
            'title',
            'config',
            'file',
            'preset',
            'groups',
        ]
        read_only_fields = ['id']

    def create(self, validated_data):
        # Extract file, preset and groups before creating task
        file = validated_data.pop('file', None)
        preset = validated_data.pop('preset', None)
        groups = validated_data.pop('groups', None)

        # Apply preset config if provided
        if preset:
            config = validated_data.get('config', {})
            preset_config = preset.config.copy()
            preset_config.update(config)
            validated_data['config'] = preset_config

        # Set original_name from file if not provided
        if file and not validated_data.get('original_name'):
            validated_data['original_name'] = file.name

        # Create the task
        task = Task.objects.create(**validated_data)

        # Always create the task directory, even for file-less tasks (e.g.
        # stacking, where inputs come from config['stack_filenames'] and
        # image.fits is produced by the stack step). Mirrors the web UI.
        os.makedirs(task.path(), exist_ok=True)

        # Copy preset files into the task folder, like the web UI does
        if preset and preset.files:
            for filename in preset.files.split('\n'):
                filename = filename.strip()
                if filename:
                    shutil.copy(filename, task.path())

        # Set initial group sharing (many-to-many, must be set after creation)
        if groups:
            task.groups.set(groups)

        # Handle file upload
        if file:
            filepath = os.path.join(task.path(), 'image.fits')
            with open(filepath, 'wb') as f:
                for chunk in file.chunks():
                    f.write(chunk)
            task.state = 'uploaded'
            task.save()

        return task


class TaskListSerializer(serializers.ModelSerializer):
    """
    Lightweight serializer for task listings.
    Excludes heavy fields like full config.
    """
    user = serializers.ReadOnlyField(source='user.username')

    class Meta:
        model = Task
        fields = [
            'id',
            'original_name',
            'title',
            'state',
            'user',
            'created',
            'modified',
        ]


class TaskStateSerializer(serializers.Serializer):
    """Serializer for task state response."""
    id = serializers.IntegerField()
    state = serializers.CharField()
    celery_id = serializers.CharField(allow_null=True)


class PresetSerializer(serializers.ModelSerializer):
    """Serializer for Preset model."""

    class Meta:
        model = Preset
        fields = ['id', 'name', 'config']


class CropRequestSerializer(serializers.Serializer):
    """Serializer for crop action requests."""
    x1 = serializers.IntegerField(required=False)
    y1 = serializers.IntegerField(required=False)
    x2 = serializers.IntegerField(required=False)
    y2 = serializers.IntegerField(required=False)


class DestripeRequestSerializer(serializers.Serializer):
    """Serializer for destripe action requests."""
    direction = serializers.ChoiceField(
        choices=['horizontal', 'vertical', 'both'],
        default='both'
    )
