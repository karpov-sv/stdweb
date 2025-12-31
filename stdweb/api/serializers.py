"""
Serializers for the STDWeb REST API.
"""

from rest_framework import serializers

from stdweb.models import Task, Preset


class TaskSerializer(serializers.ModelSerializer):
    """
    Serializer for Task model.
    Includes nested config as a JSON object.
    """
    user = serializers.ReadOnlyField(source='user.username')
    path = serializers.SerializerMethodField()

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
        """Return the task directory path."""
        return obj.path()


class TaskCreateSerializer(serializers.ModelSerializer):
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

    class Meta:
        model = Task
        fields = [
            'id',
            'original_name',
            'title',
            'config',
            'file',
            'preset',
        ]
        read_only_fields = ['id']

    def create(self, validated_data):
        # Extract file and preset before creating task
        file = validated_data.pop('file', None)
        preset = validated_data.pop('preset', None)

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

        # Handle file upload
        if file:
            import os
            os.makedirs(task.path(), exist_ok=True)
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


class ProcessRequestSerializer(serializers.Serializer):
    """Serializer for process action requests."""
    steps = serializers.ListField(
        child=serializers.ChoiceField(choices=[
            'inspect', 'photometry', 'transients', 'subtraction', 'cleanup'
        ]),
        min_length=1
    )
    config = serializers.JSONField(required=False, default=dict)


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


class FileInfoSerializer(serializers.Serializer):
    """Serializer for file information."""
    name = serializers.CharField()
    path = serializers.CharField()
    size = serializers.IntegerField()
    modified = serializers.DateTimeField()
    is_dir = serializers.BooleanField()
