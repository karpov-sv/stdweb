from rest_framework import serializers
from .models import Task, Preset


class TaskUploadSerializer(serializers.ModelSerializer):
    file = serializers.FileField(write_only=True)
    preset = serializers.IntegerField(required=False, allow_null=True)
    do_inspect = serializers.BooleanField(default=False)
    do_photometry = serializers.BooleanField(default=False)
    do_simple_transients = serializers.BooleanField(default=False)
    do_subtraction = serializers.BooleanField(default=False)
    
    class Meta:
        model = Task
        fields = [
            'file', 'title', 'original_name', 'preset', 
            'do_inspect', 'do_photometry', 'do_simple_transients', 'do_subtraction'
        ]
        read_only_fields = ['original_name']
    
    def validate_preset(self, value):
        """Validate that the preset exists if provided"""
        if value is not None:
            try:
                Preset.objects.get(id=value)
            except Preset.DoesNotExist:
                raise serializers.ValidationError("Preset with this ID does not exist.")
        return value
    
    def validate_file(self, value):
        """Validate file upload"""
        if not value:
            raise serializers.ValidationError("File is required.")
        
        # Check file extension for FITS files
        allowed_extensions = ['.fits', '.fit', '.fts']
        if not any(value.name.lower().endswith(ext) for ext in allowed_extensions):
            raise serializers.ValidationError(
                "Only FITS files (.fits, .fit, .fts) are allowed."
            )
        
        # Check file size (limit to 100MB)
        if value.size > 100 * 1024 * 1024:
            raise serializers.ValidationError("File size cannot exceed 100MB.")
        
        return value


class TaskSerializer(serializers.ModelSerializer):
    """Serializer for Task model responses"""
    user = serializers.StringRelatedField(read_only=True)
    
    class Meta:
        model = Task
        fields = [
            'id', 'original_name', 'title', 'state', 'user', 
            'created', 'modified', 'completed', 'config'
        ]
        read_only_fields = ['id', 'user', 'created', 'modified']


class PresetSerializer(serializers.ModelSerializer):
    """Serializer for Preset model"""
    
    class Meta:
        model = Preset
        fields = ['id', 'name', 'config', 'files'] 