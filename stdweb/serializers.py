from rest_framework import serializers
from .models import Task, Preset


class TaskUploadSerializer(serializers.ModelSerializer):
    file = serializers.FileField(write_only=True)
    preset = serializers.IntegerField(required=False, allow_null=True)
    do_inspect = serializers.BooleanField(default=False)
    do_photometry = serializers.BooleanField(default=False)
    do_simple_transients = serializers.BooleanField(default=False)
    do_subtraction = serializers.BooleanField(default=False)
    
    # Photometry configuration parameters
    sn = serializers.FloatField(required=False, allow_null=True, help_text="S/N Ratio")
    initial_aper = serializers.FloatField(required=False, allow_null=True, help_text="Initial aperture, pixels")
    initial_r0 = serializers.FloatField(required=False, allow_null=True, help_text="Smoothing kernel, pixels")
    bg_size = serializers.IntegerField(required=False, allow_null=True, help_text="Background mesh size")
    minarea = serializers.IntegerField(required=False, allow_null=True, help_text="Minimal object area")
    rel_aper = serializers.FloatField(required=False, allow_null=True, help_text="Relative aperture, FWHM")
    rel_bg1 = serializers.FloatField(required=False, allow_null=True, help_text="Sky inner annulus, FWHM")
    rel_bg2 = serializers.FloatField(required=False, allow_null=True, help_text="Outer annulus, FWHM")
    fwhm_override = serializers.FloatField(required=False, allow_null=True, help_text="FWHM override, pixels")
    
    filter = serializers.CharField(required=False, allow_blank=True, help_text="Filter name")
    cat_name = serializers.CharField(required=False, allow_blank=True, help_text="Reference catalog name")
    cat_limit = serializers.FloatField(required=False, allow_null=True, help_text="Catalog limiting magnitude")
    spatial_order = serializers.IntegerField(required=False, allow_null=True, help_text="Zeropoint spatial order")
    use_color = serializers.BooleanField(required=False, help_text="Use color term")
    sr_override = serializers.FloatField(required=False, allow_null=True, help_text="Matching radius, arcsec")
    
    # Boolean flags
    prefilter_detections = serializers.BooleanField(required=False, help_text="Pre-filter detections")
    filter_blends = serializers.BooleanField(required=False, help_text="Filter catalogue blends")
    diagnose_color = serializers.BooleanField(required=False, help_text="Color term diagnostics")
    refine_wcs = serializers.BooleanField(required=False, help_text="Refine astrometry")
    blind_match_wcs = serializers.BooleanField(required=False, help_text="Blind match")
    inspect_bg = serializers.BooleanField(required=False, help_text="Inspect background")
    centroid_targets = serializers.BooleanField(required=False, help_text="Centroid targets")
    nonlin = serializers.BooleanField(required=False, help_text="Non-linearity correction")

    # --- NEW flags for template-subtraction filtering ---
    filter_vizier = serializers.BooleanField(required=False, help_text="Filter Vizier catalogues")
    filter_skybot = serializers.BooleanField(required=False, help_text="Filter SkyBoT")
    filter_prefilter = serializers.BooleanField(required=False, help_text="Pre-filtering of difference detections")
    
    # Blind matching parameters
    blind_match_ps_lo = serializers.FloatField(required=False, allow_null=True, help_text="Scale lower limit, arcsec/pix")
    blind_match_ps_up = serializers.FloatField(required=False, allow_null=True, help_text="Scale upper limit, arcsec/pix")
    blind_match_center = serializers.CharField(required=False, allow_blank=True, help_text="Center position for blind match")
    blind_match_sr0 = serializers.FloatField(required=False, allow_null=True, help_text="Radius, deg")
    
    # Target specification
    target = serializers.CharField(required=False, allow_blank=True, help_text="Target name or coordinates")
    
    # Inspection parameters
    gain = serializers.FloatField(required=False, allow_null=True, help_text="Gain, e/ADU")
    saturation = serializers.FloatField(required=False, allow_null=True, help_text="Saturation level, ADU")
    time = serializers.CharField(required=False, allow_blank=True, help_text="Time")
    
    class Meta:
        model = Task
        fields = [
            'file', 'title', 'original_name', 'preset', 
            'do_inspect', 'do_photometry', 'do_simple_transients', 'do_subtraction',
            # Photometry parameters
            'sn', 'initial_aper', 'initial_r0', 'bg_size', 'minarea',
            'rel_aper', 'rel_bg1', 'rel_bg2', 'fwhm_override',
            'filter', 'cat_name', 'cat_limit', 'spatial_order', 'use_color', 'sr_override',
            'prefilter_detections', 'filter_blends', 'diagnose_color', 'refine_wcs', 
            'blind_match_wcs', 'inspect_bg', 'centroid_targets', 'nonlin',
            'blind_match_ps_lo', 'blind_match_ps_up', 'blind_match_center', 'blind_match_sr0',
            'target', 'gain', 'saturation', 'time',
            'template', 'template_catalog'
            , 'filter_vizier', 'filter_skybot', 'filter_prefilter'
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