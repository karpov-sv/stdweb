import os
import shutil
from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from celery import chain

from .models import Task, Preset
from .serializers import TaskUploadSerializer, TaskSerializer, PresetSerializer
from .views import handle_uploaded_file
from . import celery_tasks


class TaskUploadAPIView(APIView):
    """API endpoint for uploading FITS files and creating tasks"""
    
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]
    
    def post(self, request, format=None):
        serializer = TaskUploadSerializer(data=request.data)
        
        if serializer.is_valid():
            # Extract validated data
            file = serializer.validated_data['file']
            title = serializer.validated_data.get('title', '')
            preset_id = serializer.validated_data.get('preset')
            do_inspect = serializer.validated_data.get('do_inspect', False)
            do_photometry = serializer.validated_data.get('do_photometry', False)
            do_simple_transients = serializer.validated_data.get('do_simple_transients', False)
            do_subtraction = serializer.validated_data.get('do_subtraction', False)
            
            # Create task
            task = Task(
                title=title,
                original_name=file.name,
                user=request.user
            )
            task.save()  # Save to get task.id
            
            # Handle file upload
            try:
                handle_uploaded_file(file, os.path.join(task.path(), 'image.fits'))
            except Exception as e:
                task.delete()  # Clean up if file upload fails
                return Response(
                    {'error': f'File upload failed: {str(e)}'}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            # Apply config preset if provided
            if preset_id:
                try:
                    preset = Preset.objects.get(id=preset_id)
                    task.config.update(preset.config)
                    
                    # Copy preset files if they exist
                    if preset.files:
                        for filename in preset.files.split('\n'):
                            if filename.strip():
                                try:
                                    shutil.copy(filename.strip(), task.path())
                                except Exception as e:
                                    # Log error but don't fail the upload
                                    pass
                except Preset.DoesNotExist:
                    return Response(
                        {'error': 'Preset not found'}, 
                        status=status.HTTP_400_BAD_REQUEST
                    )
            
            # Extract and set all configuration parameters
            config_params = [
                # Photometry parameters
                'sn', 'initial_aper', 'initial_r0', 'bg_size', 'minarea',
                'rel_aper', 'rel_bg1', 'rel_bg2', 'fwhm_override',
                'filter', 'cat_name', 'cat_limit', 'spatial_order', 'use_color', 'sr_override',
                'prefilter_detections', 'filter_blends', 'diagnose_color', 'refine_wcs',
                'blind_match_wcs', 'inspect_bg', 'centroid_targets', 'nonlin',
                'blind_match_ps_lo', 'blind_match_ps_up', 'blind_match_center', 'blind_match_sr0',
                # Inspection parameters
                'target', 'gain', 'saturation', 'time'
            ]
            
            for param in config_params:
                value = serializer.validated_data.get(param)
                if value is not None:
                    task.config[param] = value
            
            task.state = 'uploaded'
            task.save()
            
            # Initiate processing steps if requested
            todo = []
            
            if do_inspect:
                todo.extend([
                    celery_tasks.task_set_state.subtask(args=[task.id, 'inspect'], immutable=True),
                    celery_tasks.task_inspect.subtask(args=[task.id, False], immutable=True),
                    celery_tasks.task_break_if_failed.subtask(args=[task.id], immutable=True),
                    celery_tasks.task_set_state.subtask(args=[task.id, 'inspect_done'], immutable=True)
                ])
            
            if do_photometry:
                todo.extend([
                    celery_tasks.task_set_state.subtask(args=[task.id, 'photometry'], immutable=True),
                    celery_tasks.task_photometry.subtask(args=[task.id, False], immutable=True),
                    celery_tasks.task_break_if_failed.subtask(args=[task.id], immutable=True),
                    celery_tasks.task_set_state.subtask(args=[task.id, 'photometry_done'], immutable=True)
                ])
            
            if do_simple_transients:
                todo.extend([
                    celery_tasks.task_set_state.subtask(args=[task.id, 'transients_simple'], immutable=True),
                    celery_tasks.task_transients_simple.subtask(args=[task.id, False], immutable=True),
                    celery_tasks.task_break_if_failed.subtask(args=[task.id], immutable=True),
                    celery_tasks.task_set_state.subtask(args=[task.id, 'transients_simple_done'], immutable=True)
                ])
            
            if do_subtraction:
                todo.extend([
                    celery_tasks.task_set_state.subtask(args=[task.id, 'subtraction'], immutable=True),
                    celery_tasks.task_subtraction.subtask(args=[task.id, False], immutable=True),
                    celery_tasks.task_break_if_failed.subtask(args=[task.id], immutable=True),
                    celery_tasks.task_set_state.subtask(args=[task.id, 'subtraction_done'], immutable=True)
                ])
            
            if todo:
                todo.append(celery_tasks.task_finalize.subtask(args=[task.id], immutable=True))
                task.celery_id = chain(todo).apply_async()
                task.state = 'running'
                task.save()
            
            # Return task data
            task_serializer = TaskSerializer(task)
            return Response(
                {
                    'message': 'File uploaded successfully',
                    'task': task_serializer.data
                }, 
                status=status.HTTP_201_CREATED
            )
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def task_detail_api(request, task_id):
    """Get task details by ID"""
    try:
        task = Task.objects.get(id=task_id, user=request.user)
        serializer = TaskSerializer(task)
        return Response(serializer.data)
    except Task.DoesNotExist:
        return Response(
            {'error': 'Task not found'}, 
            status=status.HTTP_404_NOT_FOUND
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def task_list_api(request):
    """List user's tasks"""
    tasks = Task.objects.filter(user=request.user).order_by('-created')
    serializer = TaskSerializer(tasks, many=True)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def preset_list_api(request):
    """List available presets"""
    presets = Preset.objects.all()
    serializer = PresetSerializer(presets, many=True)
    return Response(serializer.data) 