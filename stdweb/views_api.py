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
                'target', 'gain', 'saturation', 'time',
                # Template selection
                'template'
            ]

            # Handle `template_catalog` alias if provided (e.g., "ZTF_DR7")
            template_catalog = serializer.validated_data.get('template_catalog') if hasattr(serializer, 'validated_data') else None
            if template_catalog:
                # Normalise input
                tc_norm = str(template_catalog).lower()
                alias_map = {
                    'ztf_dr7': 'ztf',
                    'ztf': 'ztf',
                    'ps1': 'ps1',
                    'ps1_dr2': 'ps1',
                    'pan-starrs': 'ps1',
                    'pan-starrs_dr2': 'ps1',
                    'ls_dr10': 'ls',
                    'legacy': 'ls',
                }
                task.config['template'] = alias_map.get(tc_norm, tc_norm)
            
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


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def task_action_api(request, task_id):
    """Trigger processing actions on existing tasks"""
    try:
        task = Task.objects.get(id=task_id, user=request.user)
        
        # Check if task is already running
        if task.celery_id is not None:
            # Check if the task is actually running
            from . import celery
            ctask = celery.app.AsyncResult(task.celery_id)
            if ctask.state not in ['REVOKED', 'FAILURE', 'SUCCESS']:
                return Response(
                    {'error': f'Task {task_id} is already running'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
        
        action = request.data.get('action')
        if not action:
            return Response(
                {'error': 'Action parameter is required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Import celery tasks
        from . import celery_tasks

        # --- NEW: update configuration if extra parameters are supplied ---
        # Allow the same subset we accept on upload so that users can tweak
        # parameters like gain/saturation before re-running an action.
        extra_config_params = [
            # Photometry parameters
            'sn', 'initial_aper', 'initial_r0', 'bg_size', 'minarea',
            'rel_aper', 'rel_bg1', 'rel_bg2', 'fwhm_override',
            'filter', 'cat_name', 'cat_limit', 'spatial_order', 'use_color', 'sr_override',
            'prefilter_detections', 'filter_blends', 'diagnose_color', 'refine_wcs',
            'blind_match_wcs', 'inspect_bg', 'centroid_targets', 'nonlin',
            'blind_match_ps_lo', 'blind_match_ps_up', 'blind_match_center', 'blind_match_sr0',
            # Inspection parameters
            'target', 'gain', 'saturation', 'time',
            # Template selection
            'template'
        ]

        updated = False
        for param in extra_config_params:
            if param in request.data:
                # DRF gives QueryDict, convert e.g. "true"/"false" to bool where possible
                value = request.data.get(param)
                if isinstance(value, str):
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                task.config[param] = value
                updated = True
        # Handle `template_catalog` alias here as well
        if 'template_catalog' in request.data:
            tc_norm = str(request.data.get('template_catalog')).lower()
            alias_map = {
                'ztf_dr7': 'ztf',
                'ztf': 'ztf',
                'ps1': 'ps1',
                'ps1_dr2': 'ps1',
                'pan-starrs': 'ps1',
                'pan-starrs_dr2': 'ps1',
                'ls_dr10': 'ls',
                'legacy': 'ls',
            }
            task.config['template'] = alias_map.get(tc_norm, tc_norm)
            updated = True

        if updated:
            task.save()
        # --- END NEW CODE ---

        # Handle different actions
        if action == 'inspect':
            task.celery_id = celery_tasks.task_inspect.delay(task.id).id
            task.state = 'inspect'
            task.save()
            return Response({'message': f'Started inspection for task {task_id}'})
            
        elif action == 'photometry':
            task.celery_id = celery_tasks.task_photometry.delay(task.id).id
            task.state = 'photometry'
            task.save()
            return Response({'message': f'Started photometry for task {task_id}'})
            
        elif action == 'transients_simple':
            task.celery_id = celery_tasks.task_transients_simple.delay(task.id).id
            task.state = 'transients_simple'
            task.save()
            return Response({'message': f'Started simple transient detection for task {task_id}'})
            
        elif action == 'subtraction':
            task.celery_id = celery_tasks.task_subtraction.delay(task.id).id
            task.state = 'subtraction'
            task.save()
            return Response({'message': f'Started subtraction for task {task_id}'})
            
        elif action == 'cleanup':
            task.celery_id = celery_tasks.task_cleanup.delay(task.id).id
            task.state = 'cleanup'
            task.save()
            return Response({'message': f'Started cleanup for task {task_id}'})
            
        else:
            return Response(
                {'error': f'Unknown action: {action}. Valid actions are: inspect, photometry, transients_simple, subtraction, cleanup'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
            
    except Task.DoesNotExist:
        return Response(
            {'error': 'Task not found'}, 
            status=status.HTTP_404_NOT_FOUND
        ) 