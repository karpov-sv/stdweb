"""
Function-based views for the STDWeb REST API.
"""

import os
import io
import json
import shutil
from datetime import datetime

from django.conf import settings
from django.http import FileResponse, HttpResponse, Http404
from django.shortcuts import get_object_or_404

from rest_framework.decorators import api_view, permission_classes, parser_classes
from rest_framework.exceptions import ValidationError
from rest_framework.pagination import LimitOffsetPagination
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response

from django.db.models import Q

from stdweb.models import Task, Preset
from stdweb import celery_tasks
from stdweb import celery as celery_app
from stdweb.action_logging import log_action
from stdweb.processing import (
    supported_filters,
    supported_catalogs,
    supported_templates,
    fix_image,
    crop_image,
    preprocess_image,
)
from stdweb.views_celery import revoke_task_chain, find_task_by_chain_id

from .serializers import (
    TaskSerializer,
    TaskCreateSerializer,
    TaskListSerializer,
    TaskStateSerializer,
    PresetSerializer,
    CropRequestSerializer,
    DestripeRequestSerializer,
)


# Processing steps accepted by task_process, as understood by run_task_steps()
VALID_STEPS = ['stack', 'inspect', 'photometry', 'simple_transients', 'subtraction', 'cleanup']


class OptionalLimitOffsetPagination(LimitOffsetPagination):
    """Limit/offset pagination applied only when the client passes ?limit=,
    so that requests without it keep getting the plain un-paginated list."""
    default_limit = None


# =============================================================================
# Helper functions
# =============================================================================

def check_task_permission(request, task):
    """Check if user has permission to access/modify the task.

    Safe methods need view access; everything else needs edit access.
    """
    if request.method in ['GET', 'HEAD', 'OPTIONS']:
        return task.can_view(request.user)
    return task.can_edit(request.user)


def validate_task_path(task, path):
    """Validate path to prevent directory traversal."""
    basepath = os.path.normpath(task.path())
    fullpath = os.path.normpath(os.path.join(basepath, path))
    # Compare path components, not string prefixes, so that e.g. .../tasks/12
    # does not pass the check for a task rooted at .../tasks/1
    if fullpath != basepath and not fullpath.startswith(basepath + os.sep):
        raise Http404("Invalid path")
    return fullpath


def sanitize_data_path(path):
    """Sanitize path for data directory access."""
    path = path.lstrip('/')
    path = os.path.normpath(path)
    if path.startswith('..') or '/../' in path:
        raise Http404("Invalid path")
    return path


def get_numeric_param(request, name, default, cast):
    """Parse a numeric query parameter, raising a 400 error on invalid values."""
    value = request.query_params.get(name, default)
    try:
        return cast(value)
    except (TypeError, ValueError):
        raise ValidationError({name: 'Invalid numeric value'})


def save_figure_response(fig, fmt, quality, **kwargs):
    """Save a matplotlib figure into a JPEG/PNG HttpResponse."""
    buf = io.BytesIO()
    if fmt == 'png':
        fig.savefig(buf, format='png', **kwargs)
        content_type = 'image/png'
    else:
        fig.savefig(buf, format='jpeg', pil_kwargs={'quality': quality}, **kwargs)
        content_type = 'image/jpeg'

    buf.seek(0)
    return HttpResponse(buf.read(), content_type=content_type)


def render_fits_preview(request, fullpath, mask=None):
    """Render a FITS file as a JPEG/PNG image, honoring display query parameters."""
    from astropy.io import fits
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from stdpipe import plots

    # Get parameters
    ext = get_numeric_param(request, 'ext', -1, int)
    fmt = request.query_params.get('format', 'jpeg')
    quality = get_numeric_param(request, 'quality', 80, int)
    width = get_numeric_param(request, 'width', 800, int)

    # Load data
    try:
        data = fits.getdata(fullpath, ext)
    except Exception:
        raise ValidationError('Cannot read FITS data from file')

    # Calculate figure size
    aspect = data.shape[0] / data.shape[1]
    figsize = (width / 72, width * aspect / 72)

    # Create figure
    fig = Figure(dpi=72, figsize=figsize)
    ax = Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)

    # Plot image
    plots.imshow(
        data, ax=ax, mask=mask,
        show_axis=False,
        show_colorbar=False,
        origin='lower',
        interpolation='bicubic',
        cmap=request.query_params.get('cmap', 'Blues_r'),
        stretch=request.query_params.get('stretch', 'linear'),
        qq=[
            get_numeric_param(request, 'qmin', 0.5, float),
            get_numeric_param(request, 'qmax', 99.5, float)
        ],
        r0=get_numeric_param(request, 'r0', 0, float)
    )

    return save_figure_response(fig, fmt, quality)


def find_linked_task(celery_id):
    """Find the Django task linked to a Celery task ID."""
    task = Task.objects.filter(celery_id=celery_id).first()
    if not task:
        task = find_task_by_chain_id(celery_id)
    return task


def task_queue_info(task, celery_id):
    """Task-related fields to attach to a queue item."""
    info = {
        'task_id': task.id,
        'task_name': task.original_name,
    }
    if task.celery_chain_ids and celery_id in task.celery_chain_ids:
        info['chain_position'] = task.celery_chain_ids.index(celery_id) + 1
        info['chain_total'] = len(task.celery_chain_ids)
    return info


# =============================================================================
# Task endpoints
# =============================================================================

@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser, JSONParser])
def task_list(request):
    """
    GET: List all tasks for the current user (or all for staff).
    Supports filtering (?query=, ?state=, ?user=) and optional
    limit/offset pagination (?limit=, ?offset=).
    POST: Create a new task with optional file upload.
    """
    if request.method == 'GET':
        tasks = Task.accessible_to(request.user).order_by('-created')

        # Free-text search over the same fields as the web UI task list
        query = request.query_params.get('query')
        if query:
            for token in query.split():
                tasks = tasks.filter(
                    Q(original_name__icontains=token) |
                    Q(title__icontains=token) |
                    Q(user__username__icontains=token) |
                    Q(user__first_name__icontains=token) |
                    Q(user__last_name__icontains=token) |
                    Q(groups__name__icontains=token)
                )
            # M2M join can yield duplicate rows
            tasks = tasks.distinct()

        state = request.query_params.get('state')
        if state:
            tasks = tasks.filter(state=state)

        username = request.query_params.get('user')
        if username:
            tasks = tasks.filter(user__username=username)

        # Paginated response when ?limit= is given, plain list otherwise
        paginator = OptionalLimitOffsetPagination()
        page = paginator.paginate_queryset(tasks, request)
        if page is not None:
            serializer = TaskListSerializer(page, many=True)
            return paginator.get_paginated_response(serializer.data)

        serializer = TaskListSerializer(tasks, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        serializer = TaskCreateSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            task = serializer.save(user=request.user)

            details = {'original_name': task.original_name, 'access': 'api'}
            if serializer.validated_data.get('local_file'):
                details['source'] = 'local'
                details['original_path'] = serializer.validated_data['local_file']
            else:
                details['source'] = 'upload'

            log_action('task_create', task=task, request=request, details=details)
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


@api_view(['GET', 'POST', 'PATCH', 'DELETE'])
@permission_classes([IsAuthenticated])
def task_detail(request, pk):
    """
    GET: Get task details including config.
    PATCH: Update task configuration.
    DELETE: Delete the task and its files.
    """
    task = get_object_or_404(Task, pk=pk)

    if not check_task_permission(request, task):
        return Response({'error': 'Permission denied'}, status=403)

    if request.method == 'GET':
        serializer = TaskSerializer(task)
        return Response(serializer.data)

    elif request.method == 'PATCH' or request.method == 'POST':
        serializer = TaskSerializer(task, data=request.data, partial=True, context={'request': request})
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)

    elif request.method == 'DELETE':
        # Deletion is restricted to the owner or staff, like in the web UI
        if not task.can_delete(request.user):
            return Response({'error': 'Permission denied'}, status=403)

        log_action('task_delete', task=task, request=request,
                   details={'original_name': task.original_name, 'access': 'api'})
        task.delete()
        return Response(status=204)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def task_state(request, pk):
    """Lightweight task state for polling, without the full config."""
    task = get_object_or_404(Task, pk=pk)

    if not check_task_permission(request, task):
        return Response({'error': 'Permission denied'}, status=403)

    serializer = TaskStateSerializer(task)
    return Response(serializer.data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def task_duplicate(request, pk):
    """Duplicate a task with its files and config."""
    task = get_object_or_404(Task, pk=pk)

    if not check_task_permission(request, task):
        return Response({'error': 'Permission denied'}, status=403)

    new_task = Task.objects.create(
        original_name=task.original_name,
        title=f"{task.title} (copy)" if task.title else "(copy)",
        state='uploaded',
        user=request.user,
        config=task.config.copy(),
    )

    # Copy task directory
    if os.path.exists(task.path()):
        shutil.copytree(task.path(), new_task.path())

    log_action('task_duplicate', task=new_task, request=request,
               details={'source_task_id': task.id, 'access': 'api'})

    serializer = TaskSerializer(new_task)
    return Response(serializer.data, status=201)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def task_process(request, pk):
    """
    Run processing steps on the task.

    Request body:
    {
        "steps": ["inspect", "photometry", ...],
        "config": { ... }  // optional config overrides
    }
    """
    task = get_object_or_404(Task, pk=pk)

    if not check_task_permission(request, task):
        return Response({'error': 'Permission denied'}, status=403)

    if task.celery_id:
        return Response({'error': 'Task is already being processed'}, status=409)

    if hasattr(request.data, 'getlist'):
        steps = request.data.getlist('steps', [])
    else:
        steps = request.data.get('steps', [])
    config_override = request.data.get('config', {})

    if not steps:
        return Response({'error': 'No steps provided'}, status=400)

    # Validate steps
    for step in steps:
        if step not in VALID_STEPS:
            return Response({'error': f'Invalid step: {step}'}, status=400)

    # Accept config as a JSON-encoded string, for form-encoded requests
    if isinstance(config_override, str):
        try:
            config_override = json.loads(config_override)
        except ValueError:
            return Response({'error': 'Cannot parse config as JSON'}, status=400)
    if not isinstance(config_override, dict):
        return Response({'error': 'config must be a JSON object'}, status=400)

    # Update task config if provided
    if config_override:
        task.config.update(config_override)
        task.save()

    # Start processing
    celery_tasks.run_task_steps(task, steps)

    log_action('processing_start', task=task, request=request,
               details={'steps': steps, 'access': 'api'})

    return Response({
        'id': task.id,
        'state': task.state,
        'celery_id': task.celery_id,
        'steps': steps,
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def task_cancel(request, pk):
    """Cancel a running task."""
    task = get_object_or_404(Task, pk=pk)

    if not check_task_permission(request, task):
        return Response({'error': 'Permission denied'}, status=403)

    if not task.celery_id:
        return Response({'error': 'Task is not running'}, status=400)

    count = revoke_task_chain(task)

    log_action('processing_cancel', task=task, request=request,
               details={'revoked_count': count, 'access': 'api'})

    return Response({
        'id': task.id,
        'state': task.state,
        'revoked_count': count,
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def task_fix(request, pk):
    """Fix FITS header issues."""
    task = get_object_or_404(Task, pk=pk)

    if not check_task_permission(request, task):
        return Response({'error': 'Permission denied'}, status=403)

    if task.celery_id:
        return Response({'error': 'Task is already being processed'}, status=409)

    filename = os.path.join(task.path(), 'image.fits')
    if not os.path.exists(filename):
        return Response({'error': 'Image file not found'}, status=404)

    fix_image(filename, task.config, verbose=False)
    return Response({'id': task.id, 'status': 'fixed'})


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def task_crop(request, pk):
    """Crop the image to specified region."""
    task = get_object_or_404(Task, pk=pk)

    if not check_task_permission(request, task):
        return Response({'error': 'Permission denied'}, status=403)

    if task.celery_id:
        return Response({'error': 'Task is already being processed'}, status=409)

    filename = os.path.join(task.path(), 'image.fits')
    if not os.path.exists(filename):
        return Response({'error': 'Image file not found'}, status=404)

    serializer = CropRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=400)

    crop_image(
        filename,
        task.config,
        x1=serializer.validated_data.get('x1'),
        y1=serializer.validated_data.get('y1'),
        x2=serializer.validated_data.get('x2'),
        y2=serializer.validated_data.get('y2'),
        verbose=False,
    )
    return Response({'id': task.id, 'status': 'cropped'})


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def task_destripe(request, pk):
    """Remove stripes from the image."""
    task = get_object_or_404(Task, pk=pk)

    if not check_task_permission(request, task):
        return Response({'error': 'Permission denied'}, status=403)

    if task.celery_id:
        return Response({'error': 'Task is already being processed'}, status=409)

    filename = os.path.join(task.path(), 'image.fits')
    if not os.path.exists(filename):
        return Response({'error': 'Image file not found'}, status=404)

    serializer = DestripeRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=400)

    direction = serializer.validated_data['direction']
    preprocess_image(
        filename,
        task.config,
        destripe_horizontal=(direction in ['horizontal', 'both']),
        destripe_vertical=(direction in ['vertical', 'both']),
        verbose=False,
    )
    return Response({'id': task.id, 'status': 'destriped'})


# =============================================================================
# Task file endpoints
# =============================================================================

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def task_files_list(request, pk):
    """List files in the task directory."""
    task = get_object_or_404(Task, pk=pk)

    if not check_task_permission(request, task):
        return Response({'error': 'Permission denied'}, status=403)

    basepath = task.path()
    if not os.path.exists(basepath):
        return Response([])

    files = []
    for name in os.listdir(basepath):
        filepath = os.path.join(basepath, name)
        try:
            stat = os.stat(filepath)
            files.append({
                'name': name,
                'path': name,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'is_dir': os.path.isdir(filepath),
            })
        except OSError:
            continue

    files.sort(key=lambda x: x['name'])
    return Response(files)


@api_view(['GET', 'POST', 'DELETE'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def task_file(request, pk, path):
    """
    GET: Download a file from the task directory.
    POST: Upload a file to the task directory.
    DELETE: Delete a file from the task directory.
    """
    task = get_object_or_404(Task, pk=pk)

    if not check_task_permission(request, task):
        return Response({'error': 'Permission denied'}, status=403)

    fullpath = validate_task_path(task, path)

    if request.method == 'GET':
        if not os.path.exists(fullpath) or os.path.isdir(fullpath):
            raise Http404("File not found")
        return FileResponse(
            open(fullpath, 'rb'),
            as_attachment=True,
            filename=os.path.basename(path)
        )

    elif request.method == 'POST':
        file = request.FILES.get('file')
        if not file:
            return Response({'error': 'No file provided'}, status=400)

        # Create parent directories if needed
        parent_dir = os.path.dirname(fullpath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with open(fullpath, 'wb') as f:
            for chunk in file.chunks():
                f.write(chunk)

        return Response({
            'path': path,
            'size': os.path.getsize(fullpath),
        }, status=201)

    elif request.method == 'DELETE':
        if not os.path.exists(fullpath):
            raise Http404("File not found")

        if os.path.isdir(fullpath):
            shutil.rmtree(fullpath)
        else:
            os.unlink(fullpath)

        return Response(status=204)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def task_preview(request, pk, path):
    """Generate JPEG/PNG preview of a FITS file."""
    from astropy.io import fits

    task = get_object_or_404(Task, pk=pk)

    if not check_task_permission(request, task):
        return Response({'error': 'Permission denied'}, status=403)

    fullpath = validate_task_path(task, path)
    if not os.path.exists(fullpath):
        raise Http404("File not found")

    # Load custom mask if exists
    basepath = task.path()
    if path == 'image.fits' and os.path.exists(os.path.join(basepath, 'custom_mask.fits')):
        mask = fits.getdata(os.path.join(basepath, 'custom_mask.fits'), -1) > 0
    else:
        mask = None

    return render_fits_preview(request, fullpath, mask=mask)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def task_cutout(request, pk, path):
    """Generate multi-panel visualization of STDPipe cutout files."""
    from stdpipe import cutouts

    task = get_object_or_404(Task, pk=pk)

    if not check_task_permission(request, task):
        return Response({'error': 'Permission denied'}, status=403)

    fullpath = validate_task_path(task, path)
    if not os.path.exists(fullpath):
        raise Http404("File not found")

    # Get parameters
    width = get_numeric_param(request, 'width', 800, int)
    fmt = request.query_params.get('format', 'jpeg')
    quality = get_numeric_param(request, 'quality', 80, int)

    # Load and plot cutout
    cutout = cutouts.load_cutout(fullpath)
    fig = cutouts.plot_cutout(cutout, figsize=(width / 72, width / 72))

    return save_figure_response(fig, fmt, quality, bbox_inches='tight')


# =============================================================================
# Queue endpoints
# =============================================================================

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def queue_list(request):
    """List all active/pending/scheduled Celery tasks."""
    queue = []

    inspect = celery_app.app.control.inspect(timeout=0.1)
    for res, state in [
        (inspect.active(), 'active'),
        (inspect.reserved(), 'pending'),
        (inspect.scheduled(), 'scheduled')
    ]:
        if res:
            for wtasks in res.values():
                for ctask in wtasks:
                    item = {
                        'id': ctask.get('id'),
                        'name': ctask.get('name', '').split('.')[-1],
                        'full_name': ctask.get('name'),
                        'state': state,
                        'time_start': ctask.get('time_start'),
                    }

                    # Attach linked Django task info, if the user may see it
                    task = find_linked_task(ctask['id'])
                    if task and task.can_view(request.user):
                        item.update(task_queue_info(task, ctask['id']))

                    queue.append(item)

    return Response(queue)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def queue_detail(request, pk):
    """Get status of a specific Celery task."""
    ctask = celery_app.app.AsyncResult(pk)

    result = {
        'id': ctask.id,
        'state': ctask.state,
        'ready': ctask.ready(),
        'successful': ctask.successful() if ctask.ready() else None,
        'failed': ctask.failed() if ctask.ready() else None,
    }

    # Attach linked Django task info, if the user may see it
    task = find_linked_task(pk)
    if task and task.can_view(request.user):
        result.update(task_queue_info(task, pk))

    return Response(result)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def queue_terminate(request, pk):
    """Terminate a Celery task (staff or anyone able to edit the linked task)."""
    # Find Django task and revoke entire chain
    task = find_linked_task(pk)

    if task:
        if not task.can_edit(request.user):
            return Response({'error': 'No permission to terminate this task'}, status=403)

        count = revoke_task_chain(task)
        return Response({
            'id': pk,
            'task_id': task.id,
            'revoked_count': count,
            'state': 'cancelled',
        })
    else:
        # No linked Django task - only staff may revoke arbitrary Celery IDs
        if not request.user.is_staff:
            return Response({'error': 'Staff access required'}, status=403)

        # Fallback: revoke just this ID
        celery_app.app.control.revoke(pk, terminate=True, signal='SIGTERM')
        return Response({
            'id': pk,
            'revoked_count': 1,
            'state': 'terminated',
        })


# =============================================================================
# Data file endpoints
# =============================================================================

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def data_files(request, path=''):
    """List directory or download file from data directory."""
    path = sanitize_data_path(path) if path else ''
    fullpath = os.path.join(settings.DATA_PATH, path) if path else settings.DATA_PATH

    if not os.path.exists(fullpath):
        raise Http404("Path not found")

    if os.path.isdir(fullpath):
        # List directory contents
        files = []
        for name in os.listdir(fullpath):
            filepath = os.path.join(fullpath, name)
            try:
                stat = os.stat(filepath)
                files.append({
                    'name': name,
                    'path': os.path.join(path, name) if path else name,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'is_dir': os.path.isdir(filepath),
                })
            except OSError:
                continue

        files.sort(key=lambda x: (not x['is_dir'], x['name'].lower()))
        return Response(files)
    else:
        # Download file
        return FileResponse(
            open(fullpath, 'rb'),
            as_attachment=True,
            filename=os.path.basename(path)
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def data_file_preview(request, path):
    """Generate preview for a FITS file in data directory."""
    path = sanitize_data_path(path)
    fullpath = os.path.join(settings.DATA_PATH, path)

    if not os.path.exists(fullpath):
        raise Http404("File not found")

    return render_fits_preview(request, fullpath)


# =============================================================================
# Preset endpoints
# =============================================================================

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def preset_list(request):
    """List all presets."""
    presets = Preset.objects.all()
    serializer = PresetSerializer(presets, many=True)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def preset_detail(request, pk):
    """Get preset details."""
    preset = get_object_or_404(Preset, pk=pk)
    serializer = PresetSerializer(preset)
    return Response(serializer.data)


# =============================================================================
# Reference data endpoints
# =============================================================================

@api_view(['GET'])
@permission_classes([AllowAny])
def reference_data(request, data_type=None):
    """Get reference data (filters, catalogs, templates)."""
    if data_type == 'filters':
        return Response(supported_filters)
    elif data_type == 'catalogs':
        return Response(supported_catalogs)
    elif data_type == 'templates':
        return Response(supported_templates)
    elif data_type is None:
        return Response({
            'filters': list(supported_filters.keys()),
            'catalogs': list(supported_catalogs.keys()),
            'templates': list(supported_templates.keys()),
        })
    else:
        raise Http404("Unknown reference data type")
