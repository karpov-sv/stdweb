"""
Function-based views for the STDWeb REST API.
"""

import os
import io
import shutil
from datetime import datetime

from django.conf import settings
from django.http import FileResponse, HttpResponse, Http404
from django.shortcuts import get_object_or_404

from rest_framework.decorators import api_view, permission_classes, parser_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response

from stdweb.models import Task, Preset
from stdweb import celery_tasks
from stdweb import celery as celery_app
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
    PresetSerializer,
)


# =============================================================================
# Helper functions
# =============================================================================

def check_task_permission(request, task):
    """Check if user has permission to access/modify the task."""
    user = request.user
    if user.is_staff:
        return True
    if user == task.user:
        return True
    if user.has_perm('stdweb.edit_all_tasks'):
        return True
    if user.has_perm('stdweb.view_all_tasks') and request.method in ['GET', 'HEAD', 'OPTIONS']:
        return True
    return False


def validate_task_path(task, path):
    """Validate path to prevent directory traversal."""
    basepath = os.path.normpath(task.path())
    fullpath = os.path.normpath(os.path.join(basepath, path))
    if not fullpath.startswith(basepath):
        raise Http404("Invalid path")
    return fullpath


def sanitize_data_path(path):
    """Sanitize path for data directory access."""
    path = path.lstrip('/')
    path = os.path.normpath(path)
    if path.startswith('..') or '/../' in path:
        raise Http404("Invalid path")
    return path


# =============================================================================
# Task endpoints
# =============================================================================

@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser, JSONParser])
def task_list(request):
    """
    GET: List all tasks for the current user (or all for staff).
    POST: Create a new task with optional file upload.
    """
    if request.method == 'GET':
        user = request.user
        if user.is_staff or user.has_perm('stdweb.view_all_tasks'):
            tasks = Task.objects.all().order_by('-created')
        else:
            tasks = Task.objects.filter(user=user).order_by('-created')

        serializer = TaskListSerializer(tasks, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        serializer = TaskCreateSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(user=request.user)
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
        serializer = TaskSerializer(task, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)

    elif request.method == 'DELETE':
        task.delete()
        return Response(status=204)


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

    if hasattr(request.data, 'getlist'):
        steps = request.data.getlist('steps', [])
    else:
        steps = request.data.get('steps', [])
    config_override = request.data.get('config', {})

    if not steps:
        return Response({'error': 'No steps provided'}, status=400)

    # Validate steps
    valid_steps = ['inspect', 'photometry', 'transients', 'subtraction', 'cleanup']
    for step in steps:
        if step not in valid_steps:
            return Response({'error': f'Invalid step: {step}'}, status=400)

    # Update task config if provided
    if config_override:
        task.config.update(config_override)
        task.save()

    # Map step names to celery task names
    step_mapping = {
        'inspect': 'inspect',
        'photometry': 'photometry',
        'transients': 'simple_transients',
        'subtraction': 'subtraction',
        'cleanup': 'cleanup',
    }
    celery_steps = [step_mapping.get(s, s) for s in steps]

    # Start processing
    celery_tasks.run_task_steps(task, celery_steps)

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

    filename = os.path.join(task.path(), 'image.fits')
    if not os.path.exists(filename):
        return Response({'error': 'Image file not found'}, status=404)

    crop_image(
        filename,
        task.config,
        x1=request.data.get('x1'),
        y1=request.data.get('y1'),
        x2=request.data.get('x2'),
        y2=request.data.get('y2'),
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

    filename = os.path.join(task.path(), 'image.fits')
    if not os.path.exists(filename):
        return Response({'error': 'Image file not found'}, status=404)

    direction = request.data.get('direction', 'both')
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
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from stdpipe import plots

    task = get_object_or_404(Task, pk=pk)

    if not check_task_permission(request, task):
        return Response({'error': 'Permission denied'}, status=403)

    fullpath = validate_task_path(task, path)
    if not os.path.exists(fullpath):
        raise Http404("File not found")

    # Get parameters
    ext = int(request.query_params.get('ext', -1))
    fmt = request.query_params.get('format', 'jpeg')
    quality = int(request.query_params.get('quality', 80))
    width = int(request.query_params.get('width', 800))

    # Load custom mask if exists
    basepath = task.path()
    if path == 'image.fits' and os.path.exists(os.path.join(basepath, 'custom_mask.fits')):
        mask = fits.getdata(os.path.join(basepath, 'custom_mask.fits'), -1) > 0
    else:
        mask = None

    # Load data
    data = fits.getdata(fullpath, ext)

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
            float(request.query_params.get('qmin', 0.5)),
            float(request.query_params.get('qmax', 99.5))
        ],
        r0=float(request.query_params.get('r0', 0))
    )

    # Save to buffer
    buf = io.BytesIO()
    if fmt == 'png':
        fig.savefig(buf, format='png')
        content_type = 'image/png'
    else:
        fig.savefig(buf, format='jpeg', pil_kwargs={'quality':quality})
        content_type = 'image/jpeg'

    buf.seek(0)
    return HttpResponse(buf.read(), content_type=content_type)


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
    width = int(request.query_params.get('width', 800))
    fmt = request.query_params.get('format', 'jpeg')
    quality = int(request.query_params.get('quality', 80))

    # Load and plot cutout
    cutout = cutouts.load_cutout(fullpath)
    fig = cutouts.plot_cutout(cutout, figsize=(width / 72, width / 72))

    # Save to buffer
    buf = io.BytesIO()
    if fmt == 'png':
        fig.savefig(buf, format='png', bbox_inches='tight')
        content_type = 'image/png'
    else:
        fig.savefig(buf, format='jpeg', quality=quality, bbox_inches='tight')
        content_type = 'image/jpeg'

    buf.seek(0)
    return HttpResponse(buf.read(), content_type=content_type)


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

                    # Find linked Django task
                    task = Task.objects.filter(celery_id=ctask['id']).first()
                    if not task:
                        task = find_task_by_chain_id(ctask['id'])

                    if task:
                        item['task_id'] = task.id
                        item['task_name'] = task.original_name
                        if task.celery_chain_ids and ctask['id'] in task.celery_chain_ids:
                            item['chain_position'] = task.celery_chain_ids.index(ctask['id']) + 1
                            item['chain_total'] = len(task.celery_chain_ids)

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

    # Find linked Django task
    task = Task.objects.filter(celery_id=pk).first()
    if not task:
        task = find_task_by_chain_id(pk)

    if task:
        result['task_id'] = task.id
        result['task_name'] = task.original_name
        if task.celery_chain_ids and pk in task.celery_chain_ids:
            result['chain_position'] = task.celery_chain_ids.index(pk) + 1
            result['chain_total'] = len(task.celery_chain_ids)

    return Response(result)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def queue_terminate(request, pk):
    """Terminate a Celery task (staff only)."""
    if not request.user.is_staff:
        return Response({'error': 'Staff access required'}, status=403)

    # Find Django task and revoke entire chain
    task = Task.objects.filter(celery_id=pk).first()
    if not task:
        task = find_task_by_chain_id(pk)

    if task:
        count = revoke_task_chain(task)
        return Response({
            'id': pk,
            'task_id': task.id,
            'revoked_count': count,
            'state': 'cancelled',
        })
    else:
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
    from astropy.io import fits
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from stdpipe import plots

    path = sanitize_data_path(path)
    fullpath = os.path.join(settings.DATA_PATH, path)

    if not os.path.exists(fullpath):
        raise Http404("File not found")

    # Get parameters
    ext = int(request.query_params.get('ext', -1))
    fmt = request.query_params.get('format', 'jpeg')
    quality = int(request.query_params.get('quality', 80))
    width = int(request.query_params.get('width', 800))

    # Load data
    data = fits.getdata(fullpath, ext)

    # Calculate figure size
    aspect = data.shape[0] / data.shape[1]
    figsize = (width / 72, width * aspect / 72)

    # Create figure
    fig = Figure(dpi=72, figsize=figsize)
    ax = Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)

    # Plot image
    plots.imshow(
        data, ax=ax,
        show_axis=False,
        show_colorbar=False,
        origin='lower',
        interpolation='bicubic',
        cmap=request.query_params.get('cmap', 'Blues_r'),
        stretch=request.query_params.get('stretch', 'linear'),
        qq=[
            float(request.query_params.get('qmin', 0.5)),
            float(request.query_params.get('qmax', 99.5))
        ],
        r0=float(request.query_params.get('r0', 0))
    )

    # Save to buffer
    buf = io.BytesIO()
    if fmt == 'png':
        fig.savefig(buf, format='png')
        content_type = 'image/png'
    else:
        fig.savefig(buf, format='jpeg', quality=quality)
        content_type = 'image/jpeg'

    buf.seek(0)
    return HttpResponse(buf.read(), content_type=content_type)


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
    else:
        return Response({
            'filters': list(supported_filters.keys()),
            'catalogs': list(supported_catalogs.keys()),
            'templates': list(supported_templates.keys()),
        })
