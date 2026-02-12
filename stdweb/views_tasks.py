from django.http import HttpResponse, FileResponse, HttpResponseRedirect, JsonResponse, HttpResponseForbidden
from django.template.response import TemplateResponse
from django.urls import reverse
from django.db.models import Q
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import redirect_to_login
from django.views.decorators.cache import cache_page
from django.views.decorators.http import require_POST
from django.conf import settings
from django.shortcuts import get_object_or_404

import os, glob, shutil
import json
import numpy as np

from astropy.table import Table
from astropy.io import fits

from stdpipe.utils import file_write, file_read
from stdpipe import astrometry

from . import views
from . import models
from . import forms
from . import celery_tasks
from . import celery
from . import processing
from . import utils
from .action_logging import log_action


def _get_filtered_tasks(request):
    """Extract shared task filtering logic. Returns (queryset, form)."""
    tasks = models.Task.objects.all()
    tasks = tasks.order_by('-created')

    form = forms.TasksFilterForm(
        request.GET,
        show_all=request.user.is_staff or request.user.has_perm('stdweb.view_all_tasks'),
    )

    if form.is_valid():
        show_all = form.cleaned_data.get('show_all')
        if not show_all:
            tasks = tasks.filter(user=request.user)

        query = form.cleaned_data.get('query')
        if query:
            # Coordinate query?..
            ra0, dec0, sr0 = utils.resolve_coordinates(query)

            if ra0 is not None and dec0 is not None:
                pks = []
                for task in tasks:
                    if task.ra is not None and task.dec is not None and task.radius is not None:
                        dist = astrometry.spherical_distance(task.ra, task.dec, ra0, dec0)

                        if sr0 is not None and dist < sr0:
                            pks.append(task.pk)
                        elif sr0 is None and dist < sr:
                            pks.append(task.pk)

                tasks = tasks.filter(pk__in=pks)
            else:
                # Otherwise plain text search
                for token in query.split():
                    tasks = tasks.filter(
                        Q(original_name__icontains=token) |
                        Q(title__icontains=token) |
                        Q(user__username__icontains=token) |
                        Q(user__first_name__icontains=token) |
                        Q(user__last_name__icontains=token)
                    )

    return tasks, form


def tasks(request, id=None):
    context = {}

    if id:
        task = get_object_or_404(models.Task, id=id)
        path = task.path()

        # Permissions
        if request.user.is_authenticated and (
                request.user.is_staff or request.user == task.user or request.user.has_perm('stdweb.edit_all_tasks')
        ):
            context['user_may_submit'] = True
        else:
            context['user_may_submit'] = False

        # Clear the link to queued task if it was revoked
        if task.celery_id:
            ctask = celery.app.AsyncResult(task.celery_id)
            if ctask.state == 'REVOKED' or ctask.state == 'FAILURE':
                task.celery_id = None
                task.state = 'failed' # Should we do it?
                task.complete()
                task.save()

        # Prevent task operations if it is still running
        if task.celery_id is not None and request.method == 'POST':
            messages.warning(request, f"Task {id} is already running")
            return HttpResponseRedirect(request.path_info)

        all_forms = {}

        all_forms['inspect'] = forms.TaskInspectForm(request.POST or None, initial = task.config | {'raw_config': task.config})
        all_forms['photometry'] = forms.TaskPhotometryForm(request.POST or None, initial = task.config)
        all_forms['transients_simple'] = forms.TaskTransientsSimpleForm(request.POST or None, initial = task.config)
        all_forms['subtraction'] = forms.TaskSubtractionForm(request.POST or None, initial = task.config)

        for name,form in all_forms.items():
            context['form_'+name] = form

        # Form actions
        if request.method == 'POST':
            # Handle forms
            form_type = request.POST.get('form_type')
            form = all_forms.get(form_type)
            if form and form.is_valid():
                # Handle uploaded files, if any
                if 'custom_template' in request.FILES:
                    views.handle_uploaded_file(request.FILES['custom_template'],
                                               os.path.join(task.path(), 'custom_template.fits'))
                    messages.info(request, "Custom template uploaded as custom_template.fits")

                if form.has_changed() or True:
                    for name,value in form.cleaned_data.items():
                        # we do not want these to go to task.config
                        ignored_fields = [
                            'form_type',
                            'crop_x1', 'crop_y1', 'crop_x2', 'crop_y2',
                            'raw_config',
                            # 'run_photometry', 'run_simple_transients', 'run_subtraction',
                        ]
                        if name not in ignored_fields:
                            if name in form.changed_data or name not in task.config:
                                # update only changed or new fields
                                task.config[name] = value

                    task.save()

                # Special handling for in-form buttons
                if request.POST.get('action_custom_mask'):
                    return HttpResponseRedirect(reverse('task_mask', kwargs={'id': task.id, 'mode': 'template'}))

                # Handle actions
                action = request.POST.get('action')

                if action == 'delete_task':
                    # Only owner or staff may delete the task
                    if request.user.is_staff or request.user == task.user:
                        log_action('task_delete', task=task, request=request,
                                   details={'original_name': task.original_name, 'access': 'web'})
                        task.delete()
                        messages.success(request, "Task " + str(id ) + " is deleted")
                        return HttpResponseRedirect(reverse('tasks'))
                    else:
                        messages.error(request, "Cannot delete task " + str(id) + " belonging to " + task.user.username)
                        return HttpResponseRedirect(request.path_info)

                if action == 'duplicate_task':
                    old_path = task.path()
                    old_id = task.id

                    # As per https://docs.djangoproject.com/en/5.2/topics/db/queries/#copying-model-instances
                    task.pk = None
                    task._state.adding = True
                    task.user = request.user
                    task.state = 'duplicated'
                    task.title = ((task.title + '\n') if task.title else '') + f'Duplicated from task {id}'
                    task.save() # To populate new task.id

                    try:
                        os.makedirs(task.path())
                    except OSError:
                        pass

                    for name in [
                            'image.fits', 'image.wcs',
                            'custom_mask.fits', 'custom_mask.json',
                            'custom_template.fits',
                            'custom_template_mask.fits', 'custom_template_mask.json'
                    ]:
                        if os.path.exists(os.path.join(old_path, name)):
                            shutil.copyfile(os.path.join(old_path, name), os.path.join(task.path(), name))

                    log_action('task_duplicate', task=task, request=request,
                               details={'source_task_id': old_id, 'access': 'web'})
                    messages.success(request, "Task duplicated as " + str(task.id))

                    return HttpResponseRedirect(reverse('tasks', kwargs={'id': task.id}))

                # Ensure we have proper permissions for the rest of the actions
                if not context['user_may_submit']:
                    messages.error(request, "Cannot perform action " + action + " on task " + str(id) + " belonging to " + task.user.username)
                    return HttpResponseRedirect(request.path_info)

                if action == 'fix_image':
                    # TODO: move to async celery task?..
                    processing.fix_image(os.path.join(task.path(), 'image.fits'), task.config)
                    messages.success(request, "Fixed the image header for task " + str(id))

                if action == 'update_config':
                    if 'raw_config' in form.changed_data:
                        task.config = form.cleaned_data.get('raw_config')
                        task.save()
                        messages.success(request, f"Config for task {str(id)} updated")
                    else:
                        messages.success(request, f"Config for task {str(id)} unchanged")

                if action == 'make_custom_mask':
                    return HttpResponseRedirect(reverse('task_mask', kwargs={'id': task.id, 'mode': 'image'}))

                if action == 'preprocess':
                    return HttpResponseRedirect(reverse('task_preprocess', kwargs={'id': task.id}))

                if action in ['cleanup_task', 'archive_task']:
                    if action in ['cleanup_task']:
                        task.config = {} # We reset the config on cleanup but keep on archiving
                        log_action('task_cleanup', task=task, request=request, details={'access': 'web'})
                    elif action in ['archive_task']:
                        log_action('task_archive', task=task, request=request, details={'access': 'web'})
                    celery_tasks.run_task_steps(task, ['cleanup'])
                    messages.success(request, "Started cleanup for task " + str(id))

                if action == 'inspect_image':
                    steps = ['inspect']
                    if form.cleaned_data.get('run_photometry'):
                        steps.append('photometry')
                    if form.cleaned_data.get('run_subtraction'):
                        steps.append('subtraction')
                    celery_tasks.run_task_steps(task, steps)
                    log_action('processing_start', task=task, request=request,
                               details={'steps': steps, 'access': 'web'})
                    messages.success(request, "Started image inspection for task " + str(id))
                    if form.cleaned_data.get('run_photometry'):
                        messages.success(request, "Started image photometry for task " + str(id))
                    if form.cleaned_data.get('run_subtraction'):
                        messages.success(request, "Started template subtraction for task " + str(id))

                if action == 'photometry_image':
                    steps = ['photometry']
                    if form.cleaned_data.get('run_subtraction'):
                        steps.append('subtraction')
                    celery_tasks.run_task_steps(task, steps)
                    log_action('processing_start', task=task, request=request,
                               details={'steps': steps, 'access': 'web'})
                    messages.success(request, "Started image photometry for task " + str(id))
                    if form.cleaned_data.get('run_subtraction'):
                        messages.success(request, "Started template subtraction for task " + str(id))

                if action == 'transients_simple_image':
                    celery_tasks.run_task_steps(task, ['simple_transients'])
                    log_action('processing_start', task=task, request=request,
                               details={'steps': ['simple_transients'], 'access': 'web'})
                    messages.success(request, "Started simple transient detection for task " + str(id))

                if action == 'subtract_image':
                    celery_tasks.run_task_steps(task, ['subtraction'])
                    log_action('processing_start', task=task, request=request,
                               details={'steps': ['subtraction'], 'access': 'web'})
                    messages.success(request, "Started template subtraction for task " + str(id))

                return HttpResponseRedirect(request.path_info)

            elif form:
                for f,err in form.errors.items():
                    messages.error(request, f"Error in {f} field: {err}")

            return HttpResponseRedirect(request.path_info)

        # Display task
        context['task'] = task

        context['files'] = [os.path.split(_)[1] for _ in glob.glob(os.path.join(path, '*'))]

        # Target cutouts
        context['target_cutouts'] = []
        if os.path.exists(os.path.join(path, 'targets')) and 'targets' in task.config:
            for i,target in enumerate(task.config['targets']):
                if os.path.exists(os.path.join(path, f"targets/target_{i:04d}.cutout")):
                    context['target_cutouts'].append(
                        {'path': f"targets/target_{i:04d}.cutout", 'ra': target['ra'], 'dec': target['dec']}
                    )
        elif os.path.exists(os.path.join(path, 'target.cutout')):
            # Fallback to legacy path
            context['target_cutouts'].append(
                {'path': 'target.cutout', 'ra': task.config['target_ra'], 'dec': task.config['target_dec']}
            )

        # Additional info
        context['supported_filters'] = processing.supported_filters
        context['supported_catalogs'] = processing.supported_catalogs
        context['supported_templates'] = processing.supported_templates

        if 'candidates_simple.vot' in context['files']:
            candidates_simple = Table.read(os.path.join(path, 'candidates_simple.vot'))
            if candidates_simple:
                candidates_simple.sort('flux', reverse=True)
                context['candidates_simple'] = candidates_simple

        if 'candidates.vot' in context['files']:
            candidates = Table.read(os.path.join(path, 'candidates.vot'))
            if candidates:
                candidates.sort('flux', reverse=True)
                context['candidates'] = candidates

        return TemplateResponse(request, 'task.html', context=context)
    else:
        if not request.user.is_authenticated:
            return redirect_to_login(request.path)

        # List all tasks
        tasks, form = _get_filtered_tasks(request)

        context['form'] = form

        if request.method == 'GET' and form.is_valid():
            query = form.cleaned_data.get('query')
            if query:
                ra0, dec0, sr0 = utils.resolve_coordinates(query)
                if ra0 is not None and dec0 is not None:
                    if sr0 is not None:
                        messages.info(request, f"Looking for tasks inside {sr0:.2f} deg around {ra0:.4f} {dec0:+.4f}")
                    else:
                        messages.info(request, f"Looking for tasks covering {ra0:.4f} {dec0:+.4f}")

        context['tasks'] = tasks
        context['referer'] = request.path + '?' + request.GET.urlencode();

    return TemplateResponse(request, 'tasks.html', context=context)


@login_required
def tasks_skymap_data(request):
    """Return JSON with spatial data for tasks matching the current filter."""
    tasks, form = _get_filtered_tasks(request)

    # Only include tasks with spatial data
    tasks = tasks.filter(ra__isnull=False, dec__isnull=False, radius__isnull=False)[:500]

    result = []
    for task in tasks:
        entry = {
            'id': task.id,
            'original_name': task.original_name,
            'title': task.title or '',
            'state': task.state,
            'user': task.user.username,
            'created': task.created.strftime('%Y-%m-%d %H:%M'),
            'ra': task.ra,
            'dec': task.dec,
            'radius': task.radius,
        }

        if task.moc and request.GET.get('include_moc'):
            try:
                from mocpy import MOC
                moc = MOC.from_string(task.moc)
                entry['moc_json'] = moc.serialize("json")
            except Exception:
                pass

        result.append(entry)

    return JsonResponse({'tasks': result})


def tasks_actions(request):
    form = forms.TasksActionsForm(request.POST)

    if request.method == 'POST':
        if form.is_valid():
            task_ids = form.cleaned_data['tasks']
            action = request.POST.get('action')

            for id in task_ids:
                task = get_object_or_404(models.Task, id=id)

                if action == 'archive':
                    task.celery_id = celery_tasks.task_cleanup.delay(task.id).id
                    task.state = 'archive'
                    task.save()
                    log_action('task_archive', task=task, request=request, details={'access': 'web'})
                    messages.success(request, "Started archiving for task " + str(id))

                if action == 'cleanup':
                    task.celery_id = celery_tasks.task_cleanup.delay(task.id).id
                    task.config = {} # should we reset the config on cleanup?..
                    task.state = 'cleanup'
                    task.save()
                    log_action('task_cleanup', task=task, request=request, details={'access': 'web'})
                    messages.success(request, "Started cleanup for task " + str(id))

                if action == 'delete':
                    if request.user.is_staff or request.user == task.user:
                        log_action('task_delete', task=task, request=request,
                                   details={'original_name': task.original_name, 'access': 'web'})
                        task.delete()
                        messages.success(request, "Task " + str(id ) + " is deleted")
                    else:
                        messages.error(request, "Cannot delete task " + str(id) + " belonging to " + task.user.username)

            return HttpResponseRedirect(form.cleaned_data['referer'])

    return HttpResponseRedirect(reverse('tasks'))


def task_download(request, id=None, path='', **kwargs):
    task = get_object_or_404(models.Task, id=id)

    return views.download(request, path, base=task.path(), **kwargs)


@cache_page(15 * 60)
def task_preview(request, id=None, path='', **kwargs):
    task = get_object_or_404(models.Task, id=id)

    return views.preview(request, path, base=task.path(), **kwargs)


@cache_page(15 * 60)
def task_cutout(request, id=None, path='', **kwargs):
    task = get_object_or_404(models.Task, id=id)

    return views.cutout(request, path, base=task.path(), **kwargs)


@login_required
def task_files(request, id, path=''):
    """Browse files within a task folder using the generic file browser."""
    task = get_object_or_404(models.Task, id=id)

    # Permission check: owner or user with view_all_tasks permission
    if task.user != request.user and not request.user.has_perm('stdweb.view_all_tasks'):
        return HttpResponseForbidden("You don't have permission to view this task")

    # Call generic list_files with task base path
    response = views.list_files(request, path=path, base=task.path())

    # Customize context for task-specific rendering
    context = response.context_data
    context['task'] = task
    context['task_id'] = id
    context['is_task_browser'] = True

    return response


def handle_task_mask_creation(
        task, width, height, areas, inverted=False,
        image_file='image.fits',
        mask_file='custom_mask.fits',
        json_file='custom_mask.json'
):
    """
    Create binary FITS mask from rectangular areas.

    Args:
        task: Task object
        width: Display width in pixels
        height: Display height in pixels
        areas: List of rectangular area dictionaries
        inverted: If True, mask everything except selected areas
        image_file: Source FITS file to get dimensions from
        mask_file: Output FITS mask filename
        json_file: Output JSON metadata filename

    Returns:
        True if mask created, False if no areas (mask deleted)
    """
    basepath = task.path()

    if not areas:
        # Delete existing mask files if no areas selected
        if os.path.exists(os.path.join(basepath, mask_file)):
            os.unlink(os.path.join(basepath, mask_file))

        if os.path.exists(os.path.join(basepath, json_file)):
            os.unlink(os.path.join(basepath, json_file))

        return False

    if os.path.exists(os.path.join(basepath, image_file)):
        header = fits.getheader(os.path.join(basepath, image_file), -1)
        scale = header['NAXIS1'] / width # FIXME: Should we assume it is the same for y?..

        mask = np.ones(shape=(header['NAXIS2'], header['NAXIS1']), dtype=bool)
        for area in areas:
            x0 = max(0, int(np.floor(scale*area['x'])))
            y0 = max(0, int(np.floor(scale*area['y'])))

            x1 = min(header['NAXIS1'], int(np.ceil(scale*(area['x'] + area['width']))))
            y1 = min(header['NAXIS2'], int(np.ceil(scale*(area['y'] + area['height']))))

            mask[y0:y1, x0:x1] = False

        mask = mask[::-1] # Flip the mask vertically as the origin is at bottom

        if inverted:
            mask = ~mask

        processing.fits_write(os.path.join(basepath, mask_file), mask.astype(np.uint8), compress=True)

        # Store masked areas for future re-use
        file_write(
            os.path.join(basepath, json_file),
            json.dumps(
                {'areas': areas, 'inverted': True if inverted else False},
                indent=4,
                sort_keys=False
            )
        )

    return True


def task_mask(request, id=None, mode='image'):
    """
    Interactive mask creation for images or templates.

    Args:
        id: Task ID
        mode: 'image' for primary image mask, 'template' for template mask
    """
    task = get_object_or_404(models.Task, id=id)
    basepath = task.path()

    # Determine which files to use based on mode
    if mode == 'template':
        image_file = 'custom_template.fits'
        mask_file = 'custom_template_mask.fits'
        json_file = 'custom_template_mask.json'

        # Validate template exists
        if not os.path.exists(os.path.join(basepath, image_file)):
            messages.error(request, "Custom template not found. Upload a template first.")
            return HttpResponseRedirect(reverse('tasks', kwargs={'id': task.id}))
    else:
        image_file = 'image.fits'
        mask_file = 'custom_mask.fits'
        json_file = 'custom_mask.json'

    if request.method == 'POST':
        areas = json.loads(request.POST.get('areas'))

        if handle_task_mask_creation(
                task,
                int(request.POST.get('width')),
                int(request.POST.get('height')),
                areas,
                inverted=bool(request.POST.get('inverted', False)),
                image_file=image_file,
                mask_file=mask_file,
                json_file=json_file,
        ):
            messages.success(request, f"{'Custom template' if mode == 'template' else 'Image'} mask created for task {id}")
        else:
            messages.success(request, f"{'Custom template' if mode == 'template' else 'Image'} mask cleared for task {id}")

        # Only run inspection for image mask (not for template mask)
        if mode == 'image':
            task.celery_id = celery_tasks.task_inspect.delay(task.id).id
            task.state = 'inspect'
            task.save()
            messages.success(request, "Started image inspection for task " + str(id))

        return HttpResponseRedirect(reverse('tasks', kwargs={'id': task.id}))

    # GET: Load existing mask if present
    context = {
        'task': task,
        'mode': mode,
        'image_file': image_file,
    }

    areasname = os.path.join(basepath, json_file)
    if os.path.exists(areasname):
        data = json.loads(file_read(areasname))
        context['areas'] = data.get('areas')
        context['inverted'] = data.get('inverted', False)

    return TemplateResponse(request, 'task_mask.html', context=context)


def task_preprocess(request, id=None):
    """
    Interactive preprocessing for images (destripe, crop, background removal with backup/restore).

    Args:
        id: Task ID
    """
    task = get_object_or_404(models.Task, id=id)
    basepath = task.path()

    image_path = os.path.join(basepath, 'image.fits')
    orig_path = os.path.join(basepath, 'image.orig.fits')

    # Validate image exists
    if not os.path.exists(image_path):
        messages.error(request, "Image file not found for task " + str(id))
        return HttpResponseRedirect(reverse('tasks', kwargs={'id': task.id}))

    # Helper for backing up image before processing
    def _backup_image():
        if not os.path.exists(orig_path):
            shutil.copy2(image_path, orig_path)

    # Helper for actions after actual processing
    def _post_action():
        # Update the timestamp so that the image preview is instantly updated
        task.complete()
        task.save()

        # Trigger cleanup
        celery_tasks.run_task_steps(task, ['cleanup'])

    # Actual processing
    if request.method == 'POST':
        action = request.POST.get('action')

        if action in ['destripe_vertical', 'destripe_horizontal']:
            # Create backup before first modification
            _backup_image()

            # Apply preprocessing
            processing.preprocess_image(
                image_path, task.config,
                destripe_vertical=(action == 'destripe_vertical'),
                destripe_horizontal=(action == 'destripe_horizontal'),
            )

            messages.success(request, "Removed lines from the image for task " + str(id))

            # Run necessary post-activity actions
            _post_action()

        elif action == 'crop_image':
            x1 = request.POST.get('crop_x1')
            y1 = request.POST.get('crop_y1')
            x2 = request.POST.get('crop_x2')
            y2 = request.POST.get('crop_y2')

            # Validate that at least some coordinates are provided
            if not any([x1, y1, x2, y2]):
                messages.error(request, "Please provide at least one crop coordinate")
                return HttpResponseRedirect(request.path_info)

            # Create backup before first modification
            _backup_image()

            # Apply cropping
            processing.crop_image(
                image_path, task.config,
                x1=x1, y1=y1, x2=x2, y2=y2
            )

            messages.success(request, "Cropped the image for task " + str(id))

            # Run necessary post-activity actions
            _post_action()

        elif action == 'remove_background':
            bg_size = request.POST.get('bg_size')
            bg_method = request.POST.get('bg_method', 'sep')
            bg_divide = request.POST.get('bg_divide') == 'on'

            # Validate background size
            if not bg_size:
                messages.error(request, "Please provide background size")
                return HttpResponseRedirect(request.path_info)

            try:
                bg_size = int(bg_size)
                if bg_size <= 0:
                    raise ValueError("Background size must be positive")
            except ValueError as e:
                messages.error(request, f"Invalid background size: {e}")
                return HttpResponseRedirect(request.path_info)

            # Create backup before first modification
            _backup_image()

            # Apply background removal
            processing.remove_background_image(
                image_path, task.config,
                bg_size=bg_size,
                bg_method=bg_method,
                divide=bg_divide
            )

            operation = "divided by" if bg_divide else "subtracted"
            messages.success(request, f"Background {operation} for task {id} using method '{bg_method}'")

            # Run necessary post-activity actions
            _post_action()

        elif action == 'reset':
            # Restore from backup
            if not os.path.exists(orig_path):
                messages.error(request, "No backup image available to restore for task " + str(id))
                return HttpResponseRedirect(request.path_info)

            shutil.copy2(orig_path, image_path)
            messages.success(request, "Reset image to original for task " + str(id))

            # Remove the backup to preserve space
            # FIXME: is it easier to just move the file above?..
            os.remove(orig_path)

            # Run necessary post-activity actions
            _post_action()

        # Redirect back to preprocessing page for multiple operations
        return HttpResponseRedirect(request.path_info)

    # GET: Display preprocessing interface
    context = {
        'task': task,
        'has_backup': os.path.exists(orig_path),
    }

    return TemplateResponse(request, 'task_preprocess.html', context=context)


def task_candidates(request, id, filename='candidates.vot'):
    context = {}

    task = get_object_or_404(models.Task, id=id)
    path = task.path()

    context['task'] = task

    context['step'] = request.GET.get('step', 100)

    if os.path.exists(os.path.join(path, filename)):
        candidates = Table.read(os.path.join(path, filename))
        if candidates:
            candidates.sort('flux', reverse=True)
            context['candidates'] = candidates
            context['filename'] = filename

    return TemplateResponse(request, 'task_candidates.html', context=context)


def task_state(request, id):
    task = get_object_or_404(models.Task, id=id)

    return JsonResponse({'state': task.state, 'id': task.id, 'celery_id': task.celery_id})


@require_POST
def task_update_title(request, id):
    """AJAX endpoint to update task title."""
    task = get_object_or_404(models.Task, id=id)

    # Permission check
    if not request.user.is_authenticated or not (
        request.user.is_staff or request.user == task.user or request.user.has_perm('stdweb.edit_all_tasks')
    ):
        return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)

    # Don't allow updates while task is running
    if task.celery_id is not None:
        return JsonResponse({'success': False, 'error': 'Task is running'}, status=400)

    title = request.POST.get('title', '').strip()[:250]  # Max 250 chars
    task.title = title
    task.save(update_fields=['title', 'modified'])

    return JsonResponse({'success': True, 'title': title})
