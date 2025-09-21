from django.http import HttpResponse, FileResponse, HttpResponseRedirect, JsonResponse
from django.template.response import TemplateResponse
from django.urls import reverse
from django.db.models import Q
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import redirect_to_login
from django.views.decorators.cache import cache_page
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


def tasks(request, id=None):
    context = {}

    if id:
        task = get_object_or_404(models.Task, id=id)
        path = task.path()

        # Permissions
        if request.user.is_authenticated and (request.user.is_staff or request.user == task.user):
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

                # Handle actions
                action = request.POST.get('action')

                if action == 'delete_task':
                    if request.user.is_staff or request.user == task.user:
                        task.delete()
                        messages.success(request, "Task " + str(id ) + " is deleted")
                        return HttpResponseRedirect(reverse('tasks'))
                    else:
                        messages.error(request, "Cannot delete task " + str(id) + " belonging to " + task.user.username)
                        return HttpResponseRedirect(request.path_info)

                if action == 'duplicate_task':
                    old_path = task.path()

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

                    for name in ['image.fits', 'image.wcs', 'custom_mask.fits', 'custom_mask.json', 'custom_template.fits']:
                        if os.path.exists(os.path.join(old_path, name)):
                            shutil.copyfile(os.path.join(old_path, name), os.path.join(task.path(), name))

                    messages.success(request, "Task duplicated as " + str(task.id))

                    return HttpResponseRedirect(reverse('tasks', kwargs={'id': task.id}))

                if action == 'fix_image':
                    # TODO: move to async celery task?..
                    processing.fix_image(os.path.join(task.path(), 'image.fits'), task.config)
                    messages.success(request, "Fixed the image header for task " + str(id))

                if action == 'crop_image':
                    # TODO: move to async celery task?..
                    processing.crop_image(
                        os.path.join(task.path(), 'image.fits'), task.config,
                        x1=request.POST.get('crop_x1'),
                        y1=request.POST.get('crop_y1'),
                        x2=request.POST.get('crop_x2'),
                        y2=request.POST.get('crop_y2')
                    )

                    messages.success(request, "Cropped the image for task " + str(id))
                    # Now we have to cleanup, which will be handled below

                if action == 'destripe_vertical' or action == 'destripe_horizontal':
                    processing.preprocess_image(
                        os.path.join(task.path(), 'image.fits'), task.config,
                        destripe_vertical=True if action == 'destripe_vertical' else False,
                        destripe_horizontal=True if action == 'destripe_horizontal' else False,
                    )

                    messages.success(request, "Removed the lines from the image for task " + str(id))
                    # Now we have to cleanup, which will be handled below

                if action == 'update_config':
                    if 'raw_config' in form.changed_data:
                        task.config = form.cleaned_data.get('raw_config')
                        task.save()
                        messages.success(request, f"Config for task {str(id)} updated")
                    else:
                        messages.success(request, f"Config for task {str(id)} unchanged")

                if action == 'make_custom_mask':
                    return HttpResponseRedirect(reverse('task_mask', kwargs={'id': task.id}))

                if action in [
                        'cleanup_task', 'archive_task',
                        'crop_image',
                        'destripe_horizontal', 'destripe_vertical'
                ]:
                    if action in ['cleanup_task']:
                        task.config = {} # We reset the config on cleanup but keep on archiving
                    celery_tasks.run_task_steps(task, ['cleanup'])
                    messages.success(request, "Started cleanup for task " + str(id))

                if action == 'inspect_image':
                    celery_tasks.run_task_steps(task, [
                        'inspect',
                        'photometry' if form.cleaned_data.get('run_photometry') else None,
                        'subtraction' if form.cleaned_data.get('run_subtraction') else None,
                    ])
                    messages.success(request, "Started image inspection for task " + str(id))
                    if form.cleaned_data.get('run_photometry'):
                        messages.success(request, "Started image photometry for task " + str(id))
                    if form.cleaned_data.get('run_subtraction'):
                        messages.success(request, "Started template subtraction for task " + str(id))

                if action == 'photometry_image':
                    celery_tasks.run_task_steps(task, [
                        'photometry',
                        'subtraction' if form.cleaned_data.get('run_subtraction') else None,
                    ])
                    messages.success(request, "Started image photometry for task " + str(id))
                    if form.cleaned_data.get('run_subtraction'):
                        messages.success(request, "Started template subtraction for task " + str(id))

                if action == 'transients_simple_image':
                    celery_tasks.run_task_steps(task, ['simple_transients'])
                    messages.success(request, "Started simple transient detection for task " + str(id))

                if action == 'subtract_image':
                    celery_tasks.run_task_steps(task, ['subtraction'])
                    messages.success(request, "Started template subtraction for task " + str(id))

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
        tasks = models.Task.objects.all()
        tasks = tasks.order_by('-created')

        form = forms.TasksFilterForm(
            request.GET,
            show_all=request.user.is_staff or request.user.has_perm('stdweb.view_all_tasks'),
        )
        context['form'] = form

        if request.method == 'GET':
            if form.is_valid():
                show_all = form.cleaned_data.get('show_all')
                if not show_all:
                    if request.user.is_authenticated:
                        tasks = tasks.filter(user = request.user)

                query = form.cleaned_data.get('query')
                if query:
                    # Coordinate query?..
                    ra0,dec0,sr0 = utils.resolve_coordinates(query)

                    if ra0 is not None and dec0 is not None and sr0 is not None:
                        messages.info(request, f"Looking for tasks inside {sr0:.2f} deg around {ra0:.4f} {dec0:+.4f}")
                        pks = []
                        for task in tasks:
                            ra,dec,sr = [task.config.get(_) for _ in ['field_ra', 'field_dec', 'field_sr']]
                            if ra is not None and dec is not None and sr is not None:
                                # TODO: for now, ignore image size, only consider its center
                                if astrometry.spherical_distance(ra, dec, ra0, dec0) < sr0:
                                    pks.append(task.pk)

                        tasks = tasks.filter(pk__in = pks)

                    else:
                        # Otherwise plain text search
                        for token in query.split():
                            tasks = tasks.filter(
                                Q(original_name__icontains = token) |
                                Q(title__icontains = token) |
                                Q(user__username__icontains = token) |
                                Q(user__first_name__icontains = token) |
                                Q(user__last_name__icontains = token)
                            )

        context['tasks'] = tasks
        context['referer'] = request.path + '?' + request.GET.urlencode();

    return TemplateResponse(request, 'tasks.html', context=context)


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
                    messages.success(request, "Started archiving for task " + str(id))

                if action == 'cleanup':
                    task.celery_id = celery_tasks.task_cleanup.delay(task.id).id
                    task.config = {} # should we reset the config on cleanup?..
                    task.state = 'cleanup'
                    task.save()
                    messages.success(request, "Started cleanup for task " + str(id))

                if action == 'delete':
                    if request.user.is_staff or request.user == task.user:
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


def handle_task_mask_creation(task, width, height, areas, inverted=False):
    if not areas:
        if os.path.exists(os.path.join(task.path(), 'custom_mask.fits')):
            os.unlink(os.path.join(task.path(), 'custom_mask.fits'))

        if os.path.exists(os.path.join(task.path(), 'custom_mask.json')):
            os.unlink(os.path.join(task.path(), 'custom_mask.json'))

        return False

    if os.path.exists(os.path.join(task.path(), 'image.fits')):
        header = fits.getheader(os.path.join(task.path(), 'image.fits'), -1)
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

        processing.fits_write(os.path.join(task.path(), 'custom_mask.fits'), mask.astype(np.uint8), compress=True)

        # Store masked areas for future re-use
        file_write(
            os.path.join(task.path(), 'custom_mask.json'),
            json.dumps(
                {'areas': areas, 'inverted': True if inverted else False},
                indent=4,
                sort_keys=False
            )
        )

    return True


def task_mask(request, id=None, path=''):
    task = get_object_or_404(models.Task, id=id)

    if request.method == 'POST':
        areas = json.loads(request.POST.get('areas'))

        if handle_task_mask_creation(
                task,
                int(request.POST.get('width')),
                int(request.POST.get('height')),
                areas,
                inverted=bool(request.POST.get('inverted', False)),
        ):
            messages.success(request, "Custom mask created for task " + str(id))
        else:
            messages.success(request, "Custom mask cleared for task " + str(id))

        # Now we need to run image inspection
        task.celery_id = celery_tasks.task_inspect.delay(task.id).id
        task.state = 'inspect'
        task.save()
        messages.success(request, "Started image inspection for task " + str(id))

        return HttpResponseRedirect(reverse('tasks', kwargs={'id': task.id}))

    context = {}

    context['task'] = task

    areasname = os.path.join(task.path(), 'custom_mask.json')
    if os.path.exists(areasname):
        data = json.loads(file_read(areasname))
        context['areas'] = data.get('areas')
        context['inverted'] = data.get('inverted', False)

    return TemplateResponse(request, 'task_mask.html', context=context)


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
