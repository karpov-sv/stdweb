from django.http import HttpResponse, FileResponse, HttpResponseRedirect, JsonResponse
from django.template.response import TemplateResponse
from django.urls import reverse
from django.db.models import Q
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import redirect_to_login
from django.views.decorators.cache import cache_page
from django.conf import settings

import os, glob, shutil
import json
import numpy as np

from astropy.table import Table
from astropy.io import fits

from . import views
from . import models
from . import forms
from . import celery_tasks
from . import celery
from . import processing

def tasks(request, id=None):
    context = {}

    if id:
        task = models.Task.objects.get(id=id)
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
                        ]
                        if name not in ignored_fields:
                            if name in form.changed_data or name not in task.config:
                                # update only changed or new fields
                                task.config[name] = value

                    task.save()

                # Handle actions
                action = request.POST.get('action')

                print(action)

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
                    task.save() # To populate new task.id

                    try:
                        os.makedirs(task.path())
                    except OSError:
                        pass

                    for name in ['image.fits', 'image.wcs', 'custom_mask.fits', 'custom_template.fits']:
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
                    processing.crop_image(os.path.join(task.path(), 'image.fits'), task.config,
                                          x1=request.POST.get('crop_x1'),
                                          y1=request.POST.get('crop_y1'),
                                          x2=request.POST.get('crop_x2'),
                                          y2=request.POST.get('crop_y2'))

                    messages.success(request, "Cropped the image for task " + str(id))
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

                if action == 'cleanup_task' or action == 'crop_image':
                    task.celery_id = celery_tasks.task_cleanup.delay(task.id).id
                    task.config = {} # should we reset the config on cleanup?..
                    task.state = 'cleanup'
                    task.save()
                    messages.success(request, "Started cleanup for task " + str(id))

                if action == 'inspect_image':
                    task.celery_id = celery_tasks.task_inspect.delay(task.id).id
                    task.state = 'inspect'
                    task.save()
                    messages.success(request, "Started image inspection for task " + str(id))

                if action == 'photometry_image':
                    task.celery_id = celery_tasks.task_photometry.delay(task.id).id
                    task.state = 'photometry'
                    task.save()
                    messages.success(request, "Started image photometry for task " + str(id))

                if action == 'transients_simple_image':
                    task.celery_id = celery_tasks.task_transients_simple.delay(task.id).id
                    task.state = 'transients_simple'
                    task.save()
                    messages.success(request, "Started simple transient detection for task " + str(id))

                if action == 'subtract_image':
                    task.celery_id = celery_tasks.task_subtraction.delay(task.id).id
                    task.state = 'subtraction'
                    task.save()
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

        form = forms.TasksFilterForm(request.GET)
        context['form'] = form

        if request.method == 'GET':
            if form.is_valid():
                show_all = form.cleaned_data.get('show_all')
                if not show_all:
                    if request.user.is_authenticated:
                        tasks = tasks.filter(user = request.user)

                query = form.cleaned_data.get('query')
                if query:
                    for token in query.split():
                        tasks = tasks.filter(Q(original_name__icontains = token) |
                                             Q(title__icontains = token) |
                                             Q(user__username__icontains = token) |
                                             Q(user__first_name__icontains = token) |
                                             Q(user__last_name__icontains = token)
                                             )


        context['tasks'] = tasks

    return TemplateResponse(request, 'tasks.html', context=context)


def task_download(request, id=None, path='', **kwargs):
    task = models.Task.objects.get(id=id)

    return views.download(request, path, base=task.path(), **kwargs)


@cache_page(15 * 60)
def task_preview(request, id=None, path='', **kwargs):
    task = models.Task.objects.get(id=id)

    return views.preview(request, path, base=task.path(), **kwargs)


@cache_page(15 * 60)
def task_cutout(request, id=None, path='', **kwargs):
    task = models.Task.objects.get(id=id)

    return views.cutout(request, path, base=task.path(), **kwargs)


def handle_task_mask_creation(task, width, height, areas):
    if not areas:
        if os.path.exists(os.path.join(task.path(), 'custom_mask.fits')):
            os.unlink(os.path.join(task.path(), 'custom_mask.fits'))

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

        processing.fits_write(os.path.join(task.path(), 'custom_mask.fits'), mask.astype(np.int8), compress=True)

    return True


def task_mask(request, id=None, path=''):
    task = models.Task.objects.get(id=id)

    if request.method == 'POST':
        areas = json.loads(request.POST.get('areas'))

        if handle_task_mask_creation(task,
                                     int(request.POST.get('width')),
                                     int(request.POST.get('height')),
                                     areas):

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

    return TemplateResponse(request, 'task_mask.html', context=context)


def task_candidates(request, id, filename='candidates.vot'):
    context = {}

    task = models.Task.objects.get(id=id)
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
    task = models.Task.objects.get(id=id)

    return JsonResponse({'state': task.state, 'id': task.id, 'celery_id': task.celery_id})
