from django.http import HttpResponse, FileResponse, HttpResponseRedirect
from django.template.response import TemplateResponse
from django.urls import reverse

from django.contrib import messages

from django.contrib.auth.decorators import login_required

from django.views.decorators.cache import cache_page

import os, glob

from . import settings
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

        all_forms = {}

        all_forms['inspect'] = forms.TaskInspectForm(request.POST or None, initial = task.config)
        all_forms['photometry'] = forms.TaskPhotometryForm(request.POST or None, initial = task.config)
        all_forms['subtraction'] = forms.TaskSubtractionForm(request.POST or None, initial = task.config)

        for name,form in all_forms.items():
            context['form_'+name] = form

        # Form actions
        if request.method == 'POST':
            # Handle forms
            form_type = request.POST.get('form_type')
            print(form_type)
            form = all_forms.get(form_type)
            if form and not form.is_valid():
                print("imvalid")
            if form and form.is_valid():
                print(request.FILES)

                # Handle uploaded files, if any
                if 'custom_template' in request.FILES:
                    views.handle_uploaded_file(request.FILES['custom_template'],
                                               os.path.join(task.path(), 'custom_template.fits'))
                    messages.info(request, "Custom template uploaded as custom_template.fits")

                if form.has_changed():
                    for name,value in form.cleaned_data.items():
                        # we do not want these to go to task.config
                        ignored_fields = [
                            'form_type',
                            'crop_x1', 'crop_y1', 'crop_x2', 'crop_y2',
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

                if action == 'cleanup_task' or action == 'crop_image':
                    task.celery_id = celery_tasks.task_cleanup.delay(task.id).id
                    task.config = {} # should we reset the config on cleanup?..
                    task.state = 'cleaning'
                    task.save()
                    messages.success(request, "Started cleanup for task " + str(id))

                if action == 'inspect_image':
                    task.celery_id = celery_tasks.task_inspect.delay(task.id).id
                    task.state = 'inspecting'
                    task.save()
                    messages.success(request, "Started image inspection for task " + str(id))

                if action == 'photometry_image':
                    task.celery_id = celery_tasks.task_photometry.delay(task.id).id
                    task.state = 'photometry'
                    task.save()
                    messages.success(request, "Started image photometry for task " + str(id))

                if action == 'subtract_image':
                    task.celery_id = celery_tasks.task_subtraction.delay(task.id).id
                    task.state = 'subtraction'
                    task.save()
                    messages.success(request, "Started template subtraction for task " + str(id))

                return HttpResponseRedirect(request.path_info)

        # Display task
        context['task'] = task

        context['files'] = [os.path.split(_)[1] for _ in glob.glob(os.path.join(path, '*'))]

        return TemplateResponse(request, 'task.html', context=context)
    else:
        # List all tasks
        tasks = models.Task.objects.all()

        tasks = tasks.order_by('-modified')

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
