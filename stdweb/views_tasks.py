from django.http import HttpResponse, FileResponse, HttpResponseRedirect
from django.template.response import TemplateResponse
from django.urls import reverse

from django.contrib import messages

from django.contrib.auth.decorators import login_required

import os, glob

from . import settings
from . import views
from . import models
from . import celery_tasks
from . import celery

def tasks(request, id=None):
    context = {}

    if id:
        task = models.Task.objects.get(id=id)
        path = task.path()

        # Clear the link to queued task if it was revoked
        if task.celery_id:
            ctask = celery.app.AsyncResult(task.celery_id)
            if ctask.state == 'REVOKED' or ctask.state == 'FAILURE':
                task.celery_id = None
                task.state = 'failed' # Should we do it?
                task.save()

        # Form actions
        if request.method == 'POST':
            action = request.POST.get('action')

            if action == 'delete_task':
                if request.user.is_staff or request.user == task.user:
                    task.delete()
                    messages.success(request, "Task " + str(id ) + " is deleted")
                    return HttpResponseRedirect(reverse('tasks'))
                else:
                    messages.error(request, "Cannot delete task " + str(id) + " belonging to " + task.user.username)
                    return HttpResponseRedirect(request.path_info)

            if action == 'cleanup_task':
                task.celery_id = celery_tasks.task_cleanup.delay(task.id).id
                task.state = 'cleaning'
                task.save()
                messages.success(request, "Started cleanup for task " + str(id))

            if action == 'inspect_image':
                task.celery_id = celery_tasks.task_inspect.delay(task.id).id
                task.state = 'inspecting'
                task.save()
                messages.success(request, "Started image inspecion for task " + str(id))

            if action == 'photometry_image':
                task.celery_id = celery_tasks.task_photometry.delay(task.id).id
                task.state = 'processing'
                task.save()
                messages.success(request, "Started image photometry for task " + str(id))

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


def task_preview(request, id=None, path='', **kwargs):
    task = models.Task.objects.get(id=id)

    return views.preview(request, path, base=task.path(), **kwargs)
