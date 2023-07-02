from django.http import HttpResponse, FileResponse, HttpResponseRedirect
from django.template.response import TemplateResponse

from django.contrib.auth.decorators import login_required

import os

from . import settings
from . import views
from . import models

def tasks(request, id=None):
    context = {}

    if id:
        task = models.Task.objects.get(id=id)

        context['task'] = task

        return TemplateResponse(request, 'task.html', context=context)
    else:
        tasks = models.Task.objects.all()

        tasks = tasks.order_by('-modified')

        context['tasks'] = tasks

    return TemplateResponse(request, 'tasks.html', context=context)

def tasks_files(request, id=None, path=''):
    task = models.Task.objects.get(id=id)

    return views.list_files(request, path, base=os.path.join(settings.TASKS_PATH, str(task.id)))
