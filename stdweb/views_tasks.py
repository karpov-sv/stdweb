from django.http import HttpResponse, FileResponse, HttpResponseRedirect
from django.template.response import TemplateResponse

from django.contrib.auth.decorators import login_required

import os, glob

from . import settings
from . import views
from . import models

def tasks(request, id=None):
    context = {}

    if id:
        task = models.Task.objects.get(id=id)
        path = task.path()

        context['task'] = task

        context['files'] = glob.glob('*', root_dir=path)

        return TemplateResponse(request, 'task.html', context=context)
    else:
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
