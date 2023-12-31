from django.http import HttpResponse, JsonResponse, FileResponse, HttpResponseRedirect
from django.template.response import TemplateResponse

from django.contrib.auth.decorators import login_required

from django.contrib import messages

from django.core import management

from django.conf import settings

import os, glob

from . import models
from . import celery

@login_required
def view_queue(request, id=None):
    context = {}

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'terminatealltasks':
            if request.user.is_staff:
                management.call_command('terminatealltasks')
                messages.success(request, "All queued tasks are terminated")

            return HttpResponseRedirect(request.path_info)

        if action == 'cleanuplinkedtasks':
            if request.user.is_staff:
                for task in models.Task.objects.filter(celery_id__isnull=False):
                    task.celery_id = None
                    task.state = 'failed'
                    task.save()

            return HttpResponseRedirect(request.path_info)

        if action == 'terminatetask' and id:
            if request.user.is_staff or True:
                celery.app.control.revoke(id, terminate=True, signal='SIGKILL')
                messages.success(request, "Queued task " + id + " is terminated")

            return HttpResponseRedirect(request.path_info)

        if action == 'cleanuplinkedtask' and id:
            if request.user.is_staff or True:
                for task in models.Task.objects.filter(celery_id=id):
                    task.celery_id = None
                    task.state = 'failed'
                    task.save()

            return HttpResponseRedirect(request.path_info)

    if id:
        ctask = celery.app.AsyncResult(id)

        context['ctask'] = ctask

    else:
        queue = []

        inspect = celery.app.control.inspect()
        for res,state in [(inspect.active(), 'active'), (inspect.reserved(), 'pending'), (inspect.scheduled(), 'scheduled')]:
            if res:
                for wtasks in res.values():
                    for ctask in wtasks:
                        if 'name' in ctask:
                            ctask['shortname'] = ctask['name'].split('.')[-1]

                        ctask['state'] = state

                        queue.append(ctask)

        context['queue'] = queue

    return TemplateResponse(request, 'queue.html', context=context)


def get_queue(request, id):
    ctask = celery.app.AsyncResult(id)

    return JsonResponse({'state': ctask.state, 'id': ctask.id})
