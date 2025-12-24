from django.http import HttpResponse, JsonResponse, FileResponse, HttpResponseRedirect
from django.template.response import TemplateResponse

from django.contrib.auth.decorators import login_required

from django.contrib import messages

from django.core import management

from django.conf import settings

import os, glob

from . import models
from . import celery
from .celery_tasks import kill_task_processes


def find_task_by_chain_id(chain_id):
    """Find a Task that has the given chain_id in its celery_chain_ids list.
    SQLite doesn't support JSONField __contains lookup, so we filter in Python."""
    for task in models.Task.objects.exclude(celery_chain_ids=[]):
        if chain_id in task.celery_chain_ids:
            return task
    return None


def revoke_task_chain(task):
    """
    Revoke all tasks in a chain and kill associated processes.
    Also clears the task's celery_id to signal cancellation.
    """
    ids_to_revoke = []

    # Add the main celery_id
    if task.celery_id:
        ids_to_revoke.append(task.celery_id)

    # Add all chain task IDs
    if task.celery_chain_ids:
        ids_to_revoke.extend(task.celery_chain_ids)

    # Revoke all tasks - use SIGTERM first to allow cleanup
    for task_id in ids_to_revoke:
        celery.app.control.revoke(task_id, terminate=True, signal='SIGTERM')

    # Kill any external processes spawned by the task
    kill_task_processes(task)

    # Clear task state
    task.celery_id = None
    task.celery_chain_ids = []
    task.celery_pid = None
    task.state = 'cancelled'
    task.save()

    return len(ids_to_revoke)

@login_required
def view_queue(request, id=None):
    context = {}

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'terminatealltasks':
            if request.user.is_staff:
                # Terminate all tasks with proper chain revocation
                count = 0
                for task in models.Task.objects.filter(celery_id__isnull=False):
                    count += revoke_task_chain(task)
                messages.success(request, f"Terminated {count} queued tasks")

            return HttpResponseRedirect(request.path_info)

        if action == 'cleanuplinkedtasks':
            if request.user.is_staff:
                for task in models.Task.objects.filter(celery_id__isnull=False):
                    task.celery_id = None
                    task.celery_chain_ids = []
                    task.celery_pid = None
                    task.state = 'failed'
                    task.save()

            return HttpResponseRedirect(request.path_info)

        if action == 'terminatetask' and id:
            if request.user.is_staff or True:
                # Find Django task and revoke entire chain
                task = models.Task.objects.filter(celery_id=id).first()
                if not task:
                    # Try to find by chain ID
                    task = find_task_by_chain_id(id)

                if task:
                    count = revoke_task_chain(task)
                    messages.success(request, f"Terminated task chain ({count} subtasks)")
                else:
                    # Fallback: revoke just this ID
                    celery.app.control.revoke(id, terminate=True, signal='SIGTERM')
                    messages.success(request, f"Terminated task {id}")

            return HttpResponseRedirect(request.path_info)

        if action == 'cleanuplinkedtask' and id:
            if request.user.is_staff or True:
                for task in models.Task.objects.filter(celery_id=id):
                    task.celery_id = None
                    task.celery_chain_ids = []
                    task.celery_pid = None
                    task.state = 'failed'
                    task.save()

            return HttpResponseRedirect(request.path_info)

    if id:
        ctask = celery.app.AsyncResult(id)
        context['ctask'] = ctask

        # Find linked Django task
        task = models.Task.objects.filter(celery_id=id).first()
        if not task:
            task = find_task_by_chain_id(id)
        context['task'] = task

        # Show chain position if part of a chain
        if task and task.celery_chain_ids and id in task.celery_chain_ids:
            context['chain_position'] = task.celery_chain_ids.index(id) + 1
            context['chain_total'] = len(task.celery_chain_ids)

    else:
        queue = []

        inspect = celery.app.control.inspect(timeout=0.1)
        for res,state in [(inspect.active(), 'active'), (inspect.reserved(), 'pending'), (inspect.scheduled(), 'scheduled')]:
            if res:
                for wtasks in res.values():
                    for ctask in wtasks:
                        if 'name' in ctask:
                            ctask['shortname'] = ctask['name'].split('.')[-1]

                        ctask['state'] = state

                        # Find linked Django task and add chain info
                        task = models.Task.objects.filter(celery_id=ctask['id']).first()
                        if not task:
                            task = find_task_by_chain_id(ctask['id'])

                        if task:
                            ctask['task_id'] = task.id
                            ctask['task_name'] = task.original_name
                            if task.celery_chain_ids and ctask['id'] in task.celery_chain_ids:
                                ctask['chain_position'] = task.celery_chain_ids.index(ctask['id']) + 1
                                ctask['chain_total'] = len(task.celery_chain_ids)

                        queue.append(ctask)

        context['queue'] = queue

    return TemplateResponse(request, 'queue.html', context=context)


def get_queue(request, id):
    ctask = celery.app.AsyncResult(id)

    return JsonResponse({'state': ctask.state, 'id': ctask.id})
