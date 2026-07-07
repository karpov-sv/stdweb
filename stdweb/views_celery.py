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


def is_ajax(request):
    return request.headers.get('x-requested-with') == 'XMLHttpRequest'


def build_queue(user=None):
    """Collect active/pending/scheduled Celery tasks, annotated with linked Django tasks."""
    queue = []

    inspect = celery.app.control.inspect(timeout=0.5)
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

                    if user is not None:
                        ctask['can_manage'] = task.can_edit(user) if task else user.is_staff

                    queue.append(ctask)

    # Stable ordering so that entries do not jump around between refreshes
    order = {'active': 0, 'pending': 1, 'scheduled': 2}
    queue.sort(key=lambda _: (order.get(_['state'], 3), _.get('time_start') or 0, _['id']))

    return queue


@login_required
def view_queue(request, id=None):
    context = {}

    if request.method == 'POST':
        action = request.POST.get('action')
        ok = False
        message = "Unknown action"

        if action == 'terminatealltasks':
            if request.user.is_staff:
                ntasks = 0
                for task in models.Task.objects.filter(celery_id__isnull=False):
                    revoke_task_chain(task)
                    ntasks += 1
                ok = True
                message = f"Terminated {ntasks} running tasks"
            else:
                message = "Only staff may terminate all tasks"

        elif action == 'cleanuplinkedtasks':
            if request.user.is_staff:
                ntasks = 0
                for task in models.Task.objects.filter(celery_id__isnull=False):
                    task.celery_id = None
                    task.celery_chain_ids = []
                    task.celery_pid = None
                    task.state = 'failed'
                    task.save()
                    ntasks += 1
                ok = True
                message = f"Cleaned up {ntasks} linked tasks"
            else:
                message = "Only staff may cleanup all tasks"

        elif action == 'terminatetask' and id:
            # Find Django task and revoke entire chain
            task = models.Task.objects.filter(celery_id=id).first()
            if not task:
                # Try to find by chain ID
                task = find_task_by_chain_id(id)

            if task:
                if task.can_edit(request.user):
                    count = revoke_task_chain(task)
                    ok = True
                    message = f"Terminated task chain ({count} subtasks)"
                else:
                    message = f"Cannot terminate task {task.id} belonging to {task.user.username}"
            elif request.user.is_staff:
                # Fallback: revoke just this ID
                celery.app.control.revoke(id, terminate=True, signal='SIGTERM')
                ok = True
                message = f"Terminated task {id}"
            else:
                message = "Task not found"

        elif action == 'cleanuplinkedtask' and id:
            ntasks = 0
            denied = None
            for task in models.Task.objects.filter(celery_id=id):
                if task.can_edit(request.user):
                    task.celery_id = None
                    task.celery_chain_ids = []
                    task.celery_pid = None
                    task.state = 'failed'
                    task.save()
                    ntasks += 1
                else:
                    denied = task

            if denied and not ntasks:
                message = f"Cannot cleanup task {denied.id} belonging to {denied.user.username}"
            else:
                ok = True
                message = f"Cleaned up {ntasks} linked tasks"

        if is_ajax(request):
            return JsonResponse({'ok': ok, 'message': message}, status=200 if ok else 403)

        if ok:
            messages.success(request, message)
        else:
            messages.error(request, message)

        return HttpResponseRedirect(request.path_info)

    if id:
        ctask = celery.app.AsyncResult(id)
        context['ctask'] = ctask

        # Find linked Django task
        task = models.Task.objects.filter(celery_id=id).first()
        if not task:
            task = find_task_by_chain_id(id)
        context['task'] = task

        context['can_manage'] = task.can_edit(request.user) if task else request.user.is_staff

        # Show chain position if part of a chain
        if task and task.celery_chain_ids and id in task.celery_chain_ids:
            context['chain_position'] = task.celery_chain_ids.index(id) + 1
            context['chain_total'] = len(task.celery_chain_ids)

    else:
        context['queue'] = build_queue(request.user)

    return TemplateResponse(request, 'queue.html', context=context)


@login_required
def queue_list(request):
    """HTML fragment with current queue contents, for AJAX refreshing."""
    return TemplateResponse(request, 'queue_list.html', context={'queue': build_queue(request.user)})


@login_required
def get_queue(request, id):
    ctask = celery.app.AsyncResult(id)

    result = {'id': ctask.id, 'state': ctask.state, 'ready': ctask.ready()}

    # Find linked Django task
    task = models.Task.objects.filter(celery_id=id).first()
    if not task:
        task = find_task_by_chain_id(id)

    if task:
        result['task_id'] = task.id
        result['task_state'] = task.state
        result['task_running'] = task.celery_id is not None
        if task.celery_chain_ids and id in task.celery_chain_ids:
            result['chain_position'] = task.celery_chain_ids.index(id) + 1
            result['chain_total'] = len(task.celery_chain_ids)

    return JsonResponse(result)
