# Django + Celery imports
from celery import shared_task, chain

import os, glob, shutil
import signal
import time

from functools import partial

import numpy as np
from astropy.time import Time

from . import models
from . import processing


# Process group management for killing external processes
def kill_task_processes(task):
    """Kill all processes associated with a task via process group."""
    if task.celery_pid:
        try:
            # Kill entire process group
            os.killpg(os.getpgid(task.celery_pid), signal.SIGTERM)
            time.sleep(0.5)
            os.killpg(os.getpgid(task.celery_pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass


class TaskProcessContext:
    """
    Context manager for task execution with process group management.
    Handles: cancellation check, process group setup, signal handlers, cleanup.
    """
    def __init__(self, celery_task, task_id):
        self.celery_task = celery_task
        self.task_id = task_id
        self.task = None
        self.basepath = None
        self.cancelled = False
        self._old_sigterm = None

    def __enter__(self):
        self.task = models.Task.objects.get(id=self.task_id)

        # Check if task was cancelled before starting
        if not self.task.celery_id:
            self.celery_task.request.chain = None
            self.cancelled = True
            return self

        self.basepath = self.task.path()

        # Store PID in database
        self.task.celery_pid = os.getpid()
        self.task.save(update_fields=['celery_pid'])

        # Try to become process group leader so children can be killed together
        try:
            os.setpgrp()
        except OSError:
            pass  # Already a group leader

        # Set up signal handler for graceful termination
        self._old_sigterm = signal.signal(signal.SIGTERM, self._sigterm_handler)

        return self

    def _sigterm_handler(self, signum, frame):
        """Handle SIGTERM - clean up and kill process group."""
        self._cleanup_pid()
        try:
            os.killpg(os.getpgrp(), signal.SIGKILL)
        except:
            pass
        raise SystemExit(1)

    def _cleanup_pid(self):
        """Clear PID from database."""
        if self.task:
            try:
                self.task.celery_pid = None
                self.task.save(update_fields=['celery_pid'])
            except:
                pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_pid()
        # Restore old signal handler
        if self._old_sigterm is not None:
            signal.signal(signal.SIGTERM, self._old_sigterm)
        return False  # Don't suppress exceptions


def fix_config(config):
    """
    Fix non-serializable Numpy types in config
    """
    for key in config.keys():
        if type(config[key]) == np.float32:
            config[key] = float(config[key])


@shared_task(bind=True)
def task_finalize(self, id):
    task = models.Task.objects.get(id=id)
    task.celery_id = None
    task.celery_chain_ids = []
    task.complete()
    task.save()


@shared_task(bind=True)
def task_set_state(self, id, state):
    task = models.Task.objects.get(id=id)
    task.state = state
    task.save()


@shared_task(bind=True, acks_late=True, reject_on_worker_lost=True)
def task_break_if_failed(self, id):
    task = models.Task.objects.get(id=id)

    if not task.celery_id:
        print("Breaking the chain!!!")
        # Clear chain to prevent further execution
        self.request.chain = None
        raise RuntimeError("Task chain cancelled")

@shared_task(bind=True)
def task_cleanup(self, id, finalize=True):
    task = models.Task.objects.get(id=id)
    basepath = task.path()

    for path in glob.glob(os.path.join(basepath, '*')):
        if os.path.split(path)[-1] != 'image.fits':
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.unlink(path)

    if finalize:
        # End processing
        task.state = 'cleanup_done'
        task.celery_id = None
        task.complete()

    task.save()


@shared_task(bind=True, acks_late=True, reject_on_worker_lost=True)
def task_inspect(self, id, finalize=True):
    with TaskProcessContext(self, id) as ctx:
        if ctx.cancelled:
            return

        task = ctx.task
        basepath = ctx.basepath
        config = task.config

        log = partial(
            processing.print_to_file,
            logname=os.path.join(basepath, 'inspect.log'),
            time0=task.config.get('timing') and Time.now()
        )
        log(clear=True)

        # Start processing
        try:
            processing.inspect_image(os.path.join(basepath, 'image.fits'), config, verbose=log)
            task.state = 'inspect_done'
        except:
            import traceback
            log("\nError!\n", traceback.format_exc())

            task.state = 'inspect_failed'
            task.celery_id = None

        if finalize:
            # End processing
            task.celery_id = None
            task.complete()

        fix_config(config)
        task.save()


@shared_task(bind=True, acks_late=True, reject_on_worker_lost=True)
def task_photometry(self, id, finalize=True):
    with TaskProcessContext(self, id) as ctx:
        if ctx.cancelled:
            return

        task = ctx.task
        basepath = ctx.basepath
        config = task.config

        log = partial(
            processing.print_to_file,
            logname=os.path.join(basepath, 'photometry.log'),
            time0=task.config.get('timing') and Time.now()
        )
        log(clear=True)

        # Start processing
        try:
            processing.photometry_image(os.path.join(basepath, 'image.fits'), config, verbose=log)
            task.state = 'photometry_done'
        except:
            import traceback
            log("\nError!\n", traceback.format_exc())

            task.state = 'photometry_failed'
            task.celery_id = None

        if finalize:
            # End processing
            task.celery_id = None
            task.complete()

        fix_config(config)
        task.save()


@shared_task(bind=True, acks_late=True, reject_on_worker_lost=True)
def task_transients_simple(self, id, finalize=True):
    with TaskProcessContext(self, id) as ctx:
        if ctx.cancelled:
            return

        task = ctx.task
        basepath = ctx.basepath
        config = task.config

        log = partial(
            processing.print_to_file,
            logname=os.path.join(basepath, 'transients_simple.log'),
            time0=task.config.get('timing') and Time.now()
        )
        log(clear=True)

        # Start processing
        try:
            processing.transients_simple_image(os.path.join(basepath, 'image.fits'), config, verbose=log)
            task.state = 'transients_simple_done'
        except:
            import traceback
            log("\nError!\n", traceback.format_exc())

            task.state = 'transients_simple_failed'
            task.celery_id = None

        if finalize:
            # End processing
            task.celery_id = None
            task.complete()

        fix_config(config)
        task.save()


@shared_task(bind=True, acks_late=True, reject_on_worker_lost=True)
def task_subtraction(self, id, finalize=True):
    with TaskProcessContext(self, id) as ctx:
        if ctx.cancelled:
            return

        task = ctx.task
        basepath = ctx.basepath
        config = task.config

        log = partial(
            processing.print_to_file,
            logname=os.path.join(basepath, 'subtraction.log'),
            time0=task.config.get('timing') and Time.now()
        )
        log(clear=True)

        # Start processing
        try:
            processing.subtract_image(os.path.join(basepath, 'image.fits'), config, verbose=log)
            task.state = 'subtraction_done'
        except:
            import traceback
            log("\nError!\n", traceback.format_exc())

            task.state = 'subtraction_failed'
            task.celery_id = None

        if finalize:
            # End processing
            task.celery_id = None
            task.complete()

        fix_config(config)
        task.save()


@shared_task(bind=True, acks_late=True, reject_on_worker_lost=True)
def task_stacking(self, id, finalize=True):
    with TaskProcessContext(self, id) as ctx:
        if ctx.cancelled:
            return

        task = ctx.task
        basepath = ctx.basepath
        config = task.config

        log = partial(
            processing.print_to_file,
            logname=os.path.join(basepath, 'stacking.log'),
            time0=task.config.get('timing') and Time.now()
        )
        log(clear=True)

        # Start processing
        try:
            processing.stack_images(config['stack_filenames'], os.path.join(basepath, 'image.fits'), config, verbose=log)
            task.state = 'stacking_done'
        except:
            import traceback
            log("\nError!\n", traceback.format_exc())

            task.state = 'stacking_failed'
            task.celery_id = None

        if finalize:
            # End processing
            task.celery_id = None
            task.complete()

        fix_config(config)
        task.save()


# Higher-level interface for running (multiple) processing steps for the task
def run_task_steps(task, steps):
    todo = []

    for step in steps:
        print(f"Will run {step} step for task {task.id}")

        if step == 'stack':
            todo.append(task_set_state.subtask(args=[task.id, 'stacking'], immutable=True))
            todo.append(task_stacking.subtask(args=[task.id, False], immutable=True))
            todo.append(task_break_if_failed.subtask(args=[task.id], immutable=True))
            todo.append(task_set_state.subtask(args=[task.id, 'stacking_done'], immutable=True))

        elif step == 'cleanup':
            todo.append(task_set_state.subtask(args=[task.id, 'cleanup'], immutable=True))
            todo.append(task_cleanup.subtask(args=[task.id, False], immutable=True))
            todo.append(task_break_if_failed.subtask(args=[task.id], immutable=True))
            todo.append(task_set_state.subtask(args=[task.id, 'cleanup_done'], immutable=True))

        elif step == 'inspect':
            todo.append(task_set_state.subtask(args=[task.id, 'inspect'], immutable=True))
            todo.append(task_inspect.subtask(args=[task.id, False], immutable=True))
            todo.append(task_break_if_failed.subtask(args=[task.id], immutable=True))
            todo.append(task_set_state.subtask(args=[task.id, 'inspect_done'], immutable=True))

        elif step == 'photometry':
            todo.append(task_set_state.subtask(args=[task.id, 'photometry'], immutable=True))
            todo.append(task_photometry.subtask(args=[task.id, False], immutable=True))
            todo.append(task_break_if_failed.subtask(args=[task.id], immutable=True))
            todo.append(task_set_state.subtask(args=[task.id, 'photometry_done'], immutable=True))

        elif step == 'simple_transients':
            todo.append(task_set_state.subtask(args=[task.id, 'transients_simple'], immutable=True))
            todo.append(task_transients_simple.subtask(args=[task.id, False], immutable=True))
            todo.append(task_break_if_failed.subtask(args=[task.id], immutable=True))
            todo.append(task_set_state.subtask(args=[task.id, 'transients_simple_done'], immutable=True))

        elif step == 'subtraction':
            todo.append(task_set_state.subtask(args=[task.id, 'subtraction'], immutable=True))
            todo.append(task_subtraction.subtask(args=[task.id, False], immutable=True))
            todo.append(task_break_if_failed.subtask(args=[task.id], immutable=True))
            todo.append(task_set_state.subtask(args=[task.id, 'subtraction_done'], immutable=True))

        elif step:
            print(f"Unknown step: {step}")

    if todo:
        todo.append(task_finalize.subtask(args=[task.id], immutable=True))

        # Create the chain and freeze it to get task IDs before applying
        task_chain = chain(todo)
        res = task_chain.freeze()

        # Extract all task IDs from the frozen chain
        # task.celery_chain_ids = [t.id for t in todo]
        task.celery_chain_ids = list(reversed(res.as_list()))

        # Apply the chain
        result = task_chain.apply_async()
        task.celery_id = result.id
        task.state = 'running'
        task.save()
