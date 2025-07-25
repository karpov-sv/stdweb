# Django + Celery imports
from celery import shared_task

import os, glob, shutil

from functools import partial

import numpy as np

from . import models
from . import processing


def fix_config(config):
    """
    Fix non-serializable Numpy types in config
    """
    for key in list(config.keys()):
        val = config[key]
        # Convert numpy scalar types to plain Python types
        if isinstance(val, (np.float32, np.float64, np.int32, np.int64)):
            config[key] = val.item()
            val = config[key]

        # Replace NaN / Inf with None so that JSON serialization succeeds
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            config[key] = None


@shared_task(bind=True)
def task_finalize(self, id):
    task = models.Task.objects.get(id=id)
    task.celery_id = None
    task.complete()
    task.save()


@shared_task(bind=True)
def task_set_state(self, id, state):
    task = models.Task.objects.get(id=id)
    task.state = state
    task.save()


@shared_task(bind=True)
def task_break_if_failed(self, id):
    task = models.Task.objects.get(id=id)

    if not task.celery_id:
        print("Breaking the chain!!!")
        # self.request.callbacks = None
        # self.request.chain = None
        raise RuntimeError

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


@shared_task(bind=True)
def task_inspect(self, id, finalize=True):
    task = models.Task.objects.get(id=id)
    basepath = task.path()

    config = task.config

    log = partial(processing.print_to_file, logname=os.path.join(basepath, 'inspect.log'))
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


@shared_task(bind=True)
def task_photometry(self, id, finalize=True):
    task = models.Task.objects.get(id=id)
    basepath = task.path()

    config = task.config

    log = partial(processing.print_to_file, logname=os.path.join(basepath, 'photometry.log'))
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


@shared_task(bind=True)
def task_transients_simple(self, id, finalize=True):
    task = models.Task.objects.get(id=id)
    basepath = task.path()

    config = task.config

    log = partial(processing.print_to_file, logname=os.path.join(basepath, 'transients_simple.log'))
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


@shared_task(bind=True)
def task_subtraction(self, id, finalize=True):
    task = models.Task.objects.get(id=id)
    basepath = task.path()

    config = task.config

    log = partial(processing.print_to_file, logname=os.path.join(basepath, 'subtraction.log'))
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
