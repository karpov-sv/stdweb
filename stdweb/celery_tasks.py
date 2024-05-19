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
    for key in config.keys():
        if type(config[key]) == np.float32:
            config[key] = float(config[key])


@shared_task(bind=True)
def task_cleanup(self, id):
    task = models.Task.objects.get(id=id)
    basepath = task.path()

    for path in glob.glob(os.path.join(basepath, '*')):
        if os.path.split(path)[-1] != 'image.fits':
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.unlink(path)

    # End processing
    task.state = 'cleanup_done'
    task.celery_id = None
    task.complete()
    task.save()


@shared_task(bind=True)
def task_inspect(self, id):
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

    # End processing
    task.celery_id = None
    fix_config(config)
    task.complete()
    task.save()


@shared_task(bind=True)
def task_photometry(self, id):
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

        task.state = 'failed'
        task.celery_id = None

    # End processing
    task.celery_id = None
    fix_config(config)
    task.complete()
    task.save()


@shared_task(bind=True)
def task_subtraction(self, id):
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

    # End processing
    task.celery_id = None
    fix_config(config)
    task.complete()
    task.save()
