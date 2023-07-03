# Django + Celery imports
from celery import shared_task

from . import models

# Generic science stack
import os

import numpy as np
from astropy.wcs import WCS
from astropy.io import fits as fits

from astropy.table import Table, vstack
from astropy.stats import mad_std
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time

import astroscrappy

from functools import partial

# STDPipe
from stdpipe import astrometry, photometry, catalogs, cutouts
from stdpipe import templates, subtraction, plots, pipeline, utils, psf

# Disable some annoying warnings from astropy
import warnings
from astropy.wcs import FITSFixedWarning
from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter(action='ignore', category=FITSFixedWarning)
warnings.simplefilter(action='ignore', category=VerifyWarning)


def print_to_file(*args, clear=False, logname='out.log', **kwargs):
    if clear and os.path.exists(logname):
        print('Clearing', logname)
        os.unlink(logname)

    if len(args) or len(kwargs):
        print(*args, **kwargs)
        with open(logname, 'a+') as lfd:
            print(file=lfd, *args, **kwargs)


@shared_task(bind=True)
def task_inspect(self, id):
    task = models.Task.objects.get(id=id)
    basepath = task.path()

    config = task.config

    # Start processing
    log = partial(print_to_file, logname=os.path.join(basepath, 'inspect.log'))
    log(clear=True)

    try:
        inspect_image(os.path.join(basepath, 'image.fits'), config, verbose=log)
        task.state = 'inspected'
    except:
        import traceback
        traceback.print_exc()

        task.state = 'failed'
        task.celery_id = None

    # End processing
    task.celery_id = None
    task.save()

def inspect_image(filename, config=None, verbose=True):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    config = config or dict()
    config['sn'] = config.get('sn', 5)
    config['initial_aper'] = config.get('initial_aper', 3)
    config['rel_aper'] = config.get('rel_aper', 1)
    config['rel_bkgann'] = config.get('rel_bkgann', None)
    config['bg_size'] = config.get('bg_size', 256)
    config['spatial_order'] = config.get('spatial_order', 2)
    config['use_color'] = config.get('use_color', True)

    log('Inspecting ', filename)

    image,header = fits.getdata(filename).astype(np.double), fits.getheader(filename)

    log('Image shape is', image.shape)
