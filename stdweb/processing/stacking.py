"""
Image stacking and coaddition.
"""

import os

import numpy as np

import sep
import astroscrappy
import reproject
import astroalign

from astropy.wcs import WCS
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from django.conf import settings

from .inspect import mask_cosmics
from .constants import *
from .utils import *


def validate_under_data_path(filenames):
    """Ensure every stacking input path stays under DATA_PATH.

    Guards against path traversal in user-supplied stack_filenames (which may
    also arrive via the API, bypassing the upload view sanitization).

    Uses a lexical (``..``-collapsing) check rather than resolving symlinks, so
    that symlinked external folders placed under DATA_PATH remain usable, as
    they are in the file browser.
    """
    datapath = os.path.abspath(settings.DATA_PATH)

    for filename in filenames:
        path = os.path.abspath(filename)
        if path != datapath and not path.startswith(datapath + os.sep):
            raise RuntimeError(f"Stacking input path is outside of DATA_PATH: {filename}")


def stack_images(filenames, outname, config, verbose=True):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    # Safeguard: refuse to read inputs from outside the allowed data directory
    validate_under_data_path(filenames)

    if len(filenames) < 2:
        raise RuntimeError(f"Stacking requires at least 2 images, got {len(filenames)}")

    basepath = os.path.dirname(outname)

    cleanup_paths(cleanup_inspect, basepath=basepath)

    wcs0 = None
    header0 = None

    images = []

    if config.get('stack_mask_cosmics', False):
        log('Will mask cosmics in individual images while stacking')
    if config.get('stack_subtract_bg', True):
        log('Will subtract the background from individual images while stacking')

    for i,filename in enumerate(filenames):
        image,header = fits.getdata(filename, header=True)

        mask = ~np.isfinite(image)

        # Cosmics
        if config.get('stack_mask_cosmics', False):
            cmask, image = mask_cosmics(
                image, mask,
                gain=config.get('gain', 1),
                satlevel=0.05*np.nanmedian(image) + 0.95*np.nanmax(image), # med + 0.95(max-med),
                cleantype='medmask',
                get_clean=True
            )

        if config.get('stack_subtract_bg', True):
            bg = sep.Background(image.astype(np.double), mask=mask, bh=256, bw=256)
            image = 1.0*image - bg.back()

        wcs = WCS(header)
        usable_wcs = is_wcs_usable(wcs)

        # A formally-celestial but broken WCS (e.g. all-zero CRVAL/CRPIX/CD)
        # is rejected by is_wcs_usable; fall back to astroalign for it.
        if wcs is not None and wcs.is_celestial and not usable_wcs:
            log(f"({i+1}/{len(filenames)}) {filename} has a broken WCS "
                "(pixel scale too large), ignoring it")

        if header0 is None:
            header0 = header
            wcs0 = wcs if usable_wcs else None

            if wcs0 is not None:
                log(f"Using ({i+1}/{len(filenames)}) {filename} as a reference pixel grid")
            else:
                log(f"Using ({i+1}/{len(filenames)}) {filename} as a reference pixel grid "
                    "(no usable WCS, will align the rest with astroalign)")

            # First image should not be re-projected, we will keep it as-is
            images.append(image)

        elif wcs0 is not None and usable_wcs:
            log(f"Re-projecting ({i+1}/{len(filenames)}) {filename} onto the grid")

            # All other images should be re-projected to first one pixel grid
            image1,fp = reproject.reproject_adaptive((image, wcs), wcs0, images[0].shape, conserve_flux=True, parallel=True)
            image1[fp<0.5] = np.nan # Mask parts of the image not completely covered

            images.append(image1)

        else:
            # Fallback for missing WCS: register onto the reference image pixel
            # grid using source-matching (astroalign) instead of reprojection
            log(f"Aligning ({i+1}/{len(filenames)}) {filename} onto the grid with astroalign")

            try:
                image1,fp = astroalign.register(image.astype(np.double), images[0].astype(np.double))
            except Exception as e:
                raise RuntimeError(f'Cannot align {filename} with astroalign: {e}')

            image1[fp] = np.nan # Mask parts of the image not covered by the source

            images.append(image1)

    stack_method = config.get('stack_method', 'sum')

    if stack_method == 'sum':
        log(f"Computing the stack as a plain sum of {len(images)} images")
        # Plain sum: NaN propagates so only fully-covered pixels survive. This
        # is intentional - partially-covered regions would have an inconsistent
        # flux level, causing photometric problems later. Use median or
        # clipped_mean when full coverage of incomplete regions is wanted.
        coadd = np.sum(images, axis=0)

        for key in ['SATURATE', 'DATAMAX']:
            if key in header0:
                log(f"Adjusting {key} header keyword")
                header0[key] = float(header0[key]) * len(images)
    elif stack_method == 'clipped_mean':
        log(f"Computing the stack as a sigma clipped mean of {len(images)} images")
        coadd = sigma_clipped_stats(images, axis=0)[0]

    elif stack_method == 'median':
        log(f"Computing the stack as a median of {len(images)} images")
        coadd = np.nanmedian(images, axis=0)

    else:
        raise RuntimeError('Unsupported stacking method: ' + stack_method)

    fits.writeto(outname, coadd, header0, overwrite=True)

    log(f"Stacked image written to {outname}")
