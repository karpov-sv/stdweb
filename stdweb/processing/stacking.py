"""
Image stacking and coaddition.
"""

import os

import numpy as np

import sep
import astroscrappy
import reproject

from astropy.wcs import WCS
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from .inspect import mask_cosmics


def stack_images(filenames, outname, config, verbose=True):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    basepath = os.path.dirname(outname)

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
            bg = sep.Background(image.astype(np.double), mask=mask)
            image = 1.0*image - bg.back()

        wcs = WCS(header)
        if wcs is None or not wcs.is_celestial:
            raise RuntimeError('No usable WCS in first image')

        if header0 is None:
            header0 = header
            wcs0 = wcs

            log(f"Using ({i+1}/{len(filenames)}) {filename} as a reference pixel grid")

            # First image should not be re-projected, we will keep it as-is
            images.append(image)

        else:
            log(f"Re-projecting ({i+1}/{len(filenames)}) {filename} onto the grid")

            # All other images should be re-projected to first one pixel grid
            image1,fp = reproject.reproject_adaptive((image, wcs), wcs0, images[0].shape, conserve_flux=True, parallel=True)
            image1[fp<0.5] = np.nan # Mask parts of the image not completely covered

            images.append(image1)

    stack_method = config.get('stack_method', 'sum')

    if stack_method == 'sum':
        log(f"Computing the stack as a plain sum of {len(images)} images")
        coadd = np.sum(images, axis=0)

        for _ in ['SATURATE', 'DATAMAX']:
            if _ in header0:
                log(f"Adjusting {_} header keyword")
                header0[_] *= len(images)
    elif stack_method == 'clipped_mean':
        log(f"Computing the stack as a sigma clipped mean of {len(images)} images")
        coadd = np.nanmedian(images, axis=0)

    elif stack_method == 'median':
        log(f"Computing the stack as a median of {len(images)} images")
        coadd = sigma_clipped_stats(images, axis=0)[0]

    else:
        raise RuntimeError('Unsupported stacking method: ' + stack_method)

    fits.writeto(outname, coadd, header0, overwrite=True)

    log(f"Stacked image written to {outname}")
