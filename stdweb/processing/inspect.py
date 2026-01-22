"""
Image inspection and initial parameter extraction.
"""

import os

import numpy as np

from astropy.io import fits
from astropy.time import Time

from stdpipe import astrometry, cutouts, templates
from stdpipe import resolve, utils

from .constants import *
from .utils import *


import astroscrappy
import sep

def mask_cosmics(
    image, mask,
    gain=1, satlevel=np.inf,
    negative_threshold=-10,
    fwhm=None,
    get_clean=False,
    verbose=False,
    **kwargs
):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    # We will use custom noise model for astroscrappy as we do not know whether
    # the image is background-subtracted already, or how it was flatfielded
    bg = sep.Background(image.astype(np.double), mask=mask)
    rms = bg.rms()
    diff = image - bg.back()
    var = rms**2 + np.abs(diff) / gain

    # Mask negative outliers that confuse lacosmic
    mask_neg = diff < negative_threshold * rms

    kwargs = {}

    if fwhm is not None:
        kwargs.update({
            'psfmodel': 'gauss',
            'psffwhm': fwhm,
            'psfsize': 6 * (fwhm // 2) + 1,
            'fsmode': 'convolve',
        })

        # Some heuristics?
        # if fwhm < 2.3:
        #     kwargs.update({'sigclip': 5.6, 'objlim': 11.0, 'sigfrac': 0.6, 'niter': 3})
        # elif fwhm < 3.5:
        #     kwargs.update({'sigclip': 5.0, 'objlim': 8.0, 'sigfrac': 0.45, 'niter': 4})
        # else:
        #     kwargs.update({'sigclip': 4.5, 'objlim': 5.0, 'sigfrac': 0.3, 'niter': 4})

        log(f"Using convolve with gaussian kernel, fwhm {fwhm:.1f} pix")

    cmask, cimage = astroscrappy.detect_cosmics(
        image,
        inmask=mask | mask_neg,
        invar=var.astype(np.float32),
        gain=gain,
        satlevel=satlevel * gain,
        cleantype=kwargs.pop('cleantype', 'meanmask'),
        verbose=verbose,
        **kwargs
    )

    if get_clean:
        return cmask, cimage / gain
    else:
        return cmask


def inspect_image(filename, config, verbose=True, show=False):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    basepath = os.path.dirname(filename)

    # Cleanup stale plots
    cleanup_paths(cleanup_inspect, basepath=basepath)

    config['sn'] = config.get('sn', 5)
    config['initial_aper'] = config.get('initial_aper', 3)
    config['initial_r0'] = config.get('initial_r0', 0)
    config['rel_aper'] = config.get('rel_aper', 1)
    config['rel_bg1'] = config.get('rel_bg1', 5)
    config['rel_bg2'] = config.get('rel_bg2', 7)
    config['spatial_order'] = config.get('spatial_order', 2)
    config['minarea'] = config.get('minarea', 5)
    config['use_color'] = config.get('use_color', True)
    config['refine_wcs'] = config.get('refine_wcs', True)
    config['blind_match_wcs'] = config.get('blind_match_wcs', False)
    config['hotpants_extra'] = config.get('hotpants_extra', {'ko':0, 'bgo':0})
    config['sub_size'] = config.get('sub_size', 1000)
    config['sub_overlap'] = config.get('sub_overlap', 50)
    config['sub_verbose'] = config.get('sub_verbose', False)
    config['subtraction_mode'] = config.get('subtraction_mode', 'detection')

    # Fix some initial problems with the image like compression etc
    pre_fix_image(filename, verbose=verbose)

    # Load the image
    log(f'Inspecting {filename}')
    try:
        image,header = fits.getdata(filename, -1).astype(np.double), fits.getheader(filename, -1)
    except:
        raise RuntimeError('Cannot load the image')

    # Fix image Inf values?..
    image[~np.isfinite(image)] = np.nan

    log(f"Image size is {image.shape[1]} x {image.shape[0]}")

    log(f"Image Min / Median / Max : {np.nanmin(image):.2f} {np.nanmedian(image):.2f} {np.nanmax(image):.2f}")
    log(f"Image RMS: {np.nanstd(image):.2f}")

    fix_header(header)

    # Guess some parameters from keywords
    if not config.get('gain'):
        config['gain'] = float(header.get(
            'GAIN',
            header.get('CCD_GAIN', 1)
        ))
        if config['gain'] == 0:
            log("Header gain is zero, setting it to 1")
            config['gain'] = 1

        if config['gain'] == 1 and np.nanstd(image) <= 1:
            log(f"Warning: Pixel values are significantly re-scaled, guessing gain from max value")
            # Here we assume that original gain was 1 and original max value was 65535
            config['gain'] = 65535./np.nanmax(image)
    log(f"Gain is {config['gain']:.2f}")

    # Filter
    if not config.get('filter'):
        config['filter'] = 'unknown'
        for kw in ['FILTER', 'FILTERS', 'CAMFILT']:
            if kw in header:
                config['filter'] = str(header.get(kw)).strip()
                break
    log(f"Filter is {config['filter']}")

    # Normalize filters
    for fname in supported_filters.keys():
        if config['filter'] in supported_filters[fname]['aliases']:
            config['filter'] = fname
            log(f"Filter name normalized to {config['filter']}")
            break

    # Fallback filter
    if config['filter'] not in supported_filters.keys():
        log(f"Unknown filter {config['filter']}, falling back to r")
        config['filter'] = 'r'

    # Saturation
    if not config.get('saturation'):
        satlevel = header.get(
            'SATURATE',
            # header.get('DATAMAX')
        )
        if satlevel is not None:
            # Convert to float to handle numpy scalars and strings
            satlevel = float(satlevel)

            if satlevel > 0:
                log("Got saturation level from FITS header")

                if satlevel < 0.5*np.nanmax(image):
                    log(f"Warning: header saturation level ({satlevel}) is significantly smaller than image max value!")
                elif satlevel > np.nanmax(image):
                    log(f"Warning: header saturation level ({satlevel}) is larger than image max value!")
            else:
                # Treat zero or negative as missing
                satlevel = None

        if satlevel is None:
            satlevel = 0.05*np.nanmedian(image) + 0.95*np.nanmax(image) # med + 0.95(max-med)
            log("Estimating saturation level from the image max value")

        config['saturation'] = satlevel
    log(f"Saturation level is {config['saturation']:.1f}")

    # Mask
    mask = np.isnan(image)
    mask |= image >= config['saturation']

    # Custom mask
    if os.path.exists(os.path.join(basepath, 'custom_mask.fits')):
        mask |= fits.getdata(os.path.join(basepath, 'custom_mask.fits'), -1) > 0
        log("Custom mask loaded from file:custom_mask.fits")

    # Background size
    if not config.get('bg_size'):
        bg_size = 256
        if bg_size > 0.5*image.shape[0] or bg_size > 0.5*image.shape[1]:
            bg_size = int(min(image.shape[0]/2, image.shape[1]/2))
        log(f"Background mesh size set to {bg_size} x {bg_size} pixels")
        config['bg_size'] = bg_size

    # Cosmics
    if config.get('mask_cosmics', True):
        cmask = mask_cosmics(
            image, mask,
            gain=config.get('gain', 1),
            fwhm=config.get('fwhm', None),
            satlevel=config.get('saturation'),
            verbose=verbose,
        )
        log(f"Done masking cosmics, {np.sum(cmask)} ({100*np.sum(cmask)/cmask.shape[0]/cmask.shape[1]:.1f}%) pixels masked")
        mask |= cmask

    log(f"{np.sum(mask)} ({100*np.sum(mask)/mask.shape[0]/mask.shape[1]:.1f}%) pixels masked")

    if np.sum(mask) > 0.95*mask.shape[0]*mask.shape[1]:
        raise RuntimeError('More than 95% of the image is masked')

    fits_write(os.path.join(basepath, 'mask.fits'), mask.astype(np.uint8), compress=True)
    log("Mask written to file:mask.fits")

    # WCS
    wcs = get_wcs(filename, header=header, verbose=verbose)

    if wcs and wcs.is_celestial:
        ra0,dec0,sr0 = astrometry.get_frame_center(wcs=wcs, width=image.shape[1], height=image.shape[0])
        pixscale = astrometry.get_pixscale(wcs=wcs)

        log(f"Field center is at {ra0:.3f} {dec0:.3f}, radius {sr0:.2f} deg")
        log(f"Pixel scale is {3600*pixscale:.2f} arcsec/pix")

        config['refine_wcs'] = True

        if pixscale > 100.0/3600:
            log("Warning: Pixel scale is too large, most probably WCS is broken! Enabling blind matching.")
            ra0,dec0,sr0 = None,None,None
            config['blind_match_wcs'] = True

    else:
        ra0,dec0,sr0 = None,None,None
        config['blind_match_wcs'] = True
        log("No usable WCS found, blind matching enabled")

    # Target?..
    if not 'target' in config:
        config['target'] = str(header.get('TARGET'))

    if config.get('target'):
        config['targets'] = []
        for i,target_name in enumerate(config['target'].splitlines()):
            target = {'name': target_name.strip()}
            target_title = "Primary target" if i == 0 else f"Secondary target {i}"

            if target_name:
                log(f"{target_title} is {target['name']}")
                try:
                    coords = resolve.resolve(target['name'])
                    target['ra'] = coords.ra.deg
                    target['dec'] = coords.dec.deg

                    if not len(config['targets']):
                        # Keep backwards-compatible primary target coordinates
                        config['target_ra'] = target['ra']
                        config['target_dec'] = target['dec']

                    # Activate target photometry mode
                    config['subtraction_mode'] = 'target'

                    log(f"Resolved to RA={target['ra']:.4f} Dec={target['dec']:.4f}")

                    config['targets'].append(target)
                except:
                    log("Target not resolved")

        if (config.get('target_ra') or config.get('target_dec')) and wcs and wcs.is_celestial:
            if ra0 is not None and dec0 is not None and sr0 is not None:
                if astrometry.spherical_distance(ra0, dec0,
                                                 config.get('target_ra'),
                                                 config.get('target_dec')) > 2.0*sr0:
                    log("Primary target is very far from the image center!")

            try:
                x0,y0 = wcs.all_world2pix(config.get('targets')[0].get('ra'), config.get('targets')[0].get('dec'), 0)

                if x0 > 0 and y0 > 0 and x0 < image.shape[1] and y0 < image.shape[0]:
                    cutout,cheader = cutouts.crop_image_centered(image, x0, y0, 100, header=header)
                    fits.writeto(os.path.join(basepath, 'image_target.fits'), cutout, cheader, overwrite=True)
                    log(f"Primary target is at x={x0:.1f} y={y0:.1f}")
                    log("Primary target cutout written to file:image_target.fits")
                else:
                    log("Primary target is outside the image")
                    log(f"{x0} {y0}")
            except:
                pass

        # We may initialize some blind match parameters from the target position, if any
        if config.get('target_ra') is not None and config.get('blind_match_center') is None:
            config['blind_match_center'] = "{} {}".format(config.get('target_ra'), config.get('target_dec'))

    else:
        # Remove fields that are computed from the target
        config.pop('target_ra', None)
        config.pop('target_dec', None)

    # Suggested catalogue
    if not config.get('cat_name'):
        if config['filter'] in ['U', 'B', 'V', 'R', 'I']:
            config['cat_name'] = 'gaiadr3syn'
        elif config['filter'] in ['G', 'BP', 'RP']:
            config['cat_name'] = 'gaiaedr3'
        else:
            config['cat_name'] = 'ps1'

            if (dec0 is not None and dec0 < -30) or config.get('target_dec', 0) < -30:
                config['cat_name'] = 'skymapper'

        log(f"Suggested catalogue is {supported_catalogs[config['cat_name']]['name']}")

    if not config.get('cat_limit'):
        # Modest limit to restrict getting too faint stars
        config['cat_limit'] = 20.0

    # Suggested template
    if not config.get('template'):
        if ((dec0 is not None and templates.point_in_ps1(ra0, dec0)) or
            (config.get('target_dec') and templates.point_in_ps1(config.get('target_ra'), config.get('target_dec')))):
            # Always try PS1 first, even when LS is available?..
            config['template'] = 'ps1'
        elif ((dec0 is not None and templates.point_in_ls(ra0, dec0)) or
            (config.get('target_dec') and templates.point_in_ls(config.get('target_ra'), config.get('target_dec')))):
            config['template'] = 'ls'
        elif (dec0 is not None and dec0 < -30) or config.get('target_dec', 0) < -30:
            config['template'] = 'skymapper'
        else:
            config['template'] = 'ps1' # Fallback

        log(f"Suggested template is {supported_templates[config['template']]['name']}")

    # Time
    if not config.get('time'):
        time = utils.get_obs_time(header=header, verbose=verbose)

        if time is not None:
            config['time'] = time.iso

    if config.get('time'):
        log(f"Time is {config.get('time')}")
        log(f"MJD is {Time(config.get('time')).mjd}")
