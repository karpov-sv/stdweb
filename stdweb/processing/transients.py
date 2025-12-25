"""
Simple catalogue-based transient detection.
"""

import os

import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

from stdpipe import astrometry, cutouts, templates, pipeline
from stdpipe import resolve, utils

from .constants import *
from .utils import *
from .catalogs import *
from .photometry import *


def transients_simple_image(filename, config, verbose=True, show=False):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    basepath = os.path.dirname(filename)

    # Cleanup stale plots and files
    cleanup_paths(cleanup_transients_simple, basepath=basepath)

    # Image
    image,header = fits.getdata(filename, -1).astype(np.double), fits.getheader(filename, -1)

    fix_header(header)

    # Ensure all necessary files exist
    for _ in [
        'mask.fits', 'segmentation.fits', 'filtered.fits',
        'objects.vot', 'cat.vot', 'photometry.pickle',
    ]:
        if not os.path.exists(os.path.join(basepath, _)):
            raise RuntimeError(f"{_} not found, please rerun photometric calibration")

    # Mask
    mask = fits.getdata(os.path.join(basepath, 'mask.fits'), -1) > 0

    # Segmentation map
    segm = fits.getdata(os.path.join(basepath, 'segmentation.fits'), -1)

    # Filtered detection image
    fimg = fits.getdata(os.path.join(basepath, 'filtered.fits'), -1)

    # Objects
    obj = Table.read(os.path.join(basepath, 'objects.vot'))
    log(f"{len(obj)} objects loaded from file:objects.vot")

    # Catalogue
    # cat = Table.read(os.path.join(basepath, 'cat.vot'))

    # WCS
    wcs = get_wcs(filename, header=header, verbose=verbose)

    if wcs is None or not wcs.is_celestial:
        raise RuntimeError('No WCS astrometric solution')

    ra0,dec0,sr0 = astrometry.get_frame_center(wcs=wcs, width=image.shape[1], height=image.shape[0])
    pixscale = astrometry.get_pixscale(wcs=wcs)
    log(f"Field center is at {ra0:.3f} {dec0:.3f}, radius {sr0:.2f} deg, scale {3600*pixscale:.2f} arcsec/pix")

    fwhm = config.get('fwhm', 1.0)
    log(f"FWHM is {fwhm:.1f} pixels, or {3600*fwhm*pixscale:.2f} arcsec")

    # Time
    time = Time(config.get('time')) if config.get('time') else None
    if time is not None:
        log(f"Time is {time}")
        log(f"MJD is {Time(time).mjd}")

    log("\n---- Simple catalogue-based transient detection ----\n")

    # Restrict to the cone if center and radius are provided
    if config.get('simple_center') and config.get('simple_sr0'):
        center = resolve.resolve(config.get('simple_center'))
        sr0 = config.get('simple_sr0')
        log(f"Restricting the search to {sr0:.3f} deg around RA={center.ra.deg:.4f} Dec={center.dec.deg:.4f}")
        dist = astrometry.spherical_distance(obj['ra'], obj['dec'], center.ra.deg, center.dec.deg)
        obj = obj[dist < sr0]
        log(f"{len(obj)} objects inside the region")

    # Cross-check with objects detected in other tasks
    if config.get('simple_others'):
        log("Cross-checking with the objects detected in other tasks")
        for other in config.get('simple_others', '').split():
            if other.isdigit():
                otherpath = os.path.join(basepath, '..', other, 'objects.vot')
                if not os.path.exists(otherpath):
                    log(f"Task {other} has no detected objects")
                else:
                    obj1 = Table.read(otherpath)
                    oidx,_,__ = astrometry.spherical_match(obj['ra'], obj['dec'], obj1['ra'], obj1['dec'], 0.5*fwhm*pixscale)
                    idx = np.in1d(obj['NUMBER'], obj['NUMBER'][oidx])
                    obj = obj[idx]
                    log(f"Task {other}: {len(obj1)} objects, {len(obj)} matches")

        log(f"{len(obj)} objects after cross-checking")

    # Vizier catalogues to check
    vizier = guess_vizier_catalogues(ra0, dec0)
    log(f"Will check Vizier catalogues: {' '.join(vizier)}")

    # Filter based on flags and Vizier catalogs
    flagmask = 0x7fff - 0x0100 - 0x02 # Allow deblended and isophotal masked
    if not config.get('simple_prefilter', True):
        flagmask -= 0x800 # Allow pre-filtered
    else:
        log("Will reject pre-filtered detections")

    if config.get('simple_mag_diff', 2.0):
        log(f"Will only keep matches brighter than catalogue by {config.get('simple_mag_diff', 2.0):.2f} mags")
    else:
        log("Will reject all positional matches")

    # Cross-match checker
    def checker_fn(xobj, xcat, catname):
        xidx = np.ones_like(xobj, dtype=bool)

        if config.get('simple_mag_diff', 2.0):
            # Get filter used for photometric calibration
            fname = config.get('cat_col_mag')
            if fname.endswith('mag'):
                fname = fname[:-3]

            cat_col_mag, cat_col_mag_err = guess_catalogue_mag_columns(fname, xcat)

            if cat_col_mag is not None:
                mag = xobj['mag_calib']
                if fname in ['U', 'B', 'V', 'R', 'I'] and cat_col_mag not in ['Umag', 'Bmag', 'Vmag', 'Rmag', 'Imag']:
                    # Convert to AB mags if using AB reference catalogue
                    mag += filter_ab_offset.get(fname, 0)

                diff = mag - xcat[cat_col_mag]

                if len(diff[np.isfinite(diff)]) > 10:
                    # Adjust zeropoint
                    diff -= np.nanmedian(diff)

                # TODO: take errors into account?..
                xidx = diff > -config.get('simple_mag_diff', 2.0)

        return xidx

    candidates = pipeline.filter_transient_candidates(
        obj,
        sr=0.5*fwhm*pixscale,
        vizier=vizier,
        # Filter out all masked objects except for isophotal masked, and deblended
        flagged=True, flagmask=flagmask,
        # SkyBoT?..
        time=time,
        skybot=config.get('simple_skybot', True),
        vizier_checker_fn=checker_fn,
        verbose=verbose
    )

    # Additional filtering for blended stars
    # TODO: somehow integrate it into `filter_transient_candidates` proper
    if len(candidates) > 0 and config.get('simple_blends', True):
        candidates = filter_vizier_blends(
            candidates,
            sr=0.5*fwhm*pixscale,
            sr_blend=2*fwhm*pixscale,
            vizier=vizier,
            fname=config.get('filter'),
            vizier_checker_fn=checker_fn,
            verbose=verbose
        )

    # Restrict to 100 brightest ones if there are too many
    if len(candidates) > 100:
        candidates = candidates[:100]
        log(f"Warning: too many candidates, limiting to first {len(candidates)}")

    cutout_names = []

    for cand in candidates:
        cutout = cutouts.get_cutout(
            image.astype(np.double),
            cand,
            config.get('cutout_size', 30),
            header=header,
            mask=mask,
            footprint=(segm==cand['NUMBER']) if segm is not None else None,
            filtered=fimg if config.get('initial_r0') else None,
        )

        # Cutout from relevant HiPS survey
        cutout['template'] = templates.get_hips_image(
            guess_hips_survey(cand['ra'], cand['dec'], config['filter']),
            header=cutout['header'],
            get_header=False
        )

        jname = utils.make_jname(cand['ra'], cand['dec'])
        cutout_name = os.path.join('candidates_simple', jname + '.cutout')
        cutout_names.append(cutout_name)

        try:
            os.makedirs(os.path.join(basepath, 'candidates_simple'))
        except OSError:
            pass

        cutouts.write_cutout(cutout, os.path.join(basepath, cutout_name))

    log("\n---- Final list of candidates ----\n")

    if len(candidates):
        candidates['cutout_name'] = cutout_names

        log(f"{len(candidates)} candidates in total")

        candidates_extract_unstacked(filename, candidates, config, verbose=verbose, show=show)

        candidates.write(os.path.join(basepath, 'candidates_simple.vot'), format='votable', overwrite=True)
        log("Candidates written to file:candidates_simple.vot")

        write_ds9_regions(
            os.path.join(basepath, 'candidates_simple.reg'),
            candidates,
            radius=config.get('rel_aper', 1.0)*pixscale*fwhm
        )
        log("Candidates written to file:candidates_simple.reg")

    else:
        log("No candidates found")
