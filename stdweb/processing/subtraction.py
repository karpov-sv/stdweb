"""
Template subtraction and difference image analysis.
"""

import os
from functools import partial

import numpy as np

import sep
import reproject

from django.conf import settings

from astropy.wcs import WCS
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.time import Time

from stdpipe import astrometry, photometry, cutouts
from stdpipe import templates, subtraction, plots, pipeline, psf
from stdpipe import resolve, utils

from .constants import *
from .utils import *
from .catalogs import *
from .photometry import *


def subtract_image(filename, config, verbose=True, show=False):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    basepath = os.path.dirname(filename)

    if settings.STDPIPE_PS1_CACHE:
        _cachedir = settings.STDPIPE_PS1_CACHE
    else:
        # Task-local
        _cachedir = os.path.join(basepath, 'cache')

    sub_verbose = verbose if config.get('sub_verbose') else False
    subtraction_mode = config.get('subtraction_mode', 'detection')
    subtraction_method = config.get('subtraction_method', 'hotpants')

    # Cleanup stale plots and files
    cleanup_paths(cleanup_subtraction, basepath=basepath)

    # Image
    image,header = fits.getdata(filename, -1).astype(np.double), fits.getheader(filename, -1)

    fix_header(header)

    # Mask
    mask = fits.getdata(os.path.join(basepath, 'mask.fits'), -1) > 0

    # Photometric solution
    m = pickle_from_file(os.path.join(basepath, 'photometry.pickle'))

    # Objects
    obj = Table.read(os.path.join(basepath, 'objects.vot'))

    # Catalogue
    cat = Table.read(os.path.join(basepath, 'cat.vot'))

    # WCS
    wcs = get_wcs(filename, header=header, verbose=verbose)

    if wcs is None or not wcs.is_celestial:
        raise RuntimeError('No WCS astrometric solution')

    pixscale = astrometry.get_pixscale(wcs=wcs)

    # Time
    time = Time(config.get('time')) if config.get('time') else None

    log("\n---- Template selection ----\n")

    tname = config.get('template', 'ps1')
    tconf = supported_templates.get(tname)

    if tname == 'custom':
        log("Using custom template from custom_template.fits")

        if not os.path.exists(os.path.join(basepath, 'custom_template.fits')):
            raise RuntimeError("Custom template not found")

        custom_template = fits.getdata(os.path.join(basepath, 'custom_template.fits'), -1).astype(np.double)
        custom_header = fits.getheader(os.path.join(basepath, 'custom_template.fits'), -1)
        custom_wcs = WCS(custom_header)

        template_gain = config.get('custom_template_gain', 10000)
        template_saturation = config.get('custom_template_saturation', None)

        custom_mask = np.isnan(custom_template)
        if template_saturation:
            custom_mask |= custom_template >= template_saturation
    else:
        if tconf is None:
            raise RuntimeError(f"Unsupported template: {tname}")

        tfilter = None
        for _ in filter_mappings[config['filter']]:
            if _ in tconf['filters']:
                tfilter = _
                break

        template_gain = 10000 # Assume effectively noise-less

        log(f"Using {tconf['name']} in filter {tfilter} as a template")

    sub_size = config.get('sub_size', 1000)
    sub_overlap = config.get('sub_overlap', 50)

    classifier = None

    if subtraction_mode == 'detection':
        log('Transient detection mode activated')
        # We will split the image into nx x ny blocks
        nx = max(1, int(np.round(image.shape[1] / sub_size)))
        ny = max(1, int(np.round(image.shape[0] / sub_size)))
        log(f"Will split the image into {nx} x {ny} sub-images")
        split_fn = partial(pipeline.split_image, get_index=True, overlap=sub_overlap, nx=nx, ny=ny)

    elif config.get('target_ra') is not None:
        log('Forced photometry mode activated')
        # We will just crop the image
        nx, ny = 1, 1
        x0,y0 = wcs.all_world2pix(config['target_ra'], config['target_dec'], 0)
        log(f"Will crop the sub-image centered at {x0:.1f} {y0:.1f}")
        def split_fn(image, *args, **kwargs):
            result = pipeline.get_subimage_centered(image, *args, x0=x0, y0=y0, width=sub_size, **kwargs)

            yield [0] + result

    else:
        log('No target provided and transient detection is disabled, nothing to do')
        return

    if subtraction_method == 'zogy':
        log(f"\n---- Science image PSF ----\n")
        # Get global PSF model and object list with large aperture for flux normalization
        image_psf, image_psf_obj = psf.run_psfex(
            image, mask=mask,
            # Use spatially varying PSF if we have enough stars
            order=0 if len(obj[obj['flags'] == 0]) < 100 else 2,
            aper=2.0*config.get('fwhm', 3.0),
            gain=config.get('gain', 1.0),
            minarea=config.get('minarea', 3),
            r0=config.get('initial_r0', 0.0),
            sex_extra={'BACK_SIZE': config.get('bg_size', 256)},
            verbose=verbose,
            get_obj=True,
            _tmpdir=settings.STDPIPE_TMPDIR,
            _sex_exe=settings.STDPIPE_SEXTRACTOR,
            _exe=settings.STDPIPE_PSFEX)

        image_psf_obj = image_psf_obj[image_psf_obj['flags'] == 0]

    else:
        image_psf, image_psf_obj = None, None

    all_candidates = []
    cutout_names = []

    for i, x0, y0, image1, mask1, header1, wcs1, obj1, cat1, image_psf1, image_psf_obj1 in split_fn(
            image, mask, header, wcs, obj, cat, image_psf, image_psf_obj,
            get_origin=True, verbose=False):

        log(f"\n---- Sub-image {i}: {x0} {y0} - {x0 + image1.shape[1]} {y0 + image1.shape[0]} ----\n")

        fits.writeto(os.path.join(basepath, 'sub_image.fits'), image1, header1, overwrite=True)
        fits_write(os.path.join(basepath, 'sub_mask.fits'), mask1.astype(np.uint8), header1, compress=True)

        # Get the template
        if tname == 'ps1' or tname == 'ls':
            log(f"Getting the template from original {tconf['name']} archive")
            tmpl,tmask = templates.get_survey_image_and_mask(
                tfilter, survey=tname, wcs=wcs1, shape=image1.shape,
                _cachedir=_cachedir, _cache_downscale = 1 if pixscale*3600 < 0.6 else 2,
                _tmpdir=settings.STDPIPE_TMPDIR,
                _exe=settings.STDPIPE_SWARP,
                verbose=sub_verbose)
            if tmask is not None and tmpl is not None:
                if tname == 'ps1':
                    tmask = tmask > 0
                elif tname == 'ls':
                    # Bitmask for a given band, as described at https://www.legacysurvey.org/dr10/bitmasks/
                    imask = 0x0000
                    imask |= 0x0001 # not primary brick area
                    # imask |= 0x0002 # bright star nearby
                    # imask |= 0x0100 # WISE W1 (all masks)
                    # imask |= 0x0200 # WISE W2 (all masks)
                    imask |= 0x0400 # Bailed out processing
                    # imask |= 0x0800 # medium-bright star
                    # imask |= 0x1000 # SGA large galaxy
                    # imask |= 0x2000 # Globular cluster

                    if tfilter == 'g':
                        imask |= 0x0004 # g band saturated
                        imask |= 0x0020 # any ALLMASK_G bit set
                    elif tfilter == 'r':
                        imask |= 0x0008 # r band saturated
                        imask |= 0x0040 # any ALLMASK_R bit set
                    elif tfilter == 'i':
                        imask |= 0x4000 # i band saturated
                        imask |= 0x8000 # any ALLMASK_I bit set
                    elif tfilter == 'z':
                        imask |= 0x0010 # z band saturated
                        imask |= 0x0080 # any ALLMASK_Z bit set

                    tmask = (tmask & imask) > 0

                tmask |= np.isnan(tmpl)
            else:
                raise RuntimeError(f"Error getting the template from {tconf['name']}")

        elif tname == 'custom':
            log("Re-projecting custom template onto sub-image")

            tmpl,fp = reproject.reproject_adaptive((custom_template, custom_wcs), wcs1, image1.shape)
            tmask,fp = reproject.reproject_adaptive((custom_mask.astype(np.double), custom_wcs), wcs1, image1.shape)

            tmask = tmask > 0.5
            tmask |= fp < 0.5

        else:
            log("Getting the template from HiPS server")
            tmpl = templates.get_hips_image(tconf['filters'][tfilter], wcs=wcs1, shape=image1.shape,
                                            get_header=False,
                                            verbose=sub_verbose)
            if tmpl is not None:
                tmask = np.isnan(tmpl)
            else:
                raise RuntimeError(f"Error getting the template from {tconf['name']}")

        # Estimate template FWHM
        tobj,tsegm = photometry.get_objects_sextractor(
            tmpl, mask=tmask, sn=5, gain=template_gain,
            extra={
                'BACK_SIZE': config.get('bg_size', 256),
                # 'SATUR_LEVEL': template_saturation,
            },
            extra_params=['NUMBER', 'MAG_AUTO', 'ISOAREA_IMAGE'],
            checkimages=['SEGMENTATION'],
            _tmpdir=settings.STDPIPE_TMPDIR,
            _exe=settings.STDPIPE_SEXTRACTOR
        )

        # Mask the footprints of masked objects
        # for _ in tobj[(tobj['flags'] & 0x100) > 0]:
        #     tmask |= tsegm == _['NUMBER']

        fits.writeto(os.path.join(basepath, 'sub_template.fits'), tmpl, header1, overwrite=True)
        fits_write(os.path.join(basepath, 'sub_template_mask.fits'), tmask.astype(np.uint8), header1, compress=True)

        tidx = tobj['flags'] == 0
        if len(tobj) > 20:
            tidx &= filter_sextractor_detections(tobj, verbose=verbose)
        fwhm_values = 2.0*tobj['FLUX_RADIUS'] # obj['fwhm']
        template_fwhm = np.median(fwhm_values[tidx])

        if config.get('template_fwhm_override'):
            template_fwhm = config.get('template_fwhm_override')

        # Plot FWHM vs instrumental. TODO: make generic function for that?..
        with plots.figure_saver(os.path.join(basepath, 'sub_template_fwhm_mag.png'), figsize=(8, 6), show=show) as fig:
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(fwhm_values, tobj['mag'], '.', label='All objects')
            ax.plot(fwhm_values[tidx], tobj['mag'][tidx], '.', label='Used for FWHM')

            ax.axvline(template_fwhm, ls='--', color='red')
            ax.invert_yaxis()
            ax.legend()
            ax.grid(alpha=0.2)
            ax.set_title(f"FWHM: median {np.median(fwhm_values[tidx]):.2f} pix RMS {np.std(fwhm_values[tidx]):.2f} pix")
            ax.set_xlabel('FWHM, pixels')
            ax.set_ylabel('Instrumental magnitude')
            ax.set_xlim(0, np.percentile(fwhm_values, 98))

        if subtraction_method == 'zogy':
            # ZOGY

            # Estimate template PSF
            template_psf, template_psf_obj = psf.run_psfex(
                tmpl, mask=tmask,
                # Use spatially varying PSF?..
                order=0,
                aper=2.0*template_fwhm,
                gain=template_gain,
                minarea=config.get('minarea', 3),
                r0=config.get('initial_r0', 0.0),
                sex_extra={'BACK_SIZE': config.get('bg_size', 256)},
                verbose=sub_verbose,
                get_obj=True,
                _tmpdir=settings.STDPIPE_TMPDIR,
                _sex_exe=settings.STDPIPE_SEXTRACTOR,
                _exe=settings.STDPIPE_PSFEX)

            # Do the subtraction
            diff, S_corr, Fpsf, Fpsf_err = subtraction.run_zogy(
                image1, tmpl,
                mask=mask1, template_mask=tmask,
                image_gain=config.get('gain', 1.0),
                template_gain=template_gain,
                image_psf=image_psf1, template_psf=template_psf,
                image_obj=image_psf_obj1,
                template_obj=template_psf_obj,
                fit_scale=True, fit_shift=True,
                get_Fpsf=True,
                verbose=verbose
            )

            fits.writeto(os.path.join(basepath, 'sub_diff.fits'), diff, header1, overwrite=True)
            fits.writeto(os.path.join(basepath, 'sub_scorr.fits'), S_corr, header1, overwrite=True)
            fits.writeto(os.path.join(basepath, 'sub_fpsf.fits'), Fpsf, header1, overwrite=True)
            fits.writeto(os.path.join(basepath, 'sub_fpsferr.fits'), Fpsf_err, header1, overwrite=True)

            # Combined mask on the sub-image
            fullmask1 = mask1 | tmask

            conv = S_corr
            ediff = None
        else:
            # HOTPANTS
            fwhm = config.get('fwhm', 3.0)

            log(f"Using template FWHM = {template_fwhm:.1f} pix and image FWHM = {fwhm:.1f} pix")

            bg = sep.Background(
                image1.astype(np.double), mask=mask1,
                bw=config.get('bg_size', 256),
                bh=config.get('bg_size', 256),
            )
            tbg = sep.Background(
                tmpl.astype(np.double), mask=tmask,
                bw=config.get('bg_size', 256),
                bh=config.get('bg_size', 256),
            )

            res = subtraction.run_hotpants(
                image1 - bg.back(),
                tmpl - tbg.back(),
                mask=mask1,
                template_mask=tmask,
                get_convolved=True,
                get_scaled=True,
                get_noise=True,
                verbose=verbose,
                image_fwhm=fwhm,
                template_fwhm=template_fwhm,
                image_gain=config.get('gain', 1.0),
                template_gain=template_gain,
                err=True,
                extra=config.get('hotpants_extra', {'ko':0, 'bgo':0}),
                obj=obj1[obj1['flags']==0],
                _exe=settings.STDPIPE_HOTPANTS
            )

            if res is not None:
                diff,conv,sdiff,ediff = res
            else:
                # raise RuntimeError('Subtraction failed')
                log("Warning: Subtraction failed")
                continue

            dmask = diff == 1e-30 # Bad pixels

            # Combined mask on the sub-image
            fullmask1 = mask1 | tmask | dmask

            diff1 = diff.copy()
            diff1[fullmask1] = 0.0

            fits.writeto(os.path.join(basepath, 'sub_diff.fits'), diff1, header1, overwrite=True)
            fits.writeto(os.path.join(basepath, 'sub_sdiff.fits'), sdiff, header1, overwrite=True)
            fits.writeto(os.path.join(basepath, 'sub_conv.fits'), conv, header1, overwrite=True)
            fits.writeto(os.path.join(basepath, 'sub_ediff.fits'), ediff, header1, overwrite=True)

        # Post-subtraction steps

        if config.get('rel_bg1') and config.get('rel_bg2'):
            rel_bkgann = [config['rel_bg1'], config['rel_bg2']]
        else:
            rel_bkgann = None

        if config.get('target_ra') is not None and config.get('target_dec') is not None and subtraction_mode == 'target':
            # Target forced photometry
            log(f"\n---- Target forced photometry ----\n")

            target_obj = Table({'ra':[config['target_ra']], 'dec':[config['target_dec']]})
            target_obj['x'],target_obj['y'] = wcs1.all_world2pix(target_obj['ra'], target_obj['dec'], 0)

            if not (target_obj['x'] > 0 and target_obj['x'] < image1.shape[1] and
                    target_obj['y'] > 0 and target_obj['y'] < image1.shape[0]):
                raise RuntimeError("Target is outside the sub-image")

            log(f"Target position is {target_obj['ra'][0]:.3f} {target_obj['dec'][0]:.3f}"
                f" -> "
                f"{target_obj['x'][0]:.1f} {target_obj['y'][0]:.1f}")

            target_obj = photometry.measure_objects(
                target_obj, diff, mask=fullmask1,
                # FWHM should match the one used for calibration
                fwhm=config.get('fwhm'),
                aper=config.get('rel_aper', 1.0),
                bkgann=rel_bkgann,
                sn=0,
                # We assume no background
                bg=None,
                bg_size=config.get('bg_size', 256),
                # ..and known error model
                err=ediff,
                gain=config.get('gain', 1.0),
                centroid_iter=5 if config.get('centroid_targets') else 0,
                verbose=sub_verbose
            )

            if config.get('centroid_targets'):
                # Centroiding might change target pixel positions - let's update sky positions too
                target_obj['ra_orig'],target_obj['dec_orig'] = target_obj['ra'],target_obj['dec']
                target_obj['ra'],target_obj['dec'] = wcs1.all_pix2world(target_obj['x'], target_obj['y'], 0)

                dist = astrometry.spherical_distance(
                    target_obj['ra'][0], target_obj['dec'][0],
                    target_obj['ra_orig'][0], target_obj['dec_orig'][0]
                )
                log(f"Target position adjusted to {target_obj['ra'][0]:.3f} {target_obj['dec'][0]:.3f}"
                    f" which is {3600*dist:.2f} arcsec from original position")

            target_obj['mag_calib'] = target_obj['mag'] + m['zero_fn'](
                target_obj['x'] + x0,
                target_obj['y'] + y0,
                target_obj['mag']
            )

            target_obj['mag_calib_err'] = np.hypot(
                target_obj['magerr'],
                m['zero_fn'](
                    target_obj['x'] + x0,
                    target_obj['y'] + y0,
                    target_obj['mag'],
                    get_err=True
                )
            )

            # Local detection limit from background rms, if available
            if 'bg_fluxerr' in target_obj.colnames and np.any(target_obj['bg_fluxerr'] > 0):
                fluxerr = target_obj['bg_fluxerr']
            else:
                fluxerr = target_obj['fluxerr']
            target_obj['mag_limit'] = -2.5*np.log10(config.get('sn', 5)*fluxerr) + m['zero_fn'](
                target_obj['x'],
                target_obj['y'],
                target_obj['mag']
            )

            target_obj['mag_filter_name'] = m['cat_col_mag']

            if 'cat_col_mag1' in m.keys() and 'cat_col_mag2' in m.keys():
                target_obj['mag_color_name'] = '%s - %s' % (m['cat_col_mag1'], m['cat_col_mag2'])
                target_obj['mag_color_term'] = [m['color_term']]*len(target_obj)

            target_obj.write(os.path.join(basepath, 'sub_target.vot'), format='votable', overwrite=True)
            log("Measured target stored to file:sub_target.vot")

            # Create the cutout from image based on the candidate
            cutout = cutouts.get_cutout(
                image1, target_obj[0], config.get('cutout_size', 30),
                mask=fullmask1,
                header=header1,
                time=time,
                diff=diff,
                template=tmpl,
                convolved=conv,
                err=ediff
            )
            cutouts.write_cutout(cutout, os.path.join(basepath, 'sub_target.cutout'))
            log("Target cutouts stored to file:sub_target.cutout")

            log(f"Target flux is {target_obj['flux'][0]:.1f} +/- {target_obj['fluxerr'][0]:.1f} ADU")
            if target_obj['flux'][0] > 0:
                mag_string = target_obj['mag_filter_name'][0]
                if 'mag_color_name' in target_obj.colnames and 'mag_color_term' in target_obj.colnames and target_obj['mag_color_term'][0] is not None:
                    sign = '-' if target_obj['mag_color_term'][0] > 0 else '+'
                    mag_string += f" {sign} {np.abs(target_obj['mag_color_term'][0]):.2f}*({target_obj['mag_color_name'][0]})"

                log(f"Target magnitude is {mag_string} = {target_obj['mag_calib'][0]:.2f} +/- {target_obj['mag_calib_err'][0]:.2f}")
                log(f"Target detected with S/N = {1/target_obj['mag_calib_err'][0]:.2f}")
            else:
                log("Target not detected")

        elif subtraction_mode == 'detection':
            # Transient detection mode
            log(f"Starting transient detection using edge size {sub_overlap}")

            if subtraction_method == 'zogy':
                # ZOGY
                sobj,segm = photometry.get_objects_sextractor(
                    S_corr,
                    mask=fullmask1,
                    thresh=config.get('sn', 5.0),
                    wcs=wcs1, edge=sub_overlap,
                    minarea=config.get('minarea', 1),
                    extra_params=['NUMBER', 'MAG_AUTO', 'ISOAREA_IMAGE'],
                    extra={
                        'ANALYSIS_THRESH': config.get('sn', 5.0),
                        'THRESH_TYPE': 'ABSOLUTE',
                        'BACK_TYPE': 'MANUAL',
                        'BACK_VALUE': 0,
                    },
                    checkimages=['SEGMENTATION'],
                    verbose=sub_verbose,
                    _tmpdir=settings.STDPIPE_TMPDIR,
                    _exe=settings.STDPIPE_SEXTRACTOR
                )
            else:
                # HOTPANTS
                sobj,segm = photometry.get_objects_sextractor(
                    diff,
                    mask=fullmask1,
                    err=ediff,
                    wcs=wcs1, edge=sub_overlap,
                    aper=config.get('initial_aper', 3.0),
                    gain=config.get('gain', 1.0),
                    sn=config.get('sn', 5.0),
                    minarea=config.get('minarea', 3),
                    extra_params=['NUMBER', 'MAG_AUTO', 'ISOAREA_IMAGE'],
                    extra={'BACK_SIZE': config.get('bg_size', 256)},
                    checkimages=['SEGMENTATION'],
                    verbose=sub_verbose,
                    _tmpdir=settings.STDPIPE_TMPDIR,
                    _exe=settings.STDPIPE_SEXTRACTOR
                )

            sobj = photometry.measure_objects(
                sobj, diff, mask=fullmask1,
                # FWHM should match the one used for calibration
                fwhm=config.get('fwhm'),
                aper=config.get('rel_aper', 1.0),
                bkgann=rel_bkgann,
                sn=config.get('sn', 5.0),
                # We assume no background
                bg=None,
                # ..and known error model
                err=ediff,
                gain=config.get('gain', 1.0),
                verbose=sub_verbose
            )

            if len(sobj):
                sobj['mag_calib'] = sobj['mag'] + m['zero_fn'](
                    sobj['x'] + x0,
                    sobj['y'] + y0,
                    sobj['mag']
                )
                sobj['mag_calib_err'] = np.hypot(
                    sobj['magerr'],
                    m['zero_fn'](
                        sobj['x'] + x0,
                        sobj['y'] + y0,
                        sobj['mag'],
                        get_err=True
                    )
                )

                # TODO: Improve limiting mag estimate
                sobj['mag_limit'] = -2.5*np.log10(config.get('sn', 5)*sobj['fluxerr']) + m['zero_fn'](
                    sobj['x'],
                    sobj['y'],
                    sobj['mag']
                )

                sobj['mag_filter_name'] = m['cat_col_mag']

                if 'cat_col_mag1' in m.keys() and 'cat_col_mag2' in m.keys():
                    sobj['mag_color_name'] = '%s - %s' % (m['cat_col_mag1'], m['cat_col_mag2'])
                    sobj['mag_color_term'] = [m['color_term']]*len(sobj)

            log(f"{len(sobj)} transient candidates found in difference image")

            # Restrict to the cone if center and radius are provided
            if config.get('filter_center') and config.get('filter_sr0'):
                # TODO: resolve the center only once
                center = resolve.resolve(config.get('filter_center'))
                sr0 = config.get('filter_sr0')
                log(f"Restricting the search to {sr0:.3f} deg around RA={center.ra.deg:.4f} Dec={center.dec.deg:.4f}")
                dist = astrometry.spherical_distance(sobj['ra'], sobj['dec'], center.ra.deg, center.dec.deg)
                sobj = sobj[dist < sr0]
                log(f"{len(sobj)} candidates inside the region")

            # Pre-filter detections if requested
            if config.get('filter_prefilter') and len(sobj):
                if classifier is None:
                    # Prepare the classifier based on SExtractor shape parameters
                    classifier = filter_sextractor_detections(obj, verbose=False, return_classifier=True)

                fidx = filter_sextractor_detections(sobj, verbose=verbose, classifier=classifier)
                sobj = sobj[fidx]
                log(f"{len(sobj)} candidates left after pre-filtering")

            vizier = ['gaiaedr3', 'ps1', 'skymapper', ] if config.get('filter_vizier') else []

            # Filter out catalogue objects
            candidates = pipeline.filter_transient_candidates(
                sobj,
                cat=None, # cat,
                sr=0.5*pixscale*config.get('fwhm', 1.0),
                pixscale=pixscale,
                vizier=vizier,
                # Filter out any flags except for 0x100 which is isophotal masked
                flagged=True, flagmask=0x7e00,
                time=time,
                skybot=config.get('filter_skybot', False),
                verbose=verbose
            )

            # diff[fullmask1] = np.nan # For better visuals

            Ngood = 0
            for cand in candidates:
                cutout = cutouts.get_cutout(
                    image1, cand, config.get('cutout_size', 30),
                    mask=fullmask1,
                    diff=diff,
                    template=tmpl,
                    convolved=conv,
                    err=ediff,
                    footprint=(segm==cand['NUMBER']) if segm is not None else None,
                    time=time,
                    header=header1
                )

                if config.get('filter_adjust'):
                    # Try to apply some sub-pixel adjustments to fix dipoles etc
                    if cutouts.adjust_cutout(
                            cutout, max_shift=1, max_scale=1.3,
                            inner=int(np.ceil(2.0*config.get('fwhm'))),
                            normalize=False, verbose=False
                    ):
                        if cutout['meta']['adjust_pval'] > 0.01:
                            continue
                        if cutout['meta']['adjust_chi2'] < 0.33*cutout['meta']['adjust_chi2_0']:
                            continue

                jname = utils.make_jname(cand['ra'], cand['dec'])
                cutout_name = os.path.join(basepath, 'candidates', jname + '.cutout')

                try:
                    os.makedirs(os.path.join(basepath, 'candidates'))
                except OSError:
                    pass

                cutouts.write_cutout(cutout, cutout_name)

                all_candidates.append(cand)
                cutout_names.append(os.path.join('candidates', jname + '.cutout'))

                Ngood += 1

            if config.get('filter_adjust'):
                log(f"{Ngood} candidates remaining after sub-pixel adjustment routine")

    if subtraction_mode == 'detection':
        log("\n---- Final list of candidates ----\n")

        if len(all_candidates):
            all_candidates = vstack(all_candidates)
            all_candidates['cutout_name'] = cutout_names

            log(f"{len(all_candidates)} candidates in total")

            candidates_extract_unstacked(filename, all_candidates, config, verbose=verbose, show=show)

            all_candidates.write(os.path.join(basepath, 'candidates.vot'), format='votable', overwrite=True)
            log("Candidates written to file:candidates.vot")

            write_ds9_regions(
                os.path.join(basepath, 'candidates.reg'),
                candidates,
                radius=config.get('rel_aper', 1.0)*pixscale*config.get('fwhm')
            )
            log("Candidates written to file:candidates.reg as DS9 regions")

        else:
            log("No candidates found")
