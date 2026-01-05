"""
Photometric calibration and object measurement.
"""

import os

import numpy as np

from django.conf import settings

from astropy.wcs import WCS
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

from stdpipe import astrometry, photometry, catalogs, cutouts
from stdpipe import templates, plots, pipeline, artefacts

from .constants import *
from .utils import *
from .catalogs import *


def photometry_image(filename, config, verbose=True, show=False):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    basepath = os.path.dirname(filename)

    # Image
    image,header = fits.getdata(filename, -1).astype(np.double), fits.getheader(filename, -1)

    fix_header(header)

    # Mask
    mask = fits.getdata(os.path.join(basepath, 'mask.fits'), -1) > 0

    # Custom mask
    if os.path.exists(os.path.join(basepath, 'custom_mask.fits')):
        custom_mask = fits.getdata(os.path.join(basepath, 'custom_mask.fits'), -1) > 0
    else:
        custom_mask = None

    # Time
    time = Time(config.get('time')) if config.get('time') else None

    # Secondary targets - backward compatibility
    if not 'targets' in config and 'target_ra' in config and 'target_dec' in config:
        config['targets'] = [{'ra': config.get('target_ra'), 'dec': config.get('target_dec')}]

    # Cleanup stale plots
    cleanup_paths(cleanup_photometry, basepath=basepath)

    log("\n---- Object detection ----\n")

    # SExtractor does not take mask into account while computing the background
    # so we better mask custom-masked regions manually
    if custom_mask is not None:
        image_masked = image.copy()
        image_masked[custom_mask] = np.nan
    else:
        image_masked = image

    # Extract objects and get segmentation map
    obj,segm,fimg,bg,bgrms = photometry.get_objects_sextractor(
        image_masked, mask=mask,
        aper=config.get('initial_aper', 3.0),
        gain=config.get('gain', 1.0),
        extra={
            'BACK_SIZE': config.get('bg_size', 256),
            'SATUR_LEVEL': config.get('saturation')
        },
        extra_params=['NUMBER', 'MAG_AUTO', 'ISOAREA_IMAGE', 'FLUX_MAX', 'FLUX_AUTO'],
        checkimages=['SEGMENTATION', 'FILTERED', 'BACKGROUND', 'BACKGROUND_RMS'],
        minarea=config.get('minarea', 3),
        r0=config.get('initial_r0', 0.0),
        verbose=verbose,
        _tmpdir=settings.STDPIPE_TMPDIR,
        _exe=settings.STDPIPE_SEXTRACTOR
    )

    # FIXME: Filter some problematic detections
    obj = obj[obj['MAG_AUTO'] < 90]
    obj = obj[obj['fwhm'] > 0]
    obj = obj[obj['ISOAREA_IMAGE'] > config.get('minarea', 3)]

    # Ignore "deblended" flag from SExtractor, for now
    # obj['flags'] &= 0xfffd

    log(f"{len(obj)} objects found")

    fits_write(os.path.join(basepath, 'segmentation.fits'), segm, header, compress=True)
    log("Segmemtation map written to file:segmentation.fits")

    fits_write(os.path.join(basepath, 'filtered.fits'), fimg, header, compress=True)
    log("Filtered image written to file:filtered.fits")

    if config.get('inspect_bg'):
        fits_write(os.path.join(basepath, 'image_bg.fits'), bg, header, compress=True)
        log("Background map written to file:image_bg.fits")

        fits_write(os.path.join(basepath, 'image_rms.fits'), bgrms, header, compress=True)
        log("Background RMS map written to file:image_rms.fits")

    if not len(obj):
        raise RuntimeError('Cannot detect objects in the image')

    log("\n---- Object measurement ----\n")

    # FWHM
    idx = obj['flags'] == 0
    idx &= obj['magerr'] < 1/20

    if config.get('prefilter_detections', True):
        log("Pre-filtering SExtractor detections with simple shape classifier")
        fidx = artefacts.filter_sextractor_detections(obj, verbose=verbose)
        idx &= fidx
        # Also store it in the flags to exclude from photometric match later
        obj['flags'][~fidx] |= 0x800

    if not len(obj[idx]):
        raise RuntimeError("No suitable stars with S/N > 20 in the image!")

    fwhm_values = 2.0*obj['FLUX_RADIUS'] # obj['fwhm']

    fwhm = np.median(fwhm_values[idx]) # TODO: make it position-dependent
    log(f"FWHM is {fwhm:.2f} pixels")

    if config.get('fwhm_override'):
        fwhm = config.get('fwhm_override')
        log(f"Overriding with user-specified FWHM value of {fwhm:.2f} pixels")

    config['fwhm'] = fwhm

    # Plot FWHM map
    with plots.figure_saver(os.path.join(basepath, 'fwhm.png'), figsize=(8, 6), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)
        plots.binned_map(
            obj[idx]['x'], obj[idx]['y'], fwhm_values[idx],
            range=[[0, image.shape[1]], [0, image.shape[0]]],
            bins=8, statistic='median',
            show_dots=True, ax=ax
        )
        ax.set_aspect(1)
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(0, image.shape[0])
        # ax.legend()
        ax.set_title(f"FWHM: median {np.median(fwhm_values[idx]):.2f} pix RMS {np.std(fwhm_values[idx]):.2f} pix")

    # Plot FWHM vs instrumental
    with plots.figure_saver(os.path.join(basepath, 'fwhm_mag.png'), figsize=(8, 6), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(fwhm_values, obj['mag'], '.', label='All objects')
        ax.plot(fwhm_values[idx], obj['mag'][idx], '.', label='Used for FWHM')

        ax.axvline(fwhm, ls='--', color='red')
        ax.invert_yaxis()
        ax.legend()
        ax.grid(alpha=0.2)
        ax.set_title(f"FWHM: median {np.median(fwhm_values[idx]):.2f} pix RMS {np.std(fwhm_values[idx]):.2f} pix")
        ax.set_xlabel('FWHM, pixels')
        ax.set_ylabel('Instrumental magnitude')
        ax.set_xlim(0, np.percentile(fwhm_values, 98))

    log(f"FWHM diagnostic plot stored to file:fwhm_mag.png")

    # Plot pre-filtering diagnostics
    if config.get('prefilter_detections', True):
        with plots.figure_saver(os.path.join(basepath, 'prefilter.png'), figsize=(8, 8), show=show, tight_layout=False) as fig:
            features = artefacts.filter_sextractor_detections(obj, return_features=True)

            plots.cornerplot(
                features,
                # TODO: de-hardcode?..
                scales=['log', 'log', 'log'],
                lines = [fwhm/2, fwhm, None],
                subsets = [
                    {'idx': (obj['flags'] & 0x04 > 0) & (obj['flags'] & 0x100 > 0), 'label':'Saturated', 'color': 'C1'},
                    {'idx': (obj['flags'] & 0x04 == 0) & (obj['flags'] & 0x100 > 0), 'label': 'Cosmics', 'color': 'C3'},
                    {'idx': (obj['flags'] & 0x02 > 0) & (obj['flags'] & 0x100 == 0), 'label': 'Deblended', 'color': 'C4'},
                    {'idx': (obj['flags'] & (0x7fff - 0x800 - 0x100 - 0x04 - 0x02) > 0), 'label': 'Other flags', 'color': 'C5'},
                ],
                extra=lambda ax,col_x,col_y: plots.plot_outline(col_x[0][fidx], col_y[0][fidx], ax=ax, color='red'),
                alpha=0.5,
                fig=fig
            )

            fig.text(
                0.55, 0.4,
                f"Isolation forest outlier detection\n"
                f"{len(obj)} objects\n"
                f"{np.sum(idx)} flagged\n"
                f"{np.sum(fidx)} good {np.sum(~fidx)} outliers\n"
                f"FWHM {fwhm:.2f} pixels",
                ha='left', va='top'
            )

        log("Pre-filtering diagnostic plot stored to file:prefilter.png")

    if config.get('rel_bg1') and config.get('rel_bg2'):
        rel_bkgann = [config['rel_bg1'], config['rel_bg2']]
    else:
        rel_bkgann = None

    # Forced photometry at objects positions
    obj = photometry.measure_objects(obj, image, mask=mask,
                                     fwhm=fwhm,
                                     aper=config.get('rel_aper', 1.0),
                                     bkgann=rel_bkgann,
                                     sn=config.get('sn', 5.0),
                                     bg_size=config.get('bg_size', 256),
                                     gain=config.get('gain', 1.0),
                                     verbose=verbose)

    log(f"{len(obj)} objects properly measured, {np.sum(obj['flags'] == 0)} unflagged")
    if np.sum(obj['flags'] == 0) < 0.5*len(obj):
        log("Warning: more than half of objects are flagged!")

    obj.write(os.path.join(basepath, 'objects.parquet'), overwrite=True)
    log("Measured objects stored to file:objects.parquet")

    # Plot detected objects
    with plots.figure_saver(os.path.join(basepath, 'objects.png'), figsize=(8, 6), show=show,) as fig:
        ax = fig.add_subplot(1, 1, 1)
        idx = obj['flags'] == 0
        ax.plot(obj['x'][idx], obj['y'][idx], '.', label='Unflagged')
        ax.plot(obj['x'][~idx], obj['y'][~idx], '.', label='Flagged')
        ax.set_aspect(1)
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(0, image.shape[0])
        ax.legend()
        ax.set_title(f"Detected objects: {np.sum(idx)} unmasked, {np.sum(~idx)} masked")

    log("\n---- Initial astrometry ----\n")

    # Get initial WCS
    if config['blind_match_wcs']:
        # Blind match WCS
        log("Will try blind matching for WCS solution")

        # Get lowest S/N where we have at least 20 stars
        sn_vals = obj['flux']/obj['fluxerr']
        for sn0 in range(20, 1, -1):
            if np.sum(sn_vals >= sn0) > 20:
                break

        log(f"SN0 = {sn0:.1f}, N0 = {np.sum(sn_vals >= sn0)}")

        if np.sum(sn_vals >= sn0) < 10:
            raise RuntimeError('Too few good objects for blind matching')

        if config.get('blind_match_center'):
            from stdpipe import resolve
            center = resolve.resolve(config.get('blind_match_center'))
            center_ra = center.ra.deg
            center_dec = center.dec.deg
        else:
            center_ra = None
            center_dec = None

        # Exclude pre-filtered detections and limit list size
        obj_bm = obj[(obj['flags'] & 0x800) == 0]
        obj_bm = obj_bm[:500]

        wcs = astrometry.blind_match_objects(
            obj_bm,
            center_ra=center_ra,
            center_dec=center_dec,
            radius=config.get('blind_match_sr0'),
            scale_lower=config.get('blind_match_ps_lo'),
            scale_upper=config.get('blind_match_ps_up'),
            sn=sn0,
            verbose=verbose,
            _tmpdir=settings.STDPIPE_TMPDIR,
            _exe=settings.STDPIPE_SOLVE_FIELD,
            config=settings.STDPIPE_SOLVE_FIELD_CONFIG
        )

        if wcs is not None and wcs.is_celestial:
            astrometry.store_wcs(os.path.join(basepath, "image.wcs"), wcs)
            astrometry.clear_wcs(header)
            header += wcs.to_header(relax=True)
            config['blind_match_wcs'] = False
            config['refine_wcs'] = True # We need to do it as we got SIP solution
            log("Blind matched WCS stored to file:image.wcs")
        else:
            log("Blind matching failed")

    else:
        wcs = get_wcs(filename, header=header, verbose=verbose)

    if wcs is None or not wcs.is_celestial:
        raise RuntimeError('No WCS astrometric solution')

    obj['ra'],obj['dec'] = wcs.all_pix2world(obj['x'], obj['y'], 0)

    log("\n---- Reference catalogue ----\n")

    # Get reference catalogue
    ra0,dec0,sr0 = astrometry.get_frame_center(wcs=wcs, width=image.shape[1], height=image.shape[0])
    pixscale = astrometry.get_pixscale(wcs=wcs)

    log(f"Field center is at {ra0:.3f} {dec0:.3f}, radius {sr0:.2f} deg, scale {3600*pixscale:.2f} arcsec/pix")

    if config.get('cat_name') not in supported_catalogs:
        raise RuntimeError("Unsupported or not specified catalogue")

    filters = {}
    if supported_catalogs[config['cat_name']].get('limit') and config.get('cat_limit'):
        filters[supported_catalogs[config['cat_name']].get('limit')] = f"<{config['cat_limit']}"

    # Round the coordinates a bit to optimize consecutive calls to Vizier after WCS refinement
    ra00,dec00,sr00 = round_coords_to_grid(ra0, dec0, sr0)
    cat = catalogs.get_cat_vizier(ra00, dec00, sr00, config['cat_name'], filters=filters, verbose=verbose)

    if not cat or not len(cat):
        raise RuntimeError('Cannot get catalogue stars')

    log(f"Got {len(cat)} catalogue stars from {config['cat_name']}")

    cat.write(os.path.join(basepath, 'cat.parquet'), overwrite=True)
    log("Catalogue written to file:cat.parquet")

    if config.get('filter_blends', True):
        # TODO: merge blended stars, not remove them!
        cat_filtered = filter_catalogue_blends(cat, 2*fwhm*pixscale)
        log(f"{len(cat_filtered)} catalogue stars after blend filtering with {3600*fwhm*pixscale:.1f} arcsec radius")
        # cat.write(os.path.join(basepath, 'cat_filtered.parquet'), overwrite=True)
        # log("Filtered catalogue written to file:cat_filtered.parquet")
    else:
        cat_filtered = cat

    # Catalogue settings
    config['cat_col_mag'],config['cat_col_mag_err'] = guess_catalogue_mag_columns(
        config['filter'],
        cat,
        augmented_only=True
    )

    if config['cat_col_mag'] in ['Umag', 'Bmag', 'Vmag', 'Rmag', 'Imag']:
        config['cat_col_color_mag1'] = 'Bmag'
        config['cat_col_color_mag2'] = 'Vmag'
    elif config['cat_col_mag'] in ['umag', 'gmag', 'rmag', 'imag']:
        config['cat_col_color_mag1'] = 'gmag'
        config['cat_col_color_mag2'] = 'rmag'
    elif config['cat_col_mag'] in ['zmag', 'ymag']:
        config['cat_col_color_mag1'] = 'rmag'
        config['cat_col_color_mag2'] = 'imag'
    elif config['cat_col_mag'] in ['Gmag', 'BPmag', 'RPmag']:
        config['cat_col_color_mag1'] = 'BPmag'
        config['cat_col_color_mag2'] = 'RPmag'
    else:
        raise RuntimeError(f"Cannot guess magnitude columns for {config.get('cat_name')} and filter {config.get('filter')}")

    log(f"Will use catalogue column {config['cat_col_mag']} as primary magnitude ")
    log(f"Will use catalogue columns {config['cat_col_color_mag1']} and {config['cat_col_color_mag2']} for color")

    if not (config['cat_col_mag'] in cat.colnames and
            (not config.get('cat_col_color_mag1') or config['cat_col_color_mag1'] in cat.colnames) and
            (not config.get('cat_col_color_mag2') or config['cat_col_color_mag2'] in cat.colnames)):
        raise RuntimeError('Catalogue does not have required magnitudes')

    # Astrometric refinement
    if config.get('refine_wcs', False):
        log("\n---- Astrometric refinement ----\n")

        # Exclude pre-filtered detections and limit list size
        obj_ast = obj[(obj['flags'] & 0x800) == 0]

        # FIXME: make the order configurable
        wcs1 = pipeline.refine_astrometry(obj_ast, cat_filtered, fwhm*pixscale,
                                          wcs=wcs, order=config.get('refine_order', 3), method='scamp',
                                          cat_col_mag=config.get('cat_col_mag'),
                                          cat_col_mag_err=config.get('cat_col_mag_err'),
                                          verbose=verbose,
                                          _tmpdir=settings.STDPIPE_TMPDIR,
                                          _exe=settings.STDPIPE_SCAMP)
        if wcs1 is None or not wcs1.is_celestial:
            raise RuntimeError('WCS refinement failed')
        else:
            wcs = wcs1
            obj['ra'],obj['dec'] = wcs.all_pix2world(obj['x'], obj['y'], 0)
            astrometry.store_wcs(os.path.join(basepath, "image.wcs"), wcs)
            astrometry.clear_wcs(header)
            header += wcs.to_header(relax=True)
            config['refine_wcs'] = False
            log("Refined WCS stored to file:image.wcs")

    # Update field center position in the config
    config['field_ra'],config['field_dec'],config['field_sr'] = astrometry.get_frame_center(
        wcs=wcs, width=image.shape[1], height=image.shape[0]
    )
    config['pixscale'] = astrometry.get_pixscale(wcs=wcs)

    log("\n---- Photometric calibration ----\n")

    sr = config.get('sr_override')
    if sr:
        sr /= 3600 # Arcseconds to degrees

    # Photometric calibration
    m = pipeline.calibrate_photometry(
        obj, cat_filtered, sr=sr, pixscale=pixscale,
        cat_col_mag=config.get('cat_col_mag'),
        cat_col_mag_err=config.get('cat_col_mag_err'),
        cat_col_mag1=config.get('cat_col_color_mag1'),
        cat_col_mag2=config.get('cat_col_color_mag2'),
        use_color=config.get('use_color', True),
        force_color_term=config.get('force_color_term'),
        order=config.get('spatial_order', 0),
        bg_order=config.get('bg_order', None),
        nonlin=config.get('nonlin', False),
        robust=True, scale_noise=True,
        accept_flags=0x02, max_intrinsic_rms=0.01,
        verbose=verbose
    )

    if m is None:
        raise RuntimeError('Photometric match failed')

    # Check photometric correlation (instrumental vs catalogue)
    if True:
        a0, b0 = (m['omag']+m['zero_model'])[m['idx0']], m['cmag'][m['idx0']]
        c0 = np.corrcoef(a0, b0)[0, 1]

        b1 = np.array(b0.copy())
        cs = []

        for i in range(10000):
            np.random.shuffle(b1)
            cs.append(np.corrcoef(a0, b1)[0, 1])

        from scipy import stats
        qval = stats.percentileofscore(np.abs(cs), np.abs(c0))
        pval = 1 - 0.01*qval

        log(f"Instr / Cat correlation is {c0:.2f} which corresponds to p-value {pval:.2g}")
        if pval > 0.05:
            log(f"Warning: the correlation is not significant, probably the astrometry is wrong!")

    # Store photometric solution
    pickle_to_file(os.path.join(basepath, 'photometry.pickle'), m)
    log("Photometric solution stored to photometry.pickle")

    # Plot photometric solution
    with plots.figure_saver(os.path.join(basepath, 'photometry.png'), figsize=(8, 6), show=show) as fig:
        ax = fig.add_subplot(2, 1, 1)
        plots.plot_photometric_match(m, mode='mag', ax=ax)
        ax = fig.add_subplot(2, 1, 2)
        plots.plot_photometric_match(m, mode='color', ax=ax)

    with plots.figure_saver(os.path.join(basepath, 'photometry_unmasked.png'), figsize=(8, 6), show=show) as fig:
        ax = fig.add_subplot(2, 1, 1)
        plots.plot_photometric_match(m, mode='mag', show_masked=False, ax=ax)
        ax.set_ylim(-0.4, 0.4)
        ax = fig.add_subplot(2, 1, 2)
        plots.plot_photometric_match(m, mode='color', show_masked=False, ax=ax)
        ax.set_ylim(-0.4, 0.4)

    with plots.figure_saver(os.path.join(basepath, 'photometry_zeropoint.png'), figsize=(8, 6), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)
        plots.plot_photometric_match(m, mode='zero', show_dots=True, bins=8, ax=ax,
                                     range=[[0, image.shape[1]], [0, image.shape[0]]])
        ax.set_aspect(1)
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(0, image.shape[0])

    with plots.figure_saver(os.path.join(basepath, 'photometry_model.png'), figsize=(8, 6), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)
        plots.plot_photometric_match(m, mode='model', show_dots=True, bins=8, ax=ax,
                                     range=[[0, image.shape[1]], [0, image.shape[0]]])
        ax.set_aspect(1)
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(0, image.shape[0])

    with plots.figure_saver(os.path.join(basepath, 'photometry_residuals.png'), figsize=(8, 6), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)
        plots.plot_photometric_match(m, mode='residuals', show_dots=True, bins=8, ax=ax,
                                     range=[[0, image.shape[1]], [0, image.shape[0]]])
        ax.set_aspect(1)
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(0, image.shape[0])

    with plots.figure_saver(os.path.join(basepath, 'astrometry_dist.png'), figsize=(8, 6), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)
        plots.plot_photometric_match(m, mode='dist', show_dots=True, bins=8, ax=ax,
                                     range=[[0, image.shape[1]], [0, image.shape[0]]])
        ax.set_aspect(1)
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(0, image.shape[0])

    # Apply photometry to objects
    # (It should already be done in calibrate_photometry(), but let's be verbose
    zp = m['zero_fn'](obj['x'], obj['y'], obj['mag'])
    zp_err = m['zero_fn'](obj['x'], obj['y'], obj['mag'], get_err=True)
    obj['mag_calib'] = obj['mag'] + zp
    obj['mag_calib_err'] = np.hypot(obj['magerr'], zp_err)

    obj['mag_filter_name'] = m['cat_col_mag']

    if 'cat_col_mag1' in m.keys() and 'cat_col_mag2' in m.keys():
        obj['mag_color_name'] = '%s - %s' % (m['cat_col_mag1'], m['cat_col_mag2'])
    if m['color_term'] is not None:
        obj['mag_color_term'] = [m['color_term']]*len(obj)

    log(f"Mean zero point is {np.mean(zp):.3f}, estimated error {np.mean(zp_err):.2g}")

    obj.write(os.path.join(basepath, 'objects.parquet'), overwrite=True)
    log("Measured objects stored to file:objects.parquet")

    # Check the filter
    if (config.get('use_color', True) and np.any(np.abs(m['color_term']) > 0.5)) or config.get('diagnose_color'):
        if config.get('diagnose_color'):
            log("Running color term diagnostics for all possible filters")
        else:
            log("Warning: color term is too large, checking whether other filters would work better")

        for fname in supported_catalogs[config['cat_name']].get('filters', []):
            m1 = pipeline.calibrate_photometry(
                obj, cat, pixscale=pixscale,
                cat_col_mag=fname + 'mag',
                cat_col_mag_err='e_' + fname + 'mag',
                cat_col_mag1=config.get('cat_col_color_mag1'),
                cat_col_mag2=config.get('cat_col_color_mag2'),
                use_color=True,
                order=config.get('spatial_order', 0),
                robust=True, scale_noise=True,
                accept_flags=0x02, max_intrinsic_rms=0.01,
                verbose=False)

            if m1 is not None:
                log(f"filter {fname}: color term {photometry.format_color_term(m1['color_term'])}")
            else:
                log(f"filter {fname}: match failed")

    # Detection limits
    log("\n---- Global detection limit ----\n")
    sns = [10, 5, 3]
    if config.get('sn', 5) not in sns:
        sns.append(config.get('sn', 5))
    for sn in sns:
        # Just print the value
        mag0 = pipeline.get_detection_limit(obj, sn=sn, verbose=False)
        log(f"Detection limit at S/N={sn:.0f} level is {mag0:.2f}")

    mag0 = pipeline.get_detection_limit(obj, sn=config.get('sn'), verbose=False)
    config['mag_limit'] = mag0

    if 'bg_fluxerr' in obj.colnames and np.any(obj['bg_fluxerr'] > 0):
        fluxerr = obj['bg_fluxerr']
        sn = config.get('sn', 5)
        maglim = -2.5*np.log10(sn*fluxerr) + m['zero_fn'](obj['x'], obj['y'], obj['mag'])
        maglim = maglim[np.isfinite(maglim)] # Remove Inf and NaN
        log(f"Local background RMS detection limit (S/N={sn:.0f}) is {np.nanmedian(maglim):.2f} +/- {np.nanstd(maglim):.2f}")

    # Plot detection limit estimators
    with plots.figure_saver(os.path.join(basepath, 'limit_hist.png'), figsize=(8, 6), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)
        # Filter out catalogue stars outside the image
        cx,cy = wcs.all_world2pix(cat['RAJ2000'], cat['DEJ2000'], 0)
        cat_idx = (cx > 0) & (cy > 0) & (cx < image.shape[1]) & (cy < image.shape[0])
        plots.plot_mag_histogram(obj, cat[cat_idx], cat_col_mag=config['cat_col_mag'], sn=config.get('sn'), ax=ax)

    with plots.figure_saver(os.path.join(basepath, 'limit_sn.png'), figsize=(8, 6), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)
        plots.plot_detection_limit(obj, mag_name=config['cat_col_mag'], sn=config.get('sn', 3), ax=ax)

    # Target forced photometry
    if config.get('targets'):
        log("\n---- Primary and secondary targets forced photometry ----\n")

        target_obj = Table({
            'ra': [_['ra'] for _ in config['targets']],
            'dec': [_['dec'] for _ in config['targets']]
        })
        target_obj['x'],target_obj['y'] = wcs.all_world2pix(target_obj['ra'], target_obj['dec'], 0)

        # Filter out targets outside the image so that photometry routine does not crash
        def is_inside(x, y):
            return (x > 0 and x < image.shape[1] and y > 0 and y < image.shape[0])

        if not is_inside(target_obj['x'][0], target_obj['y'][0]):
            raise RuntimeError("Primary target is outside the image")

        log(f"Primary target position is {target_obj['ra'][0]:.3f} {target_obj['dec'][0]:.3f} -> {target_obj['x'][0]:.1f} {target_obj['y'][0]:.1f}")

        target_obj = photometry.measure_objects(target_obj, image, mask=mask,
                                                fwhm=fwhm,
                                                aper=config.get('rel_aper', 1.0),
                                                bkgann=rel_bkgann,
                                                sn=0,
                                                bg_size=config.get('bg_size', 256),
                                                gain=config.get('gain', 1.0),
                                                centroid_iter=5 if config.get('centroid_targets') else 0,
                                                verbose=verbose)

        target_obj['mag_calib'] = target_obj['mag'] + m['zero_fn'](target_obj['x'],
                                                                   target_obj['y'],
                                                                   target_obj['mag'])

        target_obj['mag_calib_err'] = np.hypot(target_obj['magerr'],
                                               m['zero_fn'](target_obj['x'],
                                                            target_obj['y'],
                                                            target_obj['mag'],
                                                            get_err=True))

        if config.get('centroid_targets'):
            # Centroiding might change target pixel positions - let's update sky positions too
            target_obj['ra_orig'],target_obj['dec_orig'] = target_obj['ra'],target_obj['dec']
            target_obj['ra'],target_obj['dec'] = wcs.all_pix2world(target_obj['x'], target_obj['y'], 0)

        # Local detection limit from background rms, if available
        if 'bg_fluxerr' in target_obj.colnames and np.any(target_obj['bg_fluxerr'] > 0):
            fluxerr = target_obj['bg_fluxerr']
        else:
            fluxerr = target_obj['fluxerr']
        target_obj['mag_limit'] = -2.5*np.log10(config.get('sn', 5)*fluxerr) + m['zero_fn'](target_obj['x'], target_obj['y'], target_obj['mag'])

        target_obj['mag_filter_name'] = m['cat_col_mag']

        if 'cat_col_mag1' in m.keys() and 'cat_col_mag2' in m.keys():
            target_obj['mag_color_name'] = '%s - %s' % (m['cat_col_mag1'], m['cat_col_mag2'])
            target_obj['mag_color_term'] = [m['color_term']]*len(target_obj)

        target_obj.write(os.path.join(basepath, 'target.vot'), format='votable', overwrite=True)
        log("Measured targets stored to file:target.vot")

        # Create the cutouts from image based on the targets
        for i,tobj in enumerate(target_obj):
            cutout_name = f"targets/target_{i:04d}.cutout"
            target_title = "Primary target" if i == 0 else f"Secondary target {i}"

            if (not np.isfinite(tobj['x']) or not np.isfinite(tobj['y']) or
                tobj['x'] < 0 or tobj['y'] < 0 or
                tobj['x'] > image.shape[1] or tobj['y'] > image.shape[0]):
                log(f"{target_title} is outside image, skipping")
                continue

            cutout = cutouts.get_cutout(
                image, tobj, config.get('cutout_size', 30),
                mask=mask,
                header=header,
                time=time,
                # filtered=fimg if config.get('initial_r0') else None,
            )
            # Cutout from relevant HiPS survey
            cutout['template'] = templates.get_hips_image(
                guess_hips_survey(tobj['ra'], tobj['dec'], config['filter']),
                header=cutout['header'],
                get_header=False
            )

            try:
                os.makedirs(os.path.join(basepath, 'targets'))
            except OSError:
                pass

            cutouts.write_cutout(cutout, os.path.join(basepath, cutout_name))
            log(f"{target_title} cutouts stored to file:{cutout_name}")

            log(f"{target_title} flux is {tobj['flux']:.1f} +/- {tobj['fluxerr']:.1f} ADU")
            if tobj['flux'] > 0:
                mag_string = tobj['mag_filter_name']
                if 'mag_color_name' in target_obj.colnames and 'mag_color_term' in target_obj.colnames and tobj['mag_color_term'] is not None:
                    mag_string += ' ' + photometry.format_color_term(tobj['mag_color_term'], color_name=tobj['mag_color_name'])

                log(f"{target_title} magnitude is {mag_string} = {tobj['mag_calib']:.2f} +/- {tobj['mag_calib_err']:.2f}")
                log(f"{target_title} detected with S/N = {1/tobj['mag_calib_err']:.2f}")

            else:
                log(f"{target_title} not detected")


def candidates_extract_unstacked(filename, candidates, config, verbose=True, show=False):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    basepath = os.path.dirname(filename)

    if not config.get('stack_filenames'):
        return

    log('Augmenting candidates with cutouts from original unstacked images')

    stack_cutouts = {}

    for sfilename in config['stack_filenames']:
        if os.path.exists(sfilename):
            simage,sheader = fits.getdata(sfilename, header=True)
            swcs = WCS(sheader)

            for cand in candidates:
                cid = cand['cutout_name']

                if cid not in stack_cutouts:
                    stack_cutouts[cid] = []

                sx0,sy0 = swcs.all_world2pix(cand['ra'], cand['dec'], 0)

                cut = cutouts.crop_image_centered(simage, sx0, sy0, config.get('cutout_size', 30))
                stack_cutouts[cid].append(cut)

    unstack_names = []

    for cand in candidates:
        cutout_name = cand['cutout_name']
        unstack_name = os.path.splitext(cutout_name)[0] + '.unstack.png'
        outname = os.path.join(basepath, unstack_name)

        scuts = stack_cutouts.get(cutout_name, [])

        if scuts:
            N = len(scuts)
            Nx = 5
            Ny = int(np.ceil(N/Nx))

            with plots.figure_saver(outname, figsize=(2*Nx, 2*Ny), show=show) as fig:
                for i,cut in enumerate(scuts):
                    ax = fig.add_subplot(Ny, Nx, i+1)
                    plots.imshow(
                        cut, qq=[0.5, 99.5], ax=ax,
                        cmap='Blues_r',
                        show_axis=False, show_colorbar=False, interpolation='nearest'
                    )

                fig.tight_layout()

            unstack_names.append(unstack_name)
        else:
            unstack_names.append(None)

    candidates['unstack_name'] = unstack_names
