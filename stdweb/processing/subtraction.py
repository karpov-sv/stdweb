"""
Template subtraction and difference image analysis.
"""

import os
from functools import partial

import numpy as np

try:
    import sep_x as sep
except ImportError:
    import sep
import reproject

from django.conf import settings

from astropy.wcs import WCS
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.time import Time
from astropy.stats import sigma_clip, mad_std, gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel, convolve_fft

from scipy.optimize import brentq

from stdpipe import astrometry, photometry, cutouts
from stdpipe import templates, subtraction, plots, pipeline
from stdpipe import resolve, utils, artefacts

from .constants import *
from .utils import *
from .catalogs import *
from .coverage import *
from .photometry import *


# Plausible range for the coefficient mixing the two template bands
max_mixing_coeff = 5


def fit_robust_trend(y, *args, yerr=None, sigma=5):
    """Weighted linear regression of `y` over the regressors given as positional
    arguments, with the outliers sigma-clipped away after the initial fit.

    Returns the constant term followed by the slopes, one per regressor, or None if the
    fit may not be performed.
    """
    y = np.asarray(y, dtype=float)
    A = np.column_stack([np.ones_like(y)] + [np.asarray(_, dtype=float) for _ in args])

    idx = np.isfinite(y) & np.all(np.isfinite(A), axis=1)
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=float)
        idx &= np.isfinite(yerr) & (yerr > 0)

    w = 1/yerr[idx] if yerr is not None else np.ones(np.sum(idx))

    def solve(sidx):
        if np.sum(sidx) < A.shape[1] + 2:
            return None
        return np.linalg.lstsq(A[idx][sidx]*w[sidx][:,None], y[idx][sidx]*w[sidx],
                               rcond=None)[0]

    res = solve(np.ones(np.sum(idx), dtype=bool))
    if res is None:
        return None

    # Refit without the outliers
    # Median absolute deviation, so that the outliers do not inflate the threshold
    cidx = ~sigma_clip(y[idx] - A[idx].dot(res), sigma=sigma, maxiters=3,
                       stdfunc='mad_std').mask

    return solve(cidx) if np.sum(cidx) < np.sum(idx) else res


def measure_template_fwhm(tmpl, mask=None, bg_size=256, gain=10000,
                          segmentation=False, verbose=True):
    """Detect the objects in the template image and estimate its median FWHM in pixels.

    Returns the FWHM, the detections, the mask of the ones used for the estimation, and
    the segmentation map. The latter is only produced when `segmentation` is set, as
    SExtractor has to write a full sized image for it, and we have to read it back.
    """
    res = photometry.get_objects_sextractor(
        tmpl, mask=mask, sn=5, gain=gain,
        extra={
            'BACK_SIZE': bg_size,
            # 'SATUR_LEVEL': template_saturation,
        },
        extra_params=['NUMBER', 'MAG_AUTO', 'ISOAREA_IMAGE', 'FLUX_MAX', 'FLUX_AUTO'],
        checkimages=['SEGMENTATION'] if segmentation else [],
        _tmpdir=settings.STDPIPE_TMPDIR,
        _exe=settings.STDPIPE_SEXTRACTOR
    )

    tobj,tsegm = res if segmentation else (res, None)

    tidx = tobj['flags'] == 0
    if len(tobj) > 20:
        tidx &= artefacts.filter_sextractor_detections(tobj, verbose=verbose)

    return np.median(2.0*tobj['FLUX_RADIUS'][tidx]), tobj, tidx, tsegm


def fit_mixing_coefficient(tmpls, tmask, obj, m, origin, shape, fwhm,
                           min_stars=10, bg_size=256, gain=10000, verbose=True):
    """Fit the `coeff` of `F_image = F1 + coeff*F2` on the catalogue stars used for the
    photometric calibration, as stored in `m`, by measuring them on the two PSF matched
    templates given in `tmpls`.

    Returns the coefficient, or None if it may not be fitted.
    """
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    oidx = np.asarray(m.get('oidx', []), dtype=int)

    if not len(oidx):
        return None

    x0,y0 = origin

    xs = np.asarray(obj['x'], dtype=float)[oidx] - x0
    ys = np.asarray(obj['y'], dtype=float)[oidx] - y0
    flux0 = np.asarray(obj['flux'], dtype=float)[oidx]

    margin = 3*fwhm
    idx = (xs > margin) & (ys > margin) & (xs < shape[1] - margin) & (ys < shape[0] - margin)
    idx &= np.isfinite(flux0) & (flux0 > 0) & np.asarray(m['idx'])

    if np.sum(idx) < min_stars:
        log("Too few catalogue stars in the sub-image to fit the mixing coefficient")
        return None

    # The same aperture on both PSF matched templates, so that the ratio of the two
    # fluxes is the ratio of the actual stellar fluxes in the two filters
    fluxes = [
        photometry.measure_objects(
            Table({'x': xs[idx], 'y': ys[idx]}), tmpl_, mask=tmask,
            fwhm=fwhm, aper=2.0, bkgann=[5, 7],
            sn=0, bg=None, err=None, bg_size=bg_size, gain=gain,
            verbose=False
        ) for tmpl_ in tmpls
    ]

    fidx = np.all([(_['flags'] == 0) & np.isfinite(_['flux']) & (_['flux'] > 0)
                   for _ in fluxes], axis=0)

    if np.sum(fidx) < min_stars:
        log("Too few stars measurable on both templates to fit the mixing coefficient")
        return None

    f0 = flux0[idx][fidx]
    f1,f2 = [np.asarray(_['flux'], dtype=float)[fidx] for _ in fluxes]
    color = None
    if m.get('color') is not None:
        color = np.asarray(m['color'], dtype=float)[idx][fidx]

    A = np.column_stack([f1, f2])
    # Equal weight per star in magnitude space
    w = 1/f0
    good = np.ones(len(f0), dtype=bool)

    for _ in range(5):
        c = np.linalg.lstsq(A[good]*w[good,None], f0[good]*w[good], rcond=None)[0]
        res = 1 - A.dot(c)/f0
        good1 = good & (np.abs(res) < 3*mad_std(res[good]))
        if np.all(good1 == good) or np.sum(good1) < min_stars:
            break
        good = good1

    if not (c[0] > 0 and 0 < c[1]/c[0] < max_mixing_coeff):
        log(f"Warning: fitted mixing coefficients {c[0]:.3g} and {c[1]:.3g} "
            "are not usable")
        return None

    coeff = c[1]/c[0]

    # The least squares solution above minimizes the scatter of the residuals, while
    # what the combination is actually for is nulling their dependence on the color.
    # The latter is monotonically decreasing with the coefficient, so it may just be
    # solved for.
    if color is not None and np.all(np.isfinite(color[good])) and np.ptp(color[good]) > 0.1:
        def color_slope(coeff_):
            ratio = (f1 + coeff_*f2)/f0
            return np.polyfit(color[good], (1 - ratio/np.median(ratio[good]))[good], 1)[0]

        if color_slope(0) > 0 > color_slope(max_mixing_coeff):
            coeff = brentq(color_slope, 0, max_mixing_coeff)

    ratio = (f1 + coeff*f2)/f0
    scatter = mad_std((1 - ratio/np.median(ratio[good]))[good])

    log(f"Mixing coefficient fitted as {coeff:.3f} on {np.sum(good)} stars, "
        f"they are reproduced to {scatter*100:.2f}%")

    return coeff


def combine_templates(tmpl1, tmask1, tmpl2, tmask2, obj, m, origin, shape,
                      coeff=None, bg_size=256, gain=10000, verbose=True):
    """Build a synthetic band template as a linear combination `T1 + coeff*T2` of the
    templates in the two filters bracketing the filter of the image.

    The combination has to be linear in flux, not in magnitudes, so that it stays
    additive - only then the blends, the galaxies and the transients on top of them
    combine the same way the individual stars do.

    The two templates are first brought to a common PSF by blurring the sharper one, as
    otherwise the combination would have a color dependent PSF. The mixing coefficient
    is then fitted on the catalogue stars by `fit_mixing_coefficient` below, and then
    refined so that the color dependence of the residuals is nulled, rather than just
    their scatter. Fitting it, instead of using the tabulated value, adapts
    the synthetic band to the actual bandpass of the image, including its color term,
    and needs no assumptions on the photometric scale of the two templates.

    `coeff`, if provided, is the fallback used when the fit may not be performed.

    Returns the combined template, the union of the two masks, and the coefficient
    actually used, or None if the combination may not be built at all.
    """
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    tmask = tmask1 | tmask2

    # Bring both templates to the same PSF by blurring the sharper one
    fwhm1 = measure_template_fwhm(tmpl1, mask=tmask1, bg_size=bg_size, gain=gain,
                                  verbose=False)[0]
    fwhm2 = measure_template_fwhm(tmpl2, mask=tmask2, bg_size=bg_size, gain=gain,
                                  verbose=False)[0]

    if not np.isfinite(fwhm1) or not np.isfinite(fwhm2):
        log("Warning: cannot estimate the FWHM of the templates, not combining them")
        return None

    fwhm0 = max(fwhm1, fwhm2)
    log(f"Template FWHM is {fwhm1:.2f} pix and {fwhm2:.2f} pix in the two filters")

    tmpls = []
    for tmpl_,tmask_,fwhm_ in [(tmpl1, tmask1, fwhm1), (tmpl2, tmask2, fwhm2)]:
        if fwhm0**2 - fwhm_**2 > 0.01:
            sigma = gaussian_fwhm_to_sigma*np.sqrt(fwhm0**2 - fwhm_**2)
            log(f"Blurring the template with sigma {sigma:.2f} pix to match FWHM {fwhm0:.2f} pix")
            tmpl_ = tmpl_.copy()
            tmpl_[tmask_] = np.nan
            tmpl_ = convolve_fft(tmpl_, Gaussian2DKernel(sigma), nan_treatment='interpolate',
                                 preserve_nan=False, allow_huge=True)
        tmpls.append(tmpl_)

    fit_coeff = fit_mixing_coefficient(tmpls, tmask, obj, m, origin, shape, fwhm0,
                                       bg_size=bg_size, gain=gain, verbose=verbose)

    if fit_coeff is not None:
        coeff = fit_coeff
    elif coeff is None:
        log("No mixing coefficient available, not combining the templates")
        return None
    else:
        log(f"Falling back to the tabulated mixing coefficient {coeff:.3f}")

    return tmpls[0] + coeff*tmpls[1], tmask, coeff


def plot_subtraction_photometry(filename, diff, obj, m, origin, shape, config, psf=None,
                                mask=None, err=None, show=False, verbose=True):
    """Measure the objects in the difference image and plot what is left of their
    original fluxes there, versus calibrated magnitude and catalogue color.

    Forced photometry is repeated on the difference image in the same way as the original
    one - with the method and aperture from `config`, and the PSF model `psf` - so that
    the ratio of the two fluxes is directly the fraction of flux that failed to subtract.
    The ratio is kept as a fraction, as close to zero it is directly the magnitude error
    the imperfect subtraction introduces.

    The magnitude panel needs no catalogue - `mag_calib` is known for every detection -
    so it uses all the unflagged objects inside the sub-image, and stays informative
    down to the detection limit of the image rather than to the limit of the reference
    catalogue. The color one is necessarily restricted to the stars matched during the
    photometric calibration, as stored in `m`, and fits the color dependence jointly
    with the magnitude one, so that the two correlated trends are not confused. The
    magnitude trend of the deeper sample is instead a marginal one, and absorbs
    whatever color dependence there is.

    Sparse fields are handled by degrading the diagnostics instead of dropping them:
    the median needs just `min_stars` objects, fitting a trend needs `min_stars_mag_fit`
    of them, and the joint fit of the color one needs `min_stars_fit` catalogue stars.

    Returns the dict with the median ratio (`median`), its trends with magnitude
    (`mag_slope`) and color (`color_slope`), either of which may be None, the number of
    objects the former is measured over (`nstars`) and of catalogue stars behind the
    latter (`ncolor`). Returns None if the diagnostics may not be built.
    """
    # Objects needed to show the points and the median, to fit a trend, and to fit the
    # magnitude and color ones jointly
    min_stars = 3
    min_stars_mag_fit = 5
    min_stars_fit = 10

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    if 'mag_calib' not in obj.colnames:
        return None

    x0,y0 = origin

    xs = np.asarray(obj['x'], dtype=float) - x0
    ys = np.asarray(obj['y'], dtype=float) - y0

    flux0 = np.asarray(obj['flux'], dtype=float)
    mag0 = np.asarray(obj['mag_calib'], dtype=float)
    flags = np.asarray(obj['flags'], dtype=int) if 'flags' in obj.colnames \
        else np.zeros(len(obj), dtype=int)

    # Catalogue color, and the stars actually used for the photometric calibration
    color = np.full(len(obj), np.nan)
    iscat = np.zeros(len(obj), dtype=bool)
    oidx = np.asarray(m.get('oidx', []), dtype=int)

    if len(oidx):
        color[oidx] = np.asarray(m['color'], dtype=float)
        iscat[oidx[np.asarray(m['idx'], dtype=bool)]] = True

    # Objects well inside the sub-image, with usable original photometry. The flagged
    # ones are dropped, except for the calibration stars, that are selected as they
    # were during the calibration itself.
    margin = 2*config.get('rel_aper', 1.0)*config.get('fwhm', 3.0)
    idx = (xs > margin) & (ys > margin) & (xs < shape[1] - margin) & (ys < shape[0] - margin)
    idx &= np.isfinite(flux0) & (flux0 > 0) & np.isfinite(mag0)
    idx &= (flags == 0) | iscat

    if np.sum(idx) < min_stars:
        log(f"Only {np.sum(idx)} objects in the sub-image, too few for "
            "subtraction photometry diagnostics")
        return None

    # Forced photometry in the difference image, the same way as the original one
    sobj = measure_photometry(
        Table({'x': xs[idx], 'y': ys[idx]}), diff, config, psf=psf,
        mask=mask,
        sn=0,
        # We assume no background, and known error model
        bg=None,
        err=err,
        verbose=False
    )

    ratio = np.asarray(sobj['flux'], dtype=float)/flux0[idx]
    ratio_err = np.asarray(sobj['fluxerr'], dtype=float)/flux0[idx]

    mag0,color,iscat = mag0[idx],color[idx],iscat[idx]
    unflagged = flags[idx] == 0
    # Properly measured in the difference image
    flagged = np.asarray(sobj['flags']) != 0

    # The deep sample behind the magnitude panel, and the catalogue one behind the color
    good = ~flagged & unflagged
    cgood = ~flagged & iscat & np.isfinite(color)

    if np.sum(good) < min_stars:
        log(f"Only {np.sum(good)} properly measured objects, too few for "
            "subtraction photometry diagnostics")
        return None

    median = np.median(ratio[good])

    has_color = np.sum(cgood) >= min_stars and np.ptp(color[cgood]) > 0

    # Magnitude and color are correlated in most of the fields, so on the catalogue
    # stars, where both are known, they are fitted jointly and shown in the plots as
    # partial dependences. The magnitude trend of the deeper sample is a marginal one.
    fit = fit_robust_trend(ratio[good], mag0[good], yerr=ratio_err[good]) \
        if np.sum(good) >= min_stars_mag_fit else None

    cfit = fit_robust_trend(ratio[cgood], mag0[cgood], color[cgood], yerr=ratio_err[cgood]) \
        if has_color and np.sum(cgood) >= min_stars_fit else None

    mag_median = np.median(mag0[cgood]) if has_color else None

    log(f"Subtraction leaves {median:+.3f} of flux inside the aperture, "
        f"median over {np.sum(good)} objects down to {np.max(mag0[good]):.1f} mag")
    if fit is not None:
        log(f"It depends on the magnitude as {fit[1]:+.3f} per mag"
            + (", marginally, as the color dependence is folded into it"
               if cfit is not None else ""))
    if cfit is not None:
        log(f"It depends on the color as {cfit[2]:+.3f} per mag of color, "
            f"over {np.sum(cgood)} catalogue stars")

    # Every panel is its values, the sample behind its fit, the fit itself, the mask of
    # everything worth showing, whether the fit is a partial dependence or a marginal
    # one, and the labels
    panels = [(mag0, good, fit, unflagged, False, 'All detections',
               f"Calibrated {m.get('cat_col_mag', '')} magnitude")]
    if has_color:
        panels.append(
            (color, cgood, cfit, iscat, True, 'All matched stars',
             f"Catalogue {m.get('cat_col_mag1', '')} - {m.get('cat_col_mag2', '')} color")
        )

    with plots.figure_saver(filename, figsize=(8, 6 if len(panels) > 1 else 4), show=show) as fig:
        for i,(x,sel,pfit,base,partial,alllabel,xlabel) in enumerate(panels):
            ax = fig.add_subplot(len(panels), 1, i + 1)

            shown = ~flagged & base

            # The error bars are drawn for everything shown, but only labelled
            # separately when there is something beyond the sample behind the fit
            ax.errorbar(x[shown], ratio[shown], ratio_err[shown], fmt='.', alpha=0.3,
                        label=alllabel if np.sum(shown & ~sel) else None)
            ax.plot(x[sel], ratio[sel], '.', color='red',
                    label='Used for the fit' if pfit is not None else 'Used for the median')
            if np.sum(flagged):
                ax.plot(x[flagged], ratio[flagged], 'x', color='orange', label='Masked in diff')

            ax.axhline(0, ls='--', color='black', alpha=0.3)

            if pfit is not None:
                # The partial dependence keeps the magnitude at its median
                xx = np.linspace(np.nanmin(x[sel]), np.nanmax(x[sel]), 100)
                yy = pfit[0] + (pfit[1]*mag_median if partial else 0) + pfit[-1]*xx

                ax.plot(xx, yy, '--', color='blue', alpha=0.3,
                        label=f"{pfit[-1]:+.3f} per mag"
                        + ("" if partial or cfit is None else ", marginal"))

            # Robust limits, keeping the zero level visible. The deep sample also holds
            # the objects that are not in the template at all, and thus stay in the
            # difference in full - they are for the transient detection to find, and
            # here they would just squeeze everything else into a line.
            lo,hi = np.percentile(ratio[sel], [1, 99])
            center,scale = np.median(ratio[sel]),mad_std(ratio[sel])
            if np.isfinite(scale) and scale > 0:
                lo,hi = max(lo, center - 5*scale), min(hi, center + 5*scale)
            pad = 0.2*max(hi - lo, 0.01)
            ax.set_ylim(min(0, lo) - pad, max(0, hi) + pad)

            ax.legend()
            ax.grid(alpha=0.2)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Flux left in diff, fraction')

            if not partial:
                ax.set_title(f"Subtraction photometry: {np.sum(good)} objects, "
                             f"{median:+.3f} of flux left in the difference image")

    return {'median': median,
            'nstars': int(np.sum(good)), 'ncolor': int(np.sum(cgood)),
            'mag_slope': fit[1] if fit is not None else None,
            'color_slope': cfit[2] if cfit is not None else None}


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

    # PSF model, for the photometry to be consistent with the one used for calibration
    psf_model = None
    if get_photometry_method(config) == 'psf':
        if not os.path.exists(os.path.join(basepath, 'psf.pickle')):
            raise RuntimeError('PSF photometry requested, but no PSF model is available - please re-run photometry')
        psf_model = pickle_from_file(os.path.join(basepath, 'psf.pickle'))

        if subtraction_method != 'sfft' and (config.get('hotpants_extra') or {}).get('c', 't') != 't':
            log("Warning: HOTPANTS is set to not convolve the template, "
                "the difference image PSF will not match the PSF model")

    # Objects
    obj = Table.read(os.path.join(basepath, 'objects.parquet'))

    # Catalogue
    cat = Table.read(os.path.join(basepath, 'cat.parquet'))

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

    # Template filter, and the second one used to synthesize the band closer to the one
    # of the image. Both stay unset for the custom template.
    tfilter = None
    tfilter2,tcoeff = None,None

    if tname == 'custom':
        log("Using custom template from file:custom_template.fits")

        if not os.path.exists(os.path.join(basepath, 'custom_template.fits')):
            raise RuntimeError("Custom template not found")

        custom_template = fits.getdata(os.path.join(basepath, 'custom_template.fits'), -1).astype(np.double)
        custom_header = fits.getheader(os.path.join(basepath, 'custom_template.fits'), -1)
        fix_header(custom_header, verbose=verbose)
        custom_wcs = WCS(custom_header)

        template_gain = config.get('custom_template_gain', 10000)
        template_saturation = config.get('custom_template_saturation', None)

        custom_mask = np.isnan(custom_template)
        if template_saturation:
            custom_mask |= custom_template >= template_saturation

        # Load custom template mask if exists
        template_mask_file = os.path.join(basepath, 'custom_template_mask.fits')
        if os.path.exists(template_mask_file):
            custom_template_mask = fits.getdata(template_mask_file, -1) > 0
            custom_mask |= custom_template_mask
            log("Custom template mask loaded from file:custom_template_mask.fits")

    else:
        if tconf is None:
            raise RuntimeError(f"Unsupported template: {tname}")

        for _ in filter_mappings[config['filter']]:
            if _ in tconf['filters']:
                tfilter = _
                break

        template_gain = 10000 # Assume effectively noise-less

        log(f"Using {tconf['name']} in filter {tfilter} as a template")

        if config.get('template_combine', True):
            combination = filter_combinations.get(config['filter'])

            if combination and combination[0] == tfilter and combination[1] in tconf['filters']:
                tfilter2 = combination[1]
                # Tabulated coefficients are on a common photometric scale, so they are
                # only meaningful for the surveys whose bands share it
                tcoeff = combination[2] if tconf.get('common_zp') else None

                log(f"Will combine it with the {tfilter2} band to better match "
                    f"the {config['filter']} filter of the image")

        if tname in ['ps1', 'ls']:
            # Check the survey footprint before doing anything else, so that the
            # user immediately knows that the field is just not covered by it
            for band in [tfilter, tfilter2]:
                if band is None:
                    continue

                cells = get_survey_cells(tname, wcs, image.shape, band=band)
                cellname = survey_cell_names.get(tname, 'skycells')

                if cells is not None:
                    if not len(cells):
                        ra0,dec0,sr0 = astrometry.get_frame_center(wcs=wcs, shape=image.shape)

                        if band == tfilter2:
                            # The primary band is still usable on its own
                            log(f"Warning: the field has no {tfilter2} band data in "
                                f"{tconf['name']}, will use the {tfilter} band alone")
                            tfilter2,tcoeff = None,None
                            continue

                        if has_survey_cells_any_band(tname, wcs, image.shape):
                            # The tiles are there, they just have no data in this band
                            raise RuntimeError(
                                f"The field at RA={ra0:.4f} Dec={dec0:.4f} with radius {sr0:.3f} deg "
                                f"is covered by {tconf['name']}, but has no data in {band} band there. "
                                "Please select a different filter or template."
                            )

                        raise RuntimeError(
                            f"The field at RA={ra0:.4f} Dec={dec0:.4f} with radius {sr0:.3f} deg "
                            f"is not covered by {tconf['name']} - no survey {cellname} overlap it. "
                            "Please select a different template."
                        )

                    log(f"{len(cells)} {tconf['name']} {cellname} overlap the image in {band} band")

    sub_size = config.get('sub_size', 1000)
    sub_overlap = config.get('sub_overlap', 50)

    classifier = None

    if subtraction_mode == 'detection':
        log('Transient detection mode activated')
        # Restrict the splitting if center and radius are provided
        if config.get('filter_center') and config.get('filter_sr0'):
            # TODO: resolve the center only once
            filter_center = resolve.resolve(config.get('filter_center'))
            filter_sr0 = config.get('filter_sr0')
            x0,y0 = wcs.all_world2pix(filter_center.ra.deg, filter_center.dec.deg, 0)
            r0 = max(filter_sr0/pixscale, sub_size/2)
            x1,x2 = max(0, x0 - r0), min(x0 + r0, image.shape[1])
            y1,y2 = max(0, y0 - r0), min(y0 + r0, image.shape[0])
            log(f"Will limit the search area to {x1:.0f} {y1:.0f} - {x2:.0f} {y2:.0f}")
        else:
            filter_center = None
            filter_sr0 = None
            x1,y1,x2,y2 = 0, 0, image.shape[1], image.shape[0]

        # We will split the image into nx x ny blocks
        nx = max(1, int(np.round((x2 - x1) / sub_size)))
        ny = max(1, int(np.round((y2 - y1) / sub_size)))
        log(f"Will split the image into {nx} x {ny} sub-images")
        split_fn = partial(
            pipeline.split_image,
            get_index=True, overlap=sub_overlap, nx=nx, ny=ny,
            xmin=x1, xmax=x2, ymin=y1, ymax=y2
        )

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

    all_candidates = []
    cutout_names = []

    for i, x0, y0, image1, mask1, header1, wcs1, obj1, cat1 in split_fn(
            image, mask, header, wcs, obj, cat,
            get_origin=True, verbose=False):

        log(f"\n---- Sub-image {i}: {x0} {y0} - {x0 + image1.shape[1]} {y0 + image1.shape[0]} ----\n")

        fits.writeto(os.path.join(basepath, 'sub_image.fits'), image1, header1, overwrite=True)
        fits_write(os.path.join(basepath, 'sub_mask.fits'), mask1.astype(np.uint8), header1, compress=True)

        # Get the template in a single filter
        def get_template(band):
            if tname == 'ps1' or tname == 'ls':
                log(f"Getting the template from original {tconf['name']} archive in {band} band")
                tmpl,tmask = templates.get_survey_image_and_mask(
                    band, survey=tname, wcs=wcs1, shape=image1.shape,
                    _cachedir=_cachedir, _cache_downscale = 1 if pixscale*3600 < 0.6 else 2,
                    _tmpdir=settings.STDPIPE_TMPDIR,
                    verbose=sub_verbose)
                if tmask is not None and tmpl is not None:
                    if tname == 'ps1':
                        tmask = tmask > 0
                    elif tname == 'ls':
                        # Bitmask for a given band, as described at https://www.legacysurvey.org/dr11/bitmasks/
                        imask = 0x0000
                        imask |= 0x0001 # not primary brick area
                        # imask |= 0x0002 # bright star nearby
                        # imask |= 0x0100 # WISE W1 (all masks)
                        # imask |= 0x0200 # WISE W2 (all masks)
                        imask |= 0x0400 # Bailed out processing
                        # imask |= 0x0800 # medium-bright star
                        # imask |= 0x1000 # SGA large galaxy
                        # imask |= 0x2000 # Globular cluster

                        if band == 'g':
                            imask |= 0x0004 # g band saturated
                            imask |= 0x0020 # any ALLMASK_G bit set
                        elif band == 'r':
                            imask |= 0x0008 # r band saturated
                            imask |= 0x0040 # any ALLMASK_R bit set
                        elif band == 'i':
                            imask |= 0x4000 # i band saturated
                            imask |= 0x8000 # any ALLMASK_I bit set
                        elif band == 'z':
                            imask |= 0x0010 # z band saturated
                            imask |= 0x0080 # any ALLMASK_Z bit set

                        tmask = (tmask & imask) > 0

                    tmask |= np.isnan(tmpl)
                else:
                    cells = get_survey_cells(tname, wcs1, image1.shape, band=band)
                    cellname = survey_cell_names.get(tname, 'skycells')

                    if cells is not None and not len(cells):
                        if has_survey_cells_any_band(tname, wcs1, image1.shape):
                            # The tiles are there, they just have no data in this band
                            raise RuntimeError(
                                f"Sub-image {i} is covered by {tconf['name']}, but has no data in "
                                f"{band} band there. Please select a different filter or template."
                            )

                        raise RuntimeError(
                            f"Sub-image {i} is not covered by {tconf['name']} - "
                            f"no survey {cellname} overlap it. Please select a different template."
                        )
                    elif cells is not None:
                        raise RuntimeError(
                            f"No {tconf['name']} data available for sub-image {i} in filter {band} - "
                            f"{len(cells)} {cellname} overlap it, but none of them could be retrieved. "
                            "The region may not be observed in this filter, or the downloads failed."
                        )
                    else:
                        raise RuntimeError(f"Error getting the template from {tconf['name']}")

            elif tname == 'custom':
                log("Re-projecting custom template onto sub-image")

                tmpl,fp = reproject.reproject_adaptive((custom_template, custom_wcs), wcs1, image1.shape)
                tmask,fp = reproject.reproject_adaptive((custom_mask.astype(np.double), custom_wcs), wcs1, image1.shape)

                tmask = tmask > 0.5
                tmask |= fp < 0.5

            else:
                log(f"Getting the template from HiPS server in {band} band")
                tmpl = templates.get_hips_image(tconf['filters'][band], wcs=wcs1, shape=image1.shape,
                                                get_header=False,
                                                verbose=sub_verbose)
                if tmpl is not None:
                    tmask = np.isnan(tmpl)
                else:
                    raise RuntimeError(
                        f"Error getting the template from {tconf['name']} HiPS server - "
                        "the field may not be covered by this survey"
                    )

            return tmpl,tmask

        tmpl,tmask = get_template(tfilter)

        if tfilter2 is not None:
            res = None

            try:
                # The second band is optional, so we may always fall back to the first
                # one alone if it is not available here
                tmpl2,tmask2 = get_template(tfilter2)
            except RuntimeError as e:
                log(f"Warning: cannot get the {tfilter2} band template: {e}")
                tmpl2 = None

            if tmpl2 is not None:
                log(f"\nCombining {tfilter} and {tfilter2} templates into a synthetic "
                    f"{config['filter']} band one")

                res = combine_templates(
                    tmpl, tmask, tmpl2, tmask2, obj, m, (x0, y0), image1.shape,
                    coeff=tcoeff,
                    bg_size=config.get('bg_size', 256),
                    gain=template_gain,
                    verbose=verbose
                )

            if res is not None:
                tmpl,tmask,coeff = res
                log(f"Template is {tfilter} + {coeff:.3f}*{tfilter2}")
            else:
                log(f"Warning: falling back to the {tfilter} band template alone")

        # Bail out early if the template does not cover the sub-image, otherwise
        # SExtractor would fail downstream on a fully-masked image
        tcoverage = np.sum(~tmask) / tmask.size

        if tcoverage <= 0:
            raise RuntimeError(
                f"{tconf['name'] if tconf else 'Template'} does not cover sub-image {i} - "
                "all its pixels are empty or masked. Please select a different template."
            )
        elif tcoverage < 0.9:
            log(f"Warning: template covers only {100*tcoverage:.1f}% of the sub-image")
        else:
            log(f"Template covers {100*tcoverage:.1f}% of the sub-image")

        # Estimate template FWHM
        template_fwhm,tobj,tidx,tsegm = measure_template_fwhm(
            tmpl, mask=tmask,
            bg_size=config.get('bg_size', 256),
            gain=template_gain,
            segmentation=True,
            verbose=verbose
        )

        # Mask the footprints of masked objects
        # for _ in tobj[(tobj['flags'] & 0x100) > 0]:
        #     tmask |= tsegm == _['NUMBER']

        fits.writeto(os.path.join(basepath, 'sub_template.fits'), tmpl, header1, overwrite=True)
        fits_write(os.path.join(basepath, 'sub_template_mask.fits'), tmask.astype(np.uint8), header1, compress=True)

        fwhm_values = 2.0*tobj['FLUX_RADIUS'] # obj['fwhm']

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

        if subtraction_method == 'sfft':
            # SFFT

            fwhm = config.get('fwhm', 3.0)

            log(f"Using template FWHM = {template_fwhm:.1f} pix and image FWHM = {fwhm:.1f} pix")

            # Kernel size based on larger FWHM
            kernel_hw = int(np.ceil(max(fwhm, template_fwhm)))
            # Ensure odd
            kernel_size = 2 * kernel_hw + 1
            kernel_shape = (kernel_size, kernel_size)

            res = subtraction.run_sfft(
                image1, tmpl,
                mask=mask1,
                template_mask=tmask,
                image_gain=config.get('gain', 1.0),
                template_gain=template_gain,
                kernel_shape=kernel_shape,
                kernel_poly_order=config.get('sfft_kernel_poly_order', 0),
                bg_poly_order=config.get('sfft_bg_poly_order', 0),
                flux_poly_order=config.get('sfft_flux_poly_order', 0),
                err=True,
                obj=obj1[obj1['flags']==0],
                get_convolved=True,
                get_scaled=True,
                get_noise=True,
                verbose=verbose,
            )

            if res is not None:
                diff, conv, ediff, sdiff = res
            else:
                log("Warning: SFFT subtraction failed")
                continue

            # Combined mask on the sub-image
            fullmask1 = mask1 | tmask

            diff1 = diff.copy()
            diff1[fullmask1] = 0.0

            fits.writeto(os.path.join(basepath, 'sub_diff.fits'), diff1, header1, overwrite=True)
            fits.writeto(os.path.join(basepath, 'sub_sdiff.fits'), sdiff, header1, overwrite=True)
            fits.writeto(os.path.join(basepath, 'sub_conv.fits'), conv, header1, overwrite=True)
            fits.writeto(os.path.join(basepath, 'sub_ediff.fits'), ediff, header1, overwrite=True)

        else:
            # HOTPANTS
            fwhm = config.get('fwhm', 3.0)

            log(f"Using template FWHM = {template_fwhm:.1f} pix and image FWHM = {fwhm:.1f} pix")

            # Dilate image and template masks
            mask1 = photometry.dilate(mask1, np.ones([int(np.ceil(fwhm))]*2))
            tmask = photometry.dilate(tmask, np.ones([int(np.ceil(template_fwhm))]*2))

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

        # PSF model in the coordinates of the sub-image
        psf1 = shift_psf_model(psf_model, x0, y0)

        # How well the stars actually subtracted
        if plot_subtraction_photometry(
            os.path.join(basepath, 'sub_photometry.png'),
            diff, obj, m, (x0, y0), image1.shape, config,
            psf=psf1,
            mask=fullmask1,
            err=ediff,
            show=show, verbose=verbose
        ):
            log("Subtraction photometry diagnostics stored to file:sub_photometry.png")

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

            # Photometry parameters should match the ones used for calibration
            target_obj = measure_photometry(
                target_obj, diff, config, psf=psf1,
                mask=fullmask1,
                sn=0,
                # We assume no background
                bg=None,
                # ..and known error model
                err=ediff,
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

            # Radial profile of the target in the difference image
            if config.get('inspect_profile'):
                fwhm = config.get('fwhm', 3.0)
                aper = config.get('rel_aper', 1.0)*fwhm
                bkgann = [_*fwhm for _ in rel_bkgann] if rel_bkgann else None

                # Mean stellar profile of the original image. The template is the
                # one being convolved, so the difference image keeps the original PSF
                ibg = sep.Background(
                    image1.astype(np.double), mask=mask1,
                    bw=config.get('bg_size', 256),
                    bh=config.get('bg_size', 256),
                )

                profile = measure_radial_profile(
                    image1 - ibg.back(), obj1, mask=mask1,
                    fwhm=fwhm, aper=aper, bkgann=bkgann,
                    verbose=sub_verbose
                )

                if profile is not None:
                    # How much of the stellar flux is leaking into the difference image
                    if plot_residual_radial_profile(
                            os.path.join(basepath, 'sub_profile.png'),
                            diff, obj1, mask=fullmask1,
                            fwhm=fwhm, aper=aper, bkgann=bkgann,
                            profile=profile,
                            show=show, verbose=verbose
                    ):
                        log("Subtraction residual profile stored to file:sub_profile.png")
                else:
                    log("Cannot measure the mean stellar profile of the image")

                if plot_target_radial_profile(
                        os.path.join(basepath, 'sub_target_profile.png'),
                        diff, target_obj['x'][0], target_obj['y'][0],
                        mask=fullmask1,
                        # Local background, if it was used for the target photometry
                        bg_local=target_obj['bg_local'][0] if 'bg_local' in target_obj.colnames else 0,
                        fwhm=fwhm, aper=aper, bkgann=bkgann,
                        flux=target_obj['flux'][0],
                        profile=profile,
                        title="Target radial profile in difference image",
                        show=show, verbose=sub_verbose
                ):
                    log("Target radial profile stored to file:sub_target_profile.png")

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

            sobj,segm = photometry.get_objects_sextractor(
                diff,
                mask=fullmask1,
                err=ediff,
                wcs=wcs1, edge=sub_overlap,
                aper=config.get('initial_aper', 3.0),
                gain=config.get('gain', 1.0),
                sn=config.get('sn', 5.0),
                minarea=config.get('minarea', 3),
                extra_params=['NUMBER', 'MAG_AUTO', 'ISOAREA_IMAGE', 'FLUX_MAX', 'FLUX_AUTO'],
                extra={'BACK_SIZE': config.get('bg_size', 256)},
                checkimages=['SEGMENTATION'],
                verbose=sub_verbose,
                _tmpdir=settings.STDPIPE_TMPDIR,
                _exe=settings.STDPIPE_SEXTRACTOR
            )

            # Photometry parameters should match the ones used for calibration
            sobj = measure_photometry(
                sobj, diff, config, psf=psf1,
                mask=fullmask1,
                # We assume no background
                bg=None,
                # ..and known error model
                err=ediff,
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
            if filter_center is not None and filter_sr0 is not None:
                log(f"Restricting the search to {filter_sr0:.3f} deg around RA={filter_center.ra.deg:.4f} Dec={filter_center.dec.deg:.4f}")
                dist = astrometry.spherical_distance(sobj['ra'], sobj['dec'], filter_center.ra.deg, filter_center.dec.deg)
                sobj = sobj[dist < filter_sr0]
                log(f"{len(sobj)} candidates inside the region")

            # Pre-filter detections if requested
            if config.get('filter_prefilter', True) and len(sobj):
                if classifier is None:
                    # Prepare the classifier based on SExtractor shape parameters
                    classifier = artefacts.filter_sextractor_detections(obj, verbose=False, return_classifier=True)

                fidx = classifier(sobj)
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

            os.makedirs(os.path.join(basepath, 'candidates'), exist_ok=True)

            cutout_size = config.get('cutout_size', 30)
            do_adjust = config.get('filter_adjust', True)
            adjust_inner = int(np.ceil(2.0*config.get('fwhm'))) if do_adjust else None

            def process_sub_candidate(cand):
                cutout = cutouts.get_cutout(
                    image1, cand, cutout_size,
                    mask=fullmask1,
                    diff=diff,
                    template=tmpl,
                    convolved=conv,
                    err=ediff,
                    footprint=(segm==cand['NUMBER']) if segm is not None else None,
                    time=time,
                    header=header1
                )

                if do_adjust:
                    if cutouts.adjust_cutout(
                            cutout, max_shift=1, max_scale=1.3,
                            inner=adjust_inner,
                            normalize=False, verbose=False
                    ):
                        if cutout['meta']['adjust_pval'] > 0.01:
                            return None
                        if cutout['meta']['adjust_chi2'] < 0.33*cutout['meta']['adjust_chi2_0']:
                            return None

                jname = utils.make_jname(cand['ra'], cand['dec'])
                cutout_name = os.path.join('candidates', jname + '.cutout')

                cutouts.write_cutout(cutout, os.path.join(basepath, cutout_name))

                return (cand, cutout_name)

            results = parallel_map(process_sub_candidate, candidates, verbose=verbose)

            for cand, cutout_name in results:
                all_candidates.append(cand)
                cutout_names.append(cutout_name)

            if do_adjust:
                log(f"{len(results)} candidates remaining after sub-pixel adjustment routine")

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
