"""
Photometric calibration and object measurement.
"""

import os

import numpy as np

from scipy.spatial import cKDTree
from scipy.stats import binned_statistic
from scipy.ndimage import zoom

from django.conf import settings

from astropy.wcs import WCS
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.stats import mad_std

from stdpipe import astrometry, photometry, catalogs, cutouts
from stdpipe import templates, plots, pipeline, artefacts
from stdpipe import psf as stdpipe_psf

from .constants import *
from .utils import *
from .catalogs import *


def get_profile_radius(image, fwhm, aper, bkgann=None):
    """Radius, in pixels, to accumulate the radial profiles up to."""
    rmax = max(4*fwhm, 1.2*aper)

    if bkgann is not None:
        rmax = max(rmax, 1.05*bkgann[1])

    return min(rmax, 0.45*min(image.shape))


def get_profile_pixels(image, x0, y0, rmax, mask=None, exclude=None, exclude_r=0):
    """Extract all image pixels within `rmax` from given position.

    The pixels closer than `exclude_r` to any of the positions in `exclude` - typically
    the neighbouring objects - are reported as masked ones.

    Returns their radii, values and mask flags, with the cutout clipped to image
    boundaries. Returns empty arrays if the position is outside the image.
    """
    size = int(np.ceil(rmax))
    x1,x2 = max(0, int(round(x0)) - size), min(image.shape[1], int(round(x0)) + size + 1)
    y1,y2 = max(0, int(round(y0)) - size), min(image.shape[0], int(round(y0)) + size + 1)

    if x1 >= x2 or y1 >= y2:
        return np.array([]), np.array([]), np.array([], dtype=bool)

    cutout = image[y1:y2, x1:x2]
    xs,ys = np.arange(x1, x2), np.arange(y1, y2)
    r = np.hypot((xs - x0)[None,:], (ys - y0)[:,None])

    idx = np.isfinite(cutout) & (r <= rmax)

    if mask is not None:
        cmask = mask[y1:y2, x1:x2]
    else:
        cmask = np.zeros_like(cutout, dtype=bool)

    if exclude is not None and len(exclude) and exclude_r > 0:
        cmask = cmask.copy()
        for xn,yn in exclude:
            cmask |= np.hypot((xs - xn)[None,:], (ys - yn)[:,None]) < exclude_r

    return r[idx], cutout[idx], cmask[idx]


def get_stars_profile_pixels(image, obj, fwhm, rmax, mask=None, nstars=200, verbose=True):
    """Collect the pixels around unflagged stars, normalized to their fluxes.

    The stars are selected from `obj` using their flags and magnitude errors, while the
    pixels are taken from `image` which is expected to be background subtracted. Instead
    of rejecting the stars having any neighbours nearby - too strict a cut in all but
    the sparsest fields - the pixels around the neighbouring objects are masked out, and
    only the stars that keep most of their cores are used.

    Returns the radii and normalized values of all the pixels, and the number of stars
    used, or empty arrays if there are no suitable stars.
    """
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    # Reasonably bright unflagged stars
    idx = np.asarray(obj['flags'] == 0)
    idx &= np.asarray(obj['magerr'] < 1/20)
    idx &= np.asarray(obj['flux'] > 0)

    if not np.sum(idx):
        log("No unflagged stars for measuring the radial profile")
        return np.array([]), np.array([]), 0

    xs = np.asarray(obj['x'], dtype=float)
    ys = np.asarray(obj['y'], dtype=float)
    fluxes = np.asarray(obj['flux'], dtype=float)

    # Prefer the ones with best S/N
    sidx = np.where(idx)[0]
    sidx = sidx[np.argsort(np.asarray(obj['magerr'])[sidx])][:nstars]

    # All the detected objects are potential contaminants
    tree = cKDTree(np.column_stack((xs, ys)))
    # Their wings are negligible at this distance
    exclude_r = 2*fwhm

    rs,vs = [],[]

    for i in sidx:
        nidx = [_ for _ in tree.query_ball_point([xs[i], ys[i]], rmax + exclude_r) if _ != i]

        r,v,m = get_profile_pixels(
            image, xs[i], ys[i], rmax, mask=mask,
            exclude=np.column_stack((xs[nidx], ys[nidx])) if len(nidx) else None,
            exclude_r=exclude_r
        )

        # Ignore the stars with significantly masked cores, as they would only
        # contribute their wings and thus bias the profile
        core = r < 0.5*fwhm

        if np.sum(core & ~m) > 0.5*np.sum(core):
            rs.append(r[~m])
            vs.append(v[~m]/fluxes[i])

    if not len(rs):
        log("No stars suitable for measuring the radial profile")
        return np.array([]), np.array([]), 0

    return np.concatenate(rs), np.concatenate(vs), len(rs)


def get_profile_step(fwhm):
    """Width of the radial bins used for the profiles, in pixels."""
    return max(0.25, fwhm/10)


def get_binned_profile(rs, vs, rmax, fwhm):
    """Median profile of the points in radial bins. Returns bin centers and values."""
    step = get_profile_step(fwhm)
    bins = np.arange(0, rmax + step, step)

    prof,edges,_ = binned_statistic(rs, vs, statistic='median', bins=bins)

    return 0.5*(edges[1:] + edges[:-1]), prof


def get_profile_fwhm(centers, prof):
    """FWHM from the half-maximum crossing of the binned profile normalized to unit peak.
    Returns None if the profile never drops below half maximum."""
    i_below = np.where(np.isfinite(prof) & (prof < 0.5))[0]
    if len(i_below):
        pidx = np.isfinite(prof)
        pidx[i_below[0]+1:] = False # Keep the profile monotonic up to the crossing
        if np.sum(pidx) > 1:
            return 2*np.interp(0.5, prof[pidx][::-1], centers[pidx][::-1])

    return None


def measure_radial_profile(image, obj, fwhm, aper, bkgann=None, mask=None, nstars=200,
                           verbose=True):
    """Measure the mean radial profile of the stars in background subtracted image.

    Individual stars are normalized to their measured fluxes, and then everything is
    scaled so that the median profile peaks at unity.

    Returns the dict with the median profile (`r` and `profile`), its peak value per
    unit flux (`peak`, so that `profile*peak*flux` is the expected profile of a star of
    given flux in original image units), the FWHM estimated from the profile itself
    (`fwhm`), the number of stars used (`nstars`), and the normalized pixels the profile
    is built from (`pixels_r` and `pixels_v`). Returns None if the profile may not be
    measured.
    """
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    rmax = get_profile_radius(image, fwhm, aper, bkgann)

    rs,vs,nused = get_stars_profile_pixels(image, obj, fwhm, rmax, mask=mask,
                                           nstars=nstars, verbose=verbose)

    if not nused:
        return None

    centers,prof = get_binned_profile(rs, vs, rmax, fwhm)

    # Normalize everything to the peak of the median profile
    inner = np.isfinite(prof) & (centers < 0.5*fwhm)
    peak = np.nanmax(prof[inner]) if np.sum(inner) else np.nanmax(prof)
    if not np.isfinite(peak) or peak <= 0:
        log("Cannot normalize the radial profile")
        return None

    vs = vs/peak
    prof = prof/peak

    log(f"Radial profile measured using {nused} stars")

    fwhm_prof = get_profile_fwhm(centers, prof)
    if fwhm_prof is not None:
        log(f"FWHM from the radial profile is {fwhm_prof:.2f} pixels")

    return {'r': centers, 'profile': prof, 'peak': peak, 'fwhm': fwhm_prof,
            'nstars': nused, 'rmax': rmax, 'pixels_r': rs, 'pixels_v': vs}


def plot_profile_points(ax, rs, vs, label=None, nmax=100000):
    """Scatter plot of individual pixels, sub-sampled if there are too many of them."""
    if len(rs) > nmax:
        idx = np.random.default_rng(0).choice(len(rs), nmax, replace=False)
    else:
        idx = slice(None)

    ax.plot(rs[idx], vs[idx], '.', ms=2, alpha=0.2, color='gray',
            rasterized=True, label=label)


def plot_profile_radii(ax, fwhm, aper, bkgann=None):
    """Mark the FWHM, aperture radius and background annulus on the radial profile plot."""
    ax.axvline(0.5*fwhm, ls='--', color='C3', label=f"FWHM/2 = {0.5*fwhm:.2f} pix")
    ax.axvline(aper, ls='-.', color='C2', label=f"Aperture = {aper:.2f} pix")

    if bkgann is not None:
        ax.axvspan(bkgann[0], bkgann[1], color='C1', alpha=0.15,
                   label=f"Background annulus {bkgann[0]:.2f} - {bkgann[1]:.2f} pix")


def plot_radial_profile(filename, image, obj, fwhm, aper, bkgann=None, mask=None,
                        nstars=200, title=None, show=False, verbose=True):
    """Plot the mean radial profile of the stars as a normalized scatter plot of
    pixel intensities vs distance from the centroid, along with the FWHM estimate,
    aperture radius and background annulus.

    Returns the same dict as :func:`measure_radial_profile`, or None if the profile
    may not be measured.
    """
    profile = measure_radial_profile(image, obj, fwhm, aper, bkgann=bkgann, mask=mask,
                                     nstars=nstars, verbose=verbose)

    if profile is None:
        return None

    with plots.figure_saver(filename, figsize=(8, 6), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        plot_profile_points(ax, profile['pixels_r'], profile['pixels_v'],
                            label=f"Pixels of {profile['nstars']} stars")
        ax.plot(profile['r'], profile['profile'], '-', color='C0', label='Median profile')

        ax.axhline(0.5, ls=':', lw=1, color='gray')
        plot_profile_radii(ax, fwhm, aper, bkgann)

        ax.set_xlim(0, profile['rmax'])
        ax.set_ylim(min(-0.05, np.nanpercentile(profile['pixels_v'], 1)), 1.1)
        # Keep the legend away from the peak of the profile
        ax.legend(loc='upper right')
        ax.grid(alpha=0.2)
        ax.set_xlabel('Radius, pixels')
        ax.set_ylabel('Normalized intensity')

        if title is None:
            title = f"Radial profile: {profile['nstars']} stars, FWHM {fwhm:.2f} pix"
            if profile['fwhm'] is not None:
                title += f", profile FWHM {profile['fwhm']:.2f} pix"
        ax.set_title(title)

    return profile


def plot_residual_radial_profile(filename, image, obj, fwhm, aper, profile, bkgann=None,
                                 mask=None, nstars=200, title=None, show=False, verbose=True):
    """Plot the radial profile of subtraction residuals at the positions of the stars.

    The pixels are taken from the difference image, and normalized to the peak
    intensities the same stars have in the original image, as given by the mean stellar
    `profile` measured there. So the plot directly shows the fraction of stellar flux
    leaking into the difference image.

    Returns the dict with the median residual profile (`r` and `profile`, in per cent),
    the number of stars used (`nstars`), the residual at the core (`core`), the largest
    signed one inside the aperture (`worst`), and the fraction of stellar flux leaking
    inside the aperture (`leak`, also in per cent). Returns None if it may not be
    measured.
    """
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    rmax = get_profile_radius(image, fwhm, aper, bkgann)

    rs,vs,nused = get_stars_profile_pixels(image, obj, fwhm, rmax, mask=mask,
                                           nstars=nstars, verbose=verbose)

    if not nused:
        return None

    # Per cent of the peak intensity the stars have in the original image
    vs = 100*vs/profile['peak']

    centers,prof = get_binned_profile(rs, vs, rmax, fwhm)

    fidx = np.isfinite(prof)
    if not np.sum(fidx):
        log("Cannot measure the residual profile")
        return None

    core = prof[fidx][0]

    # Largest residual inside the aperture, keeping its sign - it is often peaking
    # slightly off-center, so it is more representative than the innermost bin
    inside = fidx & (centers <= aper)
    worst = prof[inside][np.argmax(np.abs(prof[inside]))] if np.sum(inside) else np.nan

    # Integrating the median profile over the aperture area gives the fraction of
    # stellar flux that is actually leaking into it
    step = get_profile_step(fwhm)
    leak = 100*np.sum(prof[inside]/100*profile['peak']*2*np.pi*centers[inside]*step)

    log(f"Subtraction residuals at {nused} stars are {core:+.1f}% of stellar peak "
        f"intensity at the core, and up to {worst:+.1f}% inside the aperture")
    log(f"It corresponds to {leak:+.1f}% of stellar flux leaking inside the aperture")

    with plots.figure_saver(filename, figsize=(8, 6), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        plot_profile_points(ax, rs, vs, label=f"Pixels of {nused} stars")
        ax.plot(centers, prof, '-', color='C0', label='Median residual')

        ax.axhline(0, ls=':', lw=1, color='gray')
        plot_profile_radii(ax, fwhm, aper, bkgann)

        # Symmetric limits so that positive and negative residuals are comparable,
        # keeping the median profile well visible among the noise
        lim = max(1.5*np.nanmax(np.abs(prof[fidx])), 1.5*np.nanpercentile(np.abs(vs), 99.5))
        ax.set_xlim(0, rmax)
        ax.set_ylim(-1.1*lim, 1.1*lim)
        ax.legend(loc='upper right')
        ax.grid(alpha=0.2)
        ax.set_xlabel('Radius, pixels')
        ax.set_ylabel('Residual, per cent of stellar peak intensity')

        ax.set_title(title or
                     f"Subtraction residuals: {nused} stars, up to {worst:+.1f}% of peak "
                     f"intensity, {leak:+.1f}% of flux inside the aperture")

    return {'r': centers, 'profile': prof, 'nstars': nused,
            'core': core, 'worst': worst, 'leak': leak}


def plot_target_radial_profile(filename, image, x0, y0, fwhm, aper, bkgann=None, mask=None,
                               bg_local=0, flux=None, profile=None, title=None,
                               show=False, verbose=True):
    """Plot the radial profile of a single target, in original image units.

    The image is expected to be background subtracted, with `bg_local` optionally
    holding the residual local background used for the photometry of this target.

    No filtering of any kind is applied, so that the neighbours and masked pixels are
    directly visible. The mean stellar profile, if provided, is overplotted scaled to
    the measured flux of the target, to see how well the target follows it.

    Returns True if the plot has been created.
    """
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    rmax = get_profile_radius(image, fwhm, aper, bkgann)

    r,v,m = get_profile_pixels(image, x0, y0, rmax, mask=mask)

    if not len(r):
        log("Cannot measure the radial profile at the target position")
        return False

    if bg_local and np.isfinite(bg_local):
        v = v - bg_local

    with plots.figure_saver(filename, figsize=(8, 6), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(r[~m], v[~m], '.', ms=4, color='C0', label=f"Pixels ({np.sum(~m)})")
        if np.sum(m):
            ax.plot(r[m], v[m], 'x', ms=4, color='C3', label=f"Masked pixels ({np.sum(m)})")

        # Mean stellar profile scaled to the flux of the target
        if profile is not None and flux is not None and np.isfinite(flux):
            ax.plot(profile['r'], profile['profile']*profile['peak']*flux, '-', color='C4',
                    label="Mean stellar profile, scaled")

        ax.axhline(0, ls=':', lw=1, color='gray')
        plot_profile_radii(ax, fwhm, aper, bkgann)

        ax.set_xlim(0, rmax)
        ax.legend()
        ax.grid(alpha=0.2)
        ax.set_xlabel('Radius, pixels')
        ax.set_ylabel('Intensity above background, ADU')
        ax.set_title(title or "Target radial profile")

    return True


def get_photometry_method(config):
    """Photometry method selected in the config: 'aperture', 'optimal' or 'psf'."""
    method = config.get('photometry_method')
    if not method:
        # Backward compatibility with the older boolean option
        method = 'optimal' if config.get('optimal_extraction') else 'aperture'
    return method


def build_psf_model(image, obj, idx, fwhm, mask=None, degree='auto', size=None,
                    max_stars=None, verbose=True):
    """Build the PSF model from the good stars `obj[idx]` in background subtracted `image`.

    The stars are selected uniformly over the image, up to `max_stars` of them if set. The
    polynomial `degree` of the spatial variations is selected by cross-validation if set
    to 'auto', or reduced if there are too few stars to reliably constrain it otherwise.
    `size` is the model stamp size in pixels, automatically selected if not set.
    """
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    # Bright stars spread uniformly over the image, away from the edges. Their number
    # is limited by using coarser grid and fewer stars per its cell
    grid,max_per_tile = 6,25
    if max_stars:
        grid = max(1, min(grid, int(np.sqrt(max_stars))))
        max_per_tile = max(1, int(max_stars)//grid**2)
    seeds = stdpipe_psf.select_psf_seeds(
        obj[idx], image.shape, edge=int(np.ceil(3*fwhm)), grid=grid,
        max_per_tile=max_per_tile
    )

    if degree != 'auto':
        # Enough stars per polynomial coefficient for the fit to be stable. Every node of
        # oversampled model, as automatically selected by STDPipe for undersampled PSF,
        # only gets a fraction of star samples
        min_stars_per_coeff = 5
        oversampling = 1 if fwhm >= 2.5 else 2
        degree0 = degree
        while (degree > 0 and
               len(seeds)/oversampling**2 < min_stars_per_coeff*(degree + 1)*(degree + 2)//2):
            degree -= 1
        if degree != degree0:
            log(f"Only {len(seeds)} stars for the PSF model, reducing its spatial order "
                f"from {degree0} to {degree}")

    psf = stdpipe_psf.create_psf_model(
        image, seeds, fwhm=fwhm, mask=mask, degree=degree,
        size=int(size) if size else None,
        # Neighbours are modelled using all detections, not only the seeds
        neighbors_obj=obj,
        verbose=verbose
    )

    if degree == 'auto':
        log(f"Automatically selected PSF model spatial order is {psf['degree']}")

    return psf


def shift_psf_model(psf, x0, y0):
    """PSF model for the sub-image with origin at `(x0, y0)` of the original image."""
    if psf is None:
        return None

    psf = psf.copy()
    psf['x0'] = psf.get('x0', 0) - x0
    psf['y0'] = psf.get('y0', 0) - y0

    return psf


def measure_photometry(obj, image, config, psf=None, **kwargs):
    """Forced photometry of the objects using the method and parameters from the config.

    All measurements calibrated with the same photometric solution should go through
    this routine, so that they share the same definition of the flux. Extra keyword
    arguments are passed to :func:`stdpipe.photometry.measure_objects` and override
    the defaults derived from the config. `psf` is the PSF model, required for 'psf'
    method.
    """
    method = get_photometry_method(config)

    if config.get('rel_bg1') and config.get('rel_bg2'):
        rel_bkgann = [config['rel_bg1'], config['rel_bg2']]
    else:
        rel_bkgann = None

    params = dict(
        fwhm=config.get('fwhm'),
        aper=config.get('rel_aper', 1.0),
        bkgann=rel_bkgann,
        sn=config.get('sn', 5.0),
        bg_size=config.get('bg_size', 256),
        gain=config.get('gain', 1.0),
    )

    if method == 'psf':
        if psf is None:
            raise RuntimeError("PSF photometry requested, but no PSF model is available")
        # PSF fitting at fixed positions, with the PSF model instead of a Gaussian
        params.update(optimal=True, psf=psf)
        # The fit uses the pixels within the aperture, so it defines the fitting radius
        if config.get('psf_fit_radius'):
            params['aper'] = config['psf_fit_radius']
        # Nearby objects are fitted simultaneously, zero radius disables it
        if config.get('psf_group_radius') is not None:
            if config['psf_group_radius'] > 0:
                params['grouper_radius'] = config['psf_group_radius']*config.get('fwhm')
            else:
                params['group_sources'] = False
    elif method == 'optimal':
        params.update(optimal=True)
    elif method != 'aperture':
        raise ValueError(f"Unknown photometry method: {method}")

    params.update(kwargs)

    obj = photometry.measure_objects(obj, image, **params)

    if method == 'psf' and len(obj) and 'bg_fluxerr' in obj.colnames:
        # STDPipe reports the background noise summed inside the aperture, which here is
        # just the fitting region, while the one of PSF fit is sigma/sqrt(sum(P^2))
        obj['bg_fluxerr'] = get_psf_bg_fluxerr(obj, psf, params['aper']*params['fwhm'])

    return obj


def measure_targets_photometry(targets, obj, image, config, psf=None, **kwargs):
    """Forced photometry of the targets, optionally along with the objects around them.

    With PSF fitting and `psf_fit_neighbors` option set, the detected objects from `obj`
    close enough to share the fitting groups with the targets are fitted simultaneously
    with them, so that their flux does not leak into the ones of the targets. The objects
    closer than half FWHM to the targets are considered to be the targets themselves and
    ignored. Extra keyword arguments are passed to :func:`measure_photometry`.

    Returns the measured targets, and the table of neighbours measured along with them.
    """
    # Simple wrapper around print for logging in verbose mode only
    verbose = kwargs.get('verbose', False)
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    neighbours = Table({'x': np.zeros(0), 'y': np.zeros(0)})

    if get_photometry_method(config) == 'psf' and config.get('psf_fit_neighbors') and len(obj):
        fwhm = config.get('fwhm')
        fit_radius = config.get('psf_fit_radius') or config.get('rel_aper', 1.0)
        group_radius = config.get('psf_group_radius')
        if group_radius is None:
            # Default one of STDPipe
            group_radius = 2*fit_radius

        tidx = np.isfinite(targets['x']) & np.isfinite(targets['y'])

        if group_radius > 0 and np.any(tidx):
            # Within twice the grouping radius, to also catch the neighbours of neighbours
            tree = cKDTree(np.column_stack((targets['x'][tidx], targets['y'][tidx])))
            dist = tree.query(np.column_stack((obj['x'], obj['y'])))[0]
            idx = (dist < 2*group_radius*fwhm) & (dist > 0.5*fwhm)
            neighbours = Table({'x': np.asarray(obj['x'][idx], dtype=float),
                                'y': np.asarray(obj['y'][idx], dtype=float)})
            log(f"Fitting {len(neighbours)} detected objects around the targets simultaneously with them")
        elif group_radius <= 0:
            log("Warning: fitting the neighbours has no effect with grouping disabled")

    # All objects measured together, and no row may be dropped
    both = Table({'x': np.concatenate([np.asarray(targets['x'], dtype=float), neighbours['x']]),
                  'y': np.concatenate([np.asarray(targets['y'], dtype=float), neighbours['y']])})
    res = measure_photometry(both, image, config, psf=psf, **(kwargs | {'sn': 0}))

    targets = targets.copy()
    for col in res.colnames:
        targets[col] = res[col][:len(targets)]

    return targets, res[len(targets):]


def get_psf_residual_image(image, bg, psf, sources):
    """Image with the background `bg` and the PSF models of the measured `sources`
    (with `x`, `y` and `flux` columns) subtracted."""
    model = np.zeros(image.shape, dtype=float)

    for x,y,flux in zip(sources['x'], sources['y'], sources['flux']):
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(flux):
            stdpipe_psf.place_psf_stamp(model, psf, x, y, flux=flux)

    return image - bg - model


def get_psf_bg_fluxerr(obj, psf, radius):
    """Background noise of PSF fit flux for the objects, from their per-pixel background
    noise derived from `bg_fluxerr` and `npix_aper` columns, and the PSF model within the
    fitting `radius` in pixels."""
    bg_fluxerr = np.full(len(obj), np.nan)

    for i,(x,y,fluxerr,npix) in enumerate(zip(obj['x'], obj['y'], obj['bg_fluxerr'], obj['npix_aper'])):
        if not np.isfinite(x) or not np.isfinite(y) or not npix > 0:
            continue

        stamp = stdpipe_psf.get_psf_stamp(psf, x, y, normalize=True)
        h,w = stamp.shape
        r = np.hypot(np.arange(w)[None,:] - w//2 - (x - np.round(x)),
                     np.arange(h)[:,None] - h//2 - (y - np.round(y)))
        sum2 = np.sum(stamp[r <= radius]**2)

        if sum2 > 0:
            bg_fluxerr[i] = fluxerr/np.sqrt(npix*sum2)

    return bg_fluxerr


def get_psf_model_profile(psf, x, y, rmax, nsub=5):
    """Radii and values, per unit flux, of the pixels of the PSF model evaluated at `(x, y)`.

    The model is shifted over a grid of sub-pixel offsets, so that its pixels sample the
    profile the same way the pixels of real stars do.
    """
    rs,vs = [],[]

    for dy in (np.arange(nsub) + 0.5)/nsub - 0.5:
        for dx in (np.arange(nsub) + 0.5)/nsub - 0.5:
            stamp = stdpipe_psf.get_psf_stamp(psf, x, y, dx=dx, dy=dy, normalize=True)
            h,w = stamp.shape
            r = np.hypot(np.arange(w)[None,:] - w//2 - dx, np.arange(h)[:,None] - h//2 - dy)
            idx = r <= rmax
            rs.append(r[idx])
            vs.append(stamp[idx])

    return np.concatenate(rs), np.concatenate(vs)


def get_psf_model_shape(psf, x, y, upsample=4):
    """FWHM, in pixels, and ellipticity of the PSF model evaluated at `(x, y)`.

    The FWHM is the diameter of the circle having the same area as the region above half
    maximum, so it is meaningful for arbitrary PSF shapes. The ellipticity is derived from
    the second moments of the same region, that preserve the axis ratio of elliptical
    profiles.
    """
    stamp = stdpipe_psf.get_supersampled_psf_stamp(psf, x, y, normalize=True)
    # Model pixel size, in image pixels
    scale = psf['sampling']

    # Interpolate onto a finer grid so that the area above half maximum is not too coarse
    factor = max(1, int(np.round(upsample*scale)))
    if factor > 1:
        stamp = zoom(stamp, factor, order=3)
        scale /= factor

    idx = stamp > 0.5*np.max(stamp)
    fwhm = 2*np.sqrt(np.sum(idx)/np.pi)*scale

    yy,xx = np.nonzero(idx)
    w = stamp[idx]
    xx = xx - np.average(xx, weights=w)
    yy = yy - np.average(yy, weights=w)
    cov = np.cov(np.vstack([xx, yy]), aweights=w)
    l2,l1 = np.linalg.eigvalsh(cov)
    ell = 1 - np.sqrt(max(l2, 0)/l1) if l1 > 0 else 0

    return fwhm, ell


def plot_psf_model(filename, psf, image, obj, fwhm, aper, bkgann=None, mask=None,
                   show=False, verbose=True):
    """Plot the PSF model evaluated at the center of the image, and its radial profile
    compared with the mean profile of the actual stars and with the Gaussian of the same
    FWHM.

    The `image` is expected to be background subtracted. Everything is shown per unit flux,
    normalized to the peak of the stellar profile, so that the model and the stars are
    directly comparable.

    Returns the FWHM of the model radial profile, or None if it may not be measured.
    """
    x0,y0 = image.shape[1]/2, image.shape[0]/2
    rmax = get_profile_radius(image, fwhm, aper, bkgann)

    rs,vs = get_psf_model_profile(psf, x0, y0, rmax)
    centers,prof = get_binned_profile(rs, vs, rmax, fwhm)
    model_peak = np.nanmax(prof)
    fwhm_model = get_profile_fwhm(centers, prof/model_peak)

    profile = measure_radial_profile(image, obj, fwhm, aper, bkgann=bkgann, mask=mask,
                                     verbose=False)
    # Common normalization, to the peak of the stellar profile if available
    peak = profile['peak'] if profile is not None else model_peak

    with plots.figure_saver(filename, figsize=(14, 6), show=show) as fig:
        ax = fig.add_subplot(1, 2, 1)

        stamp = stdpipe_psf.get_psf_stamp(psf, x0, y0, dx=0, dy=0, normalize=True)
        h,w = stamp.shape
        im = ax.imshow(np.log10(np.maximum(stamp/np.max(stamp), 1e-4)), origin='lower',
                       interpolation='nearest', cmap='viridis',
                       extent=[-w/2, w/2, -h/2, h/2])
        plots.colorbar(im, ax=ax)
        ax.set_xlabel('dx, pixels')
        ax.set_ylabel('dy, pixels')
        ax.set_title("PSF model at the image center, log10(intensity/peak)")

        ax = fig.add_subplot(1, 2, 2)

        if profile is not None:
            plot_profile_points(ax, profile['pixels_r'], profile['pixels_v'],
                                label=f"Pixels of {profile['nstars']} stars")
            ax.plot(profile['r'], profile['profile'], '-', color='C0',
                    label='Median stellar profile')

        ax.plot(centers, prof/peak, '-', lw=2, color='C4', label='PSF model')

        sigma = fwhm/2.355
        rr = np.linspace(0, rmax, 200)
        ax.plot(rr, np.exp(-rr**2/2/sigma**2)/(2*np.pi*sigma**2)/peak, ':', lw=2,
                color='C5', label=f"Gaussian with FWHM {fwhm:.2f} pix")

        ax.axhline(0, ls=':', lw=1, color='gray')
        plot_profile_radii(ax, fwhm, aper, bkgann)

        ax.set_xlim(0, rmax)
        ax.set_ylim(-0.05, 1.1*max(1, np.nanmax(prof/peak)))
        ax.legend(loc='upper right')
        ax.grid(alpha=0.2)
        ax.set_xlabel('Radius, pixels')
        ax.set_ylabel('Normalized intensity')

        title = "PSF model radial profile"
        if fwhm_model is not None:
            title += f": FWHM {fwhm_model:.2f} pix"
        ax.set_title(title)

    return fwhm_model


def plot_psf_grid(filename, psf, shape, fwhm=None, ngrid=3, nmap=16, show=False):
    """Plot the PSF model evaluated over the grid of positions across the image, the
    difference of every grid stamp from the central one, and the maps of the model FWHM
    and ellipticity.

    Returns the ranges of the model FWHM and ellipticity over the image.
    """
    xs = shape[1]*(np.arange(ngrid) + 0.5)/ngrid
    ys = shape[0]*(np.arange(ngrid) + 0.5)/ngrid

    center = stdpipe_psf.get_psf_stamp(psf, shape[1]/2, shape[0]/2, dx=0, dy=0)
    vmax = np.max(center)

    stamps = [[stdpipe_psf.get_psf_stamp(psf, x, y, dx=0, dy=0) for x in xs] for y in ys]
    dmax = max(np.max(np.abs(_ - center)) for row in stamps for _ in row)
    dmax = max(dmax, 1e-3*vmax)

    # Model shape over the finer grid
    mxs = shape[1]*(np.arange(nmap) + 0.5)/nmap
    mys = shape[0]*(np.arange(nmap) + 0.5)/nmap
    shapes = np.array([[get_psf_model_shape(psf, x, y) for x in mxs] for y in mys])
    fwhms,ells = shapes[:,:,0],shapes[:,:,1]

    with plots.figure_saver(filename, figsize=(12, 12), show=show, tight_layout=False) as fig:
        # Stamp grids on top, sized so that their cells are nearly square, and maps below
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.1,
                              left=0.05, right=0.95, top=0.95, bottom=0.03)
        gs_top = gs[0].subgridspec(1, 2, wspace=0.08)
        gs_stamps = gs_top[0].subgridspec(ngrid, ngrid, wspace=0.05, hspace=0.15)
        gs_diffs = gs_top[1].subgridspec(ngrid, ngrid, wspace=0.05, hspace=0.15)

        for i in range(ngrid):
            for j in range(ngrid):
                # Top row of the plot is the top of the image
                row = ngrid - 1 - i

                ax = fig.add_subplot(gs_stamps[row, j])
                ax.imshow(stamps[i][j], origin='lower', interpolation='nearest',
                          cmap='viridis', vmin=0, vmax=vmax)
                ax.set_axis_off()
                ax.set_title(f"{xs[j]:.0f}, {ys[i]:.0f}", fontsize=8)

                ax = fig.add_subplot(gs_diffs[row, j])
                ax.imshow(stamps[i][j] - center, origin='lower', interpolation='nearest',
                          cmap='RdBu_r', vmin=-dmax, vmax=dmax)
                ax.set_axis_off()

        fig.text(0.27, 0.985, "PSF model across the image", ha='center', va='top')
        fig.text(0.73, 0.985, f"Difference from the center, up to {dmax/vmax:.1%} of the peak",
                 ha='center', va='top')

        gs_maps = gs[1].subgridspec(1, 2, wspace=0.2)

        extent = [0, shape[1], 0, shape[0]]

        for k,(data,title) in enumerate([
                (fwhms, f"Model FWHM: {np.min(fwhms):.2f} - {np.max(fwhms):.2f} pix" +
                 (f", photometry FWHM {fwhm:.2f}" if fwhm else "")),
                (ells, f"Model ellipticity: {np.min(ells):.2f} - {np.max(ells):.2f}"),
        ]):
            ax = fig.add_subplot(gs_maps[k])
            im = ax.imshow(data, origin='lower', interpolation='nearest', extent=extent,
                           aspect='equal')
            plots.colorbar(im, ax=ax)
            ax.set_title(title)

    return (np.min(fwhms), np.max(fwhms)), (np.min(ells), np.max(ells))


def plot_psf_residuals(filename, psf, image, bg, obj, config, mask=None, nstars=200,
                       show=False, verbose=True):
    """Plot how well the PSF model fits the stars.

    The panels show the median residual of the PSF fit for the bright unflagged stars
    in `obj`, as measured with `measure_photometry`, normalized to the model peak; the
    reduced chi2 of the fit versus instrumental magnitude; and the map of the ratio of
    the flux within the aperture predicted by the PSF fit, using the model enclosed
    flux fraction, to the one measured directly by aperture photometry. The latter
    should be unity everywhere if the model properly describes the stars.

    The `image` is the original one, and `bg` its background. Returns the dict with the
    median (`median`) and robust scatter (`rms`) of the flux ratio, and the number of
    stars used (`nstars`), or None if there are no suitable stars.
    """
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    fwhm = config.get('fwhm')
    aper = config.get('rel_aper', 1.0)*fwhm

    # Bright unflagged stars, well inside the image
    stamp0 = stdpipe_psf.get_psf_stamp(psf, image.shape[1]/2, image.shape[0]/2, dx=0, dy=0)
    size = stamp0.shape[0]//2
    idx = (obj['flags'] == 0) & (obj['magerr'] < 1/20) & (obj['flux'] > 0)
    idx &= (obj['x'] > size + 1) & (obj['x'] < image.shape[1] - size - 2)
    idx &= (obj['y'] > size + 1) & (obj['y'] < image.shape[0] - size - 2)
    sidx = np.where(idx)[0]
    sidx = sidx[np.argsort(np.asarray(obj['magerr'])[sidx])][:nstars]

    if len(sidx) < 3:
        log("Too few stars for PSF residuals diagnostics")
        return None

    image1 = image - bg

    # Median residual of the fit, relative to the model peak
    residuals = []
    for i in sidx:
        x,y,flux = obj['x'][i], obj['y'][i], obj['flux'][i]
        bg_local = obj['bg_local'][i] if 'bg_local' in obj.colnames else 0
        model = stdpipe_psf.get_psf_stamp(psf, x, y, normalize=True)
        xi,yi = int(np.round(x)), int(np.round(y))
        cut = np.s_[yi - size:yi + size + 1, xi - size:xi + size + 1]

        res = (image1[cut] - bg_local - flux*model)/flux/np.max(model)
        if mask is not None:
            res[mask[cut]] = np.nan
        residuals.append(res)

    residual = np.nanmedian(residuals, axis=0)

    # Flux within the aperture, measured directly, and predicted from the PSF fit
    aobj = measure_photometry(obj[sidx]['x', 'y'], image,
                              config | {'photometry_method': 'aperture'},
                              mask=mask, sn=0, verbose=False)
    frac = np.array([stdpipe_psf.enclosed_psf_fraction(psf, x, y, radius=aper)
                     for x,y in zip(obj['x'][sidx], obj['y'][sidx])])
    ratio = np.asarray(obj['flux'][sidx])*frac/np.asarray(aobj['flux'])

    good = np.isfinite(ratio) & (aobj['flags'] == 0)
    if np.sum(good) < 3:
        log("Too few stars with proper aperture photometry for PSF residuals diagnostics")
        return None

    median = np.median(ratio[good])
    rms = mad_std(ratio[good])

    with plots.figure_saver(filename, figsize=(18, 5.5), show=show) as fig:
        ax = fig.add_subplot(1, 3, 1)
        vmax = max(np.nanmax(np.abs(residual)), 1e-3)
        im = ax.imshow(residual, origin='lower', interpolation='nearest', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, extent=[-size - 0.5, size + 0.5, -size - 0.5, size + 0.5])
        plots.colorbar(im, ax=ax)
        ax.set_xlabel('dx, pixels')
        ax.set_ylabel('dy, pixels')
        ax.set_title(f"Median fit residual of {len(sidx)} stars, fraction of peak")

        ax = fig.add_subplot(1, 3, 2)
        if 'chi2_optimal' in obj.colnames:
            fidx = obj['flags'] == 0
            ax.plot(obj['mag'][~fidx], obj['chi2_optimal'][~fidx], '.', color='C1',
                    alpha=0.5, label='Flagged')
            ax.plot(obj['mag'][fidx], obj['chi2_optimal'][fidx], '.', color='C0',
                    alpha=0.5, label='Unflagged')
            ax.plot(obj['mag'][sidx], obj['chi2_optimal'][sidx], 'o', mfc='none',
                    color='C3', label='Used for residuals')
            ax.axhline(1, ls='--', color='gray')
            ax.set_yscale('log')
            ax.legend()
        ax.grid(alpha=0.2)
        ax.set_xlabel('Instrumental magnitude')
        ax.set_ylabel('Reduced chi2 of PSF fit')
        ax.set_title("PSF fit quality")

        ax = fig.add_subplot(1, 3, 3)
        plots.binned_map(
            obj['x'][sidx][good], obj['y'][sidx][good], ratio[good],
            range=[[0, image.shape[1]], [0, image.shape[0]]],
            bins=8, statistic='median',
            show_dots=True, ax=ax
        )
        ax.set_aspect(1)
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(0, image.shape[0])
        ax.set_title(f"PSF predicted / measured aperture flux: {median:.3f} +/- {rms:.3f}")

    return {'median': median, 'rms': rms, 'nstars': int(np.sum(good))}


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

    # SExtractor fills masked (NaN input) pixels in the FILTERED checkimage
    # with its internal -BIG sentinel (-1e30); convert these back to NaN
    fimg = fimg.astype(np.float32)
    fimg[fimg < -1e29] = np.nan

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

    # PSF model, built from the same stars used for FWHM estimation
    psf_model = None
    if get_photometry_method(config) == 'psf':
        log("\n---- PSF model ----\n")

        psf_model = build_psf_model(
            image - bg, obj, idx, fwhm,
            mask=(mask | custom_mask) if custom_mask is not None else mask,
            # Empty value means automatic selection
            degree=config.get('psf_degree') if config.get('psf_degree') not in [None, ''] else 'auto',
            size=config.get('psf_size'),
            max_stars=config.get('psf_max_stars'),
            verbose=verbose
        )

        pickle_to_file(os.path.join(basepath, 'psf.pickle'), psf_model)
        log("PSF model stored to psf.pickle")

    # Forced photometry at objects positions
    obj = measure_photometry(obj, image, config, psf=psf_model,
                             mask=mask,
                             verbose=verbose)

    log(f"{len(obj)} objects properly measured, {np.sum(obj['flags'] == 0)} unflagged")
    if np.sum(obj['flags'] == 0) < 0.5*len(obj):
        log("Warning: more than half of objects are flagged!")

    obj.write(os.path.join(basepath, 'objects.parquet'), overwrite=True)
    log("Measured objects stored to file:objects.parquet")

    # PSF model diagnostics
    if psf_model is not None:
        full_mask = (mask | custom_mask) if custom_mask is not None else mask

        fwhm_model = plot_psf_model(
            os.path.join(basepath, 'psf.png'),
            psf_model, image - bg, obj,
            fwhm=fwhm,
            aper=config.get('rel_aper', 1.0)*fwhm,
            bkgann=[_*fwhm for _ in rel_bkgann] if rel_bkgann else None,
            mask=full_mask,
            show=show, verbose=verbose
        )
        if fwhm_model is not None:
            log(f"PSF model FWHM is {fwhm_model:.2f} pixels")
        log("PSF model plot stored to file:psf.png")

        if psf_model.get('degree', 0) > 0:
            (fwhm1, fwhm2), (ell1, ell2) = plot_psf_grid(
                os.path.join(basepath, 'psf_grid.png'),
                psf_model, image.shape, fwhm=fwhm,
                show=show
            )
            log(f"PSF model FWHM varies over the image from {fwhm1:.2f} to {fwhm2:.2f} pixels, "
                f"ellipticity from {ell1:.2f} to {ell2:.2f}")
            log("PSF model variations plot stored to file:psf_grid.png")

        res = plot_psf_residuals(
            os.path.join(basepath, 'psf_residuals.png'),
            psf_model, image, bg, obj, config,
            mask=full_mask,
            show=show, verbose=verbose
        )
        if res is not None:
            log(f"PSF predicted to measured aperture flux ratio is {res['median']:.3f} "
                f"+/- {res['rms']:.3f} over {res['nstars']} stars")
            log("PSF residuals plot stored to file:psf_residuals.png")

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

    # Plot mean radial profile of the stars
    profile = None
    if config.get('inspect_profile'):
        full_mask = (mask | custom_mask) if custom_mask is not None else mask

        profile = plot_radial_profile(
            os.path.join(basepath, 'profile.png'),
            image - bg, obj, mask=full_mask,
            fwhm=fwhm,
            aper=config.get('rel_aper', 1.0)*fwhm,
            bkgann=[_*fwhm for _ in rel_bkgann] if rel_bkgann else None,
            show=show, verbose=verbose
        )

        if os.path.exists(os.path.join(basepath, 'profile.png')):
            log("Radial profile diagnostic plot stored to file:profile.png")

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

    # Catalogue settings
    config['cat_col_mag'],config['cat_col_mag_err'] = guess_catalogue_mag_columns(
        config['filter'],
        cat,
        augmented_only=True
    )

    if config['cat_col_mag'] in ['Umag', 'Bmag', 'Vmag', 'Rmag', 'Imag']:
        config['cat_col_color_mag1'] = 'Bmag'
        config['cat_col_color_mag2'] = 'Vmag'
    elif config['cat_col_mag'] in ['Jmag', 'Hmag', 'Ksmag']:
        config['cat_col_color_mag1'] = 'Jmag'
        config['cat_col_color_mag2'] = 'Ksmag'
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

    if config.get('filter_blends', True):
        # Merge the stars that are not resolved in the image into single entries, keeping
        # every magnitude the pipeline may later use consistent with the merged flux.
        # The primary one goes first, as it defines the merged centroids
        cat_col_mags = [(config['cat_col_mag'], config['cat_col_mag_err'])]
        cat_col_mags += guess_catalogue_mag_columns_all(config['cat_name'], cat)

        # A group may not end up wider than the aperture that will measure it, otherwise
        # its summed flux would include the stars the aperture never collects
        sr_blend = config.get('cat_blend_radius', 0.25)*fwhm*pixscale
        sr_blend_max = config.get('rel_aper', 1.0)*fwhm*pixscale
        cat_filtered = filter_catalogue_blends(cat, sr_blend, sr_max=sr_blend_max,
                                               cat_col_mags=cat_col_mags, verbose=verbose)
        log(f"{len(cat_filtered)} catalogue stars after merging blends closer than "
            f"{3600*sr_blend:.1f} arcsec into groups up to {3600*sr_blend_max:.1f} arcsec wide")

        # Reject the ones still significantly contaminated inside the photometric aperture
        sr_contam = fwhm*pixscale
        contamination = config.get('cat_contamination', 0.1)
        cat_filtered = filter_catalogue_contamination(
            cat_filtered, sr_contam,
            cat_col_mag=config['cat_col_mag'],
            contamination=contamination
        )
        log(f"{len(cat_filtered)} catalogue stars after rejecting the ones contaminated by more "
            f"than {100*contamination:.0f}% within {3600*sr_contam:.1f} arcsec")
    else:
        cat_filtered = cat

    # Optionally limit to a range of catalogue magnitudes
    if config.get('cat_mag_lower') is not None:
        mag_lower = float(config.get('cat_mag_lower'))
        log(f"Removing catalogue stars brighter than {mag_lower}")
        cat_filtered = cat_filtered[cat_filtered[config['cat_col_mag']] >= mag_lower]

    if config.get('cat_mag_upper') is not None:
        mag_upper = float(config.get('cat_mag_upper'))
        log(f"Removing catalogue stars fainter than {mag_upper}")
        cat_filtered = cat_filtered[cat_filtered[config['cat_col_mag']] <= mag_upper]

    # Astrometric refinement
    if config.get('refine_wcs', False):
        log("\n---- Astrometric refinement ----\n")

        # Exclude pre-filtered detections and limit list size
        obj_ast = obj[(obj['flags'] & 0x800) == 0]

        wcs1 = pipeline.refine_astrometry(obj_ast, cat_filtered, fwhm*pixscale,
                                          wcs=wcs, order=config.get('refine_order', 'auto'), method='scamp',
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

    # Update pixel scale in the config
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

        # Pixel-defined targets have authoritative x/y with sky position (re-)derived
        # from the refined WCS and stored back to the config, while sky-defined ones
        # are projected onto the image through it
        xs,ys = [],[]
        for i,target in enumerate(config['targets']):
            if target.get('x') is not None and target.get('y') is not None:
                target['ra'],target['dec'] = [
                    float(_) for _ in wcs.all_pix2world(target['x'], target['y'], 0)
                ]
                if i == 0:
                    # Keep backwards-compatible primary target coordinates
                    config['target_ra'] = target['ra']
                    config['target_dec'] = target['dec']
                x,y = target['x'],target['y']
            else:
                x,y = wcs.all_world2pix(target['ra'], target['dec'], 0)
            xs.append(x)
            ys.append(y)

        target_obj = Table({
            'ra': [_['ra'] for _ in config['targets']],
            'dec': [_['dec'] for _ in config['targets']],
            'x': xs,
            'y': ys,
        })

        # Filter out targets outside the image so that photometry routine does not crash
        def is_inside(x, y):
            return (x > 0 and x < image.shape[1] and y > 0 and y < image.shape[0])

        if not is_inside(target_obj['x'][0], target_obj['y'][0]):
            raise RuntimeError("Primary target is outside the image")

        log(f"Primary target position is {target_obj['ra'][0]:.3f} {target_obj['dec'][0]:.3f} -> {target_obj['x'][0]:.1f} {target_obj['y'][0]:.1f}")

        target_obj,target_neighbours = measure_targets_photometry(
            target_obj, obj, image, config, psf=psf_model,
            mask=mask,
            centroid_iter=5 if config.get('centroid_targets') else 0,
            verbose=verbose
        )

        # Image with fitted PSF models subtracted, to see the quality of the fit
        if psf_model is not None:
            psf_residual = get_psf_residual_image(
                image, bg, psf_model,
                Table({
                    _: np.concatenate([np.asarray(target_obj[_], dtype=float),
                                       np.asarray(target_neighbours[_], dtype=float)])
                    for _ in ['x', 'y', 'flux']
                })
            )
        else:
            psf_residual = None

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

        if config.get('inspect_profile'):
            profile_image = image - bg
            profile_mask = (mask | custom_mask) if custom_mask is not None else mask

        # Create the cutouts from image based on the targets
        for i,tobj in enumerate(target_obj):
            cutout_name = f"targets/target_{i:04d}.cutout"
            target_title = "Primary target" if i == 0 else f"Secondary target {i}"

            if (not np.isfinite(tobj['x']) or not np.isfinite(tobj['y']) or
                tobj['x'] < 0 or tobj['y'] < 0 or
                tobj['x'] > image.shape[1] or tobj['y'] > image.shape[0]):
                log(f"{target_title} is outside image, skipping")
                continue

            try:
                os.makedirs(os.path.join(basepath, 'targets'))
            except OSError:
                pass

            # Radial profile of the target itself, unfiltered
            if config.get('inspect_profile'):
                profile_name = f"targets/target_{i:04d}_profile.png"

                if plot_target_radial_profile(
                        os.path.join(basepath, profile_name),
                        profile_image, tobj['x'], tobj['y'],
                        mask=profile_mask,
                        # Local background, if it was used for the target photometry
                        bg_local=tobj['bg_local'] if 'bg_local' in target_obj.colnames else 0,
                        fwhm=fwhm,
                        aper=config.get('rel_aper', 1.0)*fwhm,
                        bkgann=[_*fwhm for _ in rel_bkgann] if rel_bkgann else None,
                        flux=tobj['flux'],
                        profile=profile,
                        title=f"{target_title} radial profile",
                        show=show, verbose=verbose
                ):
                    log(f"{target_title} radial profile stored to file:{profile_name}")

            cutout = cutouts.get_cutout(
                image, tobj, config.get('cutout_size', 30),
                mask=mask,
                header=header,
                time=time,
                # filtered=fimg if config.get('initial_r0') else None,
                # Local background, if it was used for the target photometry
                residual=(psf_residual - (tobj['bg_local'] if 'bg_local' in target_obj.colnames else 0)
                          if psf_residual is not None else None),
            )
            # Cutout from relevant HiPS survey
            cutout['template'] = templates.get_hips_image(
                guess_hips_survey(tobj['ra'], tobj['dec'], config['filter']),
                header=cutout['header'],
                get_header=False
            )

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
