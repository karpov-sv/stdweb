"""
Image inspection and initial parameter extraction.
"""

import os
import warnings

import numpy as np

from astropy.io import fits
from astropy.time import Time

from stdpipe import astrometry, cutouts, templates
from stdpipe import resolve, utils

from .constants import *
from .utils import *


import astroscrappy
try:
    import sep_x as sep
except ImportError:
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


def measure_sky_skewness(image, mask=None, verbose=False):
    """Pixel-independent skewness of the sky noise of the image.

    The skewness is measured as sigma-clipped (mean - median)/rms of the
    background-subtracted image binned by 1, 2 and 4 pixels. Skewness of
    pixel-independent noise (EM gain, low photon counts) decreases as 1/sqrt(n) when n
    pixels are averaged, while the one of the sources, correlated on PSF scales, does
    not, which separates the two. For quantized (integer-valued) images the interpolated
    median of grouped data is used, as the plain one is itself quantized, mimicking the
    skewness.

    Returns a dict with `skew` (values for the three binnings), `intrinsic` (the
    pixel-independent part of the unbinned one), `gamma` (corresponding standard
    skewness of the noise) and `rms` (sky noise rms), or None if there are too few
    unmasked pixels.
    """
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    from astropy.stats import sigma_clipped_stats, sigma_clip
    from stdpipe import photometry_measure

    image = np.array(image, dtype=np.double)
    mask = np.zeros(image.shape, dtype=bool) if mask is None else np.array(mask, dtype=bool)
    mask |= ~np.isfinite(image)

    if np.sum(~mask) < 10000:
        log("Too few unmasked pixels to measure the sky noise skewness")
        return None

    # Grid step of integer-valued data, whose plain median is itself quantized and would
    # be mistaken for the skewness - it is then replaced with interpolated median
    q = photometry_measure._quantization_step(image, mask)

    image[mask] = np.median(image[~mask])

    # Fine mesh, so that the large-scale structure does not contribute to the skewness
    bsize = int(min(64, image.shape[0]//4, image.shape[1]//4))
    bg = sep.Background(image, mask=mask, bw=bsize, bh=bsize)
    back = bg.back()
    resid = image - back

    def binned(x, b):
        h, w = (x.shape[0]//b)*b, (x.shape[1]//b)*b
        return x[:h, :w].reshape(h//b, b, w//b, b)

    skew = []
    for b in [1, 2, 4]:
        m = binned(mask, b).any(axis=(1, 3))
        a = binned(resid, b).mean(axis=(1, 3))[~m]
        if q is None:
            mean, median, std = sigma_clipped_stats(a, sigma=3, maxiters=5)
        else:
            # Averages of b x b grid values lie on the grid with q/b**2 step
            good = ~sigma_clip(a, sigma=3, maxiters=5, masked=True).mask
            raw = binned(image, b).mean(axis=(1, 3))[~m][good]
            mean, std = np.mean(a[good]), np.std(a[good])
            median = (photometry_measure._grouped_median(raw, q/b**2) -
                      np.mean(binned(back, b).mean(axis=(1, 3))[~m][good]))
        skew.append((mean - median)/std if std > 0 else 0)

    # Pixel-independent part a contributes a/b at binning b, on top of the part of sources
    # that does not decrease with binning. It is estimated from both binning steps, and
    # bounded by the skewness itself at every binning, as the sources may only add to it -
    # the smallest is conservative against sources fading with binning (undersampled
    # images), and against negative large-scale structure (over-subtracted halos etc)
    # growing with binning
    intrinsic = max(0, min(2*(skew[0] - skew[1]), 4*(skew[1] - skew[2]),
                           skew[0], 2*skew[1], 4*skew[2]))

    log(f"Background skewness (mean - median)/rms (x1 / x2 / x4 binning): {skew[0]:.3f} / {skew[1]:.3f} / {skew[2]:.3f}")
    log(f"Pixel-independent part: {intrinsic:.3f}")

    return {
        'skew': skew,
        'intrinsic': intrinsic,
        # For gamma-distributed noise, 3-sigma clipped (mean - median)/rms is close to
        # 1/7.4 of its standard skewness over 0.1 < gamma < 4, within 10%
        'gamma': 7.4*intrinsic,
        'rms': float(np.median(bg.rms())),
    }


def guess_bg_estimator(image, mask=None, npix=100, max_bias=0.5, verbose=False, skewness=None):
    """Guess the local background estimator suitable for aperture photometry of the image.

    Aperture sums of empty sky are unbiased only with the mean sky level, while mode and
    median are biased low when the noise itself is skewed (EM gain, low photon counts),
    and are better only when the skewness comes from the sources (faint stars, wings,
    nebulosity). The skewness of the noise is measured with :func:`measure_sky_skewness`,
    or may be passed as its result in `skewness`.

    Returns `('mean' or 'mode', bias)`, with `bias` being the expected bias of the
    aperture of `npix` pixels with the mode estimator, in units of its noise. 'mean' is
    suggested if it exceeds `max_bias`.
    """
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    if skewness is None:
        skewness = measure_sky_skewness(image, mask, verbose=verbose)

    if skewness is None:
        log("Cannot guess the background estimator, using 'mode'")
        return 'mode', np.nan

    # Mode estimate (3 median - 2 mean, as in stdpipe) is below the mean by 3 (mean - median)
    bias = 3*skewness['intrinsic']*np.sqrt(npix)

    estimator = 'mean' if bias > max_bias else 'mode'
    log(f"Mode local background bias: {bias:.2f} sigma in {npix} square pixel aperture")
    log(f"Suggested local background estimator: '{estimator}'")

    return estimator, bias


def guess_amplified_gain(skewness, gain, min_gamma=0.3, max_gain=0.2, min_ratio=5, verbose=False):
    """Effective gain of the image with amplified photon noise, e.g. EMCCD, from its skewness.

    Photon counts amplified by a random factor J (ADU per photoelectron) have the
    cumulant ratio kappa3/kappa2 = <J^3>/<J^2> = gamma*rms, while the effective gain
    (photoelectrons per ADU, defining the Poisson variance as flux/gain) is <J>/<J^2>.
    For the exponential amplification of EM register it gives gain = 1.5/(gamma*rms),
    independent of the sky level or bias pedestal.

    `skewness` is the result of :func:`measure_sky_skewness`. Returns the gain if the
    noise is strongly skewed (`gamma` > `min_gamma`), amplified (gain < `max_gain`), and
    the current `gain` is more than `min_ratio` times larger, otherwise None.
    """
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    if skewness is None or not skewness['gamma'] > min_gamma or not skewness['rms'] > 0:
        return None

    g = 1.5/(skewness['gamma']*skewness['rms'])
    log(f"Sky noise skewness {skewness['gamma']:.2f} at rms {skewness['rms']:.2f} "
        f"corresponds to amplified photon noise with effective gain {g:.3g}")

    if g < max_gain and gain > min_ratio*g:
        return g

    return None


def _block_noise(image, mask, size):
    """Noise of the background mesh estimates of size x size pixels, from the third
    differences of adjacent meshes, so that smooth large-scale structure (up to
    quadratic over four meshes, like vignetting) cancels out, while the noise, even
    correlated on smaller scales, does not.

    The mesh values are taken from SEP itself (sigma-clipped mode-like estimate, without
    median filtering), so that they respond to object wings and quantization the same
    way as the meshes being compared. Only full meshes with less than half of the pixels
    masked are used, as SEP replaces the others with interpolated values. NaN if there
    are too few of them.
    """
    from astropy.stats import mad_std

    ny, nx = image.shape[0]//size, image.shape[1]//size
    bg = sep.Background(image, mask=mask, bw=size, bh=size, fw=1, fh=1)
    back = bg.back()
    # Bicubic spline passes through the mesh values at the mesh centres
    c = size//2
    if size % 2:
        nodes = back[c:ny*size:size, c:nx*size:size]
    else:
        nodes = 0.25*(back[c-1:ny*size:size, c-1:nx*size:size] + back[c:ny*size:size, c-1:nx*size:size] +
                      back[c-1:ny*size:size, c:nx*size:size] + back[c:ny*size:size, c:nx*size:size])
    nodes = np.array(nodes[:ny, :nx], dtype=np.double)
    masked = mask[:ny*size, :nx*size].reshape(ny, size, nx, size).mean(axis=(1, 3))
    nodes[masked > 0.5] = np.nan

    # Third differences, with variance 20 times the one of independent meshes
    def diff3(m):
        return m[:, 3:] - 3*m[:, 2:-1] + 3*m[:, 1:-2] - m[:, :-3]

    d = np.concatenate([diff3(nodes).ravel(), diff3(nodes.T).ravel()])
    d = d[np.isfinite(d)]

    return mad_std(d)/np.sqrt(20) if d.size >= 10 else np.nan


def guess_bg_size(image, mask=None, min_size=64, max_size=None, tolerance=0.03,
                  noise_factor=1.25, max_structure=0.1, verbose=False):
    """Largest background mesh size still following the large-scale sky structure.

    The mesh is doubled from `min_size` up to `max_size` (quarter of the smaller image
    dimension by default) while it changes the background, estimated with detected
    objects masked, by less than max(`noise_factor` x its own noise, `tolerance`) of the
    sky rms per pixel, measured as Gaussian-equivalent 90th percentile of the change over
    the sky pixels. The noise is the one of the unfiltered mesh, while the compared
    backgrounds are median filtered, so the change of pure noise is ~0.4 of it (0.35-0.47
    on flat star fields). The default factor of 1.25 is chosen empirically, so that the
    resulting mesh minimizes the residual background error for synthetic vignetting,
    while flat fields still get the largest mesh. The noise of the mesh is measured from the third differences of adjacent
    blocks, so that correlated noise (fixed pattern, resampling) is not mistaken for
    the structure. Large-scale gradients (vignetting etc) are then followed, while flat images
    get large meshes, less affected by the noise and the wings of the objects.

    If already the first doubling changes the background by more than `max_structure`,
    it is dominated by extended emission (nebulae, galaxies, halos of bright stars)
    rather than by the sky, and smaller mesh would absorb its flux, so None is returned.
    """
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    from scipy.ndimage import binary_dilation

    if max_size is None:
        max_size = min(image.shape) // 4
    if max_size <= min_size:
        return int(max_size)

    image = np.array(image, dtype=np.double)
    mask = np.zeros(image.shape, dtype=bool) if mask is None else np.array(mask, dtype=bool)
    mask |= ~np.isfinite(image)
    if np.sum(~mask) < 10000:
        return None
    image[mask] = np.median(image[~mask])

    # Conservative mask of the objects, so that they do not drive the background
    bg = sep.Background(image, mask=mask, bw=min_size, bh=min_size)
    rms = bg.globalrms
    pixstack = sep.get_extract_pixstack()
    try:
        sep.set_extract_pixstack(max(pixstack, image.size))
        _,segm = sep.extract(image - bg.back(), 3.0, err=rms, mask=mask,
                             minarea=10, segmentation_map=True)
    except Exception as e:
        log(f"Cannot detect objects to estimate background mesh size: {e}")
        return None
    finally:
        sep.set_extract_pixstack(pixstack)

    mask_bg = mask | binary_dilation(segm > 0, iterations=8)
    sky = ~mask_bg

    size = min_size
    back = sep.Background(image, mask=mask_bg, bw=size, bh=size).back()
    while 2*size <= max_size:
        back2 = sep.Background(image, mask=mask_bg, bw=2*size, bh=2*size).back()
        # 90th percentile of the absolute change, in units of Gaussian sigma, so that
        # very local differences around bright stars, extended objects and masked
        # regions (few % of the image) do not dominate, while structure covering
        # larger areas, like vignetting in the corners, is still accounted for
        diff = (back2 - back)[sky]
        change = np.percentile(np.abs(diff - np.median(diff)), 90)/1.645/rms
        # Noise of the mode-like estimate over size x size pixels, per pixel sky rms -
        # at least the one of independent pixels, sqrt(2.5^2 pi/2 + 1.5^2 - 2*2.5*1.5)
        # = 2.14 for 2.5 median - 1.5 mean, or the measured one for correlated noise
        # (fixed pattern, resampling etc)
        noise = np.fmax(2.14/size, _block_noise(image, mask_bg, size)/rms)
        log(f"Background mesh {size} -> {2*size}: change {change:.3f} sigma, noise {noise:.3f}")

        if change > max(noise_factor*noise, tolerance):
            if size == min_size and change > max_structure:
                log("Background is dominated by extended emission, keeping default mesh size")
                return None
            break

        size *= 2
        back = back2

    return int(size)


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
        # Largest mesh still following the sky gradients, up to quarter of the image
        bg_size = guess_bg_size(image, mask, verbose=verbose)
        if bg_size is None:
            bg_size = int(min(256, min(image.shape)//4))
        log(f"Background mesh size set to {bg_size} x {bg_size} pixels")
        config['bg_size'] = bg_size

    # Sky noise skewness, for the gain of amplified (EMCCD) images and the local
    # background estimator. Measured before cosmics masking, that depends on the gain
    skewness = None
    if not config.get('gain') or not config.get('bg_estimator'):
        skewness = measure_sky_skewness(image, mask, verbose=verbose)

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

        # EM gain detection
        em_gain = guess_amplified_gain(skewness, config['gain'], verbose=verbose)
        if em_gain is not None:
            log(f"Warning: sky noise is amplified much more than gain {config['gain']:.3g} implies - probably EM gain")
            config['gain'] = em_gain

    log(f"Gain is {config['gain']:.3g}")

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

    # Local background estimator, from the skewness of the sky noise
    if not config.get('bg_estimator'):
        config['bg_estimator'],_ = guess_bg_estimator(image, mask, verbose=verbose,
                                                      skewness=skewness)

        # Mode-like SExtractor background is biased low for such noise, inflating the
        # sizes of fainter stars, that the pre-filtering then rejects as outliers
        if config['bg_estimator'] == 'mean' and 'prefilter_detections' not in config:
            log("Disabling pre-filtering of detections, as their shapes are biased by skewed sky noise")
            config['prefilter_detections'] = False

    # WCS
    wcs = get_wcs(filename, header=header, verbose=verbose)

    if wcs and wcs.is_celestial:
        ra0,dec0,sr0 = astrometry.get_frame_center(wcs=wcs, width=image.shape[1], height=image.shape[0])
        pixscale = astrometry.get_pixscale(wcs=wcs)

        log(f"Field center is at {ra0:.3f} {dec0:.3f}, radius {sr0:.2f} deg")
        log(f"Pixel scale is {3600*pixscale:.2f} arcsec/pix")

        config['refine_wcs'] = True

        if not is_wcs_usable(wcs):
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

            if not target_name:
                continue

            log(f"{target_title} is {target['name']}")

            xy = parse_pixel_coordinates(target['name'])
            if xy is not None:
                # Pixel coordinates - authoritative, sky position derived from WCS
                target['x'],target['y'] = xy
                log(f"Pixel position is x={target['x']:.1f} y={target['y']:.1f}")

                if is_wcs_usable(wcs):
                    try:
                        target['ra'],target['dec'] = [
                            float(_) for _ in wcs.all_pix2world(target['x'], target['y'], 0)
                        ]
                        log(f"Corresponds to RA={target['ra']:.4f} Dec={target['dec']:.4f}")
                    except:
                        log("Cannot convert pixel position to sky coordinates")
            else:
                try:
                    coords = resolve.resolve(target['name'])
                    target['ra'] = coords.ra.deg
                    target['dec'] = coords.dec.deg
                    log(f"Resolved to RA={target['ra']:.4f} Dec={target['dec']:.4f}")
                except:
                    log("Target not resolved")
                    continue

            if not len(config['targets']) and target.get('ra') is not None:
                # Keep backwards-compatible primary target coordinates
                config['target_ra'] = target['ra']
                config['target_dec'] = target['dec']

            # Activate target photometry mode
            config['subtraction_mode'] = 'target'

            config['targets'].append(target)

        if (config.get('target_ra') or config.get('target_dec')) and wcs and wcs.is_celestial:
            if ra0 is not None and dec0 is not None and sr0 is not None:
                if astrometry.spherical_distance(ra0, dec0,
                                                 config.get('target_ra'),
                                                 config.get('target_dec')) > 2.0*sr0:
                    log("Primary target is very far from the image center!")

        # Primary target pixel position - directly for pixel-defined targets, else through WCS
        primary = config['targets'][0] if config.get('targets') else None
        x0,y0 = None,None
        if primary is not None:
            if primary.get('x') is not None and primary.get('y') is not None:
                x0,y0 = primary['x'],primary['y']
            elif primary.get('ra') is not None and wcs and wcs.is_celestial:
                try:
                    x0,y0 = wcs.all_world2pix(primary['ra'], primary['dec'], 0)
                except:
                    pass

        if x0 is not None and y0 is not None:
            if x0 > 0 and y0 > 0 and x0 < image.shape[1] and y0 < image.shape[0]:
                cutout,cheader = cutouts.crop_image_centered(image, x0, y0, 100, header=header)
                fits.writeto(os.path.join(basepath, 'image_target.fits'), cutout, cheader, overwrite=True)
                log(f"Primary target is at x={x0:.1f} y={y0:.1f}")
                log("Primary target cutout written to file:image_target.fits")
            else:
                log("Primary target is outside the image")
                log(f"{x0} {y0}")

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
        elif config['filter'] in ['J', 'H', 'Ks']:
            config['cat_name'] = 'vhs' if (dec0 is not None and dec0 < 0) or config.get('target_dec', 0) < 0 else '2mass'
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
        time = None

        # DATE-AVG (FITS standard mean epoch, written e.g. by Siril and N.I.N.A.)
        # is the exposure midpoint, or the mean epoch of a stack, while DATE-OBS
        # is usually the start of the (first) exposure
        if header.get('DATE-AVG'):
            try:
                time = utils.get_obs_time(string=str(header['DATE-AVG']), verbose=verbose)
            except Exception:
                time = None

            if time is not None:
                log(f"Using DATE-AVG {header['DATE-AVG']} as observation time")

        if time is None:
            time = utils.get_obs_time(header=header, verbose=verbose)

        if time is not None:
            config['time'] = time.iso

    if config.get('time'):
        log(f"Time is {config.get('time')}")
        log(f"MJD is {Time(config.get('time')).mjd}")
