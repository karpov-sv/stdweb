import os, glob, shutil

import numpy as np
from astropy.wcs import WCS
from astropy.io import fits as fits

from astropy.table import Table, vstack
from astropy.stats import mad_std
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time

import astroscrappy
import time

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


def fix_header(header, verbose=True):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    # Fix PRISM headers
    if header.get('CTYPE2') == 'DEC---TAN':
        header['CTYPE2'] = 'DEC--TAN'
    for _ in ['CDELTM1', 'CDELTM2', 'XPIXELSZ', 'YPIXELSZ']:
        header.remove(_, ignore_missing=True)
    if header.get('CTYPE1') == 'RA---TAN':
        for _ in ['PV1_1', 'PV1_2']:
            header.remove(_, ignore_missing=True)

    if 'FOCALLEN' in header and not header.get('FOCALLEN'):
        header.remove('FOCALLEN')


def inspect_image(filename, config, verbose=True, show=False):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    basepath = os.path.dirname(filename)

    config['sn'] = config.get('sn', 5)
    config['initial_aper'] = config.get('initial_aper', 3)
    config['rel_aper'] = config.get('rel_aper', 1)
    config['rel_bkgann'] = config.get('rel_bkgann', None)
    config['bg_size'] = config.get('bg_size', 256)
    config['spatial_order'] = config.get('spatial_order', 2)
    config['minarea'] = config.get('minarea', 5)
    config['use_color'] = config.get('use_color', True)
    config['mask_cosmics'] = config.get('mask_cosmics', True)

    # Load the image
    log(f'Inspecting {filename}')
    image,header = fits.getdata(filename).astype(np.double), fits.getheader(filename)

    log(f"Image size is {image.shape[1]} x {image.shape[0]}")

    fix_header(header)

    # Guess some parameters from keywords
    if not config.get('gain'):
        config['gain'] = header.get('GAIN', 1)
    log(f"Gain is {config['gain']:.2f}")

    # Filter
    if not config.get('filter'):
        config['filter'] = header.get('FILTER', header.get('CAMFILT', 'unknown')).strip()
    log(f"Filter is {config['filter']}")

    # Normalize filters
    filter_aliases = {
        # Johnson-Cousins
        'B': [],
        'V': [],
        'R': ["Rc"],
        'I': ["Ic"],
        # Sloan-like
        'u': ["SDSS-u", "SDSS-u'", "Sloan-u"],
        'g': ["SDSS-g", "SDSS-g'", "Sloan-g"],
        'r': ["SDSS-r", "SDSS-r'", "Sloan-r"],
        'i': ["SDSS-i", "SDSS-i'", "Sloan-i"],
        'z': ["SDSS-z", "SDSS-z'", "Sloan-z"],
        }

    for fname in filter_aliases.keys():
        if config['filter'] in filter_aliases[fname]:
            config['filter'] = fname
            log(f"Filter name normalized to {config['filter']}")
            break

    # Saturation
    if not config.get('saturation'):
        satlevel = header.get('SATURATE',
                              header.get('DATAMAX'))
        if not satlevel:
            satlevel = 0.95*np.nanmax(image)
        config['saturation'] = satlevel
    log(f"Saturation level is {config['saturation']:.1f}")

    # Mask
    mask = np.isnan(image)
    mask |= image >= config['saturation']

    # Cosmics
    if config['mask_cosmics']:
        cmask, cimage = astroscrappy.detect_cosmics(image, mask, verbose=False,
                                                    gain=config['gain'],
                                                    satlevel=config['saturation'],
                                                    cleantype='medmask')
        log(f"Done masking cosmics, {np.sum(cmask)} ({100*np.sum(cmask)/cmask.shape[0]/cmask.shape[1]:.1f}%) pixels masked")
        mask |= cmask

    log(f"{np.sum(mask)} ({100*np.sum(mask)/mask.shape[0]/mask.shape[1]:.1f}%) pixels masked")

    fits.writeto(os.path.join(basepath, 'mask.fits'), mask.astype(np.int8), header, overwrite=True)
    log("Mask written to mask.fits")

    # WCS
    wcs = WCS(header)
    if wcs and wcs.is_celestial:
        ra0,dec0,sr0 = astrometry.get_frame_center(wcs=wcs, width=image.shape[1], height=image.shape[0])
        pixscale = astrometry.get_pixscale(wcs=wcs)

        log(f"Field center is at {ra0:.3f} {dec0:.3f}, radius {sr0:.2f} deg")
        log(f"Pixel scale is {3600*pixscale:.2f} arcsec/pix")
    else:
        log("No usable WCS found")

    # Target?..
    if not 'target' in config:
        config['target'] = header.get('TARGET')

    if config['target']:
        log(f"Target is {config['target']}")
        try:
            target = SkyCoord.from_name(config['target'])
            config['target_ra'] = target.ra.deg
            config['target_dec'] = target.dec.deg
            log(f"Resolved to {config['target_ra']:.3f} {config['target_ra']:.3f}")
        except:
            log("Target not resolved")


def photometry_image(filename, config, verbose=True, show=False):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    basepath = os.path.dirname(filename)

    # Image
    image,header = fits.getdata(filename).astype(np.double), fits.getheader(filename)

    fix_header(header)

    # Mask
    mask = fits.getdata(os.path.join(basepath, 'mask.fits')) > 0

    # Cleanup stale plots
    for name in ['photometry.png', 'photometry_unmasked.png',
                 'photometry_zeropoint.png', 'astrometry_dist.png']:
        fullname = os.path.join(basepath, name)
        if os.path.exists(fullname):
            os.unlink(fullname)

    # Extract objects
    obj = photometry.get_objects_sextractor(image, mask=mask,
                                            aper=config.get('initial_aper', 3.0),
                                            gain=config.get('gain', 1.0),
                                            extra={'BACK_SIZE': config.get('bg_size', 256)},
                                            minarea=config.get('minarea', 3),
                                            verbose=verbose)

    log(f"{len(obj)} objects found")

    # FWHM
    fwhm = np.median(obj['fwhm'][obj['flags'] == 0]) # TODO: make it position-dependent
    log(f"FWHM is {fwhm:.2f} pixels")
    config['fwhm'] = fwhm

    # Forced photometry at objects positions
    obj = photometry.measure_objects(obj, image, mask=mask,
                                     fwhm=fwhm,
                                     aper=config.get('rel_aper', 1.0),
                                     bkgann=config.get('rel_bkgann', None),
                                     sn=config.get('sn', 3.0),
                                     bg_size=config.get('bg_size', 256),
                                     gain=config.get('gain', 1.0),
                                     verbose=verbose)

    log(f"{len(obj)} objects properly measured")

    obj.write(os.path.join(basepath, 'objects.vot'), format='votable', overwrite=True)

    # Plot detected objects
    with plots.figure_saver(os.path.join(basepath, 'objects.png'), figsize=(8, 6), show=show,) as fig:
        ax = fig.add_subplot(1, 1, 1)
        idx = obj['flags'] == 0
        ax.plot(obj['x'][idx], obj['y'][idx], '.', label='Unflagged')
        ax.plot(obj['x'][~idx], obj['y'][~idx], '.', label='Flagged')
        ax.set_aspect(1)
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(0, image.shape[0])
        # ax.legend()
        ax.set_title(f"Detected objects: {np.sum(idx)} unmasked, {np.sum(~idx)} masked")

    # Plot FWHM
    with plots.figure_saver(os.path.join(basepath, 'fwhm.png'), figsize=(8, 6), show=True) as fig:
        ax = fig.add_subplot(1, 1, 1)
        idx = obj['flags'] == 0
        plots.binned_map(obj[idx]['x'], obj[idx]['y'], obj[idx]['fwhm'], bins=8, statistic='median', show_dots=True, ax=ax)
        ax.set_aspect(1)
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(0, image.shape[0])
        # ax.legend()
        ax.set_title(f"FWHM: median {np.median(obj[idx]['fwhm']):.2f} pix RMS {np.std(obj[idx]['fwhm']):.2f} pix")

    # Catalogue settings
    # config['cat_name'] = 'gaiadr3syn'
    config['cat_name'] = 'ps1'

    if config['cat_name'] == 'gaiadr3syn':
        config['cat_col_mag'] = config['filter'] + 'mag'
        config['cat_col_mag_err'] = 'e_' + config['cat_col_mag']

        config['cat_col_color_mag1'] = 'Bmag'
        config['cat_col_color_mag2'] = 'Vmag'
    else:
        config['cat_col_mag'] = config['filter'] + 'mag'
        config['cat_col_mag_err'] = 'e_rmag'

        config['cat_col_color_mag1'] = 'gmag'
        config['cat_col_color_mag2'] = 'rmag'

    # Get initial WCS
    wcs = WCS(header)

    if wcs is None or not wcs.is_celestial:
        raise RuntimeError('No WCS astrometric solution')

    obj['ra'],obj['dec'] = wcs.all_pix2world(obj['x'], obj['y'], 0)

    # Get reference catalogue
    ra0,dec0,sr0 = astrometry.get_frame_center(wcs=wcs, width=image.shape[1], height=image.shape[0])
    pixscale = astrometry.get_pixscale(wcs=wcs)

    log(f"Field center is at {ra0:.3f} {dec0:.3f}, radius {sr0:.2f} deg, scale {3600*pixscale:.2f} arcsec/pix")

    cat = catalogs.get_cat_vizier(ra0, dec0, sr0, config['cat_name'], filters={'rmag':'<20'})

    log(f"Got {len(cat)} catalogue stars from {config['cat_name']}")

    # Photometric calibration
    m = pipeline.calibrate_photometry(obj, cat, pixscale=pixscale,
                                      cat_col_mag=config['cat_col_mag'],
                                      cat_col_mag_err=config['cat_col_mag_err'],
                                      cat_col_mag1=config.get('cat_col_color_mag1'),
                                      cat_col_mag2=config.get('cat_col_color_mag2'),
                                      order=config['spatial_order'],
                                      robust=True, scale_noise=True,
                                      accept_flags=0x02, max_intrinsic_rms=0.01,
                                      verbose=verbose)

    if m is None:
        raise RuntimeError('Photometric match failed')

    # Plot photometric solution
    with plots.figure_saver(os.path.join(basepath, 'photometry.png'), figsize=(8, 6), show=show) as fig:
        ax = fig.add_subplot(2, 1, 1)
        plots.plot_photometric_match(m, mode='mag', ax=ax)
        ax = fig.add_subplot(2, 1, 2)
        plots.plot_photometric_match(m, mode='color', ax=ax)

    with plots.figure_saver(os.path.join(basepath, 'photometry_unmasked.png'), figsize=(8, 6), show=show) as fig:
        ax = fig.add_subplot(2, 1, 1)
        plots.plot_photometric_match(m, mode='mag', show_masked=False, ax=ax)
        ax = fig.add_subplot(2, 1, 2)
        plots.plot_photometric_match(m, mode='color', show_masked=False, ax=ax)

    with plots.figure_saver(os.path.join(basepath, 'photometry_zeropoint.png'), figsize=(8, 6), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)
        plots.plot_photometric_match(m, mode='zero', show_dots=True, bins=8, ax=ax)
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(0, image.shape[0])

    with plots.figure_saver(os.path.join(basepath, 'astrometry_dist.png'), figsize=(8, 6), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)
        plots.plot_photometric_match(m, mode='dist', show_dots=True, bins=8, ax=ax)
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(0, image.shape[0])
