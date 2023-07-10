import os, glob, shutil

from functools import partial

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

import sep

import dill as pickle

from . import settings

# STDPipe
from stdpipe import astrometry, photometry, catalogs, cutouts
from stdpipe import templates, subtraction, plots, pipeline, utils, psf

# Disable some annoying warnings from astropy
import warnings
from astropy.wcs import FITSFixedWarning
from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter(action='ignore', category=FITSFixedWarning)
warnings.simplefilter(action='ignore', category=VerifyWarning)


# Supported filters and their aliases
supported_filters = {
    # Johnson-Cousins
    'U': {'name':'Johnson-Cousins U', 'aliases':[]},
    'B': {'name':'Johnson-Cousins B', 'aliases':[]},
    'V': {'name':'Johnson-Cousins V', 'aliases':[]},
    'R': {'name':'Johnson-Cousins R', 'aliases':["Rc"]},
    'I': {'name':'Johnson-Cousins I', 'aliases':["Ic", "I'"]},
    # Sloan-like
    'u': {'name':'Sloan u', 'aliases':["sdssu", "SDSS u", "SDSS-u", "SDSS-u'", "Sloan-u", "sloanu"]},
    'g': {'name':'Sloan g', 'aliases':["sdssg", "SDSS g", "SDSS-g", "SDSS-g'", "Sloan-g", "sloang", "SG", "sG"]},
    'r': {'name':'Sloan r', 'aliases':["sdssr", "SDSS r", "SDSS-r", "SDSS-r'", "Sloan-r", "sloanr", "SR", "sR"]},
    'i': {'name':'Sloan i', 'aliases':["sdssi", "SDSS i", "SDSS-i", "SDSS-i'", "Sloan-i", "sloani", "SI", "sI"]},
    'z': {'name':'Sloan z', 'aliases':["sdssz", "SDSS z", "SDSS-z", "SDSS-z'", "Sloan-z", "sloanz"]},
    # Gaia
    'G': {'name':'Gaia G', 'aliases':[]},
    'BP': {'name':'Gaia BP', 'aliases':[]},
    'RP': {'name':'Gaia RP', 'aliases':[]},
}

supported_catalogs = {
    'gaiadr3syn': {'name':'Gaia DR3 synphot', 'filters':['U', 'B', 'V', 'R', 'I', 'g', 'r', 'i', 'z', 'y'],
                   'limit': 'Gmag'},
    'ps1': {'name':'Pan-STARRS DR1', 'filters':['B', 'V', 'R', 'I', 'g', 'r', 'i', 'z'],
            'limit':'rmag'},
    'skymapper': {'name':'SkyMapper DR1.1', 'filters':['B', 'V', 'R', 'I', 'g', 'r', 'i', 'z'],
                  'limit':'rPSF'},
    'atlas': {'name':'ATLAS-REFCAT2', 'filters':['B', 'V', 'R', 'I', 'g', 'r', 'i', 'z'],
              'limit':'rmag'},
    'gaiaedr3': {'name':'Gaia eDR3', 'filters':['G', 'BP', 'RP'],
              'limit':'Gmag'},
}

supported_templates = {
    'ps1': {'name': 'Pan-STARRS DR2', 'filters': {'g', 'r', 'i', 'z'}},
    'skymapper': {'name': 'SkyMapper DR1 (HiPS)', 'filters': {
        'u': 'CDS/P/skymapper-U',
        'g': 'CDS/P/skymapper-G',
        'r': 'CDS/P/skymapper-R',
        'i': 'CDS/P/skymapper-I',
        'z': 'CDS/P/skymapper-Z',
    }},
    'des': {'name': 'Dark Energy Survey DR2 (HiPS)', 'filters': {
        'g': 'CDS/P/DES-DR2/g',
        'r': 'CDS/P/DES-DR2/r',
        'i': 'CDS/P/DES-DR2/i',
        'z': 'CDS/P/DES-DR2/z',
    }},
    'legacy': {'name': 'DESI Legacy Surveys DR10 (HiPS)', 'filters': {
        'g': 'CDS/P/DESI-Legacy-Surveys/DR10/g',
        'i': 'CDS/P/DESI-Legacy-Surveys/DR10/i',
    }},
    'decaps': {'name': 'DECaPS DR2 (HiPS)', 'filters': {
        'g': 'CDS/P/DECaPS/DR2/g',
        'r': 'CDS/P/DECaPS/DR2/r',
        'i': 'CDS/P/DECaPS/DR2/i',
        'z': 'CDS/P/DECaPS/DR2/z',
    }},
}

# Best guess template filter mappings
filter_mappings = {
    'U': ['u', 'g'],
    'B': ['u', 'g'],
    'V': ['g'],
    'R': ['r', 'i'],
    'I': ['i', 'z'],
    'u': ['u', 'g'],
    'g': ['g', 'g'],
    'r': ['r', 'i'],
    'i': ['i', 'r'],
    'z': ['z', 'r'],
}


# Files created at every step

files_inspect = [
    'inspect.log',
    'mask.fits', 'image_target.fits'
]

files_photometry = [
    'photometry.log',
    'objects.png', 'fwhm.png',
    'photometry.png', 'photometry_unmasked.png',
    'photometry_zeropoint.png', 'photometry_model.png',
    'photometry_residuals.png', 'astrometry_dist.png',
    'photometry.pickle',
    'objects.vot', 'cat.vot',
    'limit_hist.png', 'limit_sn.png',
    'target.vot', 'target.cutout'
]

files_subtraction = [
    'subtraction.log',
    'sub_image.fits', 'sub_mask.fits',
    'sub_template.fits', 'sub_template_mask.fits',
    'sub_diff.fits', 'sub_sdiff.fits', 'sub_conv.fits', 'sub_ediff.fits',
    'sub_target.vot', 'sub_target.cutout',
]

cleanup_inspect = files_inspect + files_photometry + files_subtraction

cleanup_photometry = files_photometry + files_subtraction

cleanup_subtraction = files_subtraction


def print_to_file(*args, clear=False, logname='out.log', **kwargs):
    if clear and os.path.exists(logname):
        print('Clearing', logname)
        os.unlink(logname)

    if len(args) or len(kwargs):
        print(*args, **kwargs)
        with open(logname, 'a+') as lfd:
            print(file=lfd, *args, **kwargs)


def pickle_to_file(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def pickle_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


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


def fix_image(filename, config, verbose=True):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    basepath = os.path.dirname(filename)

    image,header = fits.getdata(filename), fits.getheader(filename)
    fix_header(header)

    if os.path.exists(os.path.join(basepath, 'image.wcs')):
        wcs = WCS(fits.getheader(os.path.join(basepath, "image.wcs")))
        astrometry.clear_wcs(header)
        header += wcs.to_header(relax=True)
        header.add_comment('WCS updated by STDWeb', before='WCSAXES')

    # Write fixed image and header back
    fits.writeto(filename, image, header, overwrite=True)


def crop_image(filename, config, x1=None, y1=None, x2=None, y2=None, verbose=True):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    basepath = os.path.dirname(filename)

    image,header = fits.getdata(filename), fits.getheader(filename)

    # Sanitize input
    try:
        x1 = int(x1)
    except:
        x1 = 0

    try:
        y1 = int(y1)
    except:
        y1 = 0

    try:
        x2 = int(x2)
    except:
        x2 = image.shape[1]

    try:
        y2 = int(y2)
    except:
        y2 = image.shape[0]

    # Interpret negative values as offsets from the top
    if x1 < 0:
        x1 = image.shape[1] + x1
    if x2 < 0:
        x2 = image.shape[1] + x2
    if y1 < 0:
        y1 = image.shape[0] + y1
    if y2 < 0:
        y2 = image.shape[0] + y2

    # Ensure we are within the image
    x1,x2 = max(0, x1), min(image.shape[1], x2)
    y1,y2 = max(0, y1), min(image.shape[0], y2)

    image,header = cutouts.crop_image(image, x1, y1, x2 - x1, y2 - y1, header=header)

    # Write cropped image and header back
    fits.writeto(filename, image, header, overwrite=True)


def inspect_image(filename, config, verbose=True, show=False):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    basepath = os.path.dirname(filename)

    # Cleanup stale plots
    for name in cleanup_inspect:
        fullname = os.path.join(basepath, name)
        if os.path.exists(fullname):
            os.unlink(fullname)

    config['sn'] = config.get('sn', 3)
    config['initial_aper'] = config.get('initial_aper', 3)
    config['initial_r0'] = config.get('initial_r0', 0)
    config['rel_aper'] = config.get('rel_aper', 1)
    config['rel_bg1'] = config.get('rel_bg1', 5)
    config['rel_bg2'] = config.get('rel_bg2', 7)
    config['bg_size'] = config.get('bg_size', 256)
    config['spatial_order'] = config.get('spatial_order', 2)
    config['minarea'] = config.get('minarea', 5)
    config['use_color'] = config.get('use_color', True)
    config['refine_wcs'] = config.get('refine_wcs', True)
    config['blind_match_wcs'] = config.get('blind_match_wcs', False)

    config['sub_size'] = config.get('sub_size', 1000)
    config['sub_overlap'] = config.get('sub_overlap', 50)
    config['sub_verbose'] = config.get('sub_verbose', False)
    config['detect_transients'] = config.get('detect_transients', False)

    # Load the image
    log(f'Inspecting {filename}')
    try:
        image,header = fits.getdata(filename).astype(np.double), fits.getheader(filename)
    except:
        raise RuntimeError('Cannot load the image')

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
    if config.get('mask_cosmics', True):
        cmask, cimage = astroscrappy.detect_cosmics(image, mask, verbose=False,
                                                    gain=config.get('gain', 1),
                                                    satlevel=config.get('saturation'),
                                                    cleantype='medmask')
        log(f"Done masking cosmics, {np.sum(cmask)} ({100*np.sum(cmask)/cmask.shape[0]/cmask.shape[1]:.1f}%) pixels masked")
        mask |= cmask

    log(f"{np.sum(mask)} ({100*np.sum(mask)/mask.shape[0]/mask.shape[1]:.1f}%) pixels masked")

    if np.sum(mask) > 0.5*mask.shape[0]*mask.shape[1]:
        raise RuntimeError('More than half of the image is masked')

    fits.writeto(os.path.join(basepath, 'mask.fits'), mask.astype(np.int8), header, overwrite=True)
    log("Mask written to mask.fits")

    # WCS
    if os.path.exists(os.path.join(basepath, "image.wcs")):
        wcs = WCS(fits.getheader(os.path.join(basepath, "image.wcs")))
        astrometry.clear_wcs(header)
        header += wcs.to_header(relax=True)
        log("WCS loaded from image.wcs")
    else:
        wcs = WCS(header)
        log("Using original WCS from FITS header")

    if wcs and wcs.is_celestial:
        ra0,dec0,sr0 = astrometry.get_frame_center(wcs=wcs, width=image.shape[1], height=image.shape[0])
        pixscale = astrometry.get_pixscale(wcs=wcs)

        log(f"Field center is at {ra0:.3f} {dec0:.3f}, radius {sr0:.2f} deg")
        log(f"Pixel scale is {3600*pixscale:.2f} arcsec/pix")
    else:
        ra0,dec0,sr0 = None,None,None
        config['blind_match_wcs'] = True
        log("No usable WCS found, blind matching enabled")

    # Target?..
    if not 'target' in config:
        config['target'] = header.get('TARGET')

    if config.get('target'):
        log(f"Target is {config['target']}")
        try:
            target = SkyCoord.from_name(config['target'])
            config['target_ra'] = target.ra.deg
            config['target_dec'] = target.dec.deg
            log(f"Resolved to RA={config['target_ra']:.4f} Dec={config['target_dec']:.4f}")
        except:
            log("Target not resolved")

        if (config.get('target_ra') or config.get('target_dec')) and wcs and wcs.is_celestial:
            if ra0 is not None and dec0 is not None and sr0 is not None:
                if astrometry.spherical_distance(ra0, dec0,
                                                 config.get('target_ra'),
                                                 config.get('target_dec')) > 2.0*sr0:
                    log("Target is very far from the image center!")

            try:
                x0,y0 = wcs.all_world2pix(config.get('target_ra'), config.get('target_dec'), 0)

                if x0 > 0 and y0 > 0 and x0 < image.shape[1] and y0 < image.shape[0]:
                    cutout,cheader = cutouts.crop_image_centered(image, x0, y0, 100, header=header)
                    fits.writeto(os.path.join(basepath, 'image_target.fits'), cutout, cheader, overwrite=True)
                    log(f"Target is at x={x0:.1f} y={y0:.1f}")
                    log("Target cutout written to image_target.fits")
                else:
                    log("Target is outside the image")
            except:
                pass

        # We may initialize some blind match parameters from the target position, if any
        if config.get('target_ra') is not None and config.get('blind_match_ra0') is None:
            config['blind_match_ra0'] = config.get('target_ra')
        if config.get('target_dec') is not None and config.get('blind_match_dec0') is None:
            config['blind_match_dec0'] = config.get('target_dec')

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
        if (dec0 is not None and dec0 < -30) or config.get('target_dec', 0) < -30:
            config['template'] = 'skymapper'
        else:
            config['template'] = 'ps1'

        log(f"Suggested template is {supported_templates[config['template']]['name']}")


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
    for name in cleanup_photometry:
        fullname = os.path.join(basepath, name)
        if os.path.exists(fullname):
            os.unlink(fullname)

    log("\n---- Object detection ----\n")

    # Extract objects
    obj = photometry.get_objects_sextractor(image, mask=mask,
                                            aper=config.get('initial_aper', 3.0),
                                            gain=config.get('gain', 1.0),
                                            extra={'BACK_SIZE': config.get('bg_size', 256)},
                                            minarea=config.get('minarea', 3),
                                            verbose=verbose,
                                            _tmpdir=settings.STDPIPE_TMPDIR,
                                            _exe=settings.STDPIPE_SEXTRACTOR)

    log(f"{len(obj)} objects found")

    if not len(obj):
        raise RuntimeError('Cannot detect objects on the image')

    log("\n---- Object measurement ----\n")

    # FWHM
    fwhm = np.median(obj['fwhm'][obj['flags'] == 0]) # TODO: make it position-dependent
    log(f"FWHM is {fwhm:.2f} pixels")

    if config.get('fwhm_override'):
        fwhm = config.get('fwhm_override')
        log(f"Overriding with user-specified FWHM value of {fwhm:.2f} pixels")

    config['fwhm'] = fwhm

    if config.get('rel_bg1') and config.get('rel_bg2'):
        rel_bkgann = [config['rel_bg1'], config['rel_bg2']]
    else:
        rel_bkgann = None

    # Forced photometry at objects positions
    obj = photometry.measure_objects(obj, image, mask=mask,
                                     fwhm=fwhm,
                                     aper=config.get('rel_aper', 1.0),
                                     bkgann=rel_bkgann,
                                     sn=config.get('sn', 3.0),
                                     bg_size=config.get('bg_size', 256),
                                     gain=config.get('gain', 1.0),
                                     verbose=verbose)

    log(f"{len(obj)} objects properly measured")

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

    log("\n---- Initial astrometry ----\n")

    # Get initial WCS
    if os.path.exists(os.path.join(basepath, "image.wcs")):
        wcs = WCS(fits.getheader(os.path.join(basepath, "image.wcs")))
        astrometry.clear_wcs(header)
        header += wcs.to_header(relax=True)
        log("WCS loaded from image.wcs")
    else:
        wcs = WCS(header)
        log("Using original WCS from FITS header")

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

        wcs = astrometry.blind_match_objects(obj[:500],
                                             center_ra=config.get('blind_match_ra0'),
                                             center_dec=config.get('blind_match_dec0'),
                                             radius=config.get('blind_match_sr0'),
                                             scale_lower=config.get('blind_match_ps_lo'),
                                             scale_upper=config.get('blind_match_ps_up'),
                                             sn=sn0,
                                             verbose=verbose,
                                             _tmpdir=settings.STDPIPE_TMPDIR,
                                             _exe=settings.STDPIPE_SOLVE_FIELD,
                                             config=settings.STDPIPE_SOLVE_FIELD_CONFIG)

        if wcs is not None and wcs.is_celestial:
            astrometry.store_wcs(os.path.join(basepath, "image.wcs"), wcs)
            astrometry.clear_wcs(header)
            header += wcs.to_header(relax=True)
            config['blind_match_wcs'] = False
            log("Blind matched WCS stored to image.wcs")
        else:
            log("Blind matching failed")

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
    cat = catalogs.get_cat_vizier(ra0, dec0, sr0, config['cat_name'], filters=filters, verbose=verbose)

    if not cat or not len(cat):
        raise RuntimeError('Cannot get catalogue stars')

    log(f"Got {len(cat)} catalogue stars from {config['cat_name']}")

    cat.write(os.path.join(basepath, 'cat.vot'), format='votable', overwrite=True)
    log("Catalogue written to cat.vot")

    # Catalogue settings
    if config.get('cat_name') == 'gaiadr3syn':
        config['cat_col_mag'] = config['filter'] + 'mag'
        config['cat_col_mag_err'] = 'e_' + config['cat_col_mag']

        config['cat_col_color_mag1'] = 'Bmag'
        config['cat_col_color_mag2'] = 'Vmag'
    elif config.get('cat_name') == 'gaiaedr3':
        config['cat_col_mag'] = config['filter'] + 'mag'
        config['cat_col_mag_err'] = 'e_' + config['cat_col_mag']

        config['cat_col_color_mag1'] = 'BPmag'
        config['cat_col_color_mag2'] = 'RPmag'
    else:
        config['cat_col_mag'] = config['filter'] + 'mag'
        config['cat_col_mag_err'] = 'e_' + config['cat_col_mag']

        config['cat_col_color_mag1'] = 'gmag'
        config['cat_col_color_mag2'] = 'rmag'

    log(f"Will use catalogue column {config['cat_col_mag']} as primary magnitude ")
    log(f"Will use catalogue columns {config['cat_col_color_mag1']} and {config['cat_col_color_mag2']} for color")

    if not (config['cat_col_mag'] in cat.colnames and
            (not config.get('cat_col_color_mag1') or config['cat_col_color_mag1'] in cat.colnames) and
            (not config.get('cat_col_color_mag2') or config['cat_col_color_mag2'] in cat.colnames)):
        raise RuntimeError('Catalogue does not have required magnitudes')

    # Astrometric refinement
    if config.get('refine_wcs', False):
        log("\n---- Astrometric refinement ----\n")

        wcs1 = pipeline.refine_astrometry(obj, cat, fwhm*pixscale,
                                          wcs=wcs, order=3, method='scamp',
                                          cat_col_mag=config.get('cat_col_mag'),
                                          cat_col_mag_err=config.get('cat_col_mag_err'),
                                          verbose=verbose,
                                          _tmpdir=settings.STDPIPE_TMPDIR,
                                          _exe=settings.STDPIPE_SCAMP)
        if wcs1 is None or not wcs1.is_celestial:
            raise RuntimeError('WCS refinement failed')
        else:
            wcs = wcs1
            astrometry.store_wcs(os.path.join(basepath, "image.wcs"), wcs)
            astrometry.clear_wcs(header)
            header += wcs.to_header(relax=True)
            config['refine_wcs'] = False
            log("Refined WCS stored to image.wcs")

    log("\n---- Photometric calibration ----\n")

    # Photometric calibration
    m = pipeline.calibrate_photometry(obj, cat, pixscale=pixscale,
                                      cat_col_mag=config.get('cat_col_mag'),
                                      cat_col_mag_err=config.get('cat_col_mag_err'),
                                      cat_col_mag1=config.get('cat_col_color_mag1'),
                                      cat_col_mag2=config.get('cat_col_color_mag2'),
                                      use_color=config.get('use_color', True),
                                      order=config.get('spatial_order', 0),
                                      robust=True, scale_noise=True,
                                      accept_flags=0x02, max_intrinsic_rms=0.01,
                                      verbose=verbose)

    if m is None:
        raise RuntimeError('Photometric match failed')

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
        ax = fig.add_subplot(2, 1, 2)
        plots.plot_photometric_match(m, mode='color', show_masked=False, ax=ax)

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
    obj['mag_calib'] = obj['mag'] + m['zero_fn'](obj['x'], obj['y'], obj['mag'])
    obj['mag_calib_err'] = np.hypot(obj['magerr'],
                                    m['zero_fn'](obj['x'], obj['y'], obj['mag'], get_err=True))

    obj.write(os.path.join(basepath, 'objects.vot'), format='votable', overwrite=True)
    log("Measured objects stored to objects.vot")

    # Detection limits
    log("\n---- Global detection limit ----\n")
    mag0 = pipeline.get_detection_limit(obj, sn=config.get('sn'), verbose=verbose)
    config['mag_limit'] = mag0

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
    if config.get('target_ra') is not None and config.get('target_dec') is not None:
        log("\n---- Target forced photometry ----\n")

        target_obj = Table({'ra':[config['target_ra']], 'dec':[config['target_dec']]})
        target_obj['x'],target_obj['y'] = wcs.all_world2pix(target_obj['ra'], target_obj['dec'], 0)

        if not (target_obj['x'] > 0 and target_obj['x'] < image.shape[1] and
                target_obj['y'] > 0 and target_obj['y'] < image.shape[0]):
            raise RuntimeError("Target is outside the image")

        log(f"Target position is {target_obj['ra'][0]:.3f} {target_obj['dec'][0]:.3f} -> {target_obj['x'][0]:.1f} {target_obj['y'][0]:.1f}")

        target_obj = photometry.measure_objects(target_obj, image, mask=mask,
                                                fwhm=fwhm,
                                                aper=config.get('rel_aper', 1.0),
                                                bkgann=rel_bkgann,
                                                sn=0,
                                                bg_size=config.get('bg_size', 256),
                                                gain=config.get('gain', 1.0),
                                                verbose=verbose)

        target_obj['mag_calib'] = target_obj['mag'] + m['zero_fn'](target_obj['x'],
                                                                   target_obj['y'],
                                                                   target_obj['mag'])

        target_obj['mag_calib_err'] = np.hypot(target_obj['magerr'],
                                               m['zero_fn'](target_obj['x'],
                                                            target_obj['y'],
                                                            target_obj['mag'],
                                                            get_err=True))

        # TODO: Improve limiting mag estimate
        target_obj['mag_limit'] = -2.5*np.log10(config.get('sn', 5)*target_obj['fluxerr']) + m['zero_fn'](target_obj['x'], target_obj['y'], target_obj['mag'])

        target_obj['mag_filter_name'] = m['cat_col_mag']

        if 'cat_col_mag1' in m.keys() and 'cat_col_mag2' in m.keys():
            target_obj['mag_color_name'] = '%s - %s' % (m['cat_col_mag1'], m['cat_col_mag2'])
            target_obj['mag_color_term'] = m['color_term']

        target_obj.write(os.path.join(basepath, 'target.vot'), format='votable', overwrite=True)
        log("Measured target stored to target.vot")

        # Create the cutout from image based on the candidate
        cutout = cutouts.get_cutout(image, target_obj[0], 30, mask=mask, header=header)
        # Cutout from relevant HiPS survey
        if target_obj['dec'][0] > -30:
            cutout['template'] = templates.get_hips_image('PanSTARRS/DR1/r', header=cutout['header'])[0]
        else:
            cutout['template'] = templates.get_hips_image('CDS/P/skymapper-R', header=cutout['header'])[0]

        cutouts.write_cutout(cutout, os.path.join(basepath, 'target.cutout'))
        log("Target cutouts stored to target.cutout")

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
    detect_transients = config.get('detect_transients', False)

    # Cleanup stale plots and files
    for name in cleanup_subtraction:
        fullname = os.path.join(basepath, name)
        if os.path.exists(fullname):
            os.unlink(fullname)

    # Image
    image,header = fits.getdata(filename).astype(np.double), fits.getheader(filename)

    fix_header(header)

    # Mask
    mask = fits.getdata(os.path.join(basepath, 'mask.fits')) > 0

    # Photometric solution
    m = pickle_from_file(os.path.join(basepath, 'photometry.pickle'))

    # Objects
    obj = Table.read(os.path.join(basepath, 'objects.vot'))

    # Catalogue
    cat = Table.read(os.path.join(basepath, 'cat.vot'))

    # WCS
    if os.path.exists(os.path.join(basepath, "image.wcs")):
        wcs = WCS(fits.getheader(os.path.join(basepath, "image.wcs")))
        astrometry.clear_wcs(header)
        header += wcs.to_header(relax=True)
        log("WCS loaded from image.wcs")
    else:
        wcs = WCS(header)
        log("Using original WCS from FITS header")

    if wcs is None or not wcs.is_celestial:
        raise RuntimeError('No WCS astrometric solution')

    log("\n---- Template selection ----\n")

    tname = config.get('template', 'ps1')
    tconf = supported_templates.get(tname)

    if tconf is None:
        raise RuntimeError(f"Unsupported template: {tname}")

    tfilter = None
    for _ in filter_mappings[config['filter']]:
        if _ in tconf['filters']:
            tfilter = _
            break

    log(f"Using {tconf['name']} in filter {tfilter} as a template")

    sub_size = config.get('sub_size', 1000)
    sub_overlap = config.get('sub_overlap', 50)

    if detect_transients:
        log('Transient detection mode activated')
        # We will split the image into nx x ny blocks
        nx = int(np.round(image.shape[1] / sub_size))
        ny = int(np.round(image.shape[0] / sub_size))
        log(f"Will split the image into {nx} x {ny} sub-images")
        split_fn = partial(pipeline.split_image, get_index=True, overlap=sub_overlap, nx=nx, ny=ny)

    elif config.get('target_ra') is not None:
        log('Forced photometry mode activated')
        # We will just crop the image
        x0,y0 = wcs.all_world2pix(config['target_ra'], config['target_dec'], 0)
        log(f"Will crop the sub-image centered at {x0:.1f} {y0:.1f}")
        def split_fn(image, **kwargs):
            result = pipeline.get_subimage_centered(image, x0, y0, sub_size, **kwargs)

            yield [0] + result

    else:
        log('No target provided and transient detection is disabled, nothing to do')
        return

    for i, x0, y0, image1, mask1, header1, wcs1, obj1, cat1 in split_fn(
            image, mask=mask, header=header, wcs=wcs, obj=obj, cat=cat,
            get_origin=True, verbose=False):

        log(f"\n---- Sub-image {i}: {x0} {y0} - {x0 + image1.shape[1]} {y0 + image1.shape[0]} ----\n")

        fits.writeto(os.path.join(basepath, 'sub_image.fits'), image1, header1, overwrite=True)
        fits.writeto(os.path.join(basepath, 'sub_mask.fits'), mask1.astype(np.int8), header1, overwrite=True)

        # Get the template
        if tname == 'ps1':
            log("Getting the template from original Pan-STARRS archive")
            tmpl = templates.get_ps1_image(tfilter, wcs=wcs1, shape=image1.shape,
                                           _cachedir=_cachedir,
                                           _tmpdir=settings.STDPIPE_TMPDIR,
                                           _exe=settings.STDPIPE_SWARP,
                                           verbose=sub_verbose)
            tmask = templates.get_ps1_image(tfilter, ext='mask', wcs=wcs1, shape=image1.shape,
                                            extra={'COMBINE_TYPE':'AND'},
                                            _cachedir=_cachedir,
                                           _tmpdir=settings.STDPIPE_TMPDIR,
                                            _exe=settings.STDPIPE_SWARP,
                                            verbose=sub_verbose)
            if tmask is not None and tmpl is not None:
                tmask = (tmask & ~0x8000) > 0
                tmask |= np.isnan(tmpl)
            else:
                raise RuntimeError('Error getting the template from Pan-STARRS')
        else:
            log("Getting the template from HiPS server")
            tmpl = templates.get_hips_image(tconf['filters'][tfilter], wcs=wcs1, shape=image1.shape,
                                            get_header=False,
                                            verbose=sub_verbose)
            if tmpl is not None:
                tmask = np.isnan(tmpl)
            else:
                raise RuntimeError(f"Error getting the template from {tconf['name']}")

        fits.writeto(os.path.join(basepath, 'sub_template.fits'), tmpl, header, overwrite=True)
        fits.writeto(os.path.join(basepath, 'sub_template_mask.fits'), tmask.astype(np.int8), header, overwrite=True)

        tobj,tsegm = photometry.get_objects_sextractor(tmpl, mask=tmask, sn=5,
                                                       extra_params=['NUMBER'],
                                                       checkimages=['SEGMENTATION'],
                                                       _tmpdir=settings.STDPIPE_TMPDIR,
                                                       _exe=settings.STDPIPE_SEXTRACTOR)
        # Mask the footprints of masked objects
        # for _ in tobj[(tobj['flags'] & 0x100) > 0]:
        #     tmask |= tsegm == _['NUMBER']
        tobj = tobj[tobj['flags'] == 0]

        t_fwhm = np.median(tobj['fwhm'])
        i_fwhm = config.get('fwhm', 3.0)

        log(f"Using template FWHM = {t_fwhm:.1f} pix and image FWHM = {i_fwhm:.1f} pix")

        bg = sep.Background(image1, mask=mask1,
                            bw=config.get('bg_size', 128),
                            bh=config.get('bg_size', 128))

        res = subtraction.run_hotpants(image1-bg, tmpl,
                                       mask=mask1, template_mask=tmask,
                                       get_convolved=True,
                                       get_scaled=True,
                                       get_noise=True,
                                       verbose=verbose,
                                       image_fwhm=i_fwhm,
                                       template_fwhm=t_fwhm,
                                       image_gain=config.get('gain', 1.0),
                                       template_gain=10000,
                                       err=True,
                                       extra={'ko':2, 'bgo':0},
                                       obj=obj1[obj1['flags']==0])

        if res is not None:
            diff,conv,sdiff,ediff = res
        else:
            raise RuntimeError('Subtraction failed')

        dmask = diff == 1e-30 # Bad pixels

        # Combined mask on the sub-image
        fullmask1 = mask1 | tmask | dmask

        fits.writeto(os.path.join(basepath, 'sub_diff.fits'), diff, header, overwrite=True)
        fits.writeto(os.path.join(basepath, 'sub_sdiff.fits'), sdiff, header, overwrite=True)
        fits.writeto(os.path.join(basepath, 'sub_conv.fits'), conv, header, overwrite=True)
        fits.writeto(os.path.join(basepath, 'sub_ediff.fits'), ediff, header, overwrite=True)

        if config.get('target_ra') is not None and config.get('target_dec') is not None and not detect_transients:
        # Target forced photometry
            log(f"\n---- Target forced photometry ----\n")

            target_obj = Table({'ra':[config['target_ra']], 'dec':[config['target_dec']]})
            target_obj['x'],target_obj['y'] = wcs1.all_world2pix(target_obj['ra'], target_obj['dec'], 0)

            if not (target_obj['x'] > 0 and target_obj['x'] < image1.shape[1] and
                    target_obj['y'] > 0 and target_obj['y'] < image1.shape[0]):
                raise RuntimeError("Target is outside the sub-image")

            log(f"Target position is {target_obj['ra'][0]:.3f} {target_obj['dec'][0]:.3f} -> {target_obj['x'][0]:.1f} {target_obj['y'][0]:.1f}")

            if config.get('rel_bg1') and config.get('rel_bg2'):
                rel_bkgann = [config['rel_bg1'], config['rel_bg2']]
            else:
                rel_bkgann = None

            target_obj = photometry.measure_objects(target_obj, diff, mask=fullmask1,
                                                    # FWHM should match the one used for calibration
                                                    fwhm=config.get('fwhm'),
                                                    aper=config.get('rel_aper', 1.0),
                                                    bkgann=rel_bkgann,
                                                    sn=0,
                                                    # We assume no background
                                                    bg=None,
                                                    # ..and known error model
                                                    err=ediff,
                                                    gain=config.get('gain', 1.0),
                                                    verbose=verbose)

            target_obj['mag_calib'] = target_obj['mag'] + m['zero_fn'](target_obj['x'] + x0,
                                                                       target_obj['y'] + y0,
                                                                       target_obj['mag'])

            target_obj['mag_calib_err'] = np.hypot(target_obj['magerr'],
                                                   m['zero_fn'](target_obj['x'] + x0,
                                                                target_obj['y'] + y0,
                                                                target_obj['mag'],
                                                                get_err=True))

            # TODO: Improve limiting mag estimate
            target_obj['mag_limit'] = -2.5*np.log10(config.get('sn', 5)*target_obj['fluxerr']) + m['zero_fn'](target_obj['x'], target_obj['y'], target_obj['mag'])

            target_obj['mag_filter_name'] = m['cat_col_mag']

            if 'cat_col_mag1' in m.keys() and 'cat_col_mag2' in m.keys():
                target_obj['mag_color_name'] = '%s - %s' % (m['cat_col_mag1'], m['cat_col_mag2'])
                target_obj['mag_color_term'] = m['color_term']

            target_obj.write(os.path.join(basepath, 'sub_target.vot'), format='votable', overwrite=True)
            log("Measured target stored to sub_target.vot")

            # Create the cutout from image based on the candidate
            cutout = cutouts.get_cutout(image1, target_obj[0], 30,
                                        mask=fullmask1, header=header1,
                                        diff=diff, template=tmpl, convolved=conv, err=ediff)
            cutouts.write_cutout(cutout, os.path.join(basepath, 'sub_target.cutout'))
            log("Target cutouts stored to sub_target.cutout")

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
