from django.conf import settings

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

import reproject

import dill as pickle

# STDPipe
from stdpipe import astrometry, photometry, catalogs, cutouts
from stdpipe import templates, subtraction, plots, pipeline, utils, psf
from stdpipe import resolve

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
    'u': {'name':'Sloan u', 'aliases':["sdssu", "SDSS u", "SDSS-u", "SDSS-u'", "Sloan-u", "sloanu", "Sloan u", "Su", "SU", "sU"]},
    'g': {'name':'Sloan g', 'aliases':["sdssg", "SDSS g", "SDSS-g", "SDSS-g'", "Sloan-g", "sloang", "Sloan g", "Sg", "SG", "sG", "ZTF_g"]},
    'r': {'name':'Sloan r', 'aliases':["sdssr", "SDSS r", "SDSS-r", "SDSS-r'", "Sloan-r", "sloanr", "Sloan r", "Sr", "SR", "sR", "ZTF_r"]},
    'i': {'name':'Sloan i', 'aliases':["sdssi", "SDSS i", "SDSS-i", "SDSS-i'", "Sloan-i", "sloani", "Sloan i", "Si", "SI", "sI", "ZTF_i"]},
    'z': {'name':'Sloan z', 'aliases':["sdssz", "SDSS z", "SDSS-z", "SDSS-z'", "Sloan-z", "sloanz", "Sloan z", "Sz", "SZ", "sZ"]},
    # Gaia
    'G': {'name':'Gaia G', 'aliases':[]},
    'BP': {'name':'Gaia BP', 'aliases':[]},
    'RP': {'name':'Gaia RP', 'aliases':[]},
}

supported_catalogs = {
    'gaiadr3syn': {'name':'Gaia DR3 synphot', 'filters':['U', 'B', 'V', 'R', 'I', 'g', 'r', 'i', 'z', 'y'],
                   'limit': 'rmag'},
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
    'custom': {'name': 'Custom template'},
    'ps1': {'name': 'Pan-STARRS DR2', 'filters': {'g', 'r', 'i', 'z'}},
    'ls': {'name': 'Legacy Survey DR10', 'filters': {'g', 'r', 'i', 'z'}},
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
    # 'legacy': {'name': 'DESI Legacy Surveys DR10 (HiPS)', 'filters': {
    #     'g': 'CDS/P/DESI-Legacy-Surveys/DR10/g',
    #     'i': 'CDS/P/DESI-Legacy-Surveys/DR10/i',
    # }},
    'decaps': {'name': 'DECaPS DR2 (HiPS)', 'filters': {
        'g': 'CDS/P/DECaPS/DR2/g',
        'r': 'CDS/P/DECaPS/DR2/r',
        'i': 'CDS/P/DECaPS/DR2/i',
        'z': 'CDS/P/DECaPS/DR2/z',
    }},
    'ztf': {'name': 'ZTF DR7 (HiPS)', 'filters': {
        'g': 'CDS/P/ZTF/DR7/g',
        'r': 'CDS/P/ZTF/DR7/r',
        'i': 'CDS/P/ZTF/DR7/i',
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
    'G': ['r', 'g'],
    'BP': ['g'],
    'RP': ['i', 'r'],
}


# Files created at every step

files_inspect = [
    'inspect.log',
    'mask.fits', 'image_target.fits',
    'image_bg.png', 'image_rms.png',
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
    'sub_scorr.fits', 'sub_fpsf.fits', 'sub_fpsferr.fits',
    'sub_target.vot', 'sub_target.cutout',
    'candidates', 'candidates.vot'
]

cleanup_inspect = files_inspect + files_photometry + files_subtraction

cleanup_photometry = files_photometry + files_subtraction

cleanup_subtraction = files_subtraction

def cleanup_paths(paths, basepath=None):
    for path in paths:
        fullpath = os.path.join(basepath, path)
        if os.path.exists(fullpath):
            if os.path.isdir(fullpath):
                shutil.rmtree(fullpath)
            else:
                os.unlink(fullpath)


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

    # Fix SCAMP headers with TAN type what are actually TPV (and thus break AstroPy WCS)
    if header.get('CTYPE1') == 'RA---TAN' and 'PV1_5' in header.keys():
        header['CTYPE1'] = 'RA---TPV'
        header['CTYPE2'] = 'DEC--TPV'

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

    image,header = fits.getdata(filename, -1), fits.getheader(filename, -1)
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

    image,header = fits.getdata(filename, -1), fits.getheader(filename, -1)

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
    cleanup_paths(cleanup_inspect, basepath=basepath)

    config['sn'] = config.get('sn', 5)
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
    config['hotpants_extra'] = config.get('hotpants_extra', {'ko':0, 'bgo':0})
    config['sub_size'] = config.get('sub_size', 1000)
    config['sub_overlap'] = config.get('sub_overlap', 50)
    config['sub_verbose'] = config.get('sub_verbose', False)
    config['subtraction_mode'] = config.get('subtraction_mode', 'detection')

    # Load the image
    log(f'Inspecting {filename}')
    try:
        image,header = fits.getdata(filename, -1).astype(np.double), fits.getheader(filename, -1)
    except:
        raise RuntimeError('Cannot load the image')

    # Handle various special cases
    if 'BSOFTEN' in header and 'BOFFSET' in header:
        # Pan-STARRS image in ASINH scaling, let's convert it to linear
        log('Detected Pan-STARRS ASINH scaled image, fixing it')
        image,header = templates.normalize_ps1_skycell(image, header, verbose=verbose)
        fits.writeto(filename, image, header, overwrite=True)

    log(f"Image size is {image.shape[1]} x {image.shape[0]}")

    log(f"Image Min / Median / Max : {np.nanmin(image):.2f} {np.nanmedian(image):.2f} {np.nanmax(image):.2f}")
    log(f"Image RMS: {np.nanstd(image):.2f}")

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
        if satlevel:
            log("Got saturation level from FITS header")

        else:
            satlevel = 0.05*np.nanmedian(image) + 0.95*np.nanmax(image) # med + 0.95(max-med)
            log("Estimating saturation level from the image max value")

        config['saturation'] = satlevel
    log(f"Saturation level is {config['saturation']:.1f}")

    # Mask
    mask = np.isnan(image)
    mask |= image >= config['saturation']

    # Custom mask
    if os.path.exists(os.path.join(basepath, 'custom_mask.fits')):
        mask |= fits.getdata(os.path.join(basepath, 'custom_mask.fits')) > 0
        log("Custom mask loaded from custom_mask.fits")

    # Cosmics
    if config.get('mask_cosmics', True):
        # We will use custom noise model for astroscrappy as we do not know whether
        # the image is background-subtracted already, or how it was flatfielded
        bg = sep.Background(image, mask=mask)
        rms = bg.rms()
        var = rms**2 + np.abs(image - bg.back())/config.get('gain', 1)
        cmask, cimage = astroscrappy.detect_cosmics(image, mask, verbose=False,
                                                    invar=var.astype(np.float32),
                                                    gain=config.get('gain', 1),
                                                    satlevel=config.get('saturation'),
                                                    cleantype='medmask')
        log(f"Done masking cosmics, {np.sum(cmask)} ({100*np.sum(cmask)/cmask.shape[0]/cmask.shape[1]:.1f}%) pixels masked")
        mask |= cmask

    log(f"{np.sum(mask)} ({100*np.sum(mask)/mask.shape[0]/mask.shape[1]:.1f}%) pixels masked")

    if np.sum(mask) > 0.95*mask.shape[0]*mask.shape[1]:
        raise RuntimeError('More than 95% of the image is masked')

    fits.writeto(os.path.join(basepath, 'mask.fits'), mask.astype(np.int8), overwrite=True)
    log("Mask written to file:mask.fits")

    # Cosmics
    if config.get('inspect_bg', False):
        bg = sep.Background(image, mask=mask)

        with plots.figure_saver(os.path.join(basepath, 'image_bg.png'), figsize=(8, 6), show=show) as fig:
            ax = fig.add_subplot(1, 1, 1)
            plots.imshow(bg.back(), ax=ax)
            ax.set_aspect(1)
            ax.set_xlim(0, image.shape[1])
            ax.set_ylim(0, image.shape[0])
            ax.set_title(f"Background: median {np.median(bg.back()):.2f} RMS {np.std(bg.back()):.2f}")

        with plots.figure_saver(os.path.join(basepath, 'image_rms.png'), figsize=(8, 6), show=show) as fig:
            ax = fig.add_subplot(1, 1, 1)
            plots.imshow(bg.rms(), ax=ax)
            ax.set_aspect(1)
            ax.set_xlim(0, image.shape[1])
            ax.set_ylim(0, image.shape[0])
            ax.set_title(f"Background RMS: median {np.median(bg.rms()):.2f} RMS {np.std(bg.rms()):.2f}")

    # WCS
    if os.path.exists(os.path.join(basepath, "image.wcs")):
        wcs = WCS(fits.getheader(os.path.join(basepath, "image.wcs")))
        astrometry.clear_wcs(header)
        header += wcs.to_header(relax=True)
        log("WCS loaded from file:image.wcs")
    else:
        wcs = WCS(header)
        log("Using original WCS from FITS header")

    if wcs and wcs.is_celestial:
        ra0,dec0,sr0 = astrometry.get_frame_center(wcs=wcs, width=image.shape[1], height=image.shape[0])
        pixscale = astrometry.get_pixscale(wcs=wcs)

        log(f"Field center is at {ra0:.3f} {dec0:.3f}, radius {sr0:.2f} deg")
        log(f"Pixel scale is {3600*pixscale:.2f} arcsec/pix")

        config['refine_wcs'] = True
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
            target = resolve.resolve(config['target'])
            config['target_ra'] = target.ra.deg
            config['target_dec'] = target.dec.deg

            # Activate target photometry mode
            config['subtraction_mode'] = 'target'

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
                    log("Target cutout written to file:image_target.fits")
                else:
                    log("Target is outside the image")
                    log(f"{x0} {y0}")
            except:
                pass

        # We may initialize some blind match parameters from the target position, if any
        if config.get('target_ra') is not None and config.get('blind_match_ra0') is None:
            config['blind_match_ra0'] = config.get('target_ra')
        if config.get('target_dec') is not None and config.get('blind_match_dec0') is None:
            config['blind_match_dec0'] = config.get('target_dec')
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


def photometry_image(filename, config, verbose=True, show=False):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    basepath = os.path.dirname(filename)

    # Image
    image,header = fits.getdata(filename, -1).astype(np.double), fits.getheader(filename, -1)

    fix_header(header)

    # Mask
    mask = fits.getdata(os.path.join(basepath, 'mask.fits')) > 0

    # Time
    time = Time(config.get('time')) if config.get('time') else None

    # Cleanup stale plots
    cleanup_paths(cleanup_photometry, basepath=basepath)

    log("\n---- Object detection ----\n")

    # Extract objects
    obj = photometry.get_objects_sextractor(image, mask=mask,
                                            aper=config.get('initial_aper', 3.0),
                                            gain=config.get('gain', 1.0),
                                            extra={'BACK_SIZE': config.get('bg_size', 256)},
                                            minarea=config.get('minarea', 3),
                                            r0=config.get('initial_r0', 0.0),
                                            mask_to_nans=True,
                                            verbose=verbose,
                                            _tmpdir=settings.STDPIPE_TMPDIR,
                                            _exe=settings.STDPIPE_SEXTRACTOR)

    log(f"{len(obj)} objects found")

    if not len(obj):
        raise RuntimeError('Cannot detect objects in the image')

    log("\n---- Object measurement ----\n")

    # FWHM
    idx = obj['flags'] == 0
    idx &= obj['magerr'] < 1/20

    if not len(obj[idx]):
        raise RuntimeError("No stars with S/N > 20 in the image!")

    fwhm = np.median(obj['fwhm'][idx]) # TODO: make it position-dependent
    log(f"FWHM is {fwhm:.2f} pixels")

    if config.get('fwhm_override'):
        fwhm = config.get('fwhm_override')
        log(f"Overriding with user-specified FWHM value of {fwhm:.2f} pixels")

    config['fwhm'] = fwhm

    # Plot FWHM map
    with plots.figure_saver(os.path.join(basepath, 'fwhm.png'), figsize=(8, 6), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)
        plots.binned_map(obj[idx]['x'], obj[idx]['y'], obj[idx]['fwhm'], bins=8, statistic='median', show_dots=True, ax=ax)
        ax.set_aspect(1)
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(0, image.shape[0])
        # ax.legend()
        ax.set_title(f"FWHM: median {np.median(obj[idx]['fwhm']):.2f} pix RMS {np.std(obj[idx]['fwhm']):.2f} pix")

    # Plot FWHM vs instrumental
    with plots.figure_saver(os.path.join(basepath, 'fwhm_mag.png'), figsize=(8, 6), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(obj['fwhm'], obj['mag'], '.', label='All objects')
        ax.plot(obj['fwhm'][idx], obj['mag'][idx], '.', label='Used for FWHM')

        ax.axvline(fwhm, ls='--', color='red')
        ax.invert_yaxis()
        ax.legend()
        ax.grid(alpha=0.2)
        ax.set_title(f"FWHM: median {np.median(obj[idx]['fwhm']):.2f} pix RMS {np.std(obj[idx]['fwhm']):.2f} pix")
        ax.set_xlabel('FWHM, pixels')
        ax.set_ylabel('Instrumental magnitude')
        ax.set_xlim(0, np.percentile(obj['fwhm'], 98))

    log(f"FWHM diagnostic plot stored to file:fwhm_mag.png")

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

    log(f"{len(obj)} objects properly measured")

    obj.write(os.path.join(basepath, 'objects.vot'), format='votable', overwrite=True)
    log("Measured objects stored to file:objects.vot")

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
            config['refine_wcs'] = True # We need to do it as we got SIP solution
            log("Blind matched WCS stored to file:image.wcs")
        else:
            log("Blind matching failed")

    elif os.path.exists(os.path.join(basepath, "image.wcs")):
        wcs = WCS(fits.getheader(os.path.join(basepath, "image.wcs")))
        astrometry.clear_wcs(header)
        header += wcs.to_header(relax=True)
        log("WCS loaded from image.wcs")

    else:
        wcs = WCS(header)
        log("Using original WCS from FITS header")


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
    log("Catalogue written to file:cat.vot")

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

        if config['filter'] == 'z':
            config['cat_col_color_mag1'] = 'rmag'
            config['cat_col_color_mag2'] = 'imag'
        else:
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

        # FIXME: make the order configurable
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

    sr = config.get('sr_override')
    if sr:
        sr /= 3600 # Arcseconds to degrees

    # Photometric calibration
    m = pipeline.calibrate_photometry(obj, cat, sr=sr, pixscale=pixscale,
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
    obj['mag_calib'] = obj['mag'] + m['zero_fn'](obj['x'], obj['y'], obj['mag'])
    obj['mag_calib_err'] = np.hypot(obj['magerr'],
                                    m['zero_fn'](obj['x'], obj['y'], obj['mag'], get_err=True))

    obj.write(os.path.join(basepath, 'objects.vot'), format='votable', overwrite=True)
    log("Measured objects stored to file:objects.vot")

    # Check the filter
    if config.get('use_color', True) and np.abs(m['color_term']) > 0.5:
        log("Warning: color term is too large, checking whether other filters would work better")

        for fname in supported_catalogs[config['cat_name']].get('filters', []):
            m1 = pipeline.calibrate_photometry(
                obj, cat, pixscale=pixscale,
                cat_col_mag=fname + 'mag',
                cat_col_mag_err='e_' + fname + 'mag',
                cat_col_mag1=config.get('cat_col_color_mag1'),
                cat_col_mag2=config.get('cat_col_color_mag2'),
                use_color=config.get('use_color', True),
                order=config.get('spatial_order', 0),
                robust=True, scale_noise=True,
                accept_flags=0x02, max_intrinsic_rms=0.01,
                verbose=False)

            if m1 is not None:
                log(f"filter {fname}: color term {m1['color_term']:.2f}")
            else:
                log(f"filter {fname}: match failed")

    # Detection limits
    log("\n---- Global detection limit ----\n")
    mag0 = pipeline.get_detection_limit(obj, sn=config.get('sn'), verbose=verbose)
    config['mag_limit'] = mag0

    if 'bg_fluxerr' in obj.colnames and np.any(obj['bg_fluxerr'] > 0):
        fluxerr = obj['bg_fluxerr']
        maglim = -2.5*np.log10(config.get('sn', 5)*fluxerr) + m['zero_fn'](obj['x'], obj['y'], obj['mag'])
        maglim = maglim[np.isfinite(maglim)] # Remove Inf and NaN
        log(f"Local background RMS detection limit is {np.nanmedian(maglim):.2f} +/- {np.nanstd(maglim):.2f}")

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

        # Local detection limit from background rms, if available
        if 'bg_fluxerr' in target_obj.colnames and np.any(target_obj['bg_fluxerr'] > 0):
            fluxerr = target_obj['bg_fluxerr']
        else:
            fluxerr = target_obj['fluxerr']
        target_obj['mag_limit'] = -2.5*np.log10(config.get('sn', 5)*fluxerr) + m['zero_fn'](target_obj['x'], target_obj['y'], target_obj['mag'])

        target_obj['mag_filter_name'] = m['cat_col_mag']

        if 'cat_col_mag1' in m.keys() and 'cat_col_mag2' in m.keys():
            target_obj['mag_color_name'] = '%s - %s' % (m['cat_col_mag1'], m['cat_col_mag2'])
            target_obj['mag_color_term'] = m['color_term']

        target_obj.write(os.path.join(basepath, 'target.vot'), format='votable', overwrite=True)
        log("Measured target stored to file:target.vot")

        # Create the cutout from image based on the candidate
        cutout = cutouts.get_cutout(image, target_obj[0], 30, mask=mask, header=header, time=time)
        # Cutout from relevant HiPS survey
        if target_obj['dec'][0] > -30:
            cutout['template'] = templates.get_hips_image('PanSTARRS/DR1/r', header=cutout['header'])[0]
        else:
            cutout['template'] = templates.get_hips_image('CDS/P/skymapper-R', header=cutout['header'])[0]

        cutouts.write_cutout(cutout, os.path.join(basepath, 'target.cutout'))
        log("Target cutouts stored to file:target.cutout")

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
    subtraction_mode = config.get('subtraction_mode', 'detection')
    subtraction_method = config.get('subtraction_method', 'hotpants')

    # Cleanup stale plots and files
    cleanup_paths(cleanup_subtraction, basepath=basepath)

    # Image
    image,header = fits.getdata(filename, -1).astype(np.double), fits.getheader(filename, -1)

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
        fits.writeto(os.path.join(basepath, 'sub_mask.fits'), mask1.astype(np.int8), header1, overwrite=True)

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
            tmpl, mask=tmask, sn=5,
            extra_params=['NUMBER'],
            checkimages=['SEGMENTATION'],
            _tmpdir=settings.STDPIPE_TMPDIR,
            _exe=settings.STDPIPE_SEXTRACTOR
        )

        # Mask the footprints of masked objects
        # for _ in tobj[(tobj['flags'] & 0x100) > 0]:
        #     tmask |= tsegm == _['NUMBER']
        tobj = tobj[tobj['flags'] == 0]
        template_fwhm = np.median(tobj['fwhm'])

        fits.writeto(os.path.join(basepath, 'sub_template.fits'), tmpl, header1, overwrite=True)
        fits.writeto(os.path.join(basepath, 'sub_template_mask.fits'), tmask.astype(np.int8), header1, overwrite=True)

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
                bw=32 if fwhm < 4 else 64,
                bh=32 if fwhm < 4 else 64
            )
            tbg = sep.Background(
                tmpl.astype(np.double), mask=tmask,
                bw=32 if template_fwhm < 4 else 64,
                bh=32 if template_fwhm < 4 else 64
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
                # ..and known error model
                err=ediff,
                gain=config.get('gain', 1.0),
                verbose=sub_verbose
            )

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
                target_obj['mag_color_term'] = m['color_term']

            target_obj.write(os.path.join(basepath, 'sub_target.vot'), format='votable', overwrite=True)
            log("Measured target stored to file:sub_target.vot")

            # Create the cutout from image based on the candidate
            cutout = cutouts.get_cutout(
                image1, target_obj[0], 30,
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
                    extra_params=['NUMBER'],
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
                    extra_params=['NUMBER'],
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
                    sobj['mag_color_term'] = m['color_term']

            log(f"{len(sobj)} transient candidates found in difference image")

            vizier = ['ps1', 'skymapper', ] if config.get('filter_vizier') else []

            # Filter out catalogue objects
            candidates = pipeline.filter_transient_candidates(
                sobj,
                cat=None, # cat,
                sr=0.5*pixscale*config.get('fwhm', 1.0),
                pixscale=pixscale,
                vizier=vizier,
                # Filter out any flags except for 0x100 which is isophotal masked
                flagged=True, flagmask=0xfe00,
                time=time,
                skybot=config.get('filter_skybot', False),
                verbose=verbose
            )

            # diff[fullmask1] = np.nan # For better visuals

            Ngood = 0
            for cand in candidates:
                cutout = cutouts.get_cutout(
                    image1, cand, 30,
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
                            cutout, max_shift=1, max_scale=1.1,
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

            all_candidates.write(os.path.join(basepath, 'candidates.vot'), format='votable', overwrite=True)
            log("Candidates written to file:candidates.vot")
        else:
            log("No candidates found")
