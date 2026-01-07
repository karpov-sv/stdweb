"""
Utility functions for image processing.
Includes file I/O, WCS handling, image preprocessing, and helper functions.
"""

import os
import shutil

import numpy as np

from astropy.wcs import WCS
from astropy.io import fits
from astropy.time import Time

import dill as pickle

from stdpipe import astrometry, cutouts, templates


def cleanup_paths(paths, basepath=None):
    for path in paths:
        fullpath = os.path.join(basepath, path)
        if os.path.exists(fullpath):
            if os.path.isdir(fullpath):
                shutil.rmtree(fullpath)
            else:
                os.unlink(fullpath)


def print_to_file(*args, clear=False, logname='out.log', time0=None, **kwargs):
    if clear and os.path.exists(logname):
        print('Clearing', logname)
        os.unlink(logname)

    if time0 is not None:
        prefix = "{:6.2f}s ".format((Time.now() - time0).sec)
    else:
        prefix = None

    if len(args) or len(kwargs):
        if prefix is not None:
            print(prefix, end='')

        print(*args, **kwargs)

        if logname is not None:
            with open(logname, 'a+') as lfd:
                if prefix is not None:
                    print(prefix, end='', file=lfd)
                print(file=lfd, *args, **kwargs)


def pickle_to_file(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def pickle_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def fits_write(filename, image, header=None, compress=False):
    """Store image with or without header to FITS file, with or without compression"""
    if compress:
        hdu = fits.CompImageHDU(image, header)
    else:
        hdu = fits.PrimaryHDU(image, header)

    hdu.writeto(filename, overwrite=True)


def get_wcs(filename, header=None, verbose=True):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    basepath = os.path.dirname(filename)

    # Load the WCS from separate file, if it exists
    if os.path.exists(os.path.join(basepath, "image.wcs")):
        wcs = WCS(fits.getheader(os.path.join(basepath, "image.wcs")))

        if header is not None:
            # Update the header in-place with this new solution
            astrometry.clear_wcs(header)
            header += wcs.to_header(relax=True) # in-place update

        log("WCS loaded from file:image.wcs")
    else:
        if header is None:
            header = fits.getheader(filename, -1)
        wcs = WCS(header)
        log("Using original WCS from FITS header")

    return wcs


def fix_header(header, verbose=True):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    # Fix FITS standard errors in the header
    for _ in header.cards:
        _.verify('silentfix')
        __ = str(_) # it runs self.image()

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

    # Fix some IRAF stuff that breaks astropy WCS
    for _ in ['WCSDIM', 'LTM1_1', 'LTM2_2', 'WAT0_001', 'WAT1_001', 'WAT2_001']:
        header.remove(_, ignore_missing=True)

    # Ensure WCS keywords are numbers, not strings
    for kw in ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'CDELT1', 'CDELT2']:
        if kw in header:
            header[kw] = float(header[kw])

    if 'FOCALLEN' in header and not header.get('FOCALLEN'):
        header.remove('FOCALLEN')

    # Fix extra NAXIS headers
    for _ in ['NAXIS3']:
        if _ in header:
            header.remove(_, ignore_missing=True)


def pre_fix_image(filename, verbose=True):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    # First check for compressed data
    hdus = fits.open(filename)
    if len(hdus) > 1:
        for i,hdu in enumerate(hdus):
            if hdu.is_image and len(hdu.shape) == 2:
                log(f"Keeping first usable plane ({i}: {hdu.name}) from multi-extension or tile compressed image")
                fits.writeto(filename, hdu.data, hdu.header, overwrite=True)
                break

    hdus.close()

    # Handle various special cases
    image,header = fits.getdata(filename), fits.getheader(filename)
    if 'BSOFTEN' in header and 'BOFFSET' in header:
        # Pan-STARRS image in ASINH scaling, let's convert it to linear
        log('Detected Pan-STARRS ASINH scaled image, fixing it')
        image,header = templates.normalize_ps1_skycell(image, header, verbose=verbose)
        fits.writeto(filename, image, header, overwrite=True)


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
    fix_header(header)

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


def preprocess_image(filename, config, destripe_horizontal=False, destripe_vertical=False, verbose=True):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    basepath = os.path.dirname(filename)

    image,header = fits.getdata(filename, -1), fits.getheader(filename, -1)
    # fix_header(header)

    if destripe_vertical:
        # Vertical stripes
        image1 = image.copy()
        med = np.nanmedian(image1)

        for _ in range(image1.shape[1]):
            image1[:,_] += med - np.nanmedian(image1[:,_])

        image = image1

    if destripe_horizontal:
        # horizontal stripes
        image1 = image.copy()
        med = np.nanmedian(image1)

        for _ in range(image1.shape[0]):
            image1[_,:] += med - np.nanmedian(image1[_,:])

        image = image1

    # Write the image and header back
    fits.writeto(filename, image, header, overwrite=True)


from astropy_healpix import healpy

def round_coords_to_grid(ra0, dec0, sr0, nside=None):
    """Tries to round the coordinates to nearest HEALPix pixel center"""
    if nside is None:
        for n in range(1, 16):
            nside = 2**n
            res = healpy.nside_to_pixel_resolution(nside).to('deg').value
            if res < 0.05*sr0:
                break
    else:
        res = healpy.nside_to_pixel_resolution(nside).to('deg').value

    ipix = healpy.ang2pix(nside, ra0, dec0, lonlat=True)
    ra1,dec1 = healpy.pix2ang(nside, ipix, lonlat=True)
    sr1 = (np.floor(sr0/res) + 1)*res

    return ra1, dec1, sr1


import regions
from astropy.coordinates import SkyCoord
from astropy import units as u

def write_ds9_regions(filename, objs, radius=None):
    regs = []

    if radius is None:
        radius = 5/3600 # 5 arcsec

    for row in objs:
        regs.append(regions.CircleSkyRegion(SkyCoord(row['ra'], row['dec'], unit='deg'), radius*u.deg))

    regs = regions.Regions(regs)

    regs.write(filename, format='ds9', overwrite=True)


from mocpy import MOC

def _get_moc_order(shape, pixscale, frac=1/100):
    area_sky = 4*np.pi*(180/np.pi)**2
    area = shape[0] * shape[1] * pixscale**2

    for order in range(3, 12):
        cell_area = area_sky / MOC.n_cells(order)
        if cell_area < area * frac:
            break

    return order


def get_moc_for_wcs(wcs, shape, get_moc=False):
    """
    Get MOC representation for a given wcs and image shape
    """
    ra,dec = wcs.all_pix2world(
        [0, 0, shape[1], shape[1], 0],
        [0, shape[0], shape[0], 0, 0],
        0
    )

    order = _get_moc_order(shape, astrometry.get_pixscale(wcs=wcs))

    moc = MOC.from_polygon_skycoord(
        SkyCoord(ra, dec, unit='deg', frame='icrs'),
        complement=False,
        max_depth=order
    )

    if get_moc:
        return moc
    else:
        return moc.to_string()
