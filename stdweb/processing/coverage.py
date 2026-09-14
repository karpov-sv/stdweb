"""
Checking whether the image field is covered by the template surveys.

The checks are cheap and do not download any actual imaging data, so they may be
used to warn the user before the subtraction is even started.
"""

import numpy as np

from django.core.cache import cache

from astropy.coordinates import SkyCoord

from stdpipe import astrometry, templates

from .constants import *


# Human-readable names of the survey tiles
survey_cell_names = {'ps1': 'skycells', 'ls': 'bricks'}

# Approximate radii of survey tiles, in degrees, as used inside STDPipe
survey_cell_radii = {'ps1': 0.3, 'ls': 0.186}

# How long to keep the survey coverage maps in Django cache
moc_cache_time = 7*24*3600


def get_survey_cells(survey, wcs, shape, band=None, ext='image'):
    """
    Get the list of survey (Pan-STARRS or Legacy Survey) tiles overlapping the
    footprint of the image defined by its WCS and shape.

    Legacy Survey bricks are not all observed in all the bands, and STDPipe
    skips the ones with no data in the requested one, so the result depends on
    the band. Use :func:`has_survey_cells_any_band` to tell a band gap from a
    genuinely uncovered field.

    Returns None if the coverage may not be checked for some reason, so that the
    caller may fall back to a generic error message.
    """

    try:
        ra0, dec0, sr0 = astrometry.get_frame_center(wcs=wcs, width=shape[1], height=shape[0])

        return templates.find_skycells(
            ra0, dec0, sr0, band=band, ext=ext, survey=survey,
            wcs=wcs, width=shape[1], height=shape[0]
        )
    except Exception:
        return None


def has_survey_cells_any_band(survey, wcs, shape):
    """
    Whether any survey tiles overlap the field, irrespective of the bands they
    are covered in.

    Legacy Survey masks are common for all the bands, so requesting them is the
    way to check the coverage with no band filtering applied.
    """

    cells = get_survey_cells(survey, wcs, shape, ext='mask')

    return cells is not None and len(cells) > 0


# Lazily loaded centers of the survey tiles, keyed by survey name
__survey_cell_centers = {}


def get_survey_cell_centers(survey):
    """
    Get the centers of all survey tiles, from the tables bundled with STDPipe.
    """

    if survey not in __survey_cell_centers:
        from stdpipe import utils as stdpipe_utils
        from astropy.table import Table

        if survey == 'ps1':
            table = Table.read(stdpipe_utils.get_data_path('ps1skycells.txt'), format='ascii')
            centers = (np.asarray(table['ra0']), np.asarray(table['dec0']))
        else:
            table = Table.read(
                stdpipe_utils.get_data_path('legacysurvey_bricks.fits.gz'), format='fits'
            )
            centers = (np.asarray(table['ra']), np.asarray(table['dec']))

        __survey_cell_centers[survey] = centers

    return __survey_cell_centers[survey]


def get_hips_moc(hips):
    """
    Get the coverage map (MOC) of a given HiPS survey from CDS MOCServer.

    The result is cached in Django cache, as it is not supposed to change often.
    Returns None if the coverage map may not be acquired.
    """

    key = f"hips_moc:{hips}"
    value = cache.get(key)

    if value is not None:
        # Cached failures are stored as False to avoid hammering the server
        return value or None

    try:
        from astroquery.mocserver import MOCServer

        moc = MOCServer.query_region(criteria=f"ID={hips}", return_moc=True)
    except Exception:
        moc = None

    cache.set(key, moc if moc is not None else False, moc_cache_time)

    return moc


def sample_footprint(wcs, shape, num=32):
    """
    Sample the image footprint on a regular grid of pixel positions, and return
    the sky coordinates of the samples.
    """

    xx,yy = np.meshgrid(
        np.linspace(0, shape[1] - 1, num),
        np.linspace(0, shape[0] - 1, num)
    )

    # Forward projection only - it never needs the distortion inversion
    ra,dec = wcs.all_pix2world(xx.ravel(), yy.ravel(), 0)

    return ra, dec


def get_template_filter(tconf, filter_name):
    """
    Get the template filter best matching the image photometric filter, or None
    if the template has nothing suitable.
    """

    for _ in filter_mappings.get(filter_name, []):
        if _ in tconf['filters']:
            return _

    return None


def get_template_coverage(tname, wcs, shape, filter_name=None):
    """
    Check whether the field defined by the WCS and shape is covered by a given
    template survey.

    The check is based on the survey tile lists bundled with STDPipe for
    Pan-STARRS and Legacy Survey, and on the coverage maps from CDS MOCServer
    for HiPS based templates. It does not know whether the actual imaging data
    are usable, so the result is just an estimate.

    Returns the dict with the following fields:

    - `status` - one of `ok`, `partial`, `none`, `noband`, `nofilter` or `unknown`
    - `fraction` - estimated fraction of the image covered by the template, or None
    - `filter` - template filter that will be used for the image one
    - `message` - human-readable summary of the result
    """

    tconf = supported_templates.get(tname)

    result = {'status': 'unknown', 'fraction': None, 'filter': None, 'message': ''}

    if tconf is None or tname == 'custom':
        return result

    tfilter = get_template_filter(tconf, filter_name) if filter_name else None
    result['filter'] = tfilter

    if filter_name and tfilter is None:
        result['status'] = 'nofilter'
        result['message'] = f"{tconf['name']} has no filter matching {filter_name}"
        return result

    noband = False

    try:
        ra,dec = sample_footprint(wcs, shape)

        if tname in ['ps1', 'ls']:
            cells = get_survey_cells(tname, wcs, shape, band=tfilter)
            cellname = survey_cell_names.get(tname)

            if cells is None:
                return result

            if not len(cells):
                # Authoritative check - STDPipe will not find anything to download
                fraction = 0

                if tfilter and has_survey_cells_any_band(tname, wcs, shape):
                    # The tiles do overlap the field, they just have no data in this
                    # band - most notably, Legacy Survey has no i band in the north
                    noband = True
                    result['message'] = (
                        f"{tconf['name']} covers the field, "
                        f"but has no data in {tfilter} band there"
                    )
                else:
                    result['message'] = f"No {tconf['name']} {cellname} overlap the field"
            else:
                # Estimate the covered fraction from the distances to the tile centers
                cell_ra, cell_dec = get_survey_cell_centers(tname)
                idx,_,_ = astrometry.spherical_match(
                    ra, dec, cell_ra, cell_dec, survey_cell_radii[tname]
                )
                # The tiles do exist, so never report the field as fully uncovered here
                fraction = max(len(np.unique(idx)) / len(ra), 0.01)

        else:
            hips = tconf['filters'].get(tfilter) if tfilter else None

            if hips is None:
                # Without the image filter we may only check the union of all bands
                mocs = [get_hips_moc(_) for _ in tconf['filters'].values()]
                mocs = [_ for _ in mocs if _ is not None]

                if not mocs:
                    return result

                covered = np.zeros(len(ra), dtype=bool)
                for moc in mocs:
                    covered |= moc.contains_skycoords(SkyCoord(ra, dec, unit='deg'))
            else:
                moc = get_hips_moc(hips)

                if moc is None:
                    return result

                covered = moc.contains_skycoords(SkyCoord(ra, dec, unit='deg'))

            fraction = np.sum(covered) / len(covered)

            if fraction <= 0:
                result['message'] = f"The field is outside the {tconf['name']} footprint"

        result['fraction'] = fraction

        if fraction <= 0:
            result['status'] = 'noband' if noband else 'none'
        elif fraction < 0.9:
            result['status'] = 'partial'
            result['message'] = f"{tconf['name']} covers only {100*fraction:.0f}% of the field"
        else:
            result['status'] = 'ok'
            result['message'] = f"{tconf['name']} covers the field"

    except Exception:
        return result

    return result


def get_all_template_coverage(wcs, shape, filter_name=None):
    """
    Check the coverage of the given field by all supported templates.
    Returns the dict keyed by template name.
    """

    result = {}

    for tname in supported_templates.keys():
        if tname == 'custom':
            continue

        result[tname] = get_template_coverage(tname, wcs, shape, filter_name=filter_name)

    return result
