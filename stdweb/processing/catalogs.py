"""
Catalog query and filtering functions.
Includes HiPS survey selection, Vizier catalog handling, and blend filtering.
"""

import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.cluster import AgglomerativeClustering

from stdpipe import astrometry, catalogs

from .constants import *


def guess_hips_survey(ra, dec, filter_name='R'):
    survey_filter = filter_mappings.get(filter_name, 'r')[0]

    # TODO: add Legacy Survey?..

    if dec > -30:
        if survey_filter == 'u':
            survey_filter = 'g'

        survey = f"PanSTARRS/DR1/{survey_filter}"

    else:
        if survey_filter == 'u':
            survey_filter = 'g'
        elif survey_filter == 'z' or survey_filter == 'y':
            survey_filter = 'i'

        survey = f"CDS/P/Skymapper/DR4/{survey_filter}"

    return survey


def guess_vizier_catalogues(ra, dec):
    vizier = ['gaiaedr3'] # All-sky

    if dec > -30:
        vizier.append('ps1')

    if dec < 0:
        vizier.append('skymapper')
        vizier.append('II/371/des_dr2')

    return vizier


def guess_catalogue_mag_columns(fname, cat, augmented_only=False):
    cat_col_mag = None
    cat_col_mag_err = None

    # Most of augmented catalogues
    if f"{fname}mag" in cat.colnames:
        cat_col_mag = f"{fname}mag"

        if f"e_{fname}mag" in cat.colnames:
            cat_col_mag_err = f"e_{fname}mag"

    elif augmented_only:
        raise RuntimeError(f"Unsupported filter {fname} for this catalogue")

    # Non-augmented PS1 etc
    elif "gmag" in cat.colnames and "rmag" in cat.colnames:
        if fname in ['U', 'B', 'V', 'BP']:
            cat_col_mag = "gmag"
        if fname in ['R', 'G']:
            cat_col_mag = "rmag"
        if fname in ['I', 'RP']:
            cat_col_mag = "imag"

        if f"e_{cat_col_mag}" in cat.colnames:
            cat_col_mag_err = f"e_{cat_col_mag}"

    # SkyMapper
    elif f"{fname}PSF" in cat.colnames:
        cat_col_mag = f"{fname}PSF"

        if f"e_{fname}PSF" in cat.colnames:
            cat_col_mag_err = f"e_{fname}PSF"

    # Gaia DR2/eDR3/DR3 from Vizier
    elif "BPmag" in cat.colnames and "RPmag" in cat.colnames and "Gmag" in cat.colnames:
        if fname in ['U', 'B', 'V', 'R', 'u', 'g', 'r', 'BP']:
            cat_col_mag = "BPmag"
        elif fname in ['I', 'i', 'z', 'RP']:
            cat_col_mag = "RPmag"
        else:
            cat_col_mag = "Gmag"

        if f"e_{cat_col_mag}" in cat.colnames:
            cat_col_mag_err = f"e_{cat_col_mag}"

    # Gaia DR2/eDR3/DR3 from XMatch
    elif "phot_bp_mean_mag" in cat.colnames and "phot_rp_mean_mag" in cat.colnames and "phot_g_mean_mag" in cat.colnames:
        if fname in ['U', 'B', 'V', 'R', 'u', 'g', 'r', 'BP']:
            cat_col_mag = "phot_bp_mean_mag"
        elif fname in ['I', 'i', 'z', 'RP']:
            cat_col_mag = "phot_rp_mean_mag"
        else:
            cat_col_mag = "phot_g_mean_mag"

        if f"{cat_col_mag}_error" in cat.colnames:
            cat_col_mag_err = f"{cat_col_mag}_error"

    # else:
    #     raise RuntimeError(f"Unsupported filter {fname} and/or catalogue")

    return cat_col_mag, cat_col_mag_err


def guess_catalogue_radec_columns(cat):
    cat_col_ra = None
    cat_col_dec = None

    # Find relevant coordinate columns
    if 'RAJ2000' in cat.keys():
        cat_col_ra = 'RAJ2000'
        cat_col_dec = 'DEJ2000'

    elif '_RAJ2000' in cat.keys():
        cat_col_ra = '_RAJ2000'
        cat_col_dec = '_DEJ2000'

    elif 'RA_ICRS' in cat.keys():
        cat_col_ra = 'RA_ICRS'
        cat_col_dec = 'DE_ICRS'

    # SkyMapper 1.1
    elif 'RAICRS' in cat.keys():
        cat_col_ra = 'RAICRS'
        cat_col_dec = 'DEICRS'

    # SkyMapper 4
    elif 'RAdeg' in cat.keys():
        cat_col_ra = 'RAdeg'
        cat_col_dec = 'DEdeg'

    # cross-match with Gaia eDR3
    elif 'ra' in cat.keys():
        cat_col_ra = 'ra'
        cat_col_dec = 'dec'

    elif 'ra_2' in cat.keys():
        cat_col_ra = 'ra_2'
        cat_col_dec = 'dec_2'

    # else:
    #     raise RuntimeError(f"Cannot find coordinate columns for the catalogue")

    return cat_col_ra, cat_col_dec


def filter_catalogue_blends(
        cat_in,
        sr,
        cat_col_ra='RAJ2000',
        cat_col_dec='DEJ2000',
        cat_col_mag=None,
        cat_col_mag_err=None
):
    # Clustering fails if we have less than 2 stars. And it is meaningless anyway
    if len(cat_in) < 2:
        return cat_in

    x,y,z = astrometry.radectoxyz(cat_in[cat_col_ra], cat_in[cat_col_dec])

    # Cluster into groups using sr radius
    cids = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=np.deg2rad(sr),
        linkage='single'
    ).fit_predict(np.array([x, y, z]).T)

    # Unique clusters
    uid,uids,urids,ucnt = np.unique(cids, return_index=True, return_inverse=True, return_counts=True)

    # Copy of the catalogue to work on
    cat = cat_in.copy()
    # cat['__blend__'] = False
    cat['__remove__'] = False

    for i,row in enumerate(cat):
        uid1 = urids[i]

        if row['__remove__']:
            continue

        if ucnt[uid1] > 1:
            ids = np.where(urids == uid1)[0]

            if cat_col_mag is not None:
                x1,y1,z1 = astrometry.radectoxyz(cat_in[cat_col_ra][ids], cat_in[cat_col_dec][ids])
                flux1 = 10**(-0.4*cat[cat_col_mag][ids])
                flux1 = np.ma.filled(flux1, np.nan)
                x0,y0,z0 = [np.nansum(_*flux1)/np.nansum(flux1) for _ in [x1,y1,z1]]
                ra,dec = astrometry.xyztoradec([x0,y0,z0])

                if not np.isfinite(ra) or not np.isfinite(dec):
                    # No usable fluxes at all?..
                    continue

                cat[cat_col_ra][ids[0]],cat[cat_col_dec][ids[0]] = ra, dec
                cat[cat_col_mag][ids[0]] = -2.5*np.log10(np.nansum(flux1))

                # cat['__blend__'][ids[0]] = True
                cat['__remove__'][ids[1:]] = True
            else:
                cat['__remove__'][ids] = True

        else:
            pass

    cat = cat[~cat['__remove__']]
    cat.remove_column('__remove__')
    # cat.remove_column('__blend__')

    return cat


def filter_vizier_blends(
    obj,
    sr,
    sr_blend=None,
    obj_col_ra='ra',
    obj_col_dec='dec',
    fname=None,
    vizier=[],
    col_id=None,
    vizier_checker_fn=None,
    verbose=False,
):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    log(
        'Blend filtering routine started with %d initial candidates, %.1f arcsec blending radius and %.1f arcsec matching radius'
        % (len(obj), sr_blend * 3600, sr * 3600)
    )
    cand_idx = np.ones(len(obj), dtype=bool)

    if col_id is None:
        col_id = 'stdpipe_id'

    if col_id not in obj.keys():
        obj = obj.copy()
        obj[col_id] = np.arange(len(obj))

    if sr_blend is None:
        sr_blend = 4*sr # It assumes sr to be half FWHM

    for catname in vizier or []:
        if not np.any(cand_idx):
            break

        xcat = catalogs.xmatch_objects(
            obj[cand_idx][[col_id, obj_col_ra, obj_col_dec]],
            catname,
            sr_blend,
            col_ra=obj_col_ra,
            col_dec=obj_col_dec,
        )

        if fname is not None:
            # Find relevant magnitude and coordinate columns
            cat_col_mag,_ = guess_catalogue_mag_columns(fname, xcat)
            cat_col_ra,cat_col_dec = guess_catalogue_radec_columns(xcat)

            if cat_col_ra is None:
                log("Cannot guess catalogue coordinate columns, skipping")
                log(xcat.keys())
                continue

            if cat_col_mag:
                xcat = filter_catalogue_blends(
                    xcat,
                    sr_blend,
                    cat_col_ra=cat_col_ra,
                    cat_col_dec=cat_col_dec,
                    cat_col_mag=cat_col_mag
                )

                oidx,xidx,_ = astrometry.spherical_match(
                    obj[cand_idx][obj_col_ra],
                    obj[cand_idx][obj_col_dec],
                    xcat[cat_col_ra],
                    xcat[cat_col_dec],
                    sr,
                )
                xcat = xcat[xidx]

                if xcat is not None and len(xcat):
                    if callable(vizier_checker_fn):
                        # Pass matched results through user-supplied checker
                        xobj = obj[[np.where(obj[col_id] == _)[0][0] for _ in xcat[col_id]]]
                        xidx = vizier_checker_fn(xobj, xcat, catname)
                        xcat = xcat[xidx]

                    cand_idx &= ~np.in1d(obj[col_id], xcat[col_id])

        log(
            np.sum(cand_idx),
            'remains after matching blends with',
            catalogs.catalogs.get(catname, {'name': catname})['name'],
        )

    return obj[cand_idx]
