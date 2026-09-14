"""
Catalog query and filtering functions.
Includes HiPS survey selection, Vizier catalog handling, and blend filtering.
"""

import numpy as np

from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from sklearn.cluster import AgglomerativeClustering

from astropy import units as u

from stdpipe import astrometry, catalogs

from .constants import *


def guess_hips_survey(ra, dec, filter_name='R'):
    survey_filter = filter_mappings.get(filter_name, 'r')[0]

    # TODO: add Legacy Survey?..

    # 2MASS for NIR filters
    if survey_filter in ['J', 'H', 'Ks']:
        survey = f"CDS/P/2MASS/{survey_filter[0]}"

    elif dec > -30:
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


def guess_vizier_catalogues(ra, dec, filter_name=None):
    """Vizier catalogues to check the transient candidates against.

    Gaia is always there, being the deepest all-sky one, and the rest are added by the
    sky position. When `filter_name` is known, the catalogues actually covering that
    band are added too, so that the candidate magnitudes are compared against a similar
    band instead of a distant proxy - Gaia eDR3 alone would have to stand in for the
    near infrared with its G magnitude.
    """
    vizier = ['gaiaedr3'] # All-sky

    if filter_name in ['J', 'H', 'Ks']:
        vizier.append('2mass') # All-sky, and the only one Gaia may not stand in for

        if dec < 0:
            vizier.append('vhs')
    else:
        # Synthetic photometry in the proper bands, much shallower than Gaia itself
        vizier.append('gaiadr3syn')

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


def filters_by_distance(fname):
    """Return the known filters ordered by how close they are to `fname` in wavelength.

    The distance is measured in log scale, as the bands span more than a decade.
    """
    if fname not in filter_wavelengths:
        return sorted(filter_wavelengths, key=lambda _: filter_wavelengths[_])

    return sorted(
        filter_wavelengths,
        key=lambda _: abs(np.log(filter_wavelengths[_] / filter_wavelengths[fname]))
    )


def guess_catalogue_mags_any(cat, fname=None):
    """Return the catalogue magnitudes for the band closest to `fname` that is actually
    usable for every individual entry, along with the column every value came from.

    Looking up a single column is not enough, as the catalogues are routinely incomplete:
    about a third of the Pan-STARRS entries have no r magnitude while having i or g. An
    entry whose magnitude we may not compare is an entry whose candidate gets rejected,
    so we keep looking in the neighbouring bands instead of giving up on it.

    The bands sit at different offsets from our instrumental system, so the caller has to
    bring them together - the returned column names tell which values belong together.
    """
    mags = np.full(len(cat), np.nan)
    cols = np.full(len(cat), '', dtype=object)
    seen = set()

    for fname1 in filters_by_distance(fname):
        if np.all(np.isfinite(mags)):
            break

        cat_col_mag,_ = guess_catalogue_mag_columns(fname1, cat)

        if cat_col_mag is None or cat_col_mag in seen:
            continue

        seen.add(cat_col_mag)
        values = _column_values(cat[cat_col_mag])
        idx = ~np.isfinite(mags) & np.isfinite(values)

        mags[idx] = values[idx]
        cols[idx] = cat_col_mag

    return mags, cols


def guess_catalogue_mag_columns_all(cat_name, cat):
    """Return the (magnitude, magnitude error) column pairs for every filter the
    catalogue is known to provide, so that they may all be kept consistent."""
    return [
        guess_catalogue_mag_columns(fname, cat)
        for fname in supported_catalogs.get(cat_name, {}).get('filters', [])
    ]


def guess_catalogue_radec_columns(cat, exclude=None):
    """Guess the columns holding the catalogue coordinates.

    `exclude` lists the columns belonging to the objects the catalogue was cross-matched
    with. XMatch keeps them under their original names and renames the clashing
    catalogue ones, so without it the object positions would be taken for the catalogue
    ones - which for Gaia, whose columns are named just `ra` and `dec`, silently
    collapses every match onto the object it was matched to.
    """
    cat_col_ra = None
    cat_col_dec = None

    exclude = exclude or []

    def has(*names):
        return all(_ in cat.keys() and _ not in exclude for _ in names)

    # Find relevant coordinate columns
    if has('RAJ2000', 'DEJ2000'):
        cat_col_ra = 'RAJ2000'
        cat_col_dec = 'DEJ2000'

    elif has('_RAJ2000', '_DEJ2000'):
        cat_col_ra = '_RAJ2000'
        cat_col_dec = '_DEJ2000'

    elif has('RA_ICRS', 'DE_ICRS'):
        cat_col_ra = 'RA_ICRS'
        cat_col_dec = 'DE_ICRS'

    # SkyMapper 1.1
    elif has('RAICRS', 'DEICRS'):
        cat_col_ra = 'RAICRS'
        cat_col_dec = 'DEICRS'

    # SkyMapper 4
    elif has('RAdeg', 'DEdeg'):
        cat_col_ra = 'RAdeg'
        cat_col_dec = 'DEdeg'

    # cross-match with Gaia eDR3, where XMatch had to rename the clashing columns
    elif has('ra2', 'dec2'):
        cat_col_ra = 'ra2'
        cat_col_dec = 'dec2'

    elif has('ra_2', 'dec_2'):
        cat_col_ra = 'ra_2'
        cat_col_dec = 'dec_2'

    elif has('ra', 'dec'):
        cat_col_ra = 'ra'
        cat_col_dec = 'dec'

    # else:
    #     raise RuntimeError(f"Cannot find coordinate columns for the catalogue")

    return cat_col_ra, cat_col_dec


def _column_values(col):
    """Return the column values as a plain float array, with masked and non-finite
    entries set to NaN."""
    return np.asarray(np.ma.filled(col, np.nan), dtype=float)


def _column_fluxes(col):
    """Return the linear fluxes corresponding to the magnitudes stored in the column,
    with unusable entries set to zero so that they do not contribute to the sums."""
    mag = _column_values(col)
    flux = np.zeros_like(mag)
    good = np.isfinite(mag)
    flux[good] = 10**(-0.4*mag[good])

    return flux


def _invalidate_column(col, idx):
    """Mark the given rows of the column as unusable, either by masking them out, or by
    setting them to NaN if the column does not support masking."""
    if not np.any(idx):
        return

    if np.ma.isMaskedArray(col):
        col.mask[idx] = True
    elif col.dtype.kind == 'f':
        col[idx] = np.nan


def _catalogue_xyz(cat, cat_col_ra, cat_col_dec):
    """Return the catalogue positions as an (N, 3) array of unit vectors."""
    return np.array([
        np.asarray(_, dtype=float)
        for _ in astrometry.radectoxyz(cat[cat_col_ra], cat[cat_col_dec])
    ]).T


def _close_pairs(points, sr):
    """Return the indices of all pairs of points closer than `sr` degrees.

    The threshold is applied to the chord between the unit vectors, which for the
    separations we care about is the angle itself to better than a part in a million.
    A KD-tree is used instead of :func:`astrometry.spherical_match` as the catalogues
    here routinely have hundreds of thousands of entries.
    """
    return cKDTree(points).query_pairs(np.deg2rad(sr), output_type='ndarray')


def _group_bounds(labels, ngroups):
    """Return the per-group member indices as a flat array, with the start of every
    group inside it."""
    order = np.argsort(labels, kind='stable')

    return order, np.searchsorted(labels[order], np.arange(ngroups))


# Groups larger than that are not worth splitting, as it costs O(N^2) memory
MAX_SPLITTABLE_GROUP = 5000


def group_catalogue_stars(cat, sr, sr_max=None, cat_col_ra='RAJ2000', cat_col_dec='DEJ2000',
                          verbose=False):
    """Group the catalogue stars so that every pair closer than `sr` degrees ends up in
    the same group.

    The result is identical to single-linkage clustering with `sr` distance threshold,
    but it is computed as the connected components of the pair graph built using a
    KD-tree, which scales to the large catalogues typical for crowded fields.

    Single linkage chains the stars, so the groups may be arbitrarily larger than `sr`.
    If `sr_max` is set, they are additionally split, using complete linkage, so that no
    two members of a group are farther than `sr_max` degrees from each other. It keeps
    the groups within the size a fixed photometric aperture may actually collect, and so
    prevents merging the stars the aperture will never see together.

    Returns the array of per-star group indices.
    """
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    points = _catalogue_xyz(cat, cat_col_ra, cat_col_dec)
    pairs = _close_pairs(points, sr)

    graph = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
        shape=(len(cat), len(cat))
    )

    _,labels = connected_components(graph, directed=False)

    if sr_max is None:
        return labels

    threshold = np.deg2rad(sr_max)

    ngroups = labels.max() + 1
    cnt = np.bincount(labels, minlength=ngroups)
    order,starts = _group_bounds(labels, ngroups)
    ends = np.append(starts[1:], len(order))

    extra = ngroups
    nunsplittable = 0

    for g in np.where(cnt > 1)[0]:
        ids = order[starts[g]:ends[g]]

        if len(ids) > MAX_SPLITTABLE_GROUP:
            # Too large to split - safer to not merge these stars at all
            labels[ids] = extra + np.arange(len(ids))
            extra += len(ids)
            nunsplittable += len(ids)
            continue

        sub = points[ids]

        if pdist(sub).max() <= threshold:
            continue # Already compact enough

        split = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            linkage='complete'
        ).fit_predict(sub)

        labels[ids] = extra + split
        extra += split.max() + 1

    if nunsplittable:
        log(f"{nunsplittable} stars are chained into groups too large to split, "
            f"and will not be merged")

    # Renumber to keep the labels contiguous
    return np.unique(labels, return_inverse=True)[1]


# Number of merged members, stored into every catalogue returned by the blend merging
COL_NBLEND = '__nblend__'


def filter_catalogue_blends(
        cat_in,
        sr,
        cat_col_ra='RAJ2000',
        cat_col_dec='DEJ2000',
        cat_col_mags=None,
        sr_max=None,
        verbose=False,
):
    """Merge the groups of catalogue stars closer than `sr` degrees to each other into
    single entries, so that they represent what is actually seen in the image.

    `sr_max`, when set, caps the size of a group, so that the stars chained into
    something wider than the photometric aperture are not merged together - see
    :func:`group_catalogue_stars`.

    Every magnitude listed in `cat_col_mags`, as a list of (magnitude, magnitude error)
    column name pairs, is replaced with the sum of the fluxes of all group members, with
    the errors added in quadrature. The first pair is the primary one, and defines the
    weights for the flux weighted centroid the position is replaced with. Number of
    merged members is stored into `COL_NBLEND` column.

    Any other magnitude-like column would still hold the value of an arbitrary group
    member, and is thus masked out on merged entries so that it may not be silently used
    downstream.

    The merged magnitude is masked out if any group member lacks a usable value for it,
    as the sum would then miss a part of the group flux.

    If no usable magnitude columns are provided, the blends may not be merged, and all
    members of every group are rejected instead.
    """
    # Clustering fails if we have less than 2 stars. And it is meaningless anyway
    if len(cat_in) < 2:
        return cat_in

    # Magnitude columns to merge, without duplicates and unknown ones. The caller may
    # well list the same column twice, e.g. as both the primary and a colour one
    cols = {}
    for cm,ce in cat_col_mags or []:
        if cm and cm in cat_in.colnames and cm not in cols:
            cols[cm] = ce if ce and ce in cat_in.colnames else None

    labels = group_catalogue_stars(cat_in, sr, sr_max=sr_max, verbose=verbose,
                                   cat_col_ra=cat_col_ra, cat_col_dec=cat_col_dec)
    ngroups = labels.max() + 1
    nmem = np.bincount(labels, minlength=ngroups)

    if not cols:
        # Without magnitudes we may not merge the blends, so we just reject them
        cat = cat_in[nmem[labels] == 1]
        cat[COL_NBLEND] = 1

        return cat

    # First member of every group, to be used as a template for the merged entry
    order,starts = _group_bounds(labels, ngroups)

    cat = cat_in[order[starts]]
    cat[COL_NBLEND] = nmem

    idx = nmem > 1 # Groups that actually need merging

    if not np.any(idx):
        return cat

    fluxes = {cm: _column_fluxes(cat_in[cm]) for cm in cols}

    # Flux weighted centroids, falling back to plain ones for the groups where no member
    # has a usable flux at all
    primary = fluxes[next(iter(cols))]
    psum = np.bincount(labels, weights=primary, minlength=ngroups)
    weight = np.where(psum[labels] > 0, primary, 1.0)
    wsum = np.where(psum > 0, psum, nmem)

    points = _catalogue_xyz(cat_in, cat_col_ra, cat_col_dec)
    xyz = np.array([
        np.bincount(labels, weights=weight*points[:, i], minlength=ngroups)
        for i in range(3)
    ]) / wsum

    ra,dec = astrometry.xyztoradec(xyz)
    cat[cat_col_ra][idx] = ra[idx]
    cat[cat_col_dec][idx] = dec[idx]

    for cm,ce in cols.items():
        flux = fluxes[cm]
        fsum = np.bincount(labels, weights=flux, minlength=ngroups)
        # Members with no usable magnitude, that would make the merged flux incomplete.
        # It typically happens for spurious catalogue entries around bright stars
        nmiss = np.bincount(labels, weights=(flux <= 0).astype(float), minlength=ngroups)

        good = idx & (fsum > 0) & (nmiss == 0)
        cat[cm][good] = -2.5*np.log10(fsum[good])
        # Nothing usable to merge, or only a part of the group flux - the value would be wrong
        _invalidate_column(cat[cm], idx & ~good)

        if ce:
            # Magnitude errors converted to flux ones and added in quadrature
            err = _column_values(cat_in[ce])
            ferr = np.zeros_like(err)
            good_err = np.isfinite(err)
            ferr[good_err] = 0.4*np.log(10)*flux[good_err]*err[good_err]
            fesum = np.sqrt(np.bincount(labels, weights=ferr**2, minlength=ngroups))

            cat[ce][good] = 2.5/np.log(10)*fesum[good]/fsum[good]
            _invalidate_column(cat[ce], idx & ~good)

    # Mask out the magnitudes we did not merge, as they still correspond to a single
    # arbitrarily chosen member of the group
    merged = [_ for pair in cols.items() for _ in pair if _]
    for cname in cat.colnames:
        if cname not in merged and getattr(cat[cname], 'unit', None) == u.mag:
            _invalidate_column(cat[cname], idx)

    return cat


def filter_catalogue_contamination(
        cat,
        sr,
        cat_col_ra='RAJ2000',
        cat_col_dec='DEJ2000',
        cat_col_mag=None,
        contamination=0.1,
):
    """Reject the catalogue entries that are significantly contaminated by their
    neighbours inside `sr` degrees radius, i.e. the ones where the total flux of all
    other entries exceeds `contamination` fraction of their own flux.

    Unlike rejecting everything that merely has a neighbour, it keeps the stars that
    dominate their photometric aperture, which is what actually matters for the
    calibration, and thus works much better in crowded fields.

    Entries with no usable magnitude, as well as the ones having such a neighbour whose
    contribution may not be estimated, are rejected as well.
    """
    if len(cat) < 2 or not cat_col_mag or cat_col_mag not in cat.colnames:
        return cat

    pairs = _close_pairs(_catalogue_xyz(cat, cat_col_ra, cat_col_dec), sr)

    flux = _column_fluxes(cat[cat_col_mag])

    def sum_over_neighbours(values):
        return (np.bincount(pairs[:, 0], weights=values[pairs[:, 1]], minlength=len(cat)) +
                np.bincount(pairs[:, 1], weights=values[pairs[:, 0]], minlength=len(cat)))

    # Total flux of the neighbours of every entry
    neighbours = sum_over_neighbours(flux)

    # Neighbours with no usable magnitude contribute an unknown amount of flux, so the
    # entries they may contaminate have to be rejected instead of being deemed clean
    unknown = sum_over_neighbours((flux <= 0).astype(float))

    return cat[(flux > 0) & (unknown == 0) & (neighbours <= contamination*flux)]


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
            cat_col_mag,cat_col_mag_err = guess_catalogue_mag_columns(fname, xcat)
            cat_col_ra,cat_col_dec = guess_catalogue_radec_columns(
                xcat, exclude=[obj_col_ra, obj_col_dec]
            )

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
                    cat_col_mags=[(cat_col_mag, cat_col_mag_err)]
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

                    cand_idx &= ~np.isin(obj[col_id], xcat[col_id])

        log(
            np.sum(cand_idx),
            'remains after matching blends with',
            catalogs.catalogs.get(catname, {'name': catname})['name'],
        )

    return obj[cand_idx]
