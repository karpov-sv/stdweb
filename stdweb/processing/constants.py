"""
Constants and configuration data for astronomical processing.
Includes filter definitions, catalog mappings, and template sources.
"""

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
    'g': {'name':'Pan-STARRS g', 'aliases':["sdssg", "SDSS g", "SDSS-g", "SDSS-g'", "Sloan-g", "sloang", "Sloan g", "Sg", "SG", "sG", "ZTF_g"]},
    'r': {'name':'Pan-STARRS r', 'aliases':["sdssr", "SDSS r", "SDSS-r", "SDSS-r'", "Sloan-r", "sloanr", "Sloan r", "Sr", "SR", "sR", "ZTF_r"]},
    'i': {'name':'Pan-STARRS i', 'aliases':["sdssi", "SDSS i", "SDSS-i", "SDSS-i'", "Sloan-i", "sloani", "Sloan i", "Si", "SI", "sI", "ZTF_i"]},
    'z': {'name':'Pan-STARRS z', 'aliases':["sdssz", "SDSS z", "SDSS-z", "SDSS-z'", "Sloan-z", "sloanz", "Sloan z", "Sz", "SZ", "sZ"]},
    'y': {'name':'Pan-STARRS y', 'aliases':["sdssy", "SDSS y", "SDSS-y", "SDSS-y'", "Sloan-y", "sloany", "Sloan y", "Sy", "SY", "sY"]},
    # Gaia
    'G': {'name':'Gaia G', 'aliases':[]},
    'BP': {'name':'Gaia BP', 'aliases':[]},
    'RP': {'name':'Gaia RP', 'aliases':[]},
    # 2MASS
    'J': {'name':'2MASS J', 'aliases':[]},
    'H': {'name':'2MASS H', 'aliases':[]},
    'Ks': {'name':'2MASS Ks', 'aliases':['K']},
}

# Approximate effective wavelengths in nm, used to pick the band closest to the one we
# need when a catalogue does not provide it
filter_wavelengths = {
    'u': 354,
    'U': 365,
    'B': 445,
    'g': 477,
    'BP': 511,
    'V': 551,
    'G': 622,
    'r': 623,
    'R': 658,
    'i': 762,
    'RP': 777,
    'I': 806,
    'z': 913,
    'y': 962,
    'J': 1235,
    'H': 1662,
    'Ks': 2159,
}

supported_catalogs = {
    'gaiadr3syn': {'name':'Gaia DR3 synphot', 'filters':['U', 'B', 'V', 'R', 'I', 'u', 'g', 'r', 'i', 'z', 'y'],
                   'limit': 'rmag'},
    'ps1': {'name':'Pan-STARRS DR1', 'filters':['B', 'V', 'R', 'I', 'g', 'r', 'i', 'z', 'y'],
            'limit':'rmag'},
    'skymapper': {'name':'SkyMapper DR4', 'filters':['B', 'V', 'R', 'I', 'g', 'r', 'i', 'z', 'y'],
                  'limit':'rPSF'},
    'sdss': {'name':'SDSS DR16', 'filters':['u', 'g', 'r', 'i', 'z'],
             'limit':'rmag'},
    'atlas': {'name':'ATLAS-REFCAT2', 'filters':['B', 'V', 'R', 'I', 'g', 'r', 'i', 'z'],
              'limit':'rmag'},
    'gaiaedr3': {'name':'Gaia eDR3', 'filters':['G', 'BP', 'RP'],
              'limit':'Gmag'},
    '2mass': {'name':'2MASS PSC', 'filters':['J', 'H', 'Ks'],
              'limit':'Jmag'},
    'vhs': {'name':'VHS DR5', 'filters':['J', 'H', 'Ks'],
              'limit':'Jap3'},
}

supported_catalogs_transients = {
    **supported_catalogs,
    'II/371/des_dr2': {'name':'DES DR2', 'filters':['g', 'r', 'i', 'z'],
            'limit': 'rmag'},
}

supported_templates = {
    'custom': {'name': 'Custom template'},
    # `common_zp` marks the surveys whose bands are on a common photometric scale, so
    # that tabulated combination coefficients may be applied to their pixel values
    'ps1': {'name': 'Pan-STARRS DR2', 'filters': {'g', 'r', 'i', 'z'}, 'common_zp': True},
    'ls': {'name': 'Legacy Survey DR11', 'filters': {'g', 'r', 'i', 'z'}, 'common_zp': True},
    'skymapper': {'name': 'SkyMapper DR4 (HiPS)', 'filters': {
        'u': 'CDS/P/skymapper-U', # DR1 fallback
        'g': 'CDS/P/Skymapper/DR4/g',
        'r': 'CDS/P/Skymapper/DR4/r',
        'i': 'CDS/P/Skymapper/DR4/i',
        'z': 'CDS/P/skymapper-Z', # DR1 fallback
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
    '2mass': {'name': '2MASS (HiPS)', 'filters': {
        'J': 'CDS/P/2MASS/J',
        'H': 'CDS/P/2MASS/H',
        'Ks': 'CDS/P/2MASS/K',
    }},
}

# Best guess template filter mappings
filter_mappings = {
    'U': ['u', 'g'],
    'B': ['u', 'g'],
    'V': ['g'],
    'R': ['r', 'i'],
    'I': ['i', 'z'],
    'J': ['J', 'i', 'r'],
    'H': ['H', 'i', 'r'],
    'Ks': ['K', 'i', 'r'],
    'u': ['u', 'g'],
    'g': ['g', 'g'],
    'r': ['r', 'i'],
    'i': ['i', 'r'],
    'z': ['z', 'i', 'r'],
    'y': ['y', 'z', 'i', 'r'],
    'G': ['r', 'g'],
    'BP': ['g'],
    'RP': ['i', 'r'],
}


# Image filters that fall between two template bands, and thus may be approximated by
# their linear combination, `T = T1 + coeff*T2`. Such a synthetic band cancels most of
# the color dependence the subtraction residuals have when a single, spectrally
# mismatched template band is used.
# The coefficients are the fallbacks used when they may not be fitted on the actual
# images, and correspond to the pure filters. They are the ones nulling the color trend
# of the stellar fluxes in Gaia DR3 synthetic photometry - for V it leaves 1.3% of
# scatter with no trend, against 9.5% and 0.40 per mag of (g-r) for the g band alone;
# for R, 0.40% against 2.2% and 0.23; for I, 0.25% against 2.1% and 0.42.
# They are close to `a/((1 - a)*F2/F1)`, the tangent to the `m = m1 - a*(m1 - m2)` color
# transform at the typical color of the field, which is 0.59 for V.
# The bands bluer than g (U, u, B) would need a negative coefficient, i.e. an
# extrapolation outside the pair, and are thus not listed here.
filter_combinations = {
    'V': ('g', 'r', 0.47),
    'R': ('r', 'i', 0.23),
    'I': ('i', 'z', 0.77),
}


# Conversion to AB mags, from https://www.astronomy.ohio-state.edu/martini.10/usefuldata.html
filter_ab_offset = {
    'U': 0.79,
    'B': -0.09,
    'V': 0.02,
    'R': 0.21,
    'I': 0.45,
    'J': 0.91,
    'H': 1.39,
    'Ks': 1.85,
    'u': 0,
    'g': 0,
    'r': 0,
    'i': 0,
    'z': 0,
    'y': 0,
    'G': 0,
    'BP': 0,
    'RP': 0,
}


# Object flags set by SExtractor and STDPipe during the photometry, and by the
# pre-filtering step. Note that 0x800 is also used by STDPipe optimal extraction to
# report a failure, so it is only unambiguous with plain aperture photometry.
object_flags = {
    0x0001: 'bad pixels',
    0x0002: 'deblended',
    0x0004: 'saturated',
    0x0008: 'edge',
    0x0010: 'incomplete',
    0x0020: 'iso incomplete',
    0x0040: 'deblend overflow',
    0x0080: 'extract overflow',
    0x0100: 'masked footprint',
    0x0200: 'masked aperture',
    0x0400: 'bad position',
    0x0800: 'prefiltered',
    0x1000: 'bad fit',
    0x2000: 'moved',
}


def describe_object_flags(flags):
    """Return a human readable, comma separated list of the object flags that are set."""
    if not flags:
        return ''

    names = [name for bit,name in object_flags.items() if int(flags) & bit]
    unknown = int(flags) & ~sum(object_flags.keys())

    if unknown:
        names.append(f"0x{unknown:x}")

    return ', '.join(names)


# Files created at every step

files_inspect = [
    'inspect.log',
    'mask.fits', 'image_target.fits',
]

files_photometry = [
    'photometry.log',
    'objects.png', 'fwhm.png', 'profile.png',
    'segmentation.fits',
    'image_bg.fits', 'image_rms.fits',
    'photometry.png', 'photometry_unmasked.png',
    'photometry_zeropoint.png', 'photometry_model.png',
    'photometry_residuals.png', 'astrometry_dist.png',
    'photometry.pickle',
    'psf.pickle', 'psf.png', 'psf_grid.png', 'psf_residuals.png',
    'objects.parquet', 'cat.parquet',
    'limit_hist.png', 'limit_sn.png',
    'target.vot', 'target.cutout', 'targets'
]

files_transients_simple = [
    'transients_simple.log',
    'candidates_simple', 'candidates_simple.vot', 'candidates_simple.reg'
]

files_subtraction = [
    'subtraction.log',
    'sub_image.fits', 'sub_mask.fits',
    'sub_template.fits', 'sub_template_mask.fits',
    'sub_diff.fits', 'sub_sdiff.fits', 'sub_conv.fits', 'sub_ediff.fits',
    'sub_scorr.fits', 'sub_fpsf.fits', 'sub_fpsferr.fits',
    'sub_target.vot', 'sub_target.cutout',
    'sub_profile.png', 'sub_target_profile.png', 'sub_photometry.png',
    'candidates', 'candidates.vot', 'candidates.reg'
]

cleanup_inspect = files_inspect + files_photometry + files_transients_simple + files_subtraction

cleanup_photometry = files_photometry + files_transients_simple + files_subtraction

cleanup_transients_simple = files_transients_simple

cleanup_subtraction = files_subtraction
