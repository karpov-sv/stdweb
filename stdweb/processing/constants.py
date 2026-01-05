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
}

supported_catalogs_transients = {
    **supported_catalogs,
    'II/371/des_dr2': {'name':'DES DR2', 'filters':['g', 'r', 'i', 'z'],
            'limit': 'rmag'},
}

supported_templates = {
    'custom': {'name': 'Custom template'},
    'ps1': {'name': 'Pan-STARRS DR2', 'filters': {'g', 'r', 'i', 'z'}},
    'ls': {'name': 'Legacy Survey DR10', 'filters': {'g', 'r', 'i', 'z'}},
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
    'z': ['z', 'i', 'r'],
    'y': ['y', 'z', 'i', 'r'],
    'G': ['r', 'g'],
    'BP': ['g'],
    'RP': ['i', 'r'],
}


# Conversion to AB mags, from https://www.astronomy.ohio-state.edu/martini.10/usefuldata.html
filter_ab_offset = {
    'U': 0.79,
    'B': -0.09,
    'V': 0.02,
    'R': 0.21,
    'I': 0.45,
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


# Files created at every step

files_inspect = [
    'inspect.log',
    'mask.fits', 'image_target.fits',
]

files_photometry = [
    'photometry.log',
    'objects.png', 'fwhm.png',
    'segmentation.fits',
    'image_bg.fits', 'image_rms.fits',
    'photometry.png', 'photometry_unmasked.png',
    'photometry_zeropoint.png', 'photometry_model.png',
    'photometry_residuals.png', 'astrometry_dist.png',
    'photometry.pickle',
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
    'candidates', 'candidates.vot', 'candidates.reg'
]

cleanup_inspect = files_inspect + files_photometry + files_transients_simple + files_subtraction

cleanup_photometry = files_photometry + files_transients_simple + files_subtraction

cleanup_transients_simple = files_transients_simple

cleanup_subtraction = files_subtraction
