"""
Tests for subtraction diagnostics, in particular the post-subtraction photometry.
"""

import os

import numpy as np
import pytest

from astropy.table import Table

from stdweb.processing.subtraction import fit_robust_trend, plot_subtraction_photometry


def _make_leaking_field(shape=(600, 600), fwhm=4.0, n=120, seed=3,
                        leak0=5.0, color_slope=10.0):
    """Render a field of stars along with the difference image where a color dependent
    fraction of their flux is left unsubtracted."""
    rng = np.random.default_rng(seed)
    sigma = fwhm/2.355

    xs = rng.uniform(40, shape[1] - 40, n)
    ys = rng.uniform(40, shape[0] - 40, n)
    fluxes = 10**rng.uniform(4, 5.5, n)
    colors = rng.uniform(0.3, 1.6, n)

    # Per cent of the flux left in the difference image
    leak = leak0 + color_slope*(colors - 1.0)

    yy,xx = np.mgrid[0:shape[0], 0:shape[1]]
    diff = np.zeros(shape)
    for x0,y0,flux,frac in zip(xs, ys, fluxes, leak/100):
        diff += frac*flux/(2*np.pi*sigma**2)*np.exp(-((xx - x0)**2 + (yy - y0)**2)/(2*sigma**2))

    diff += rng.normal(0, 1.0, shape)

    obj = Table({
        'x': xs, 'y': ys, 'flux': fluxes,
        'mag_calib': -2.5*np.log10(fluxes) + 25,
        'flags': np.zeros(n, dtype=int),
        'magerr': np.full(n, 0.01),
    })

    m = {
        'oidx': np.arange(n),
        'color': colors,
        'idx': np.ones(n, dtype=bool),
        'cat_col_mag': 'Vmag', 'cat_col_mag1': 'Bmag', 'cat_col_mag2': 'Vmag',
    }

    return diff, obj, m


def test_fit_robust_trend():
    rng = np.random.default_rng(1)
    x1 = rng.uniform(0, 10, 200)
    x2 = rng.uniform(0, 1, 200)
    y = 3.0 + 0.5*x1 - 2.0*x2 + rng.normal(0, 0.01, 200)

    # The outliers should not spoil the fit
    y[:10] += 100

    c = fit_robust_trend(y, x1, x2, yerr=np.full(len(y), 0.01))

    assert c == pytest.approx([3.0, 0.5, -2.0], abs=0.05)

    # Not enough points to fit
    assert fit_robust_trend(y[:3], x1[:3], x2[:3]) is None


def test_subtraction_photometry(tmp_path):
    fwhm = 4.0
    diff,obj,m = _make_leaking_field(fwhm=fwhm, leak0=5.0, color_slope=10.0)

    filename = str(tmp_path / 'sub_photometry.png')
    res = plot_subtraction_photometry(filename, diff, obj, m, (0, 0), diff.shape,
                                      {'fwhm': fwhm, 'rel_aper': 1.0}, verbose=False)

    assert os.path.exists(filename)
    assert res['nstars'] > 100
    # Aperture of 1 FWHM radius holds most, but not all, of the leaking flux
    assert res['color_slope'] == pytest.approx(0.1, rel=0.15)
    assert res['mag_slope'] == pytest.approx(0.0, abs=0.005)


def test_subtraction_photometry_few_stars(tmp_path):
    """Too few stars to separate the magnitude and color trends - the magnitude one
    should still be fitted, marginally."""
    fwhm = 4.0
    diff,obj,m = _make_leaking_field(fwhm=fwhm, n=7, leak0=5.0, color_slope=10.0)

    filename = str(tmp_path / 'sub_photometry.png')
    res = plot_subtraction_photometry(filename, diff, obj, m, (0, 0), diff.shape,
                                      {'fwhm': fwhm, 'rel_aper': 1.0}, verbose=False)

    assert os.path.exists(filename)
    assert res['nstars'] == 7
    assert res['median'] == pytest.approx(0.05, abs=0.03)
    assert res['mag_slope'] is not None
    assert res['color_slope'] is None


def test_subtraction_photometry_very_few_stars(tmp_path):
    """Not even enough for a fit - the median should still be reported."""
    fwhm = 4.0
    diff,obj,m = _make_leaking_field(fwhm=fwhm, n=4, leak0=5.0, color_slope=10.0)

    filename = str(tmp_path / 'sub_photometry.png')
    res = plot_subtraction_photometry(filename, diff, obj, m, (0, 0), diff.shape,
                                      {'fwhm': fwhm, 'rel_aper': 1.0}, verbose=False)

    assert os.path.exists(filename)
    assert res['nstars'] == 4
    assert res['median'] == pytest.approx(0.05, abs=0.03)
    assert res['mag_slope'] is None
    assert res['color_slope'] is None


def test_subtraction_photometry_without_color(tmp_path):
    """No usable colors - only the magnitude panel, fitted alone."""
    fwhm = 4.0
    diff,obj,m = _make_leaking_field(fwhm=fwhm, leak0=5.0, color_slope=0.0)
    m['color'] = np.full(len(m['color']), np.nan)

    filename = str(tmp_path / 'sub_photometry.png')
    res = plot_subtraction_photometry(filename, diff, obj, m, (0, 0), diff.shape,
                                      {'fwhm': fwhm, 'rel_aper': 1.0}, verbose=False)

    assert os.path.exists(filename)
    assert res['mag_slope'] is not None
    assert res['color_slope'] is None


def test_subtraction_photometry_without_catalogue(tmp_path):
    """The magnitude panel needs no catalogue at all, and should survive it."""
    diff,obj,m = _make_leaking_field()

    # Nothing is matched with the catalogue
    m['oidx'] = np.array([], dtype=int)
    m['color'] = np.array([])
    m['idx'] = np.array([], dtype=bool)

    filename = str(tmp_path / 'sub_photometry.png')
    res = plot_subtraction_photometry(filename, diff, obj, m, (0, 0), diff.shape,
                                      {'fwhm': 4.0, 'rel_aper': 1.0}, verbose=False)

    assert os.path.exists(filename)
    assert res['nstars'] > 100
    assert res['ncolor'] == 0
    assert res['mag_slope'] is not None
    assert res['color_slope'] is None


def test_subtraction_photometry_deeper_than_catalogue(tmp_path):
    """Only a handful of objects are in the catalogue, the rest are fainter - the
    magnitude panel should use them all, and the color one just the matched ones."""
    fwhm = 4.0
    diff,obj,m = _make_leaking_field(fwhm=fwhm, leak0=5.0, color_slope=10.0)

    # Only the 12 brightest objects are matched with the catalogue
    bright = np.argsort(obj['mag_calib'])[:12]
    m['oidx'] = bright
    m['color'] = m['color'][bright]
    m['idx'] = np.ones(len(bright), dtype=bool)

    filename = str(tmp_path / 'sub_photometry.png')
    res = plot_subtraction_photometry(filename, diff, obj, m, (0, 0), diff.shape,
                                      {'fwhm': fwhm, 'rel_aper': 1.0}, verbose=False)

    assert os.path.exists(filename)
    assert res['nstars'] > 100
    assert res['ncolor'] == 12
    assert res['color_slope'] == pytest.approx(0.1, rel=0.3)


def test_subtraction_photometry_without_objects(tmp_path):
    diff,obj,m = _make_leaking_field()

    # Nothing at all is detected
    obj = obj[:0]
    m['oidx'] = np.array([], dtype=int)
    m['color'] = np.array([])
    m['idx'] = np.array([], dtype=bool)

    filename = str(tmp_path / 'sub_photometry.png')
    assert plot_subtraction_photometry(filename, diff, obj, m, (0, 0), diff.shape,
                                       {'fwhm': 4.0, 'rel_aper': 1.0}, verbose=False) is None
    assert not os.path.exists(filename)
