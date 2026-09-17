"""
Tests for photometry diagnostics, in particular the radial profile plot.
"""

import os

import numpy as np
import pytest

from astropy.table import Table

from stdweb.processing.photometry import (plot_radial_profile,
                                          plot_residual_radial_profile,
                                          plot_target_radial_profile)


def _make_starfield(shape=(512, 512), fwhm=4.0, n=60, seed=1, extra=[]):
    """Render a field of Gaussian stars with known FWHM."""
    rng = np.random.default_rng(seed)
    sigma = fwhm/2.355

    xs = rng.uniform(30, shape[1] - 30, n)
    ys = rng.uniform(30, shape[0] - 30, n)
    fluxes = rng.uniform(5e4, 2e5, n)

    # Additional stars at exactly known positions
    for x0,y0,flux in extra:
        xs,ys,fluxes = np.append(xs, x0),np.append(ys, y0),np.append(fluxes, flux)

    yy,xx = np.mgrid[0:shape[0], 0:shape[1]]
    image = np.zeros(shape)
    for x0,y0,flux in zip(xs, ys, fluxes):
        image += flux/(2*np.pi*sigma**2)*np.exp(-((xx - x0)**2 + (yy - y0)**2)/(2*sigma**2))

    image += rng.normal(0, 1.0, shape)

    obj = Table({
        'x': xs, 'y': ys, 'flux': fluxes,
        'flags': np.zeros(len(xs), dtype=int),
        'magerr': np.full(len(xs), 0.01),
    })

    return image, obj


def test_radial_profile_recovers_fwhm(tmp_path):
    fwhm = 4.0
    image,obj = _make_starfield(fwhm=fwhm)

    filename = str(tmp_path / 'profile.png')
    profile = plot_radial_profile(filename, image, obj, fwhm=fwhm, aper=1.0*fwhm,
                                  bkgann=[5*fwhm, 7*fwhm], verbose=False)

    assert os.path.exists(filename)
    assert profile['fwhm'] == pytest.approx(fwhm, abs=0.3)
    # Peak of the profile, per unit flux, should match the Gaussian one
    assert profile['peak'] == pytest.approx(1/(2*np.pi*(fwhm/2.355)**2), rel=0.1)


def test_radial_profile_without_suitable_stars(tmp_path):
    image,obj = _make_starfield()

    # Nothing to measure - everything is flagged
    obj['flags'] = 1

    filename = str(tmp_path / 'profile.png')
    assert plot_radial_profile(filename, image, obj, fwhm=4.0, aper=4.0, verbose=False) is None
    assert not os.path.exists(filename)


def test_target_radial_profile(tmp_path):
    fwhm = 4.0
    # Target with a close neighbour, both at known positions
    image,obj = _make_starfield(fwhm=fwhm, extra=[(200.0, 200.0, 1e5), (206.0, 200.0, 5e4)])

    filename = str(tmp_path / 'target_profile.png')
    mask = np.zeros(image.shape, dtype=bool)
    mask[200, 197] = True

    assert plot_target_radial_profile(filename, image, 200.0, 200.0, fwhm=fwhm, aper=1.0*fwhm,
                                      bkgann=[5*fwhm, 7*fwhm], mask=mask, flux=1e5,
                                      verbose=False)
    assert os.path.exists(filename)

    # Nothing to plot outside the image
    filename = str(tmp_path / 'outside.png')
    assert not plot_target_radial_profile(filename, image, -100.0, -100.0, fwhm=fwhm,
                                          aper=1.0*fwhm, verbose=False)
    assert not os.path.exists(filename)


def test_residual_radial_profile(tmp_path):
    fwhm = 4.0
    image,obj = _make_starfield(fwhm=fwhm)

    profile = plot_radial_profile(str(tmp_path / 'profile.png'), image, obj,
                                  fwhm=fwhm, aper=1.0*fwhm, verbose=False)

    # Difference image where 5% of the stellar flux is left unsubtracted
    res = plot_residual_radial_profile(str(tmp_path / 'residuals.png'), 0.05*image, obj,
                                       fwhm=fwhm, aper=1.0*fwhm, profile=profile,
                                       verbose=False)

    assert os.path.exists(str(tmp_path / 'residuals.png'))
    assert res['core'] == pytest.approx(5.0, abs=0.5)
    assert res['worst'] == pytest.approx(5.0, abs=0.5)
    # Integrated leakage is the same 5%, less the flux outside the aperture
    assert res['leak'] == pytest.approx(4.7, abs=0.5)

    # Perfect subtraction leaves nothing but the noise
    rng = np.random.default_rng(2)
    res = plot_residual_radial_profile(str(tmp_path / 'residuals_clean.png'),
                                       rng.normal(0, 1.0, image.shape), obj,
                                       fwhm=fwhm, aper=1.0*fwhm, profile=profile,
                                       verbose=False)
    assert np.abs(res['core']) < 0.5
