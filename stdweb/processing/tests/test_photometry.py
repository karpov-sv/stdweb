"""
Tests for photometry diagnostics, in particular the radial profile plot.
"""

import os

import numpy as np
import pytest

from astropy.table import Table, vstack

from stdpipe import psf as stdpipe_psf

from stdweb.processing.photometry import (plot_radial_profile,
                                          plot_residual_radial_profile,
                                          plot_target_radial_profile,
                                          get_photometry_method,
                                          build_psf_model,
                                          shift_psf_model,
                                          measure_photometry,
                                          get_psf_model_shape,
                                          plot_psf_model,
                                          plot_psf_grid,
                                          plot_psf_residuals,
                                          measure_targets_photometry,
                                          get_psf_residual_image)


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


def test_photometry_method():
    assert get_photometry_method({}) == 'aperture'
    assert get_photometry_method({'optimal_extraction': True}) == 'optimal'
    assert get_photometry_method({'photometry_method': 'psf', 'optimal_extraction': True}) == 'psf'


def test_psf_photometry_recovers_total_flux():
    fwhm = 4.0
    image,obj = _make_starfield(fwhm=fwhm, n=80)

    psf = build_psf_model(image, obj, np.ones(len(obj), dtype=bool), fwhm, verbose=False)

    config = {'fwhm': fwhm, 'rel_aper': 1.0, 'bg_size': 128}

    # Aperture of 1 FWHM radius misses part of the flux
    res = measure_photometry(obj['x', 'y'], image, config, sn=0)
    assert np.median(res['flux']/obj['flux']) < 0.95

    # PSF fitting should get all of it
    res = measure_photometry(obj['x', 'y'], image, config | {'photometry_method': 'psf'},
                             psf=psf, sn=0)
    assert np.all(res['flags'] == 0)
    assert np.median(res['flux']/obj['flux']) == pytest.approx(1.0, abs=0.01)

    # No PSF model to fit with
    with pytest.raises(RuntimeError):
        measure_photometry(obj['x', 'y'], image, config | {'photometry_method': 'psf'})


def test_shift_psf_model():
    # Position dependent PSF model, with linear terms in both coordinates
    rng = np.random.default_rng(1)
    psf = {
        'width': 15, 'height': 15, 'fwhm': 4.0, 'sampling': 1.0,
        'ncoeffs': 3, 'degree': 1,
        'x0': 256.0, 'y0': 256.0, 'sx': 256.0, 'sy': 256.0,
        'data': rng.uniform(0, 1, (3, 15, 15)),
    }

    x0,y0 = 100,150
    spsf = shift_psf_model(psf, x0, y0)

    # The original model is unchanged
    assert psf['x0'] == 256.0 and psf['y0'] == 256.0

    # PSF in the sub-image should be the one at the corresponding position of the image
    for x,y in [(10.3, 20.7), (200.0, 50.5)]:
        stamp = stdpipe_psf.get_psf_stamp(psf, x + x0, y + y0)
        assert np.allclose(stdpipe_psf.get_psf_stamp(spsf, x, y), stamp)
        assert not np.allclose(stdpipe_psf.get_psf_stamp(psf, x, y), stamp)

    assert shift_psf_model(None, x0, y0) is None


@pytest.mark.parametrize('fwhm', [2.0, 4.0])
def test_psf_model_shape(fwhm):
    # Undersampled PSF gets oversampled model, so both model pixel scales are checked
    image,obj = _make_starfield(fwhm=fwhm, n=80)
    psf = build_psf_model(image, obj, np.ones(len(obj), dtype=bool), fwhm, degree=0,
                          verbose=False)

    mfwhm,ell = get_psf_model_shape(psf, 256, 256)

    assert mfwhm == pytest.approx(fwhm, rel=0.1)
    assert ell < 0.05


def test_psf_diagnostics(tmp_path):
    fwhm = 4.0
    image,obj = _make_starfield(fwhm=fwhm, n=80)
    psf = build_psf_model(image, obj, np.ones(len(obj), dtype=bool), fwhm, degree=1,
                          verbose=False)

    config = {'fwhm': fwhm, 'rel_aper': 1.0, 'bg_size': 128, 'photometry_method': 'psf'}
    mobj = measure_photometry(obj['x', 'y'], image, config, psf=psf, sn=0)
    mobj['magerr'] = np.full(len(mobj), 0.01)

    filename = str(tmp_path / 'psf.png')
    mfwhm = plot_psf_model(filename, psf, image, mobj, fwhm=fwhm, aper=fwhm,
                           bkgann=[5*fwhm, 7*fwhm], verbose=False)
    assert os.path.exists(filename)
    assert mfwhm == pytest.approx(fwhm, rel=0.1)

    filename = str(tmp_path / 'psf_grid.png')
    (fwhm1,fwhm2),(ell1,ell2) = plot_psf_grid(filename, psf, image.shape, fwhm=fwhm)
    assert os.path.exists(filename)
    # Constant PSF should be recovered as such
    assert fwhm1 == pytest.approx(fwhm, rel=0.1)
    assert fwhm2 == pytest.approx(fwhm, rel=0.1)

    filename = str(tmp_path / 'psf_residuals.png')
    res = plot_psf_residuals(filename, psf, image, np.zeros_like(image), mobj, config,
                             verbose=False)
    assert os.path.exists(filename)
    assert res['nstars'] > 50
    # Correct model predicts aperture fluxes exactly
    assert res['median'] == pytest.approx(1.0, abs=0.01)
    assert res['rms'] < 0.01


def test_psf_model_degree_reduction():
    fwhm = 4.0

    # Enough stars for the requested order
    image,obj = _make_starfield(fwhm=fwhm, n=80)
    psf = build_psf_model(image, obj, np.ones(len(obj), dtype=bool), fwhm, degree=2,
                          verbose=False)
    assert psf['degree'] == 2

    # Too few stars for quadratic spatial dependence, but enough for linear one
    image,obj = _make_starfield(fwhm=fwhm, n=20)
    psf = build_psf_model(image, obj, np.ones(len(obj), dtype=bool), fwhm, degree=2,
                          verbose=False)
    assert psf['degree'] == 1

    # Undersampled PSF gets oversampled model, with fewer star samples per model node
    image,obj = _make_starfield(fwhm=2.0, n=80)
    psf = build_psf_model(image, obj, np.ones(len(obj), dtype=bool), 2.0, degree=2,
                          verbose=False)
    assert psf['degree'] == 1


def test_psf_photometry_options():
    fwhm = 4.0
    image,obj = _make_starfield(fwhm=fwhm, n=80)
    psf = build_psf_model(image, obj, np.ones(len(obj), dtype=bool), fwhm, verbose=False)

    config = {'fwhm': fwhm, 'rel_aper': 1.0, 'bg_size': 128, 'photometry_method': 'psf'}

    # Larger fit radius should still recover the fluxes, with better accuracy
    res1 = measure_photometry(obj['x', 'y'], image, config, psf=psf, sn=0)
    res3 = measure_photometry(obj['x', 'y'], image, config | {'psf_fit_radius': 3.0},
                              psf=psf, sn=0)
    assert np.median(res3['flux']/obj['flux']) == pytest.approx(1.0, abs=0.01)
    assert np.median(res3['fluxerr']) < np.median(res1['fluxerr'])
    assert np.max(res3['npix_optimal']) > np.max(res1['npix_optimal'])

    # Grouping is on by default, and zero radius disables it
    assert 'group_id' in res1.colnames
    res = measure_photometry(obj['x', 'y'], image, config | {'psf_group_radius': 0},
                             psf=psf, sn=0)
    assert 'group_id' not in res.colnames
    assert np.median(res['flux']/obj['flux']) == pytest.approx(1.0, abs=0.01)

    # Explicit grouping radius, large enough to join some of the stars
    res = measure_photometry(obj['x', 'y'], image, config | {'psf_group_radius': 10},
                             psf=psf, sn=0)
    assert np.max(res['group_size']) > np.max(res1['group_size'])


def test_psf_bg_fluxerr():
    """Background noise of PSF fit should not depend on the fitting radius, and match the
    analytic value for Gaussian PSF: sigma_pix*sqrt(4*pi)*sigma_psf"""
    fwhm = 4.0
    image,obj = _make_starfield(fwhm=fwhm, n=80)
    psf = build_psf_model(image, obj, np.ones(len(obj), dtype=bool), fwhm, verbose=False)

    config = {'fwhm': fwhm, 'rel_aper': 1.0, 'bg_size': 128, 'photometry_method': 'psf'}
    expected = np.sqrt(4*np.pi)*fwhm/2.355

    for radius in [1.0, 3.0]:
        res = measure_photometry(obj['x', 'y'], image, config | {'psf_fit_radius': radius},
                                 psf=psf, sn=0)
        assert np.median(res['bg_fluxerr']) == pytest.approx(expected, rel=0.05)


def test_psf_model_auto_degree():
    # Constant PSF should get constant model
    image,obj = _make_starfield(fwhm=4.0, n=150)
    psf = build_psf_model(image, obj, np.ones(len(obj), dtype=bool), 4.0, verbose=False)
    assert psf['degree'] == 0

    # PSF width changing across the image should get spatially varying one
    shape = (512, 512)
    rng = np.random.default_rng(2)
    xs = rng.uniform(30, shape[1] - 30, 150)
    ys = rng.uniform(30, shape[0] - 30, 150)
    fluxes = rng.uniform(5e4, 2e5, 150)
    yy,xx = np.mgrid[0:shape[0], 0:shape[1]]
    image = rng.normal(0, 1.0, shape)
    for x0,y0,flux in zip(xs, ys, fluxes):
        sigma = (3.0 + 2.0*x0/shape[1])/2.355
        image += flux/(2*np.pi*sigma**2)*np.exp(-((xx - x0)**2 + (yy - y0)**2)/(2*sigma**2))
    obj = Table({'x': xs, 'y': ys, 'flux': fluxes, 'flags': np.zeros(150, dtype=int)})

    psf = build_psf_model(image, obj, np.ones(len(obj), dtype=bool), 4.0, verbose=False)
    assert psf['degree'] > 0
    assert 'degree_selection' in psf


@pytest.mark.parametrize('max_stars', [10, 40])
def test_psf_model_max_stars(max_stars):
    fwhm = 4.0
    image,obj = _make_starfield(fwhm=fwhm, n=150)
    psf = build_psf_model(image, obj, np.ones(len(obj), dtype=bool), fwhm, degree=0,
                          max_stars=max_stars, size=25, verbose=False)
    assert psf['width'] == 25

    # Number of stars actually used is only logged, so capture it
    lines = []
    build_psf_model(image, obj, np.ones(len(obj), dtype=bool), fwhm, degree=0,
                    max_stars=max_stars, verbose=lines.append)
    nstars = [int(_.split()[1]) for _ in lines if _.startswith('Extracted')][0]
    assert 0.5*max_stars <= nstars <= max_stars


def _make_target_pair(fwhm=4.0, flux=2e4, fratio=5, dist=1.0):
    """Star field with the target, and a brighter neighbour at `dist` FWHM from it,
    placed in the empty region. The neighbour is in the object list, the target is not."""
    image,obj = _make_starfield(fwhm=fwhm, n=60)
    sigma = fwhm/2.355
    x0,y0 = 60.3, 160.6
    yy,xx = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    for x,y,f in [(x0, y0, flux), (x0 + dist*fwhm, y0, fratio*flux)]:
        image += f/(2*np.pi*sigma**2)*np.exp(-((xx - x)**2 + (yy - y)**2)/(2*sigma**2))

    psf = build_psf_model(image, obj, np.ones(len(obj), dtype=bool), fwhm, degree=0,
                          verbose=False)

    obj.add_row({'x': x0 + dist*fwhm, 'y': y0, 'flux': fratio*flux, 'flags': 0, 'magerr': 0.01})
    target = Table({'ra': [10.0], 'dec': [20.0], 'x': [x0], 'y': [y0]})

    return image, obj, psf, target


def test_psf_target_neighbours():
    fwhm,flux = 4.0,2e4
    image,obj,psf,target = _make_target_pair(fwhm=fwhm, flux=flux)

    config = {'fwhm': fwhm, 'rel_aper': 1.0, 'bg_size': 128, 'photometry_method': 'psf'}

    # Neighbour flux leaks into the target
    res,nb = measure_targets_photometry(target, obj, image, config, psf=psf)
    assert len(nb) == 0
    assert res['flux'][0]/flux > 1.5
    # Target columns are kept
    assert res['ra'][0] == 10.0

    # And it does not when fitted simultaneously
    config['psf_fit_neighbors'] = True
    res,nb = measure_targets_photometry(target, obj, image, config, psf=psf)
    assert len(nb) == 1
    assert res['flux'][0]/flux == pytest.approx(1.0, abs=0.02)
    assert nb['flux'][0]/(5*flux) == pytest.approx(1.0, abs=0.02)

    # The detection at the target position is the target itself
    obj.add_row({'x': target['x'][0] + 0.3, 'y': target['y'][0], 'flux': flux, 'flags': 0, 'magerr': 0.01})
    res,nb = measure_targets_photometry(target, obj, image, config, psf=psf)
    assert len(nb) == 1
    assert res['flux'][0]/flux == pytest.approx(1.0, abs=0.02)

    # No grouping - no neighbours
    res,nb = measure_targets_photometry(target, obj, image, config | {'psf_group_radius': 0}, psf=psf)
    assert len(nb) == 0

    # Only for PSF photometry
    res,nb = measure_targets_photometry(target, obj, image, config | {'photometry_method': 'aperture'})
    assert len(nb) == 0


def test_psf_residual_image():
    fwhm,flux = 4.0,2e4
    image,obj,psf,target = _make_target_pair(fwhm=fwhm, flux=flux, dist=2.0)
    config = {'fwhm': fwhm, 'rel_aper': 1.0, 'bg_size': 128, 'photometry_method': 'psf',
              'psf_fit_neighbors': True}

    res,nb = measure_targets_photometry(target, obj, image, config, psf=psf)
    x0,y0 = int(round(target['x'][0])), int(round(target['y'][0]))
    box = np.s_[y0 - 12:y0 + 13, x0 - 8:x0 + 17]
    peak = np.max(image[box])

    # Target and neighbour are subtracted down to the noise
    residual = get_psf_residual_image(image, np.zeros_like(image), psf, vstack([res['x', 'y', 'flux'], nb]))
    assert np.max(np.abs(residual[box])) < 0.02*peak

    # Neighbour that was not fitted is left
    residual = get_psf_residual_image(image, np.zeros_like(image), psf, res)
    assert np.max(residual[box]) > 0.5*peak
