import numpy as np
import pytest

from stdweb.processing.inspect import guess_bg_estimator, measure_sky_skewness, guess_amplified_gain


def _add_stars(image, n, flux, fwhm, rng):
    """Faint Gaussian stars at random positions"""
    sigma = fwhm/2.355
    yy, xx = np.mgrid[:image.shape[0], :image.shape[1]]
    for x0, y0, f in zip(rng.uniform(0, image.shape[1], n), rng.uniform(0, image.shape[0], n),
                         rng.exponential(flux, n)):
        sl = np.s_[max(0, int(y0) - 15):int(y0) + 16, max(0, int(x0) - 15):int(x0) + 16]
        image[sl] += f/(2*np.pi*sigma**2)*np.exp(-((xx[sl] - x0)**2 + (yy[sl] - y0)**2)/2/sigma**2)


@pytest.mark.parametrize('kind,nstars,expected', [
    # Sum of two exponential (EM gain) frames - pixel-independent skewness
    ('gamma', 0, 'mean'),
    ('gamma', 1000, 'mean'),
    # Gaussian noise with many faint stars - skewness correlated on PSF scale
    ('gauss', 3000, 'mode'),
    ('gauss', 0, 'mode'),
])
def test_guess_bg_estimator(kind, nstars, expected):
    rng = np.random.default_rng(1)
    shape = (512, 512)

    if kind == 'gamma':
        image = 100 + 50*rng.gamma(2.0, 1.0, shape)
    else:
        image = 100 + 70*rng.normal(size=shape)

    if nstars:
        _add_stars(image, nstars, 2000, 6.0, rng)

    estimator, bias = guess_bg_estimator(image)
    assert estimator == expected
    if expected == 'mean':
        assert bias > 1


def test_guess_bg_estimator_masked():
    """Masked pixels and NaNs are ignored, too few good ones fall back to mode"""
    rng = np.random.default_rng(2)
    image = 100 + 50*rng.gamma(2.0, 1.0, (256, 256))
    mask = np.zeros(image.shape, dtype=bool)
    mask[:, :64] = True
    image[:, :64] = np.nan
    assert guess_bg_estimator(image, mask)[0] == 'mean'

    mask[:] = True
    assert guess_bg_estimator(image, mask) == ('mode', pytest.approx(np.nan, nan_ok=True))


@pytest.mark.parametrize('seed', range(10))
def test_guess_bg_estimator_quantized(seed):
    """Rounding of Gaussian noise to integers is not mistaken for its skewness"""
    rng = np.random.default_rng(seed)
    image = np.round(100.3 + 3*rng.normal(size=(512, 512)))
    assert guess_bg_estimator(image)[0] == 'mode'

    # Integer-valued skewed noise is still recognized
    image = np.round(100 + 5*rng.gamma(2.0, 1.0, (512, 512)))
    assert guess_bg_estimator(image)[0] == 'mean'


def _em_image(nphot, em_gain, shape=(512, 512), readnoise=1.0, seed=1):
    """EMCCD sky: Poisson photoelectrons, each amplified by exponential random factor"""
    rng = np.random.default_rng(seed)
    n = rng.poisson(nphot, shape)
    image = np.zeros(shape)
    image[n > 0] = rng.gamma(n[n > 0], em_gain)
    return 100 + image + readnoise*rng.normal(size=shape)


@pytest.mark.parametrize('nphot,em_gain', [(2, 50), (5, 20), (10, 100)])
def test_guess_amplified_gain_em(nphot, em_gain):
    """Effective gain of EM-amplified noise, 1/(2 em_gain), is recovered from the skewness"""
    image = _em_image(nphot, em_gain)
    g = guess_amplified_gain(measure_sky_skewness(image), 1.0)
    assert g == pytest.approx(1/(2*em_gain), rel=0.35)
    # Already consistent gain is kept
    assert guess_amplified_gain(measure_sky_skewness(image), 1/(2*em_gain)) is None


@pytest.mark.parametrize('kind', ['poisson', 'gauss_stars', 'too_few'])
def test_guess_amplified_gain_normal(kind):
    """Ordinary photon noise, faint stars, or unmeasurable skewness do not change the gain"""
    rng = np.random.default_rng(3)
    if kind == 'poisson':
        # Low sky of 20 e- with unit gain - skewed, but consistent with the gain
        image = rng.poisson(20, (512, 512)).astype(float)
    elif kind == 'gauss_stars':
        image = 100 + 70*rng.normal(size=(512, 512))
        _add_stars(image, 1000, 2000, 6.0, rng)
    else:
        image = rng.normal(size=(50, 50))
    assert guess_amplified_gain(measure_sky_skewness(image), 1.0) is None


@pytest.mark.parametrize('kind,preset,expected', [
    ('gamma', {}, False),
    ('gamma', {'prefilter_detections': True}, True),
    ('gauss', {}, None),
])
def test_inspect_disables_prefilter_for_skewed_noise(tmp_path, kind, preset, expected):
    """Guessing 'mean' background also disables pre-filtering, unless it is set explicitly"""
    from astropy.io import fits
    from stdweb.processing.inspect import inspect_image

    rng = np.random.default_rng(1)
    if kind == 'gamma':
        image = 100 + 50*rng.gamma(2.0, 1.0, (256, 256))
    else:
        image = 100 + 70*rng.normal(size=(256, 256))
    filename = str(tmp_path / 'image.fits')
    fits.writeto(filename, image.astype(np.float32))

    config = {'mask_cosmics': False, 'target': ''} | preset
    inspect_image(filename, config, verbose=False)

    assert config['bg_estimator'] == ('mean' if kind == 'gamma' else 'mode')
    assert config.get('prefilter_detections') == expected
