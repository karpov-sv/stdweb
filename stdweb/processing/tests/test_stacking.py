"""
Tests for image stacking, in particular the WCS-less astroalign fallback.
"""

import os
import glob

import numpy as np
import pytest

from astropy.io import fits
from astropy.wcs import WCS

from stdweb.processing.stacking import stack_images

# Real task 229 frames without any WCS. These are large and not tracked in
# git, so the fixture-based test skips gracefully when they are absent.
FIXTURE_DIR = os.path.join(
    'data', 'tests', 'stacking', 'nowcs', 'GCN-260706_161221_R.O.A.D.'
)


def _make_starfield(shape=(256, 256), n=20, shift=(0.0, 0.0), seed=1):
    """Render a field of Gaussian stars, optionally translated by `shift`."""
    rng = np.random.default_rng(seed)
    margin = 30  # keep stars off the edges so they survive the shift
    xs = rng.uniform(margin, shape[1] - margin, n)
    ys = rng.uniform(margin, shape[0] - margin, n)
    fluxes = rng.uniform(2000, 20000, n)

    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    img = np.zeros(shape, dtype=np.float64)
    sigma = 2.0
    for x, y, f in zip(xs, ys, fluxes):
        img += f * np.exp(-(((xx - x - shift[0]) ** 2 +
                             (yy - y - shift[1]) ** 2) / (2 * sigma ** 2)))

    return img, xs, ys, fluxes


def _write_nowcs_fits(path, data):
    fits.PrimaryHDU(data.astype(np.float32)).writeto(path, overwrite=True)


def _write_brokenwcs_fits(path, data):
    """Write a FITS with a formally celestial but all-zero (broken) WCS.

    Mirrors task 232 inputs: CTYPE says RA/DEC TAN so ``is_celestial`` is True,
    but all CRVAL/CRPIX/CD are zero, giving a nonsensical pixel scale.
    """
    hdu = fits.PrimaryHDU(data.astype(np.float32))
    hdu.header['CTYPE1'] = 'RA---TAN'
    hdu.header['CTYPE2'] = 'DEC--TAN'
    for key in ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2',
                'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']:
        hdu.header[key] = 0.0
    hdu.writeto(path, overwrite=True)


def _peak_at(arr, x, y, r=3):
    x, y = int(round(x)), int(round(y))
    return np.nanmax(arr[y - r:y + r + 1, x - r:x + r + 1])


def test_stack_without_wcs_uses_astroalign(tmp_path, settings):
    """Two translated star fields without WCS must be aligned and coadded."""
    pytest.importorskip('astroalign')

    # Inputs must live under DATA_PATH to satisfy the stacking safeguard
    settings.DATA_PATH = str(tmp_path)

    shift = (7.0, -5.0)
    img0, xs, ys, fluxes = _make_starfield(seed=1)
    img1, *_ = _make_starfield(seed=1, shift=shift)  # same field, translated

    f0 = str(tmp_path / 'a.fits')
    f1 = str(tmp_path / 'b.fits')
    _write_nowcs_fits(f0, img0)
    _write_nowcs_fits(f1, img1)

    # Sanity: the inputs really do lack a usable WCS
    assert not WCS(fits.getheader(f0)).is_celestial

    out = str(tmp_path / 'stack.fits')
    stack_images([f0, f1], tmp_path, out,
                 {'stack_method': 'sum', 'stack_subtract_bg': False},
                 verbose=False)

    assert os.path.exists(out)
    coadd = fits.getdata(out)
    assert coadd.shape == img0.shape

    # Alignment check on the brightest star: both frames should stack
    # coherently at its reference position (~2x single peak), while the
    # un-corrected "ghost" position (offset by `shift`) stays empty.
    idx = int(np.argmax(fluxes))
    px, py = xs[idx], ys[idx]

    single_peak = _peak_at(img0, px, py)
    ref_peak = _peak_at(coadd, px, py)
    ghost_peak = _peak_at(coadd, px + shift[0], py + shift[1])

    assert ref_peak > 1.5 * single_peak    # coherent stack -> aligned
    assert ghost_peak < 0.5 * single_peak   # no doubled star -> aligned


def test_stack_with_broken_wcs_falls_back_to_astroalign(tmp_path, settings):
    """Formally-celestial but all-zero WCS must be treated as absent."""
    pytest.importorskip('astroalign')

    settings.DATA_PATH = str(tmp_path)

    shift = (7.0, -5.0)
    img0, xs, ys, fluxes = _make_starfield(seed=1)
    img1, *_ = _make_starfield(seed=1, shift=shift)

    f0 = str(tmp_path / 'a.fits')
    f1 = str(tmp_path / 'b.fits')
    _write_brokenwcs_fits(f0, img0)
    _write_brokenwcs_fits(f1, img1)

    # Sanity: the WCS is formally celestial yet broken (huge pixel scale)
    assert WCS(fits.getheader(f0)).is_celestial

    out = str(tmp_path / 'stack.fits')
    stack_images([f0, f1], tmp_path, out,
                 {'stack_method': 'sum', 'stack_subtract_bg': False},
                 verbose=False)

    assert os.path.exists(out)
    coadd = fits.getdata(out)
    assert coadd.shape == img0.shape

    # If the broken WCS were used for reprojection the stars would not align;
    # astroalign fallback should coadd them coherently.
    idx = int(np.argmax(fluxes))
    px, py = xs[idx], ys[idx]

    single_peak = _peak_at(img0, px, py)
    ref_peak = _peak_at(coadd, px, py)
    ghost_peak = _peak_at(coadd, px + shift[0], py + shift[1])

    assert ref_peak > 1.5 * single_peak
    assert ghost_peak < 0.5 * single_peak


@pytest.mark.skipif(not os.path.isdir(FIXTURE_DIR),
                    reason="task 229 no-WCS fixtures not present")
def test_stack_task229_nowcs_fixtures(tmp_path):
    """Regression test on the real task 229 frames that triggered the fix."""
    pytest.importorskip('astroalign')

    files = sorted(glob.glob(os.path.join(FIXTURE_DIR, '*.fit')))
    assert len(files) >= 2

    # Frames have no WCS, so this exercises the astroalign fallback
    assert not WCS(fits.getheader(files[0])).is_celestial

    out = str(tmp_path / 'image.fits')
    stack_images(files, tmp_path, out, {'stack_method': 'median'}, verbose=False)

    assert os.path.exists(out)
    coadd = fits.getdata(out)
    # Real dithered frames align with only small border losses
    assert np.isfinite(coadd).mean() > 0.9
