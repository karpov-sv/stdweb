Processing Workflow
===================

This section describes the image processing workflow implemented in STDWeb. The processing pipeline consists of several sequential steps, each producing diagnostic outputs and intermediate results.

.. code-block:: text

   Image Upload/Import
          ↓
   Preprocessing (optional: destripe, fringe removal, crop, background removal)
          ↓
   Initial Inspection & Masking
          ↓
   Object Detection
          ↓
   Astrometric Calibration
          ↓
   Photometric Calibration
          ↓
   Transient Detection (optional)
          ├── Simple (catalogue-based)
          └── Image Subtraction (HOTPANTS or SFFT)

Results from many tasks covering the same sky position can afterwards be
combined into :ref:`light curves <light-curves>`, and the full task collection
can be browsed on an interactive :ref:`sky map <sky-map>`.

Image Import
------------

STDWeb accepts astronomical images in FITS format. Images can be:

- **Uploaded** directly via the web browser
- **Selected from local storage** via the built-in file browser (useful for telescope archives)

Upon import, a dedicated **task** is created that holds:

- The image file
- Processing configuration
- Intermediate results
- Operation logs

Tasks can be revisited at any time to review results or re-process with different parameters.

.. _presets:

Configuration Presets
~~~~~~~~~~~~~~~~~~~~~~

When importing an image you may select a **preset** - a named, reusable bundle
of initial configuration values (and, optionally, files to copy into the new
task). Presets are convenient for applying a consistent setup across many images
from the same instrument: filter, reference catalog, detection thresholds, and
so on. They are managed by administrators through the Django admin interface,
and the same presets are also available through the :doc:`REST API <api>` and
the :doc:`command-line tool <cli>`.

Image Stacking
~~~~~~~~~~~~~~

Several images covering the same field can be combined into a single deeper task
at import time. Select multiple files in the browser and choose **Stack and
Process** instead of importing them individually.

The first image defines the reference pixel grid; the rest are reprojected onto
it (flux-conserving adaptive resampling) and then combined using one of:

- **Sum** - plain co-addition; the saturation level is scaled by the number of
  images accordingly
- **Clipped mean** - sigma-clipped mean, robust against outliers
- **Median** - median combination

Optionally, the background can be subtracted from each frame before combining,
and cosmic rays masked per frame. The resulting stack becomes an ordinary task
and proceeds through the normal pipeline. Remember to set the gain to reflect the
stacking (see the :doc:`handbook`).

Preprocessing
-------------

Before the main pipeline runs, an optional interactive preprocessing page lets
you clean up the uploaded image. All operations rewrite ``image.fits`` in place;
the original is backed up automatically (as ``image.orig.fits``) so it can be
restored at any time with the **Reset** action.

Available operations:

- **Destriping** - removes horizontal or vertical banding by equalising the
  per-row or per-column median to the global image median
- **Fringe removal** - subtracts a fringe pattern model (using the custom mask,
  if present, to exclude sources); the fitted model is saved as
  ``fringe_model.fits``
- **Cropping** - trims the image to a user-specified pixel box
- **Background removal** - estimates a smooth background (``sep``, ``photutils``,
  or ``morphology`` method, with a configurable mesh size) and either subtracts
  it or divides by it (useful for strong gradients or large-scale flat-field
  residuals)

Each operation triggers a cleanup of downstream results so the pipeline is
re-run from a consistent state.

Initial Inspection and Masking
------------------------------

The first processing step analyzes the image and prepares it for further analysis.

Header Analysis
~~~~~~~~~~~~~~~

The software automatically extracts information from FITS headers:

- **Timestamp** - observation time
- **Saturation level** - detector saturation threshold
- **Gain** - detector gain (e/ADU)
- **Existing WCS** - astrometric solution if present

All values can be manually specified if header information is missing or incorrect. The software performs basic sanity checks to detect issues like stacked or rescaled images.

WCS Sanitization
~~~~~~~~~~~~~~~~

The software fixes common WCS header problems that break AstroPy's WCS module, removing or modifying problematic keywords produced by some imaging software.

Automatic Masking
~~~~~~~~~~~~~~~~~

A pixel mask is created to exclude problematic regions:

- **Saturated pixels** - based on saturation level and gain
- **Cosmic ray hits** - detected using AstroSCRAPPY (LACosmic algorithm)

The noise model for cosmic ray detection is automatically constructed from:

- Empirically estimated background noise
- Poissonian noise contribution for sources above background

Cosmic ray masking can be disabled for undersampled images where the algorithm may incorrectly flag stellar cores.

Custom Masking
~~~~~~~~~~~~~~

Users can interactively create additional masks for:

- Vignetted regions
- Overscan areas
- Imaging artifacts
- Significant reflections

Target Specification
~~~~~~~~~~~~~~~~~~~~

A list of targets for forced photometry can be provided as:

- Coordinate strings (various formats supported)
- SIMBAD resolvable names
- TNS (Transient Name Server) names

Object Detection
----------------

Object detection uses SExtractor for performance and memory efficiency on large images.

SExtractor Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

Minimal configuration is constructed from user-specified parameters:

- Smoothing kernel size
- Background mesh size
- Detection threshold

Custom mask regions are set to NaN to exclude them from background estimation.

Diagnostic Outputs
~~~~~~~~~~~~~~~~~~

The following diagnostic images are generated:

- Segmentation map
- Estimated background map
- Background RMS map
- Filtered (smoothed) image

Shape-Based Pre-filtering
~~~~~~~~~~~~~~~~~~~~~~~~~

Three SExtractor parameters are used to identify non-stellar detections:

- ``FWHM_IMAGE`` - Full Width at Half Maximum (pixels)
- ``FLUX_RADIUS`` - Half Flux Radius (pixels)
- ``MAG_APER - MAG_AUTO`` - difference between fixed circular and Kron-like elliptical aperture magnitudes

An isolation forest anomaly detection algorithm (scikit-learn) identifies outliers in this parameter space, flagging artifacts and blended sources.

Object Flags
~~~~~~~~~~~~

Objects are flagged at various processing stages:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Bitmask
     - Meaning
   * - ``0x001``
     - Aperture flux affected by nearby stars or bad pixels
   * - ``0x002``
     - Object is deblended
   * - ``0x004``
     - Object is saturated
   * - ``0x008``
     - Object footprint is truncated
   * - ``0x010``
     - Object aperture data are incomplete
   * - ``0x020``
     - Object isophotal data are incomplete
   * - ``0x100``
     - Object footprint contains masked pixels
   * - ``0x200``
     - Photometric aperture contains masked pixels
   * - ``0x400``
     - Local background annulus lacks sufficient good pixels
   * - ``0x800``
     - Object classified as outlier by pre-filtering

Aperture Photometry
~~~~~~~~~~~~~~~~~~~

Additional photometry is performed using photutils routines:

- **Aperture radius**: automatically set to image FWHM
- **Sky annulus**: placed between 5 and 7 FWHM
- **FWHM estimation**: median of 2×FLUX_RADIUS for unflagged objects with S/N > 20

This step also performs forced photometry at user-specified target positions.

Optionally, **optimal (PSF-weighted) extraction** can be enabled. Instead of a
plain aperture sum, fluxes are measured with inverse-variance weighting matched
to the image PSF, which improves the signal-to-noise ratio for faint sources at
the cost of additional computation.

S/N Filtering
~~~~~~~~~~~~~

Detections with photometric errors exceeding the user-specified threshold (default S/N > 5, corresponding to magnitude errors < 0.2 mag) are excluded from further analysis.

Astrometric Calibration
-----------------------

Blind Solving
~~~~~~~~~~~~~

If the FITS header lacks a usable WCS solution, blind solving is performed using a local Astrometry.Net instance with 2MASS indices:

- Uses detected object positions (flagged objects excluded)
- Optional user-specified constraints on sky position and pixel scale
- Runs twice to produce WCS with second-order SIP distortions

WCS Refinement
~~~~~~~~~~~~~~

Once a preliminary solution exists (from blind solving or header), refinement is performed using SCAMP:

- Uses the photometric reference catalog
- Matches unflagged detected objects
- Ensures accuracy required for photometric calibration and image subtraction

Photometric Calibration
-----------------------

Reference Catalogs
~~~~~~~~~~~~~~~~~~

STDWeb supports multiple reference catalogs from CDS Vizier:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Catalog
     - Coverage
   * - Pan-STARRS DR1
     - Northern sky (δ > -30°), faint stars
   * - SkyMapper DR4
     - Southern sky, faint stars
   * - SDSS DR16
     - Footprint-limited, ugriz bands
   * - Gaia DR3 synphot
     - All-sky, brighter stars (synthetic photometry from spectra)
   * - Gaia eDR3
     - All-sky, G/BP/RP bands only
   * - ATLAS-REFCAT2
     - All-sky compilation

Catalogs are queried on-the-fly with optional magnitude limits to reduce download size.

Supported Filter Systems
~~~~~~~~~~~~~~~~~~~~~~~~

- **Pan-STARRS**: g, r, i, z, y
- **Johnson-Cousins**: U, B, V, R, I
- **Gaia**: G, BP, RP

Transformation equations between systems are derived from Landolt standards.

Photometric Model
~~~~~~~~~~~~~~~~~

The calibration model accounts for multiple effects:

.. code-block:: text

   m_calib = m_instr + ZP(x, y) + C × Color

Where:

- ``m_calib`` - magnitude in catalog system
- ``m_instr`` - instrumental magnitude (-2.5 × log₁₀(ADU))
- ``ZP(x, y)`` - positionally-dependent zero point (spatial polynomial)
- ``C`` - color term
- ``Color`` - object color from catalog (g-r, B-V, or BP-RP depending on filter)

The model is fit using a robust fitter with iteratively rescaled errors. Close stellar pairs (< 2×FWHM separation) are excluded to avoid blended objects.

Color Term Diagnostics
~~~~~~~~~~~~~~~~~~~~~~

The software can test all supported filters as primary calibration filter and compute the corresponding color terms. The filter minimizing the color term best matches the instrumental passband.

Detection Limit
---------------

The detection limit is estimated by fitting a noise model to the S/N vs magnitude relationship:

- Combines constant background noise with Poissonian source noise
- Reports mean limit over the whole image
- Local limits may vary due to positional zero point dependence

Diagnostic plots show:

- Magnitude histograms (detected objects vs catalog stars)
- S/N vs magnitude with fitted noise model

Simple Transient Detection
--------------------------

For brighter transients not on complex backgrounds, catalog cross-matching is sufficient.

Filtering Steps
~~~~~~~~~~~~~~~

1. **Flag rejection** - exclude saturated, cosmic ray, and shape-outlier objects
2. **Positional filtering** - optional restriction to error box region
3. **Multi-image mode** - require detection in multiple uploaded images
4. **Catalog cross-match** - matching against Vizier catalogues chosen by sky position: Gaia eDR3 (all-sky), Pan-STARRS DR1 (δ > -30°), and SkyMapper DR4 plus DES DR2 (δ < 0°)
5. **Brightness comparison** - reject matches where object is not significantly brighter than catalog star (default: 2 mag)
6. **Solar System check** - optional SkyBoT query to reject known minor planets

Output
~~~~~~

Candidates are presented with cutouts showing:

- Original image
- Reference image (Pan-STARRS or SkyMapper)
- Detection footprint (segmentation map)
- Mask image

Results are downloadable in tabular format.

Image Subtraction
-----------------

Image subtraction detects transients by removing static sources via template subtraction.

Template Sources
~~~~~~~~~~~~~~~~

Templates can be obtained from:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Source
     - Notes
   * - Pan-STARRS DR2
     - Northern sky, automatic download and mosaicking
   * - DESI Legacy Surveys
     - Wide coverage, automatic download
   * - Dark Energy Survey
     - Southern sky, via HiPS2FITS
   * - SkyMapper
     - Southern sky, via HiPS2FITS
   * - ZTF
     - Northern sky, via HiPS2FITS
   * - Custom upload
     - User-provided template FITS file

HiPS2FITS sources lack mask information, limiting their reliability.

Subtraction Method
~~~~~~~~~~~~~~~~~~

Two subtraction methods are available, selectable per task:

**HOTPANTS** (Alard-Lupton image differencing):

- Automatic parameter selection based on FWHM and saturation level
- Noise model constructed from background RMS and gain
- Template always convolved to match image (preserves zero point)
- Image split into overlapping chunks to handle PSF variations
- Convolution kernel and background spatial orders adjustable via the
  ``hotpants_extra`` parameters (``ko``, ``bgo``, ...)

**SFFT** (Saccadic Fast Fourier Transform):

- Fourier-domain image differencing, provided by STDPipe (no extra binaries required)
- Spatially varying solution controlled by three polynomial orders:
  kernel order, background order, and flux (photometric) order
- A good alternative to HOTPANTS for difficult fields or large PSF variations

Common to both methods:

- The template is always matched to the science image so the difference image
  keeps the science-image zero point
- The image is split into overlapping sub-images to handle PSF variations; when
  a search position is specified, splitting is restricted to that region

Candidate Detection
~~~~~~~~~~~~~~~~~~~

After subtraction:

1. Noise-weighted detection in difference image (SExtractor)
2. Shape-based pre-filtering using classifier from original image
3. Aperture photometry at candidate positions
4. Optional catalog cross-matching and SkyBoT filtering

Artifact Rejection
~~~~~~~~~~~~~~~~~~

**Dipole artifacts** (from positional misalignment or proper motion) are detected via:

- Sub-pixel shift optimization between convolved template and image
- Minor flux scale adjustment (up to 30%)
- Rejection if χ² reduces by factor > 3 or final p-value > 0.01

Forced Photometry Mode
~~~~~~~~~~~~~~~~~~~~~~

If the transient position is known, image subtraction can skip detection and directly perform forced photometry at the specified position in the difference image.

Output
~~~~~~

Candidates are presented with six cutouts:

- Original image
- Template image
- Convolved template
- Difference image
- Detection footprint
- Mask image

Reviewing Candidates
--------------------

Both transient-detection paths (simple and image-subtraction) produce a sortable
candidate table, by default ordered by brightness. Each row links to its cutout
panels (the four-panel set for simple transients, the six-panel set for image
subtraction described above), so spurious detections - hot pixels, residual
dipoles, masked artifacts - can be told apart from genuine sources at a glance.

For a closer look, the cutout viewer offers a **blink** mode that flips between
the relevant frames (e.g. science and difference image), making real variability
stand out from static structure. Candidate lists are downloadable as VOTable
(``candidates.vot`` / ``candidates_simple.vot``) for further analysis, and as DS9
region files (``.reg``) for overlay on the original image. See
:doc:`output_files` for the full set of files produced.

.. _light-curves:

Light Curves
------------

Photometry stored in individual tasks can be combined into a light curve for a
given target across all images that cover its position.

Searching
~~~~~~~~~

The light curve view resolves a sky position from an object name (SIMBAD/TNS) or
coordinate string and gathers all matching tasks within a configurable search
radius. Results can be:

- Restricted to the current user or expanded to all accessible tasks
  (own tasks plus those shared via groups)
- Filtered by free-text criteria (filename, title, username, or group name)
- Limited to target (forced) photometry only, or include all detected objects

Output
~~~~~~

Matching measurements are presented as an interactive light curve with optional
image cutouts, and can be downloaded as a VOTable for further analysis.

.. _sky-map:

Sky Map
-------

The task list offers an interactive all-sky map showing the positions (and MOC
coverage) of all accessible tasks, making it easy to locate observations of a
given field. Task centre coordinates and footprints are stored on the task model
to support fast positional search.

Task Sharing
------------

By default a task is private to the user who created it. Tasks can be shared
with one or more **user groups**: every member of a listed group gains read
access to the task and its files (including via the light curve and sky-map
views). Sharing is managed per-task from the task page, and in bulk from the
task list. Group membership is administered through the Django admin interface.

External Dependencies
---------------------

STDWeb relies on several external tools:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Tool
     - Purpose
     - Reference
   * - SExtractor
     - Object detection
     - Bertin & Arnouts (1996)
   * - Astrometry.Net
     - Blind astrometric solving
     - Lang et al. (2010)
   * - SCAMP
     - Astrometric refinement
     - Bertin (2006)
   * - SWarp
     - Image reprojection
     - Bertin et al. (2002)
   * - HOTPANTS
     - Image subtraction (Alard-Lupton)
     - Becker (2015)
   * - AstroSCRAPPY
     - Cosmic ray detection
     - McCully et al. (2018)
