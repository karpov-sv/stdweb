Processing Workflow
===================

This section describes the image processing workflow implemented in STDWeb. The processing pipeline consists of several sequential steps, each producing diagnostic outputs and intermediate results.

.. code-block:: text

   Image Upload/Import
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
          └── Image Subtraction

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
   * - Gaia DR3 Syntphot
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
4. **Catalog cross-match** - sequential matching with Gaia eDR3, Pan-STARRS DR1, SkyMapper DR4
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

STDWeb uses HOTPANTS (Alard-Lupton method) for image subtraction:

- Automatic parameter selection based on FWHM and saturation level
- Noise model constructed from background RMS and gain
- Template always convolved to match image (preserves zero point)
- Image split into overlapping chunks to handle PSF variations

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
     - Image subtraction
     - Becker (2015)
   * - AstroSCRAPPY
     - Cosmic ray detection
     - McCully et al. (2018)
