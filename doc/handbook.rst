Analysis Handbook
=================

This section provides practical guidance for image analysis, including parameter selection, troubleshooting, and diagnostic interpretation.

Image Preparation
-----------------

Pre-processing Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

STDWeb expects science-ready images with instrument signatures removed:

- Bias/dark subtraction completed
- Flat-fielding applied
- Overscan regions removed or masked

Raw images may still work if you properly mask bad regions, but results will vary.

.. tip::
   Always use the custom mask editor to remove artifacts or heavily vignetted regions on the edges. This typically improves calibration quality.

Setting the Gain Correctly
~~~~~~~~~~~~~~~~~~~~~~~~~~

The gain (electrons per ADU) is critical for noise estimation, affecting detection, calibration accuracy, and template subtraction performance.

**Common issues:**

1. **Header value may be wrong** - Some cameras store gain presets rather than physical values
2. **Stacking changes effective gain** - If you average N images, multiply gain by N

   Example: Original gain 1.6, stacked 20 images with averaging → effective gain = 32

3. **Rescaling changes effective gain** - If pixel values are downscaled by factor F, multiply gain by F

**Quick check:** Compare the image pixel value range (shown in the log) with the estimated gain. If values are small with tiny RMS, the gain should be large. Incorrect gain causes the code to fail at star detection.

Choosing Reference Catalogs
---------------------------

Catalog selection is critical for accurate photometric calibration. Consider:

1. **Filter system match** - Catalog must include your observation band
2. **Depth match** - Catalog should not be much deeper than your image (causes spurious matches)
3. **Sky coverage** - Different catalogs cover different regions

Decision Tree
~~~~~~~~~~~~~

.. code-block:: text

   Is filter Sloan-like (g, r, i, z)?
   ├── Yes → Use Pan-STARRS
   └── No (Johnson-Cousins) →
       Is target fainter than ~18 mag?
       ├── Yes →
       │   Dec > -30°?
       │   ├── Yes → Use Pan-STARRS
       │   └── No → Use SkyMapper
       └── No → Use Gaia Synphot

   For unfiltered images → Use Gaia G band

**Catalog characteristics:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Catalog
     - Notes
   * - Pan-STARRS
     - Northern sky (Dec > -30°), deep (~23 mag), excellent for Sloan filters
   * - SkyMapper
     - Southern hemisphere, ~20 mag depth, only option for far south
   * - Gaia Synphot
     - All-sky, very accurate, but only to ~18 mag
   * - ATLAS-REFCAT2
     - All-sky merged catalog, less uniform than individual sources

Non-Standard Filters
~~~~~~~~~~~~~~~~~~~~

For non-standard filters (RGB, L, or custom):

1. Try different catalog bands
2. Use the one that minimizes the color term
3. STDWeb automatically searches if color term exceeds 0.5

Color Term Guidance
~~~~~~~~~~~~~~~~~~~

The color term describes the difference between your instrumental system and the catalog system.

**When to keep color term enabled:**

- Always start with it enabled to measure its value
- Keep enabled if the value is significant (> 0.1)

**When to disable color term:**

- If measured value is small (< 0.1) and your filter matches the standard
- Disabling assumes your target color equals the mean catalog star color

**Important:** If color term is significant, it must be reported and accounted for in final light curve construction.

Zero Point Spatial Order
~~~~~~~~~~~~~~~~~~~~~~~~

Use spatial polynomial for zero point when:

- Field of view is large
- Image has uncorrected vignetting
- PSF varies across the image (affects aperture correction)
- You have many (hundreds of) good stars

Use constant zero point (order=0) when:

- Narrow-field telescope
- No visible vignetting
- Stable PSF across image
- Only a handful of stars available

Diagnostic Plots
----------------

Photometry Calibration Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``photometry.png`` plot shows instrumental vs. catalog magnitude residuals.

**Good calibration:**

- Upper panel shows constant scatter around zero
- Orange points (flagged) are correctly excluded

**Common issues:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Symptom
     - Cause
   * - Orange points at bright end curving down
     - Saturation correctly detected and masked
   * - Faint end curving up
     - Detection bias (only positive noise fluctuations detected)
   * - Strong non-linearity across range
     - Detector non-linearity or pre-processing issue
   * - Large scatter
     - Poor calibration, check catalog choice and masking

FWHM Diagnostic Plot
~~~~~~~~~~~~~~~~~~~~

The ``fwhm_mag.png`` plot shows FWHM vs. magnitude for all detections.

**What to look for:**

- Stellar locus should be horizontal (constant FWHM)
- STDWeb estimates FWHM from unflagged objects with S/N > 20

**If FWHM is wrong:**

- Check the diagnostic plot
- If estimate deviates from the stellar locus, use the FWHM override field
- Common in crowded fields with many blended stars

Limiting Magnitude Plot
~~~~~~~~~~~~~~~~~~~~~~~

The ``limit_sn.png`` plot shows S/N vs. magnitude with fitted noise model.

**Good calibration:**

- Smooth curve fitting the data
- Reasonable scatter around the fit

**Problem indicators:**

- Highly scattered data indicates non-uniform background/noise
- Usually caused by unmasked artifacts or overscans
- Solution: Improve masking and reprocess

Template Subtraction Tips
-------------------------

HOTPANTS Parameters
~~~~~~~~~~~~~~~~~~~

The most important parameters to adjust if subtraction fails:

- **ko** - Spatial order of convolution kernel
- **bgo** - Background order inside model regions

These can be specified as a JSON string in the interface.

Template Source Selection
~~~~~~~~~~~~~~~~~~~~~~~~~

**Best quality:**

- Pan-STARRS DR2 (northern sky)
- Legacy Survey DR10 (wide coverage)

**HiPS sources (SkyMapper, DES, DECaPS, ZTF):**

- Accessed via HiPS2FITS service
- Lack proper masks for saturated regions
- May have other artifacts
- Results less reliable than native sources

Custom Template Tips
~~~~~~~~~~~~~~~~~~~~

When using your own template FITS file:

- Specify correct gain and saturation level
- Template should be deeper than the science image
- Ensure good astrometric alignment

Troubleshooting
---------------

Blind Matching Fails
~~~~~~~~~~~~~~~~~~~~

If Astrometry.Net blind solving fails:

1. Adjust parameters: sky region constraints, pixel scale range
2. Use the original `Astrometry.Net web service <https://nova.astrometry.net/upload>`_ directly
3. Upload the solved image to STDWeb

The original web service is more robust and handles problematic images better.

Cosmic Ray Detection Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The AstroSCRAPPY (LACosmic) algorithm may incorrectly flag stellar cores in:

- Undersampled images
- Images with very sharp PSF

Solution: Disable cosmic ray masking for such images.

Calibration Non-Linearity
~~~~~~~~~~~~~~~~~~~~~~~~~

If the photometry plot shows non-linearity:

- Check if the image is properly pre-processed
- Non-linearity at the faint end may indicate detection bias (normal)
- Non-linearity at the bright end should be masked by saturation flag
- If non-linearity spans the whole range, contact the observer about pre-processing

When working with non-linear regions:

- Increase reported uncertainties
- Note the limitation in your analysis

Log Error Messages
~~~~~~~~~~~~~~~~~~

Always check the bottom of processing logs for error messages (often highlighted in red).

Common errors:

- Blind matching failure → adjust parameters or use Astrometry.Net directly
- Insufficient stars → check masking, try different catalog
- WCS refinement failure → check initial WCS quality

If error messages are cryptic Python crashes rather than clear diagnostics, report the issue.
