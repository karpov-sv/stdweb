Output Files
============

Each processing step writes its results into the task directory (``tasks/{id}/``
for web tasks, or alongside the image for the :doc:`command-line tool <cli>`).
This page describes the files produced at every stage. Re-running a step first
removes the files of that step and all later ones, so the directory always
reflects a single consistent run.

The authoritative lists live in ``stdweb/processing/constants.py`` (the
``files_*`` definitions).

Image and Inspection
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Contents
   * - ``image.fits``
     - The working image (possibly preprocessed/stacked)
   * - ``image.orig.fits``
     - Backup of the image before preprocessing (used by **Reset**)
   * - ``mask.fits``
     - Combined pixel mask (saturation, cosmic rays, custom regions)
   * - ``custom_mask.fits``
     - User-drawn mask regions, if any
   * - ``image_target.fits``
     - Cropped image around the target(s), for display
   * - ``inspect.log``
     - Log of the inspection step

Photometry
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Contents
   * - ``objects.parquet``
     - Detected objects with measured photometry
   * - ``cat.parquet``
     - Reference catalog stars used for calibration
   * - ``segmentation.fits``
     - SExtractor segmentation map
   * - ``image_bg.fits``, ``image_rms.fits``
     - Background and background-RMS maps (when *Inspect background* is enabled)
   * - ``objects.png``, ``fwhm.png``
     - Detected-object overlay and FWHM-vs-magnitude diagnostic
   * - ``photometry.png``, ``photometry_unmasked.png``
     - Calibration residual plots
   * - ``photometry_zeropoint.png``, ``photometry_model.png``, ``photometry_residuals.png``
     - Zero-point map, fitted model, and residuals
   * - ``astrometry_dist.png``
     - Astrometric residual distribution
   * - ``limit_hist.png``, ``limit_sn.png``
     - Magnitude histogram and S/N-vs-magnitude limit fit
   * - ``photometry.pickle``
     - Serialized calibration result
   * - ``target.vot``
     - Photometry of the requested target(s)
   * - ``target.cutout``, ``targets``
     - Target cutout(s) and per-target outputs
   * - ``photometry.log``
     - Log of the photometry step

Simple Transients
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Contents
   * - ``candidates_simple.vot``
     - Candidate table (VOTable)
   * - ``candidates_simple.reg``
     - DS9 region file for overlay
   * - ``candidates_simple``
     - Per-candidate cutouts
   * - ``transients_simple.log``
     - Log of the simple-transients step

Template Subtraction
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Contents
   * - ``sub_image.fits``, ``sub_mask.fits``
     - Science image and mask used for subtraction
   * - ``sub_template.fits``, ``sub_template_mask.fits``
     - Template image and mask
   * - ``sub_conv.fits``
     - Convolved (PSF-matched) template
   * - ``sub_diff.fits``
     - Difference image
   * - ``sub_sdiff.fits``, ``sub_ediff.fits``
     - Noise-scaled and error difference images
   * - ``sub_scorr.fits``
     - Correlation/significance image
   * - ``sub_fpsf.fits``, ``sub_fpsferr.fits``
     - Forced-PSF flux and error maps
   * - ``sub_target.vot``, ``sub_target.cutout``
     - Forced photometry of the target in the difference image
   * - ``candidates.vot``, ``candidates.reg``
     - Candidate table and DS9 region file
   * - ``candidates``
     - Per-candidate six-panel cutouts
   * - ``subtraction.log``
     - Log of the subtraction step

Tabular files (``.parquet`` for catalogs and object lists, ``.vot`` VOTables for
targets and candidates) can be opened directly in the file browser, and FITS
images have on-the-fly previews.
