Quickstart
==========

This walkthrough takes a single FITS image from upload to calibrated photometry
and (optionally) a transient search. It assumes STDWeb is already running (see
:doc:`installation` and :doc:`configuration`).

For the concepts behind each step, see the :doc:`workflow`; for parameter
details, see :doc:`task_config`; for practical advice, see the :doc:`handbook`.

1. Import an image
------------------

From the home page, **upload** a FITS file or **select** one from the server's
data directory via the file browser. A new *task* is created to hold the image,
its configuration, results, and logs. Optionally pick a :ref:`preset <presets>`
to pre-fill the configuration, or select several frames and **Stack and Process**
them into one task.

2. Inspect
----------

Run the **Inspect** step. STDWeb reads the FITS header (time, gain, saturation,
existing WCS), sanitizes the WCS, and builds a pixel mask (saturated pixels and
cosmic rays).

- Check that **gain** and **saturation** were detected correctly - a wrong gain
  is the most common cause of later failures (see :doc:`handbook`).
- Optionally add a **target** (coordinates, or a SIMBAD/TNS name) for forced
  photometry, and draw a **custom mask** over vignetting or artifacts.

3. Measure photometry
---------------------

Run the **Photometry** step. STDWeb detects sources with SExtractor, refines the
astrometry, queries a reference catalog, and fits the photometric zero point.

- Pick a **filter** and a **reference catalog** suited to your band and sky
  position (the :doc:`handbook` has a decision tree).
- Inspect the diagnostic plots (``photometry.png``, ``fwhm.png``, ``limit_sn.png``)
  to confirm a clean calibration. The fitted zero point and limiting magnitude
  are reported in the log and stored in the task config.
- Use the :doc:`interactive image controls <image_display>` to overlay detected
  objects and catalog stars on the image and verify they line up.

If you only need photometry of known targets, you are done - the measurements are
in ``target.vot``.

4. Find transients (optional)
-----------------------------

For transient detection, run either:

- **Simple transients** - cross-matches detected sources against reference
  catalogs and flags those with no counterpart. Best for bright transients on
  uncomplicated backgrounds.
- **Subtraction** - downloads (or accepts an uploaded) template, subtracts it
  with HOTPANTS or SFFT, and detects residual sources in the difference image.
  Best for crowded or structured fields.

Review the resulting candidate table and cutouts (see
:doc:`workflow` → *Reviewing Candidates*), and download the results as VOTable.

5. Next steps
-------------

- Combine measurements of the same target across many tasks into a
  :ref:`light curve <light-curves>`.
- Share a task with collaborators via :doc:`groups <workflow>`.
- Automate the same pipeline over many files with the
  :doc:`command-line tool <cli>` or the :doc:`REST API <api>`.
