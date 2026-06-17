File Browser
============

STDWeb includes a built-in file browser used both for the shared **data
directory** (configured via ``DATA_PATH``) and for the working directory of an
individual **task**. It lets you navigate folders and inspect most file types
that the pipeline reads or produces without leaving the browser.

Browsing
--------

- **Data directory** - reachable from the *Files* link; useful for picking
  images out of telescope archives mounted under ``DATA_PATH``.
- **Task directory** - the *Files* tab of a task lists everything in
  ``tasks/{id}/`` (see :doc:`output_files`). Access requires permission to view
  the task (see :doc:`accounts`).

Directory listings show a breadcrumb trail and per-entry type, size, and
modification time.

Viewing Files
-------------

The browser renders a file according to its type rather than just offering a
download:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Type
     - Rendering
   * - FITS images
     - On-the-fly preview with adjustable stretch, colormap, and scaling, plus a
       view of the FITS header. ``.wcs`` (header-only) files are recognized too.
   * - STDPipe cutouts (``.cutout``)
     - Multi-panel visualization (e.g. image / template / difference / mask)
   * - Tables (``.vot``, ``.parquet``)
     - Shown as a sortable table. For backward compatibility a requested ``.vot``
       transparently falls back to the matching ``.parquet`` file if present.
   * - Text
     - Displayed inline
   * - Other images (PNG, JPEG)
     - Displayed inline

Every file can also be **viewed** inline or **downloaded** as an attachment.
FITS previews and cutout renderings are generated server-side and accept the
same parameters as the corresponding :doc:`REST API <api>` preview endpoints
(width, colormap, stretch, percentile limits, smoothing). FITS previews carry
the :doc:`interactive image controls <image_display>` for stretching, zooming,
and overlays.

Importing from the data browser
-------------------------------

When viewing a FITS file in the data directory, the browser also offers to
import it directly as a new task (optionally applying a :ref:`preset <presets>`),
providing a quick path from an archive file to a processing task.
