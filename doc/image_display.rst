Interactive Image Viewer
========================

Images shown in STDWeb - the image panels on the task page, FITS previews in the
:doc:`file browser <file_browser>`, and similar displays - are interactive. A
small set of controls appears as an overlay in the corner of each image; changing
any of them re-renders the image **server-side** from the full-resolution data,
so zooming, smoothing, and re-stretching stay sharp rather than scaling a
thumbnail.

Controls
--------

Which controls are available depends on the image; a typical science image
offers all of them:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Control
     - Effect
   * - **Stretch**
     - Intensity transfer function: ``linear``, ``asinh``, ``log``, ``sqrt``,
       ``sinh``, ``power``, or ``histeq`` (histogram equalization)
   * - **Scale**
     - Upper percentile used for scaling (90% - 100%); lower values increase
       contrast on faint structure
   * - **Zoom**
     - Magnification from x1 to x32. While zoomed in, click near an edge of the
       image to pan in that direction; selecting x1 returns to the full frame
   * - **Smooth**
     - Gaussian smoothing radius (0 - 4 px), useful for revealing faint sources
   * - **Mark** (|bullseye|)
     - Toggle a marker at a position of interest (e.g. the target), optionally
       with aperture / annulus circles
   * - **Grid** (|grid|)
     - Overlay a coordinate (WCS) grid
   * - **Objects** (|objects|)
     - Overlay the detected objects
   * - **Catalogue** (|catalogue|)
     - Overlay the reference-catalog stars used for calibration

.. |bullseye| unicode:: U+25CE
.. |grid| unicode:: U+229E
.. |objects| unicode:: U+2606
.. |catalogue| unicode:: U+2605

Toggling **Objects** against **Catalogue** is a quick way to confirm that
detections line up with known stars and that the astrometric solution is sound;
combining **Zoom** and **Smooth** helps inspect faint candidates.

The same rendering options are also available programmatically through the
preview endpoints of the :doc:`REST API <api>` (stretch, scale/percentile limits,
smoothing radius, grid, object and catalog overlays, and target markers), so the
exact view you see can be reproduced or scripted.
