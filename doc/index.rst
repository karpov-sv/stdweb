STDWeb Documentation
====================

**STDWeb** (Simple Transient Detection for the Web) is a web-based tool for quick-look photometry and transient detection in astronomical images. It is based on the `STDPipe <https://github.com/karpov-sv/stdpipe>`_ library and implements a self-consistent, mostly automatic data analysis workflow.

Features
--------

- Upload your own FITS files or analyze files already on the server
- Interactive pre-processing (destriping, fringe removal, cropping, background removal) and masking
- Object detection with astrometric calibration (blind matching or WCS refinement)
- Photometric calibration using multiple reference catalogs
- Template subtraction (HOTPANTS or SFFT) with automatically downloaded or user-provided templates
- Forced photometry for specified targets in original or difference images
- Transient detection via catalogue cross-matching or image subtraction
- Multi-task light curve assembly across images covering the same position
- Sky-map overview of all tasks and group-based task sharing

For details on the underlying routines, see the `STDPipe documentation <https://stdpipe.readthedocs.io/>`_ and `example notebooks <https://github.com/karpov-sv/stdpipe/tree/master/notebooks>`_.

Supported Photometric Filters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Johnson-Cousins**: U, B, V, R, I
- **Sloan/Pan-STARRS**: u, g, r, i, z, y
- **Gaia**: G, BP, RP

Reference Catalogs
~~~~~~~~~~~~~~~~~~~

- Gaia DR3 synthetic photometry
- Gaia eDR3
- Pan-STARRS DR1
- SkyMapper DR4
- SDSS DR16
- ATLAS-REFCAT2
- DES DR2 (transient cross-matching)

Template Sources
~~~~~~~~~~~~~~~~~

- Pan-STARRS DR2
- Legacy Survey DR10
- SkyMapper DR4
- DES DR2
- DECaPS DR2
- ZTF DR7


Citation
--------

If you use STDWeb in your work, please cite:

   Karpov, S. (2025). **STDweb: simple transient detection pipeline for the web**.
   *Acta Polytechnica*, 65(1), 50-64.
   https://doi.org/10.14311/AP.2025.65.0050

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   configuration
   architecture
   workflow
   handbook
   REST API <api>
   task_config

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
