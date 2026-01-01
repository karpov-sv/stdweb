STDWeb Documentation
====================

**STDWeb** (Simple Transient Detection for the Web) is a web-based tool for quick-look photometry and transient detection in astronomical images. It is based on the `STDPipe <https://github.com/karpov-sv/stdpipe>`_ library and implements a self-consistent, mostly automatic data analysis workflow.

Features
--------

- Upload your own FITS files or analyze files already on the server
- Basic pre-processing and masking
- Object detection with astrometric calibration (blind matching or WCS refinement)
- Photometric calibration using multiple reference catalogs
- Template subtraction with automatically downloaded or user-provided templates
- Forced photometry for specified targets in original or difference images
- Transient detection in difference images

For details on the underlying routines, see the `STDPipe documentation <https://stdpipe.readthedocs.io/>`_ and `example notebooks <https://github.com/karpov-sv/stdpipe/tree/master/notebooks>`_.

Supported Photometric Filters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Johnson-Cousins**: U, B, V, R, I
- **Sloan/Pan-STARRS**: u, g, r, i, z, y
- **Gaia**: G, BP, RP

Reference Catalogs
~~~~~~~~~~~~~~~~~~

- Gaia DR3 synthetic photometry
- Pan-STARRS DR1
- SkyMapper DR4
- SDSS DR16
- ATLAS-REFCAT2
- DES DR2

Template Sources
~~~~~~~~~~~~~~~~

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
   api
   workflow
   handbook

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
