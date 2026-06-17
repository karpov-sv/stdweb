Command-Line Processing
=======================

In addition to the web interface and the :doc:`REST API <api>`, STDWeb ships a
standalone command-line tool, ``process.py``, for batch processing of FITS files
without the database or task machinery. It runs the same processing functions as
the web pipeline, reading and writing configuration alongside each image.

Basic Usage
-----------

.. code-block:: bash

   python process.py [options] filename ...

One or more FITS files can be given; each is processed independently. The
processing steps are selected with explicit flags:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Action
   * - ``--inspect``
     - Header analysis, WCS sanitization, masking
   * - ``--photometry``
     - Source detection and photometric calibration
   * - ``--simple-transients``
     - Catalog-based transient detection
   * - ``--subtract``
     - Template subtraction and candidate detection
   * - ``--cleanup``
     - Remove all generated files (keeps only ``image.fits``) and reset config
   * - ``-v``, ``--verbose``
     - Print progress and diagnostic messages

Configuration
-------------

Each image directory keeps its configuration in a ``.config`` JSON file, which is
read before processing and written back afterwards. This means results and
settings accumulate across invocations, exactly as a task does in the web UI.

Individual parameters are supplied as ``KEY=VALUE`` pairs with ``-c`` (repeatable).
Values are parsed as booleans, integers, or floats where possible, otherwise kept
as strings. See :doc:`task_config` for the full parameter list.

.. code-block:: bash

   # Inspect and calibrate, specifying filter and catalog
   python process.py --inspect --photometry \
       -c filter=R -c cat_name=ps1 -c sn=5 image.fits

A reusable :ref:`preset <presets>` can be loaded from a JSON file with
``--preset``; command-line ``-c`` values are applied on top of it:

.. code-block:: bash

   python process.py --preset my_setup.json --inspect --photometry image.fits

Examples
--------

.. code-block:: bash

   # Full pipeline including subtraction, verbose
   python process.py -v --inspect --photometry --subtract \
       -c filter=r -c cat_name=ps1 -c template=ps1 image.fits

   # Re-run photometry with a different catalog (config is preserved between runs)
   python process.py --photometry -c cat_name=skymapper image.fits

   # Batch over many files with a shared preset
   python process.py --preset survey.json --inspect --photometry *.fits

   # Start over
   python process.py --cleanup image.fits

Output files are written next to each image, the same set as in a web task; see
:doc:`output_files`.
