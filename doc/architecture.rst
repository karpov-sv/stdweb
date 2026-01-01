Architecture
============

STDWeb is a Django-based web application with Celery for distributed task processing.

Overview
--------

.. code-block:: text

   Django Views → Task Model (DB) → Celery Queue (Redis) → Processing Pipeline → STDPipe Library

Key Components
--------------

Django Application (``stdweb/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Core modules:**

- ``views.py``, ``views_tasks.py``, ``views_celery.py`` - Request handlers
- ``forms.py`` - Multi-step form handling for task configuration
- ``models.py`` - Task and Preset database models
- ``celery_tasks.py`` - Async task definitions with ``TaskProcessContext``
- ``celery.py`` - Celery app configuration
- ``api/`` - REST API (see :doc:`api`)

Processing Package (``stdweb/processing/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Purpose
   * - ``constants.py``
     - Filter definitions (Johnson-Cousins, Sloan, Gaia), catalog specs, template sources
   * - ``utils.py``
     - File I/O, WCS handling, FITS header fixes, image preprocessing
   * - ``catalogs.py``
     - Vizier queries, blend filtering, magnitude column detection
   * - ``inspect.py``
     - Image inspection, cosmic ray removal, initial WCS refinement
   * - ``photometry.py``
     - Source extraction (SExtractor), photometric calibration, zero-point fitting
   * - ``transients.py``
     - Simple transient detection via catalog cross-matching
   * - ``subtraction.py``
     - Template acquisition, HOTPANTS image subtraction, candidate detection
   * - ``stacking.py``
     - Multi-image stacking and combination
   * - ``__init__.py``
     - Re-exports all public functions for backward compatibility

Processing Pipeline
-------------------

.. code-block:: text

   inspect_image()           → Loads FITS, fixes header, creates mask, refines WCS
          ↓
   photometry_image()        → Runs SExtractor, calibrates photometry, measures targets
          ↓
   transients_simple_image() → Cross-matches objects against Vizier catalogs (optional)
          ↓
   subtract_image()          → Downloads template, runs HOTPANTS, detects candidates (optional)

Task Workflow
-------------

1. User uploads FITS file and configures processing steps via web forms
2. Task object created in database with JSON config
3. Celery executes chained tasks: inspect → photometry → transients → subtraction
4. Results saved to ``tasks/{id}/`` directory

Celery Task Handling
--------------------

Chain Tracking
~~~~~~~~~~~~~~

Tasks are executed as Celery chains. The ``Task`` model stores:

- ``celery_id`` - ID of the chain's first task
- ``celery_chain_ids`` - List of all subtask IDs in the chain
- ``celery_pid`` - PID of the current Celery worker process (for killing external processes)

Process Group Management
~~~~~~~~~~~~~~~~~~~~~~~~

Task functions use ``TaskProcessContext`` context manager which:

- Checks for cancellation before starting (via ``celery_id`` being cleared)
- Creates a process group (``os.setpgrp()``) so child processes can be killed together
- Stores PID in ``task.celery_pid`` for external process tracking
- Registers SIGTERM handler for graceful cleanup
- Clears PID from database on exit

Adding New Task Types
~~~~~~~~~~~~~~~~~~~~~

Follow this pattern:

.. code-block:: python

   @shared_task(bind=True, acks_late=True, reject_on_worker_lost=True)
   def task_example(self, id, finalize=True):
       with TaskProcessContext(self, id) as ctx:
           if ctx.cancelled:
               return
           task = ctx.task
           basepath = ctx.basepath
           # ... task logic ...

Task Termination
~~~~~~~~~~~~~~~~

Use ``revoke_task_chain(task)`` from ``views_celery.py`` to:

1. Revoke all tasks in the chain (not just the current one)
2. Kill the process group (terminates external binaries like HOTPANTS, SExtractor)
3. Mark task as cancelled

External Dependencies
---------------------

- **STDPipe** - Python astronomy toolkit (core processing library)
- **Astrometry.Net** - Blind WCS solving
- **SExtractor** - Source extraction
- **SCAMP** - Astrometric calibration
- **PSFEx** - PSF modeling
- **SWarp** - Image resampling
- **HOTPANTS** - Image subtraction

Command-Line Interface
----------------------

STDWeb includes a standalone CLI tool for batch processing:

.. code-block:: bash

   # Basic processing
   python process.py --inspect --photometry filename.fits

   # Full pipeline with options
   python process.py --simple-transients --subtract -c key=value filename.fits

   # Using a preset configuration
   python process.py --preset config.json filename.fits
