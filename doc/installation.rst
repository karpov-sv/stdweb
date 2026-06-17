Installation
============

Requirements
------------

The main requirement for STDWeb is `STDPipe <https://github.com/karpov-sv/stdpipe>`_, which provides the underlying processing routines. STDWeb also requires a running Redis instance for Celery task queue management.

In addition, several external astronomical binaries are used by the processing pipeline. They are technically optional, but most of the functionality depends on them (see :ref:`binary-dependencies` below).

.. _binary-dependencies:

Binary Dependencies
-------------------

The pipeline relies on the following external packages (the same ones used by STDPipe):

- `SExtractor <https://github.com/astromatic/sextractor>`__ - source detection
- `SCAMP <https://github.com/astromatic/scamp>`__ - astrometric refinement
- `PSFEx <https://github.com/astromatic/psfex>`__ - PSF modelling
- `SWarp <https://github.com/astromatic/swarp>`__ - image resampling
- `HOTPANTS <https://github.com/acbecker/hotpants>`__ - image subtraction
- `Astrometry.Net <https://github.com/dstndstn/astrometry.net>`__ - blind WCS solving

Most are available from system or conda package managers.

Ubuntu / Debian:

.. code-block:: bash

   sudo apt install sextractor scamp psfex swarp

Conda:

.. code-block:: bash

   conda install -c conda-forge astromatic-source-extractor astromatic-scamp astromatic-psfex astromatic-swarp

HOTPANTS
~~~~~~~~

HOTPANTS is not available from package managers and must be compiled manually from `its repository <https://github.com/acbecker/hotpants>`__.

.. note::

   If compilation fails at the linking stage with ``multiple definition of`` errors, this is a `known issue <https://github.com/acbecker/hotpants/issues/5>`__ with recent GCC defaults. Add the ``-fcommon`` switch to the ``COPTS`` line in the ``Makefile``. STDPipe ships an ``install_hotpants.sh`` helper script that downloads, patches, compiles, and installs the binary automatically.

On Ubuntu/Debian you may first need the build prerequisites:

.. code-block:: bash

   sudo apt install gcc make libcfitsio-dev

Astrometry.Net
~~~~~~~~~~~~~~

Astrometry.Net is only needed for *blind* WCS solving (images that arrive with no usable astrometry). It can be installed from package managers, but also requires downloading index files matched to your field of view, which can take significant disk space.

.. code-block:: bash

   sudo apt install astrometry.net

After installing, point STDWeb at the ``solve-field`` binary and its index configuration via the :doc:`configuration` settings ``STDPIPE_SOLVE_FIELD`` and ``STDPIPE_SOLVE_FIELD_CONFIG``.

Installing STDWeb
-----------------

Clone the repository:

.. code-block:: bash

   git clone https://github.com/karpov-sv/stdweb.git

Install Python dependencies (preferably in a dedicated environment):

.. code-block:: bash

   cd stdweb
   pip install -r requirements.txt

Installing Redis
----------------

Ubuntu/Debian:

.. code-block:: bash

   sudo apt install redis-server

Conda:

.. code-block:: bash

   conda install redis

Deployment
==========

Running STDWeb requires three components running simultaneously:

1. Redis Server
---------------

Start Redis (if not running system-wide):

.. code-block:: bash

   redis-server

2. Celery Worker
----------------

The data processing backend uses Celery:

.. code-block:: bash

   python -m celery -A stdweb worker --loglevel=info

For development, use the auto-reloading helper script (requires ``watchdog``):

.. code-block:: bash

   pip install watchdog
   ./run_celery.sh

3. Django Web Server
--------------------

Initialize the database and start the server:

.. code-block:: bash

   python manage.py migrate
   python manage.py runserver

Creating an Admin User
----------------------

To create a superuser for the admin panel:

.. code-block:: bash

   python manage.py createsuperuser

Production Deployment
---------------------

For production deployment, follow the `Django deployment documentation <https://docs.djangoproject.com/en/5.0/howto/deployment/>`_.
