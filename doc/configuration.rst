Configuration
=============

STDWeb uses environment variables for site-specific configuration. All settings should be placed in a ``.env`` file in the project root.

Environment Variables
---------------------

Core Django Settings
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``SECRET_KEY``
     - Django secret key (generate with ``python -c "from django.core.management import utils; print(utils.get_random_secret_key())"``)
   * - ``DEBUG``
     - Enable Django debug mode (``True`` or ``False``, default: ``False``)

Path Settings
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``DATA_PATH``
     - Directory for user-accessible data files (file browser root)
   * - ``TASKS_PATH``
     - Directory for task working directories (uploads, results, caches)

STDPipe Binary Paths
~~~~~~~~~~~~~~~~~~~~

These are optional. If not specified, STDPipe will search standard paths.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``STDPIPE_TMPDIR``
     - Temporary folder for processing
   * - ``STDPIPE_SOLVE_FIELD``
     - Path to Astrometry.Net ``solve-field`` executable
   * - ``STDPIPE_SOLVE_FIELD_CONFIG``
     - Path to Astrometry.Net index configuration
   * - ``STDPIPE_SEXTRACTOR``
     - Path to SExtractor executable
   * - ``STDPIPE_SCAMP``
     - Path to SCAMP executable
   * - ``STDPIPE_PSFEX``
     - Path to PSFEx executable
   * - ``STDPIPE_HOTPANTS``
     - Path to HOTPANTS executable
   * - ``STDPIPE_SWARP``
     - Path to SWarp executable
   * - ``STDPIPE_PS1_CACHE``
     - Path for Pan-STARRS download cache (if not set, uses task-local cache)

External Services
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``SKYPORTAL_TOKEN``
     - SkyPortal API token for transient submission

Example Configuration
---------------------

.. code-block:: bash

   SECRET_KEY = 'your-django-secret-key-goes-here'

   DEBUG = True

   DATA_PATH = /opt/stdweb/data/
   TASKS_PATH = /opt/stdweb/tasks/

   # STDPipe settings
   STDPIPE_HOTPANTS = /usr/local/bin/hotpants
   STDPIPE_SOLVE_FIELD = /usr/local/astrometry/bin/solve-field
   STDPIPE_SOLVE_FIELD_CONFIG = /usr/local/astrometry/etc/astrometry.cfg

Supported Systems
-----------------

Photometric Filters
~~~~~~~~~~~~~~~~~~~

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
