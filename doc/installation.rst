Installation
============

Requirements
------------

The main requirement for STDWeb is `STDPipe <https://github.com/karpov-sv/stdpipe>`_. Please refer to its documentation for installation of binary dependencies like SExtractor, HOTPANTS, and Astrometry.Net.

STDWeb also requires a running Redis instance for Celery task queue management.

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
