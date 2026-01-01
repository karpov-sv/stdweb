REST API
========

STDWeb provides a REST API for programmatic access to image processing functionality.

For complete API documentation with examples, see the `REST API Reference <https://github.com/karpov-sv/stdweb/blob/master/REST_API.md>`_.

Authentication
--------------

The API uses token-based authentication. Obtain a token:

.. code-block:: bash

   curl -X POST /api/auth/token/ \
     -d "username=<username>&password=<password>"

Include the token in subsequent requests:

.. code-block:: bash

   curl -H "Authorization: Token <your-token>" /api/tasks/

Creating API Tokens
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Via management command
   python manage.py drf_create_token <username>

   # Or via Django shell
   python manage.py shell
   >>> from rest_framework.authtoken.models import Token
   >>> from django.contrib.auth.models import User
   >>> user = User.objects.get(username='myuser')
   >>> token, created = Token.objects.get_or_create(user=user)
   >>> print(token.key)

Main Endpoints
--------------

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Endpoint
     - Description
   * - ``GET /api/tasks/``
     - List tasks for current user
   * - ``POST /api/tasks/``
     - Create new task with file upload
   * - ``GET /api/tasks/{id}/``
     - Get task details
   * - ``PATCH /api/tasks/{id}/``
     - Update task configuration
   * - ``DELETE /api/tasks/{id}/``
     - Delete task and files
   * - ``POST /api/tasks/{id}/duplicate/``
     - Duplicate task

Processing
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Endpoint
     - Description
   * - ``POST /api/tasks/{id}/process/``
     - Start processing steps
   * - ``POST /api/tasks/{id}/cancel/``
     - Cancel running task

Available processing steps: ``cleanup``, ``inspect``, ``photometry``, ``simple_transients``, ``subtraction``

Files
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Endpoint
     - Description
   * - ``GET /api/tasks/{id}/files/``
     - List task files
   * - ``GET /api/tasks/{id}/files/{path}``
     - Download file
   * - ``POST /api/tasks/{id}/files/{path}``
     - Upload file
   * - ``DELETE /api/tasks/{id}/files/{path}``
     - Delete file
   * - ``GET /api/tasks/{id}/preview/{path}``
     - Generate FITS preview image

Queue
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Endpoint
     - Description
   * - ``GET /api/queue/``
     - List Celery queue status
   * - ``GET /api/queue/{celery_id}/``
     - Get task status
   * - ``POST /api/queue/{celery_id}/terminate/``
     - Terminate task (staff only)

Reference Data
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Endpoint
     - Description
   * - ``GET /api/reference/``
     - List all reference data
   * - ``GET /api/reference/filters/``
     - Available photometric filters
   * - ``GET /api/reference/catalogs/``
     - Available reference catalogs
   * - ``GET /api/reference/templates/``
     - Available template sources
   * - ``GET /api/presets/``
     - List configuration presets

Quick Example
-------------

.. code-block:: bash

   # Get token
   TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/token/ \
     -d "username=myuser&password=mypass" | jq -r .token)

   # Upload and create task
   TASK_ID=$(curl -s -X POST http://localhost:8000/api/tasks/ \
     -H "Authorization: Token $TOKEN" \
     -F "file=@image.fits" \
     -F 'config={"filter":"R"}' | jq -r .id)

   # Run processing
   curl -X POST "http://localhost:8000/api/tasks/$TASK_ID/process/" \
     -H "Authorization: Token $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"steps":["inspect","photometry"]}'

   # Check status
   curl "http://localhost:8000/api/tasks/$TASK_ID/" \
     -H "Authorization: Token $TOKEN"

Python Client
-------------

.. code-block:: python

   import requests

   BASE_URL = "http://localhost:8000/api"

   # Authenticate
   response = requests.post(f"{BASE_URL}/auth/token/",
       data={"username": "myuser", "password": "mypass"})
   token = response.json()["token"]

   session = requests.Session()
   session.headers["Authorization"] = f"Token {token}"

   # Upload file
   with open("image.fits", "rb") as f:
       response = session.post(f"{BASE_URL}/tasks/",
           files={"file": f},
           data={"config": '{"filter": "R"}'})
   task_id = response.json()["id"]

   # Process
   session.post(f"{BASE_URL}/tasks/{task_id}/process/",
       json={"steps": ["inspect", "photometry"]})

   # Get results
   task = session.get(f"{BASE_URL}/tasks/{task_id}/").json()
   print(f"Zero point: {task['config'].get('zp')}")
