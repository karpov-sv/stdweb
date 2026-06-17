Accounts and Access Control
===========================

STDWeb is a multi-user application. Access to tasks and their files is governed
by ownership, group sharing, and a small set of permissions.

User Accounts
-------------

Accounts are created by administrators - there is no open self-registration.
A superuser is created during setup (see :doc:`installation`):

.. code-block:: bash

   python manage.py createsuperuser

Further users (and group memberships) are managed through the Django admin
interface at ``/admin/``. Once an account exists, users can:

- **Log in / out** and **change their password** through the web interface
- Visit their **profile** page to view and manage their personal **API token**
  (including regenerating it)

Access Control
--------------

By default a task is private to the user who created it. Beyond the owner,
access is determined as follows:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Capability
     - Who
     - Notes
   * - View
     - Owner; staff; members of a group the task is shared with; holders of
       ``view_all_tasks`` or ``edit_all_tasks``
     - Includes the task's files via the :doc:`file browser <file_browser>`
   * - Edit / run
     - Owner; staff; shared-group members; holders of ``edit_all_tasks``
     - Submitting and re-processing
   * - Delete
     - Owner or staff only
     -

Group Sharing
~~~~~~~~~~~~~

Tasks can be shared with one or more **user groups**; every member of a listed
group gains view (and run) access. Sharing is managed per task from the task
page, in bulk from the task list, and through the :doc:`REST API <api>`; you may
only share with groups you belong to (staff may use any group). Group
membership itself is administered in the Django admin. See the *Task Sharing*
section of the :doc:`workflow` for the user-facing workflow.

Permissions
~~~~~~~~~~~

The application defines a few custom permissions (assignable to users or groups
in the admin):

- ``view_all_tasks`` - view every task regardless of ownership
- ``edit_all_tasks`` - view and modify every task
- ``skyportal_upload`` - upload task results to SkyPortal

Staff users implicitly have full access; task deletion remains restricted to the
owner and staff.

API Authentication
------------------

Programmatic access uses token authentication. Obtain a token from your profile
page, or via the management command:

.. code-block:: bash

   python manage.py drf_create_token <username>

See the :doc:`REST API <api>` for usage.

Audit Log
---------

User actions on tasks (creation, processing, sharing, deletion, etc.) are
recorded in an audit log. Staff can review recent entries through a dedicated
read-only view, and the full log is available in the Django admin.
