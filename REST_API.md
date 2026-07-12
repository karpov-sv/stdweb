# STDWeb REST API

This document describes the REST API for STDWeb, enabling programmatic access to astronomical image processing functionality.

## Base URL

```
/api/
```

## Authentication

The API uses token-based authentication. To obtain a token:

```bash
curl -X POST /api/auth/token/ \
  -d "username=<username>&password=<password>"
```

Response:
```json
{"token": "your-auth-token-here"}
```

Include the token in all subsequent requests:
```bash
curl -H "Authorization: Token <your-token>" /api/tasks/
```

Session authentication is also supported for browser-based access.

---

## Endpoints

### Tasks

#### List Tasks
```
GET /api/tasks/
```

Returns the list of tasks the current user can access: their own tasks plus any
tasks shared with a group they belong to (staff users see all tasks).

**Query Parameters (all optional):**
- `query` - Free-text search; every whitespace-separated token must match the
  original filename, title, owner username/first/last name, or a shared group
  name (case-insensitive substring match, same as the web UI task list)
- `state` - Filter by exact task state (e.g. `photometry_done`)
- `user` - Filter by exact owner username
- `limit` / `offset` - Paginate the results (see below)

**Response:**
```json
[
  {
    "id": 123,
    "original_name": "image.fits",
    "title": "My observation",
    "state": "photometry_done",
    "user": "username",
    "created": "2024-01-15T10:30:00Z",
    "modified": "2024-01-15T10:35:00Z"
  }
]
```

Without `limit` the full list is returned as a plain JSON array, as above. When
`limit` is given, the response is paginated:

```
GET /api/tasks/?query=image&limit=10&offset=20
```

```json
{
  "count": 123,
  "next": "http://.../api/tasks/?limit=10&offset=30&query=image",
  "previous": "http://.../api/tasks/?limit=10&offset=10&query=image",
  "results": [ ... ]
}
```

#### Create Task
```
POST /api/tasks/
```

Create a new task with optional file upload. The `file` may be omitted to create
a file-less task (e.g. for [stacking](#stacking-tasks), where `image.fits` is
produced from `config['stack_filenames']` by the `stack` step).

**Request (multipart/form-data):**
- `file` - FITS file to upload (optional)
- `original_name` - Filename (optional, derived from file if not provided)
- `title` - Optional title/comment
- `config` - JSON configuration object
- `preset` - Preset ID to apply (optional)
- `groups` - List of group names to share the task with (optional). Subject to
  the same restriction as [Update Task](#update-task): you may only share with
  groups you belong to (staff may use any group).

**Example:**
```bash
curl -X POST /api/tasks/ \
  -H "Authorization: Token <token>" \
  -F "file=@image.fits" \
  -F 'config={"filter":"R","cat_name":"ps1"}'
```

To set initial sharing, send a JSON body instead so `groups` is a proper list:

```bash
curl -X POST /api/tasks/ \
  -H "Authorization: Token <token>" \
  -H "Content-Type: application/json" \
  -d '{"original_name":"image.fits","config":{"filter":"R"},"groups":["my-team"]}'
```

#### Get Task
```
GET /api/tasks/{id}/
```

**Response:**
```json
{
  "id": 123,
  "original_name": "image.fits",
  "title": "My observation",
  "state": "photometry_done",
  "user": "username",
  "groups": ["my-team"],
  "created": "2024-01-15T10:30:00Z",
  "modified": "2024-01-15T10:35:00Z",
  "completed": "2024-01-15T10:35:00Z",
  "config": {
    "filter": "R",
    "cat_name": "ps1",
    "fwhm": 3.5,
    "zp": 25.2,
    ...
  },
  "celery_id": null,
  "path": "123"
}
```

The `groups` field lists the user groups this task is shared with, by group name.
Every member of a listed group can view the task and its files. See
[Update Task](#update-task) for how to change it.

The `path` field is the task directory, relative to the server's tasks base
directory.

#### Get Task State
```
GET /api/tasks/{id}/state/
```

Lightweight state endpoint for polling during processing — returns only the
state without the (potentially large) config.

**Response:**
```json
{
  "id": 123,
  "state": "photometry",
  "celery_id": "abc123-def456"
}
```

The task is running while `celery_id` is not null.

#### Update Task
```
PATCH /api/tasks/{id}/
```
or
```
POST /api/tasks/{id}/
```

Update task configuration.

**Request:**
```json
{
  "title": "Updated title",
  "config": {
    "filter": "V",
    "cat_name": "gaiaedr3"
  }
}
```

**Writable fields:** `original_name`, `title`, `config`, `groups`.

##### Sharing with Groups

Set the `groups` field to a list of group names to share the task with those
groups. Every member of a listed group gains view access to the task and its
files. The list replaces the current sharing set, so send the full desired list
(an empty list un-shares the task).

```json
{
  "groups": ["my-team", "transients-wg"]
}
```

You may only share with **groups you belong to**; staff users may share with any
group. Sharing with an unknown group, or with a group you are not a member of,
is rejected with a `400` error. Groups can also be set when creating a task —
see [Create Task](#create-task).

#### Delete Task
```
DELETE /api/tasks/{id}/
```

Deletes the task and all associated files. Only the task owner or staff may
delete a task — group members with edit access cannot.

#### Duplicate Task
```
POST /api/tasks/{id}/duplicate/
```

Creates a copy of the task with all files.

---

### Processing Actions

#### Run Processing
```
POST /api/tasks/{id}/process/
```

Queue processing steps for the task.

**Request:**
```json
{
  "steps": ["inspect", "photometry", "subtraction"],
  "config": {
    "filter": "R"
  }
}
```

**Available steps:**
- `stack` - Coadd several images into `image.fits` (see [Stacking Tasks](#stacking-tasks)); usually the first step, before `inspect`
- `cleanup` - Remove intermediate files
- `inspect` - Image inspection, WCS refinement, cosmic ray masking
- `photometry` - Source extraction and photometric calibration
- `simple_transients` - Simple transient detection via catalog cross-matching
- `subtraction` - Template subtraction and candidate detection

**Response:**
```json
{
  "id": 123,
  "state": "running",
  "celery_id": "abc123-def456",
  "steps": ["inspect", "photometry", "subtraction"]
}
```

Returns `409` if the task is already being processed. Poll
[Get Task State](#get-task-state) and wait for `celery_id` to become null, or
[cancel](#cancel-task) the run first.

#### Stacking Tasks

Stacking coadds several existing images into a single `image.fits`, which the
remaining steps then process. Unlike a normal task, a stacking task is created
**without** uploading a file — the inputs are listed in the config, and
`image.fits` is produced by the `stack` step.

**Requirements:**
- `stack_filenames` — a list of **absolute paths** to the input images. Every
  path must reside under the server's `DATA_PATH` (paths outside it are rejected
  for security). These are server-side paths, so the files must already exist in
  the data directory (e.g. previously uploaded or otherwise placed there).
- At least two input images.
- Include `stack` as the first entry in `steps` (before `inspect`).

The stacking parameters (`stack_method`, `stack_subtract_bg`,
`stack_mask_cosmics`) may be supplied at any point before the `stack` step runs —
at task creation, via [Update Task](#update-task), or in the `config` override of
the process request. See [Stacking](#stacking) for the full parameter list.

**Example — create a file-less task and stack:**
```bash
# 1. Create a task with no file, providing the input list in config
TASK_ID=$(curl -s -X POST /api/tasks/ \
  -H "Authorization: Token <token>" \
  -H "Content-Type: application/json" \
  -d '{
        "title": "My stack",
        "config": {
          "stack_filenames": [
            "/data/night1/img_001.fits",
            "/data/night1/img_002.fits",
            "/data/night1/img_003.fits"
          ],
          "stack_method": "clipped_mean"
        }
      }' | jq -r .id)

# 2. Run the stack step (optionally chained with further processing)
curl -X POST /api/tasks/$TASK_ID/process/ \
  -H "Authorization: Token <token>" \
  -H "Content-Type: application/json" \
  -d '{"steps": ["stack", "inspect", "photometry"]}'
```

The input list may equally be provided or overridden in the process request
itself:
```bash
curl -X POST /api/tasks/$TASK_ID/process/ \
  -H "Authorization: Token <token>" \
  -H "Content-Type: application/json" \
  -d '{
        "steps": ["stack"],
        "config": {"stack_filenames": ["/data/a.fits", "/data/b.fits"]}
      }'
```

#### Cancel Task
```
POST /api/tasks/{id}/cancel/
```

Cancel a running task and terminate all associated processes.

**Response:**
```json
{
  "id": 123,
  "state": "cancelled",
  "revoked_count": 3
}
```

#### Fix FITS Header
```
POST /api/tasks/{id}/fix/
```

Fix common FITS header issues.

This and the other image-modifying actions below return `409` if the task is
currently being processed.

#### Crop Image
```
POST /api/tasks/{id}/crop/
```

**Request:**
```json
{
  "x1": 100,
  "y1": 100,
  "x2": 900,
  "y2": 900
}
```

Negative values are interpreted as offsets from the edge.

#### Remove Stripes
```
POST /api/tasks/{id}/destripe/
```

**Request:**
```json
{
  "direction": "horizontal"
}
```

Options: `horizontal`, `vertical`, `both`

---

### Task Files

#### List Files
```
GET /api/tasks/{id}/files/
```

**Response:**
```json
[
  {
    "name": "image.fits",
    "path": "image.fits",
    "size": 16777216,
    "modified": "2024-01-15T10:30:00Z",
    "is_dir": false
  },
  {
    "name": "objects.vot",
    "path": "objects.vot",
    "size": 45678,
    "modified": "2024-01-15T10:35:00Z",
    "is_dir": false
  }
]
```

#### Download File
```
GET /api/tasks/{id}/files/{path}
```

Returns the file as an attachment.

#### Upload File
```
POST /api/tasks/{id}/files/{path}
```

Upload a file to the task directory.

**Request (multipart/form-data):**
- `file` - The file to upload

**Example:**
```bash
curl -X POST /api/tasks/123/files/custom_mask.fits \
  -H "Authorization: Token <token>" \
  -F "file=@mask.fits"
```

The `path` is relative to the task directory. Some filenames have special
meaning to the processing pipeline, so uploading them is the API way of
providing custom inputs:

- `image.fits` - The image to be processed (replaces the original upload)
- `custom_mask.fits` - Custom mask; pixels with non-zero values are excluded
  from processing (applied on top of the automatically generated mask)
- `custom_template.fits` - Custom template image for subtraction; used when
  `config['template']` is set to `custom`. Template gain and saturation level
  may be provided via the `custom_template_gain` and
  `custom_template_saturation` config options
- `custom_template_mask.fits` - Optional mask for the custom template; pixels
  with non-zero values are excluded

#### Delete File
```
DELETE /api/tasks/{id}/files/{path}
```

Delete a file from the task directory.

---

### Task Previews

#### Image Preview
```
GET /api/tasks/{id}/preview/{path}
```

Generate JPEG/PNG preview of a FITS file.

**Query Parameters:**
- `width` - Image width in pixels (default: 800)
- `format` - Output format: `jpeg` or `png` (default: jpeg)
- `quality` - JPEG quality 1-100 (default: 80)
- `ext` - FITS extension (default: -1, last)
- `cmap` - Colormap (default: Blues_r)
- `stretch` - Stretch function: `linear`, `log`, `sqrt`, `asinh` (default: linear)
- `qmin` - Lower percentile for scaling (default: 0.5)
- `qmax` - Upper percentile for scaling (default: 99.5)
- `r0` - Smoothing radius in pixels (default: 0)

**Example:**
```bash
curl "/api/tasks/123/preview/image.fits?width=1024&stretch=log" \
  -H "Authorization: Token <token>" > preview.jpg
```

#### Cutout Preview
```
GET /api/tasks/{id}/cutout/{path}
```

Generate multi-panel visualization of STDPipe cutout files.

**Query Parameters:**
- `width` - Image width in pixels (default: 800)
- `format` - Output format: `jpeg` or `png` (default: jpeg)
- `quality` - JPEG quality 1-100 (default: 80)

---

### Queue Management

#### List Queue
```
GET /api/queue/
```

List all active/pending/scheduled Celery tasks.

**Response:**
```json
[
  {
    "id": "abc123-def456",
    "name": "task_photometry",
    "full_name": "stdweb.celery_tasks.task_photometry",
    "state": "active",
    "time_start": 1705312200.0,
    "task_id": 123,
    "task_name": "image.fits",
    "chain_position": 2,
    "chain_total": 4
  }
]
```

#### Get Queue Task Status
```
GET /api/queue/{celery_id}/
```

**Response:**
```json
{
  "id": "abc123-def456",
  "state": "STARTED",
  "ready": false,
  "successful": null,
  "failed": null,
  "task_id": 123,
  "task_name": "image.fits",
  "chain_position": 2,
  "chain_total": 4
}
```

#### Terminate Queue Task
```
POST /api/queue/{celery_id}/terminate/
```

Allowed for staff, or for anyone able to edit the linked task (owner or member of
a group the task is shared with). Celery tasks not linked to any task may only be
terminated by staff.

**Response:**
```json
{
  "id": "abc123-def456",
  "task_id": 123,
  "revoked_count": 3,
  "state": "cancelled"
}
```

---

### Data Files

Access files in the shared data directory.

#### List/Download Data Files
```
GET /api/files/
GET /api/files/{path}
```

If path is a directory, returns file listing. If path is a file, downloads it.

#### Preview Data File
```
GET /api/files/{path}/preview/
```

Generate preview of a FITS file in the data directory. Same query parameters as task preview.

---

### Presets

#### List Presets
```
GET /api/presets/
```

**Response:**
```json
[
  {
    "id": 1,
    "name": "Default Photometry",
    "config": {
      "filter": "R",
      "cat_name": "ps1",
      "sn": 5
    }
  }
]
```

#### Get Preset
```
GET /api/presets/{id}/
```

When a preset is passed to [Create Task](#create-task), its config is used as
the base for the task config (explicitly provided config values win), and any
files associated with the preset are copied into the task directory.

---

### Reference Data

No authentication required.

#### List All Reference Data
```
GET /api/reference/
```

**Response:**
```json
{
  "filters": ["U", "B", "V", "R", "I", "u", "g", "r", "i", "z", "y", "G", "BP", "RP"],
  "catalogs": ["gaiadr3syn", "ps1", "skymapper", "sdss", "atlas", "gaiaedr3"],
  "templates": ["custom", "ps1", "ls", "skymapper", "des", "decaps", "ztf"]
}
```

#### Get Filters
```
GET /api/reference/filters/
```

**Response:**
```json
{
  "U": {"name": "Johnson-Cousins U", "aliases": []},
  "B": {"name": "Johnson-Cousins B", "aliases": []},
  "V": {"name": "Johnson-Cousins V", "aliases": []},
  ...
}
```

#### Get Catalogs
```
GET /api/reference/catalogs/
```

#### Get Templates
```
GET /api/reference/templates/
```

---

## Configuration Options

The task `config` object accepts these parameters:

### Stacking
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stack_filenames` | list | - | Absolute paths of input images to coadd (must be under `DATA_PATH`, at least two) |
| `stack_method` | string | "sum" | Coaddition method: `sum`, `clipped_mean`, or `median` |
| `stack_subtract_bg` | bool | true | Subtract background from each input before stacking |
| `stack_mask_cosmics` | bool | false | Mask cosmic rays in each input before stacking |

### Image Inspection
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target` | string | - | Target name or coordinates |
| `time` | string | - | Observation time (ISO format) |
| `gain` | float | - | Detector gain (e/ADU) |
| `saturation` | float | - | Saturation level (ADU) |
| `mask_cosmics` | bool | true | Enable cosmic ray masking |

### Photometry
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filter` | string | "R" | Photometric filter |
| `cat_name` | string | - | Reference catalog name |
| `cat_limit` | float | - | Catalog limiting magnitude |
| `sn` | float | 5 | Detection S/N threshold |
| `fwhm_override` | float | - | Force FWHM value (pixels) |
| `spatial_order` | int | 2 | Polynomial order for zero point |
| `use_color` | bool | true | Use color calibration term |
| `refine_wcs` | bool | true | Refine WCS with catalog |
| `blind_match_wcs` | bool | false | Perform blind WCS matching |

### Transient Detection
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `simple_skybot` | bool | false | Check SkyBoT asteroid catalog |
| `simple_blends` | bool | true | Reject blended objects |
| `simple_center` | string | - | Center position for search |
| `simple_sr0` | float | - | Search radius (degrees) |

### Template Subtraction
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `template` | string | - | Template source name (`custom` uses an uploaded `custom_template.fits`, see [Upload File](#upload-file)) |
| `custom_template_gain` | float | 10000 | Gain of the custom template |
| `custom_template_saturation` | float | - | Saturation level of the custom template |
| `subtraction_method` | string | "hotpants" | Method: "hotpants" or "sfft" |
| `subtraction_mode` | string | "detection" | Mode: "target" or "detection" |
| `hotpants_extra` | object | {} | Extra HOTPANTS parameters (when method is "hotpants") |
| `sfft_kernel_poly_order` | int | 0 | SFFT kernel spatial order (when method is "sfft") |
| `sfft_bg_poly_order` | int | 0 | SFFT background spatial order (when method is "sfft") |
| `sfft_flux_poly_order` | int | 0 | SFFT flux spatial order (when method is "sfft") |

---

## Error Responses

All errors return JSON with an `error` or `detail` field:

```json
{
  "detail": "Authentication credentials were not provided."
}
```

**HTTP Status Codes:**
- `400` - Bad request (invalid parameters)
- `401` - Authentication required
- `403` - Permission denied
- `404` - Resource not found
- `409` - Conflict (task is already being processed)
- `500` - Server error

---

## Example Workflow

```bash
# 1. Get authentication token
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/token/ \
  -d "username=myuser&password=mypass" | jq -r .token)

# 2. Upload a FITS file and create task
TASK=$(curl -s -X POST http://localhost:8000/api/tasks/ \
  -H "Authorization: Token $TOKEN" \
  -F "file=@observation.fits" \
  -F 'config={"filter":"R","cat_name":"ps1"}')

TASK_ID=$(echo $TASK | jq -r .id)
echo "Created task $TASK_ID"

# 3. Run processing
curl -X POST "http://localhost:8000/api/tasks/$TASK_ID/process/" \
  -H "Authorization: Token $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"steps":["inspect","photometry"]}'

# 4. Poll for completion
while true; do
  CELERY_ID=$(curl -s "http://localhost:8000/api/tasks/$TASK_ID/state/" \
    -H "Authorization: Token $TOKEN" | jq -r .celery_id)
  if [[ "$CELERY_ID" == "null" ]]; then break; fi
  sleep 5
done

# 5. List result files
curl -s "http://localhost:8000/api/tasks/$TASK_ID/files/" \
  -H "Authorization: Token $TOKEN" | jq .

# 6. Download detected objects catalog
curl "http://localhost:8000/api/tasks/$TASK_ID/files/objects.vot" \
  -H "Authorization: Token $TOKEN" -o objects.vot

# 7. Get image preview
curl "http://localhost:8000/api/tasks/$TASK_ID/preview/image.fits?width=1024" \
  -H "Authorization: Token $TOKEN" -o preview.jpg
```

---

## Python Example

```python
import requests
import time

BASE_URL = "http://localhost:8000/api"

# 1. Authenticate and get token
response = requests.post(f"{BASE_URL}/auth/token/", data={
    "username": "myuser",
    "password": "mypass"
})
token = response.json()["token"]

# Create session with auth header
session = requests.Session()
session.headers.update({"Authorization": f"Token {token}"})

# 2. Upload FITS file and create task
with open("observation.fits", "rb") as f:
    response = session.post(f"{BASE_URL}/tasks/", files={
        "file": ("observation.fits", f, "application/fits")
    }, data={
        "config": '{"filter": "R", "cat_name": "ps1"}'
    })
task = response.json()
task_id = task["id"]
print(f"Created task {task_id}")

# 3. Run processing
response = session.post(f"{BASE_URL}/tasks/{task_id}/process/", json={
    "steps": ["inspect", "photometry"]
})
print(f"Processing started: {response.json()}")

# 4. Poll for completion
while True:
    response = session.get(f"{BASE_URL}/tasks/{task_id}/state/")
    status = response.json()
    print(f"State: {status['state']}")
    if status["celery_id"] is None:
        break
    time.sleep(5)

# 5. List result files
response = session.get(f"{BASE_URL}/tasks/{task_id}/files/")
files = response.json()
print("Files:", [f["name"] for f in files])

# 6. Download objects catalog
response = session.get(f"{BASE_URL}/tasks/{task_id}/files/objects.vot")
with open("objects.vot", "wb") as f:
    f.write(response.content)

# 7. Download image preview
response = session.get(f"{BASE_URL}/tasks/{task_id}/preview/image.fits", params={
    "width": 1024,
    "stretch": "log"
})
with open("preview.jpg", "wb") as f:
    f.write(response.content)

# 8. Get task config and results
response = session.get(f"{BASE_URL}/tasks/{task_id}/")
task_data = response.json()
print(f"Zero point: {task_data['config'].get('zp')}")
print(f"FWHM: {task_data['config'].get('fwhm')}")
```

### Helper Class

For convenience, you can use this wrapper class:

```python
import requests
import time


class STDWebClient:
    """Simple client for STDWeb REST API."""

    def __init__(self, base_url, username=None, password=None, token=None):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

        if token:
            self.session.headers["Authorization"] = f"Token {token}"
        elif username and password:
            self.authenticate(username, password)

    def authenticate(self, username, password):
        """Get auth token and set up session."""
        response = requests.post(f"{self.base_url}/auth/token/", data={
            "username": username,
            "password": password
        })
        response.raise_for_status()
        token = response.json()["token"]
        self.session.headers["Authorization"] = f"Token {token}"
        return token

    def list_tasks(self):
        """List all tasks."""
        response = self.session.get(f"{self.base_url}/tasks/")
        response.raise_for_status()
        return response.json()

    def get_task(self, task_id):
        """Get task details."""
        response = self.session.get(f"{self.base_url}/tasks/{task_id}/")
        response.raise_for_status()
        return response.json()

    def create_task(self, filepath, config=None, title=None):
        """Create task with FITS file upload."""
        with open(filepath, "rb") as f:
            data = {}
            if config:
                data["config"] = config if isinstance(config, str) else str(config)
            if title:
                data["title"] = title

            response = self.session.post(f"{self.base_url}/tasks/", files={
                "file": (filepath.split("/")[-1], f, "application/fits")
            }, data=data)
        response.raise_for_status()
        return response.json()

    def process(self, task_id, steps, config=None):
        """Run processing steps."""
        payload = {"steps": steps}
        if config:
            payload["config"] = config
        response = self.session.post(
            f"{self.base_url}/tasks/{task_id}/process/",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def wait_for_completion(self, task_id, poll_interval=5, timeout=600):
        """Wait for task processing to complete."""
        start = time.time()
        while time.time() - start < timeout:
            response = self.session.get(f"{self.base_url}/tasks/{task_id}/state/")
            response.raise_for_status()
            status = response.json()
            if status["celery_id"] is None:
                return self.get_task(task_id)
            time.sleep(poll_interval)
        raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")

    def list_files(self, task_id):
        """List files in task directory."""
        response = self.session.get(f"{self.base_url}/tasks/{task_id}/files/")
        response.raise_for_status()
        return response.json()

    def download_file(self, task_id, filename, local_path=None):
        """Download file from task."""
        response = self.session.get(
            f"{self.base_url}/tasks/{task_id}/files/{filename}"
        )
        response.raise_for_status()
        local_path = local_path or filename
        with open(local_path, "wb") as f:
            f.write(response.content)
        return local_path

    def get_preview(self, task_id, filename="image.fits", **params):
        """Get image preview as bytes."""
        response = self.session.get(
            f"{self.base_url}/tasks/{task_id}/preview/{filename}",
            params=params
        )
        response.raise_for_status()
        return response.content


# Usage example
if __name__ == "__main__":
    client = STDWebClient(
        "http://localhost:8000/api",
        username="myuser",
        password="mypass"
    )

    # Create and process task
    task = client.create_task("observation.fits", config={"filter": "R"})
    client.process(task["id"], ["inspect", "photometry"])

    # Wait and get results
    result = client.wait_for_completion(task["id"])
    print(f"Completed with state: {result['state']}")
    print(f"Zero point: {result['config'].get('zp')}")

    # Download results
    client.download_file(task["id"], "objects.vot")
```

---

## Setup

### Create API Token

```bash
# Via management command
python manage.py drf_create_token <username>

# Or via Django shell
python manage.py shell
>>> from rest_framework.authtoken.models import Token
>>> from django.contrib.auth.models import User
>>> user = User.objects.get(username='myuser')
>>> token, created = Token.objects.get_or_create(user=user)
>>> print(token.key)
```
