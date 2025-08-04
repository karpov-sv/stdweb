# STDWeb API Documentation

This document describes the REST API endpoints available for the STDWeb project.

## Authentication

The API uses token-based authentication. You need to obtain an API token first.

### Get API Token

**Endpoint:** `POST /api/auth/token/`

**Request:**
```bash
curl -X POST http://your-domain/api/auth/token/ \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'
```

**Response:**
```json
{
  "token": "your_api_token_here"
}
```

### Using the Token

Include the token in the Authorization header for all API requests:
```
Authorization: Token your_api_token_here
```

## REST API – Quick Usage

The remainder of this document goes into detail for every endpoint, but if you
just want the **TL;DR** for the most common operations via cURL here it is.

### Upload a FITS file and start inspection, photometry & transient detection

```bash
curl -X POST http://your-domain/api/tasks/upload/ \
  -H "Authorization: Token your_api_token_here" \
  -F "file=@path/to/your/image.fits" \
  -F "title=Full processing run" \
  -F "do_inspect=true" \
  -F "do_photometry=true" \
  -F "do_simple_transients=true"
```

The response will contain the `id` of the newly-created task.  Use that ID to
check progress or results later.

### Consult / check task status

```bash
curl -X GET http://your-domain/api/tasks/{task_id}/ \
  -H "Authorization: Token your_api_token_here"
```

Replace `{task_id}` with the numeric ID returned by the upload call.  The JSON
response includes the current `state` field plus any configuration and timing
information.

## Endpoints

### 1. Upload FITS Image

**Endpoint:** `POST /api/tasks/upload/`

**Description:** Upload a FITS file and create a new processing task.

**Request:**
```bash
curl -X POST http://your-domain/api/tasks/upload/ \
  -H "Authorization: Token your_api_token_here" \
  -F "file=@path/to/your/image.fits" \
  -F "title=My Image Analysis" \
  -F "target=NGC1234" \
  -F "preset=1" \
  -F "do_inspect=true" \
  -F "do_photometry=true"
```

**Example with photometry parameters (disabling color term):**
```bash
curl -X POST http://your-domain/api/tasks/upload/ \
  -H "Authorization: Token your_api_token_here" \
  -F "file=@path/to/your/image.fits" \
  -F "title=Photometry without color term" \
  -F "do_photometry=true" \
  -F "use_color=false" \
  -F "sn=5.0" \
  -F "spatial_order=2" \
  -F "filter=r" \
  -F "cat_name=gaiaedr3"
```

**Example with gain and saturation parameters:**
```bash
curl -X POST http://your-domain/api/tasks/upload/ \
  -H "Authorization: Token your_api_token_here" \
  -F "file=@path/to/your/image.fits" \
  -F "title=Analysis with custom gain" \
  -F "do_inspect=true" \
  -F "do_photometry=true" \
  -F "gain=2.5" \
  -F "saturation=60000" \
  -F "use_color=true"
```

**Parameters:**

**Basic Parameters:**
- `file` (required): FITS file to upload (.fits, .fit, .fts extensions)
- `title` (optional): Title or description for the task
- `target` (optional): Target name or coordinates
- `preset` (optional): Preset ID to apply configuration
- `do_inspect` (optional): Automatically run inspection (default: false)
- `do_photometry` (optional): Automatically run photometry (default: false)
- `do_simple_transients` (optional): Automatically run simple transient detection (default: false)
- `do_subtraction` (optional): Automatically run image subtraction (default: false)

**Inspection Parameters:**
- `gain` (optional): Gain in e-/ADU
- `saturation` (optional): Saturation level in ADU
- `time` (optional): Time parameter

**Photometry Configuration Parameters:**
- `sn` (optional): S/N Ratio for detection threshold
- `initial_aper` (optional): Initial aperture in pixels
- `initial_r0` (optional): Smoothing kernel in pixels
- `bg_size` (optional): Background mesh size
- `minarea` (optional): Minimal object area
- `rel_aper` (optional): Relative aperture in FWHM units
- `rel_bg1` (optional): Sky inner annulus in FWHM units
- `rel_bg2` (optional): Outer annulus in FWHM units
- `fwhm_override` (optional): FWHM override in pixels
- `filter` (optional): Filter name (e.g., 'r', 'g', 'i')
- `cat_name` (optional): Reference catalog name (e.g., 'gaiaedr3')
- `cat_limit` (optional): Catalog limiting magnitude
- `spatial_order` (optional): Zeropoint spatial order (0-3)
- `use_color` (optional): Use color term in photometric calibration (true/false)
- `sr_override` (optional): Matching radius in arcseconds
- `prefilter_detections` (optional): Pre-filter detections (true/false)
- `filter_blends` (optional): Filter catalogue blends (true/false)
- `diagnose_color` (optional): Run color term diagnostics (true/false)
- `refine_wcs` (optional): Refine astrometry (true/false)
- `blind_match_wcs` (optional): Enable blind astrometric matching (true/false)
- `inspect_bg` (optional): Inspect background (true/false)
- `centroid_targets` (optional): Centroid targets (true/false)
- `nonlin` (optional): Apply non-linearity correction (true/false)

**Blind Matching Parameters:**
- `blind_match_ps_lo` (optional): Scale lower limit in arcsec/pix
- `blind_match_ps_up` (optional): Scale upper limit in arcsec/pix
- `blind_match_center` (optional): Center position for blind match
- `blind_match_sr0` (optional): Search radius in degrees

**Response:**
```json
{
  "message": "File uploaded successfully",
  "task": {
    "id": 123,
    "original_name": "image.fits",
    "title": "My Image Analysis",
    "state": "uploaded",
    "user": "username",
    "created": "2024-01-15T10:30:00Z",
    "modified": "2024-01-15T10:30:00Z",
    "completed": "2024-01-15T10:30:00Z",
    "config": {
      "target": "NGC1234"
    }
  }
}
```

### Python example

Below is a minimal Python snippet that performs the same upload — requesting
inspection, photometry, and simple transient detection in one go — using the
popular `requests` library.  Replace the placeholders with your own values.

```python
import requests

API_TOKEN = "your_api_token_here"  # <-- put your token here
API_URL = "http://your-domain/api/tasks/upload/"
FITS_PATH = "path/to/your/image.fits"

# Form-data payload (all values must be **strings**)  
# Booleans are sent as the strings "true" / "false".
data = {
    "title": "My Image Analysis",
    "do_inspect": "true",
    "do_photometry": "true",
    "do_simple_transients": "true",
}

# Open the FITS file in binary mode for upload
with open(FITS_PATH, "rb") as fh:
    files = {"file": (FITS_PATH, fh, "application/fits")}

    # The token goes in the Authorization header
    headers = {"Authorization": f"Token {API_TOKEN}"}

    response = requests.post(API_URL, headers=headers, files=files, data=data, timeout=60)

print("Status:", response.status_code)
print(response.json())
```

If the upload succeeds you will receive a JSON response similar to the cURL
examples above, containing the newly created task’s ID and its initial state.

### 2. List Tasks

**Endpoint:** `GET /api/tasks/`

**Description:** Get a list of all tasks for the authenticated user.

**Request:**
```bash
curl -X GET http://your-domain/api/tasks/ \
  -H "Authorization: Token your_api_token_here"
```

**Response:**
```json
[
  {
    "id": 123,
    "original_name": "image.fits",
    "title": "My Image Analysis",
    "state": "completed",
    "user": "username",
    "created": "2024-01-15T10:30:00Z",
    "modified": "2024-01-15T10:35:00Z",
    "completed": "2024-01-15T10:35:00Z",
    "config": {
      "target": "NGC1234"
    }
  }
]
```

### 3. Get Task Details

**Endpoint:** `GET /api/tasks/{task_id}/`

**Description:** Get details of a specific task.

**Request:**
```bash
curl -X GET http://your-domain/api/tasks/123/ \
  -H "Authorization: Token your_api_token_here"
```

**Response:**
```json
{
  "id": 123,
  "original_name": "image.fits",
  "title": "My Image Analysis",
  "state": "completed",
  "user": "username",
  "created": "2024-01-15T10:30:00Z",
  "modified": "2024-01-15T10:35:00Z",
  "completed": "2024-01-15T10:35:00Z",
  "config": {
    "target": "NGC1234"
  }
}
```

### 4. List Presets

**Endpoint:** `GET /api/presets/`

**Description:** Get a list of available configuration presets.

**Request:**
```bash
curl -X GET http://your-domain/api/presets/ \
  -H "Authorization: Token your_api_token_here"
```

**Response:**
```json
[
  {
    "id": 1,
    "name": "Default FITS Processing",
    "config": {
      "some_setting": "value"
    },
    "files": "path/to/preset/file1.txt\npath/to/preset/file2.txt"
  }
]
```

## Task States

Tasks can have the following states:
- `initial`: Task created but not yet processed
- `uploaded`: File uploaded successfully
- `running`: Processing in progress
- `inspect`: Running inspection
- `inspect_done`: Inspection completed
- `photometry`: Running photometry
- `photometry_done`: Photometry completed
- `transients_simple`: Running simple transient detection
- `transients_simple_done`: Simple transient detection completed
- `subtraction`: Running image subtraction
- `subtraction_done`: Image subtraction completed
- `completed`: All processing finished
- `failed`: Processing failed

## Error Responses

API errors are returned with appropriate HTTP status codes and error messages:

```json
{
  "error": "Error description"
}
```

Common HTTP status codes:
- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Missing or invalid authentication
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

## File Size Limits

- Maximum file size: 100MB
- Supported formats: .fits, .fit, .fts

## Management Commands

### Create API Token for User

```bash
python manage.py create_api_token username
```

This command creates or retrieves an API token for the specified user. 

### Python example: full workflow (inspection → photometry → template subtraction)

The first snippet uploads and leaves the rest of the processing to the server.  
Sometimes you prefer to trigger subsequent steps manually after the previous
one finishes – for instance, run photometry only after inspection, and run a
template subtraction after photometry.  The code below does exactly that,
using simple polling to wait for each step to complete:

```python
import time
import requests

API  = "http://your-domain"          # <-- change to http://86.253.141.183:7000 for the public demo
TOK  = "your_api_token_here"
FITS = "/absolute/path/image.fits"

# ---------------------------------------------------------------------------
H = {"Authorization": f"Token {TOK}"}

# 1) upload FITS and ask for inspection only
with open(FITS, "rb") as fh:
    r = requests.post(f"{API}/api/tasks/upload/",
                      headers=H,
                      files={"file": (FITS, fh, "application/fits")},
                      data={"title": "Full run", "do_inspect": "true"},
                      timeout=120)
    r.raise_for_status()
    task_id = r.json()["task"]["id"]
print("Task", task_id, "created → inspection running…")


# helper: wait until the task reaches a target state
def wait_for(state):
    while True:
        cur_state = requests.get(f"{API}/api/tasks/{task_id}/", headers=H).json()["state"]
        print("state =", cur_state)
        if cur_state == state:
            return
        time.sleep(10)


# 2) wait for inspection to finish, then launch photometry
wait_for("inspect_done")
print("→ launching photometry…")

requests.post(f"{API}/api/tasks/{task_id}/action/",
              headers={**H, "Content-Type": "application/json"},
              json={"action": "photometry"}).raise_for_status()

# 3) wait for photometry to finish, then launch template subtraction
wait_for("photometry_done")
print("→ launching template subtraction…")

requests.post(f"{API}/api/tasks/{task_id}/action/",
              headers={**H, "Content-Type": "application/json"},
              json={"action": "subtraction",
                    "template_catalog": "ZTF_DR7"}).raise_for_status()

# 4) final wait
wait_for("subtraction_done")
print("All processing steps finished ✔")
```

The script uses the public‐facing steps `inspect`, `photometry`, and
`subtraction`.  Adjust `template_catalog` if you need a different survey (e.g.
`PS1`, `LS_DR10`). 

### Running `full_workflow.py` step-by-step

You’ll find a self-contained example in `examples/full_workflow.py`.  It
uploads an image, waits for **inspection**, then starts **photometry** and
finally **template subtraction**.

```text
1) Create a virtual environment
   python3 -m venv venv

2) Activate it
   source venv/bin/activate        # prompt becomes (venv)$

3) Install the only dependency
   pip3 install requests

4) Export the API url, your token and the FIT/FITS file to upload
   export STDWEB_API_URL="http://86.253.141.183:7000"
   export STDWEB_API_TOKEN="<your_STDWEB_token>"
   export STDWEB_FITS_FILE="/path/to/image/res.fit"
   # Optional: supply detector gain (e-/ADU)
   # If your image is in full 16-bit range (0-65535):
   #   export STDWEB_GAIN="2.3"
   # If your image is normalised 0-1 and you want the script to scale:
   #   export STDWEB_GAIN="2.3"
   #   export STDWEB_AUTOSCALE_GAIN="true"

5) Run the script
   python3 examples/full_workflow.py
```

What the script does
* uploads the image (`upload`) with `do_inspect=true`;
* polls until the task reaches `inspect_done`;
* triggers photometry (`photometry`) and waits for `photometry_done`;
* triggers template subtraction (`subtraction`) and waits for
  `subtraction_done`.

All credentials / paths are provided via environment variables so nothing
sensitive is hard-coded in the script. 