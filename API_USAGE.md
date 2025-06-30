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