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

**Parameters:**
- `file` (required): FITS file to upload (.fits, .fit, .fts extensions)
- `title` (optional): Title or description for the task
- `target` (optional): Target name or coordinates
- `preset` (optional): Preset ID to apply configuration
- `do_inspect` (optional): Automatically run inspection (default: false)
- `do_photometry` (optional): Automatically run photometry (default: false)
- `do_simple_transients` (optional): Automatically run simple transient detection (default: false)
- `do_subtraction` (optional): Automatically run image subtraction (default: false)

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