#!/usr/bin/env python3
"""Full STDWeb workflow example

Uploads a FITS image, waits for inspection to finish, then launches
photometry and template subtraction sequentially.

Configuration is supplied via environment variables so that secrets and
file paths are not hard-coded:

STDWEB_API_URL   – base URL of the STDWeb instance, e.g. http://86.253.141.183:7000
STDWEB_API_TOKEN – your personal API token (required)
STDWEB_FITS_FILE – absolute path to the FITS file to upload (required)
STDWEB_TEMPLATE  – template catalog alias (optional, default ZTF_DR7)
"""
import os
import time
from pathlib import Path

import requests

API = os.getenv("STDWEB_API_URL", "http://86.253.141.183:7000")
TOKEN = os.getenv("STDWEB_API_TOKEN")
FITS_PATH = os.getenv("STDWEB_FITS_FILE")
TEMPLATE = os.getenv("STDWEB_TEMPLATE", "ZTF_DR7")
GAIN_ENV = os.getenv("STDWEB_GAIN")  # optional user-supplied gain (e/ADU)
AUTOSCALE_GAIN = os.getenv("STDWEB_AUTOSCALE_GAIN", "false").lower() == "true"

if not TOKEN or not FITS_PATH:
    raise SystemExit("Set STDWEB_API_TOKEN and STDWEB_FITS_FILE environment variables before running.")

fits_file = Path(FITS_PATH)
if not fits_file.is_file():
    raise SystemExit(f"FITS file not found: {fits_file}")

HEADERS = {"Authorization": f"Token {TOKEN}"}

# ---------------------------------------------------------------------------
# Helper to compute gain if autoscaling is requested
def compute_gain():
    if GAIN_ENV is None:
        return None

    try:
        gain_val = float(GAIN_ENV)
    except ValueError:
        print("[warn] STDWEB_GAIN is not a number – ignoring")
        return None

    if not AUTOSCALE_GAIN:
        return gain_val

    # autoscale requested: inspect image range
    try:
        from astropy.io import fits
        import numpy as np
        sample = fits.getdata(fits_file, 0, memmap=True)
        # sample a subset to speed up
        subset = sample.ravel()[:: max(1, sample.size // 1_000_000)]
        vmin, vmax = np.nanmin(subset), np.nanmax(subset)
        if vmax <= 1.0:
            print("[info] Image appears normalised 0..1 – scaling gain by 65535")
            gain_val *= 65535
    except Exception as exc:
        print(f"[warn] Could not auto-scale gain ({exc}); using raw value")

    return gain_val

GAIN_VALUE = compute_gain()

def wait_for(task_id: int, target_state: str, poll: int = 10):
    """Poll the task until it reaches *target_state*."""
    while True:
        resp = requests.get(f"{API}/api/tasks/{task_id}/", headers=HEADERS, timeout=30)
        resp.raise_for_status()
        state = resp.json()["state"]
        print("state=", state)
        if state == target_state:
            return
        time.sleep(poll)

# 1) upload & inspection -----------------------------------------------------------------
print("Uploading", fits_file, "to", API)
with fits_file.open("rb") as fh:
    resp = requests.post(
        f"{API}/api/tasks/upload/",
        headers=HEADERS,
        files={"file": (fits_file.name, fh, "application/fits")},
        data={
            "title": "Full run",
            "do_inspect": "true",
            **({"gain": str(GAIN_VALUE)} if GAIN_VALUE is not None else {})
        },
        timeout=300,
    )
resp.raise_for_status()
task_id = resp.json()["task"]["id"]
print("Task", task_id, "created – inspection running…")

# 2) photometry ---------------------------------------------------------------------------
wait_for(task_id, "inspect_done")
print("Launching photometry…")
requests.post(
    f"{API}/api/tasks/{task_id}/action/",
    headers={**HEADERS, "Content-Type": "application/json"},
    json={"action": "photometry", **({"gain": GAIN_VALUE} if GAIN_VALUE is not None else {})},
    timeout=60,
).raise_for_status()

# 3) subtraction --------------------------------------------------------------------------
wait_for(task_id, "photometry_done")
print("Launching subtraction (", TEMPLATE, ")…")
requests.post(
    f"{API}/api/tasks/{task_id}/action/",
    headers={**HEADERS, "Content-Type": "application/json"},
    json={"action": "subtraction", "template_catalog": TEMPLATE},
    timeout=60,
).raise_for_status()

# 4) final wait ---------------------------------------------------------------------------
wait_for(task_id, "subtraction_done")
print("All processing steps finished ✔") 