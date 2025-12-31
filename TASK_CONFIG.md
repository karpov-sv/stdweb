# Task Configuration Parameters

This document describes all configuration parameters available in the task `config` dictionary. Parameters are organized by processing stage.

---

## Table of Contents

- [Image Inspection](#image-inspection)
- [Photometry](#photometry)
- [Simple Transient Detection](#simple-transient-detection)
- [Template Subtraction](#template-subtraction)
- [Image Stacking](#image-stacking)
- [Computed Output Parameters](#computed-output-parameters)

---

## Image Inspection

Parameters used during the initial image inspection stage (`inspect_image()`).

### Basic Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target` | str | None | Target name(s) or coordinates. Supports multiple newline-separated targets. Resolved via Sesame/Simbad. |
| `time` | str | Auto | Observation time in ISO format. Auto-detected from FITS header (`DATE-OBS`, `MJD-OBS`, etc.). |
| `gain` | float | Auto | Detector gain in e-/ADU. Auto-detected from FITS header (`GAIN`, `EGAIN`). |
| `saturation` | float | Auto | Saturation level in ADU. Estimated from image if not provided. |
| `filter` | str | Auto | Photometric filter name. Auto-detected and normalized to canonical names (U, B, V, R, I, u, g, r, i, z, etc.). |

### Cosmic Ray Masking

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mask_cosmics` | bool | True | Enable cosmic ray detection and masking using astroscrappy. |

### Catalog Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cat_name` | str | Auto | Reference catalog name. Auto-suggested based on sky position and filter. Options: `ps1`, `gaiaedr3`, `gaiadr3syn`, `skymapper`, `sdss`, `atlas`, `apass`. |
| `cat_limit` | float | 20.0 | Magnitude limit for reference catalog query. |

### Template Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `template` | str | Auto | Template image source. Auto-suggested based on sky position. Options: `ps1`, `ls`, `skymapper`, `des`, `decaps`, `ztf`, `custom`. |

---

## Photometry

Parameters used during photometric calibration (`photometry_image()`).

### Detection Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sn` | float | 5 | Signal-to-noise ratio threshold for object detection. |
| `initial_aper` | float | 3 | Initial aperture diameter for SExtractor (pixels). |
| `initial_r0` | float | 0 | Gaussian smoothing kernel radius for detection (pixels). 0 = no smoothing. |
| `minarea` | int | 5 | Minimum connected pixel area for detection. |
| `bg_size` | int | Auto | Background mesh size for SExtractor (pixels). Auto-calculated if not set. |
| `fwhm_override` | float | None | Override auto-measured FWHM with fixed value (pixels). |
| `prefilter_detections` | bool | True | Apply isolation forest outlier detection to filter spurious detections. |

### Aperture Photometry

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rel_aper` | float | 1.0 | Photometry aperture radius in FWHM units. |
| `rel_bg1` | float | 5.0 | Inner background annulus radius in FWHM units. |
| `rel_bg2` | float | 7.0 | Outer background annulus radius in FWHM units. |

### Photometric Calibration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spatial_order` | int | 2 | Polynomial order for zero point spatial variations. |
| `use_color` | bool | True | Enable color term in photometric calibration. |
| `force_color_term` | float | None | Force specific color term value instead of fitting. |
| `nonlin` | bool | False | Include detector nonlinearity term in photometric solution. |
| `bg_order` | int | None | Polynomial order for background spatial variations. None = disabled. |
| `filter_blends` | bool | True | Exclude blended stars from photometric calibration. |
| `sr_override` | float | None | Override automatic matching radius (arcsec). |

### WCS Refinement

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `refine_wcs` | bool | True | Refine WCS astrometry using SCAMP with catalog matches. |
| `refine_order` | int | 3 | Polynomial order for WCS distortion correction. |

### Blind WCS Matching

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `blind_match_wcs` | bool | False | Enable blind WCS solving when initial WCS is missing or bad. |
| `blind_match_center` | str | None | Center position to constrain blind matching (RA Dec or object name). |
| `blind_match_sr0` | float | 2.0 | Search radius for blind matching (degrees). |
| `blind_match_ps_lo` | float | 0.2 | Lower limit for pixel scale guess (arcsec/pixel). |
| `blind_match_ps_up` | float | 4.0 | Upper limit for pixel scale guess (arcsec/pixel). |

### Target Handling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `centroid_targets` | bool | False | Refine target positions via centroiding. |
| `cutout_size` | int | 30 | Size of cutout images around targets and candidates (pixels). |

### Diagnostics

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `diagnose_color` | bool | False | Run photometric diagnostics for all available filter combinations. |
| `inspect_bg` | bool | False | Write background and RMS maps to FITS files. |

---

## Simple Transient Detection

Parameters for catalog-based transient detection (`transients_simple_image()`).

### Search Region

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `simple_center` | str | None | Center position to restrict search region (RA Dec or object name). |
| `simple_sr0` | float | None | Search radius around center (degrees). None = entire field. |

### Candidate Filtering

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `simple_mag_diff` | float | 2.0 | Magnitude difference threshold for rejecting catalog matches. Objects within this many magnitudes of a catalog star are rejected. Set to 0 to reject all matches. |
| `simple_prefilter` | bool | True | Reject detections flagged by isolation forest pre-filter. |
| `simple_blends` | bool | True | Reject detections near blended catalog stars. |
| `simple_skybot` | bool | True | Query SkyBoT to reject known solar system objects. |
| `simple_others` | str | '' | Space-separated task IDs to cross-check detections against. Detections present in other tasks are rejected. |

---

## Template Subtraction

Parameters for image subtraction and difference imaging (`subtract_image()`).

### Template Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `template` | str | 'ps1' | Template source: `ps1`, `ls` (Legacy Survey), `skymapper`, `des`, `decaps`, `ztf`, `custom`. |
| `template_fwhm_override` | float | None | Override template FWHM measurement (pixels). |
| `custom_template_gain` | float | 10000 | Assumed gain for custom template images (e-/ADU). |
| `custom_template_saturation` | float | None | Saturation level for custom template (ADU). |

### Subtraction Method

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `subtraction_mode` | str | 'detection' | Mode: `detection` for transient search, `target` for forced photometry at known position. |
| `subtraction_method` | str | 'hotpants' | Algorithm: `hotpants` (Alard-Lupton) or `zogy`. |

### Image Splitting

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sub_size` | int | 1000 | Sub-image size for processing large images (pixels). |
| `sub_overlap` | int | 50 | Overlap between sub-images to avoid edge artifacts (pixels). |
| `sub_verbose` | bool | False | Enable verbose output during subtraction. |

### HOTPANTS Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hotpants_extra` | dict | `{'ko':0, 'bgo':0}` | Additional parameters passed to HOTPANTS. Common options: `ko` (kernel order), `bgo` (background order), `nsx`/`nsy` (stamp grid). |

### Candidate Filtering

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filter_center` | str | None | Center position to restrict candidate region (RA Dec or name). |
| `filter_sr0` | float | 1.0 | Search radius around center (degrees). |
| `filter_prefilter` | bool | True | Apply shape-based classifier to filter artifacts. |
| `filter_adjust` | bool | True | Apply sub-pixel adjustment to fix dipole artifacts. |
| `filter_vizier` | bool | False | Cross-match candidates against Vizier catalogs. |
| `filter_skybot` | bool | False | Query SkyBoT to reject known solar system objects. |

---

## Image Stacking

Parameters for co-adding multiple images (`stack_images()`).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stack_method` | str | 'sum' | Combination method: `sum`, `clipped_mean`, or `median`. |
| `stack_subtract_bg` | bool | True | Subtract background from individual images before stacking. |
| `stack_mask_cosmics` | bool | False | Mask cosmic rays in individual images before stacking. |
| `stack_filenames` | list | [] | List of image filenames to include in stacking. |

---

## Computed Output Parameters

These parameters are computed during processing and stored in the config for use by subsequent stages or for reference.

### From Inspection

| Parameter | Type | Description |
|-----------|------|-------------|
| `target_ra` | float | Primary target RA (degrees), resolved from `target` name. |
| `target_dec` | float | Primary target Dec (degrees), resolved from `target` name. |
| `targets` | list | List of all resolved targets, each with `name`, `ra`, `dec` keys. |

### From Photometry

| Parameter | Type | Description |
|-----------|------|-------------|
| `fwhm` | float | Measured PSF FWHM (pixels). |
| `pixscale` | float | Pixel scale (degrees/pixel). |
| `field_ra` | float | Field center RA (degrees). |
| `field_dec` | float | Field center Dec (degrees). |
| `field_sr` | float | Field radius (degrees). |
| `mag_limit` | float | Detection magnitude limit at specified S/N. |
| `zp` | float | Photometric zero point (mag). |
| `zp_err` | float | Zero point uncertainty (mag). |
| `cat_col_mag` | str | Catalog column used for primary magnitude. |
| `cat_col_mag_err` | str | Catalog column for magnitude error. |
| `cat_col_color_mag1` | str | First magnitude column for color term. |
| `cat_col_color_mag2` | str | Second magnitude column for color term. |

---

## Parameter Types Reference

| Type | Examples | Notes |
|------|----------|-------|
| **bool** | `True`, `False` | Enable/disable flags |
| **int** | `5`, `1000` | Pixel counts, polynomial orders |
| **float** | `3.5`, `0.2` | Measurements, thresholds |
| **str** | `'ps1'`, `'R'` | Names, identifiers, coordinates |
| **list** | `['file1.fits']` | Collections |
| **dict** | `{'ko': 0}` | Complex parameter sets |

---

## Supported Values

### Filters (`filter`)

Standard photometric filters:
- Johnson-Cousins: `U`, `B`, `V`, `R`, `I`
- SDSS/Pan-STARRS: `u`, `g`, `r`, `i`, `z`, `y`
- Gaia: `G`, `BP`, `RP`
- Clear/unfiltered: `C`, `N`, `CV`, `CR`

### Catalogs (`cat_name`)

| Catalog | Coverage | Filters | Notes |
|---------|----------|---------|-------|
| `ps1` | Dec > -30° | grizy | Pan-STARRS DR1 |
| `gaiaedr3` | All-sky | G, BP, RP | Gaia EDR3 photometry |
| `gaiadr3syn` | All-sky | All | Gaia DR3 synthetic photometry |
| `skymapper` | Dec < +10° | uvgriz | SkyMapper DR2 |
| `sdss` | Limited | ugriz | SDSS DR12 |
| `atlas` | All-sky | griz | ATLAS Refcat2 |
| `apass` | All-sky | BVgri | APASS DR10 |

### Templates (`template`)

| Template | Coverage | Filters | Resolution |
|----------|----------|---------|------------|
| `ps1` | Dec > -30° | grizy | 0.25"/pix |
| `ls` | Dec > -20° | grz | 0.26"/pix |
| `skymapper` | Dec < +5° | uvgriz | 0.5"/pix |
| `des` | Limited | grizY | 0.26"/pix |
| `decaps` | Galactic plane | grizY | 0.26"/pix |
| `ztf` | Dec > -30° | gri | 1.0"/pix |
| `custom` | User-provided | Any | Any |

---

## Example Configurations

### Basic Photometry
```json
{
  "filter": "R",
  "cat_name": "ps1",
  "sn": 5,
  "use_color": true
}
```

### Blind WCS Solving
```json
{
  "blind_match_wcs": true,
  "blind_match_center": "M31",
  "blind_match_sr0": 5.0,
  "blind_match_ps_lo": 0.5,
  "blind_match_ps_up": 2.0
}
```

### Transient Detection
```json
{
  "filter": "r",
  "cat_name": "ps1",
  "simple_skybot": true,
  "simple_blends": true,
  "simple_mag_diff": 2.0
}
```

### Template Subtraction
```json
{
  "template": "ps1",
  "subtraction_method": "hotpants",
  "subtraction_mode": "detection",
  "filter_prefilter": true,
  "filter_skybot": true
}
```

### Forced Photometry at Target
```json
{
  "target": "AT2024abc",
  "subtraction_mode": "target",
  "centroid_targets": true
}
```
