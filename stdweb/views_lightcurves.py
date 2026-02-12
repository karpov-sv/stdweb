from django.http import HttpResponse, HttpResponseRedirect, JsonResponse, FileResponse
from django.template.response import TemplateResponse
from django.template.loader import render_to_string
from django.contrib import messages
from django.contrib.auth.views import redirect_to_login
from django.views.decorators.cache import cache_page
from django.shortcuts import get_object_or_404
from django.db.models import Q, F

import io
import os
import numpy as np

from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord

from mocpy import MOC

from stdpipe import astrometry, resolve, photometry

from . import models
from . import forms


def _find_matching_tasks(user, coordinates, extra, targets_only, show_all, radius_arcsec):
    """Find tasks covering a specified sky position.

    Returns:
        (ra0, dec0, task_data_list) where task_data_list is a list of
        {'task', 'distance', 'distance_arcsec'} dicts.

    Raises:
        ValueError: If coordinates cannot be resolved.
    """
    coords = resolve.resolve(coordinates)
    ra0, dec0 = coords.ra.deg, coords.dec.deg

    # Get all tasks user can access
    tasks = models.Task.objects.all().order_by('-id')

    # Apply permission filtering
    if not show_all or not (user.is_staff or user.has_perm('stdweb.view_all_tasks')):
        tasks = tasks.filter(user=user)

    if extra:
        for token in extra.split():
            tasks = tasks.filter(
                Q(original_name__icontains=token) |
                Q(title__icontains=token) |
                Q(user__username__icontains=token) |
                Q(user__first_name__icontains=token) |
                Q(user__last_name__icontains=token)
            )

    # Approximate position search - dec only
    tasks = tasks.filter(
        dec__gte=dec0 - F('radius'),
        dec__lte=dec0 + F('radius')
    )

    # Filter tasks by field coverage
    task_data = []

    for task in tasks:
        if task.ra is not None and task.dec is not None and task.radius is not None and task.moc is not None:
            is_good = True

            # Calculate angular distance between search position and field center
            dist = astrometry.spherical_distance(task.ra, task.dec, ra0, dec0)
            if dist > task.radius:
                is_good = False

            # Check target info if targets_only is set
            if targets_only and is_good:
                targets = task.config.get('targets', [])
                if not len(targets):
                    is_good = False
                else:
                    for target in targets:
                        if astrometry.spherical_distance(
                                target['ra'], target['dec'],
                                ra0, dec0
                        ) > radius_arcsec/3600:
                            is_good = False
                            break

            # Final check based on MOC
            if is_good:
                moc = MOC.from_string(task.moc)

                if not moc.contains_skycoords(SkyCoord(ra0, dec0, unit='deg', frame='icrs')):
                    is_good = False

            if is_good:
                task_data.append({
                    'task': task,
                    'distance': dist,  # in degrees
                    'distance_arcsec': dist * 3600,  # in arcseconds
                })

    return ra0, dec0, task_data


def lightcurves(request):
    """
    Search for tasks covering a specified sky position.

    Displays tasks that have observations covering the input coordinates,
    sorted by distance from the search position.

    GET Parameters:
        coordinates (str): Sky position in various formats:
            - Decimal degrees: "200.5 8.2"
            - HMS/DMS: "13:22:00 +08:12:00"
            - With radius: "200.5 8.2 0.1" (radius in degrees)

    Returns:
        TemplateResponse: Rendered lightcurves.html template with:
            - form: LightcurveSearchForm instance
            - task_data: List of dicts with task and distance info
            - search_ra, search_dec: Parsed coordinates

    Permissions:
        - Authenticated users see their own tasks
        - Staff and users with view_all_tasks see all tasks
    """
    context = {}

    if not request.user.is_authenticated:
        return redirect_to_login(request.path)

    # Initialize form
    form = forms.LightcurveSearchForm(
        request.GET or None,
        show_all=request.user.is_staff or request.user.has_perm('stdweb.view_all_tasks'),
    )

    context['form'] = form

    if request.method == 'GET' and form.is_valid():
        coordinates = form.cleaned_data['coordinates']
        extra = form.cleaned_data.get('extra')
        targets_only = form.cleaned_data.get('targets_only')
        radius_arcsec = float(form.cleaned_data.get('radius', 5.0))

        try:
            ra0, dec0, task_data = _find_matching_tasks(
                request.user,
                coordinates,
                extra,
                targets_only,
                form.cleaned_data.get('show_all'),
                radius_arcsec,
            )

            context['search_ra'] = ra0
            context['search_dec'] = dec0
            context['search_sr0'] = radius_arcsec
            context['sub_sr'] = max(float(radius_arcsec), 30)/3600
            context['tasks'] = task_data
            context['show_images'] = form.cleaned_data.get('show_images')
            context['targets_only'] = targets_only

        except Exception:
            messages.error(request, f"Could not resolve coordinates: {coordinates}")

    return TemplateResponse(request, 'lightcurves.html', context=context)


def _get_photometry_data(task, ra0, dec0, radius_arcsec, filename='objects.parquet'):
    """
    Load and filter photometric measurements for a task.

    Returns:
        tuple: (filtered_table, distance_array, metadata_dict) or (None, None, error_dict)
    """
    # Determine which file to load
    filepath = os.path.join(task.path(), filename)

    if not os.path.exists(filepath):
        return None, None, {'error': f'{filename} not found'}

    try:
        # Load VOTable
        obj = Table.read(filepath)

        # Spatial filtering
        radius_deg = radius_arcsec / 3600.0
        dist = astrometry.spherical_distance(
            obj['ra'], obj['dec'],
            ra0, dec0
        )
        mask = dist < radius_deg
        obj_filtered = obj[mask]
        dist_filtered = dist[mask]

        # Metadata
        metadata = {
            'filename': filename,
            'filter': task.config.get('filter'),
            'time': task.config.get('time'),
            'count': len(obj_filtered),
            'radius_arcsec': radius_arcsec,
        }

        return obj_filtered, dist_filtered, metadata

    except Exception as e:
        return None, None, {'error': str(e)}


def _compute_band_column(obj):
    """Add 'band' column to table from color term / filter name info."""
    if 'mag_color_term' in obj.colnames and obj['mag_color_term'][0] and obj['mag_color_term'][0] != 'None':
        obj['band'] = [photometry.format_color_term(
            _['mag_color_term'],
            name=_['mag_filter_name'],
            color_name=_['mag_color_name']
        ) for _ in obj]
    elif 'mag_filter_name' in obj.colnames:
        obj['band'] = obj['mag_filter_name']
    else:
        obj['band'] = None


def task_photometry_html(request, id):
    """
    Return photometric measurements as pre-rendered HTML.

    GET Parameters:
        ra (float): Center RA in degrees
        dec (float): Center Dec in degrees
        radius (float): Search radius in arcseconds
        targets_only (bool): If true, return target photometry data only, ignoring detections

    Returns:
        HttpResponse: Pre-rendered HTML table
    """
    # Get parameters
    try:
        ra0 = float(request.GET.get('ra'))
        dec0 = float(request.GET.get('dec'))
        radius_arcsec = float(request.GET.get('radius', 5.0))
        targets_only = request.GET.get('targets_only', 'false').lower() == 'true'
    except (TypeError, ValueError):
        return HttpResponse('<div class="alert alert-danger">Invalid parameters</div>', status=400)

    # Load task and check permissions
    task = get_object_or_404(models.Task, id=id)

    if not request.user.is_authenticated:
        return HttpResponse('<div class="alert alert-danger">Authentication required</div>', status=403)

    if not (request.user.is_staff or request.user == task.user or
            request.user.has_perm('stdweb.view_all_tasks')):
        return HttpResponse('<div class="alert alert-danger">Permission denied</div>', status=403)

    html = ""

    files = ['target.vot', 'sub_target.vot']
    if not targets_only:
        files = ['objects.parquet'] + files

    for filename in files:
        # Get filtered data
        obj, dist, metadata = _get_photometry_data(task, ra0, dec0, radius_arcsec, filename=filename)

        if obj is None:
            if 'error' in metadata and not html:
                return HttpResponse(f'<div class="alert alert-warning">{metadata["error"]}</div>', status=200)

        elif len(obj) > 0:
            obj['distance_arcsec'] = dist * 3600
            _compute_band_column(obj)

        if obj is not None:
            # Render template
            html += render_to_string('lightcurves_photometry_table.html', {
                'measurements': obj,
                'metadata': metadata,
            })

    if not html:
        return HttpResponse(f'<div class="alert alert-info">No detections</div>', status=200)

    return HttpResponse(html)


def download_votable(request):
    """
    Download combined photometry from all matching tasks as a single VOTable.

    Uses the same search parameters as the lightcurves view.
    """
    if not request.user.is_authenticated:
        return redirect_to_login(request.path)

    form = forms.LightcurveSearchForm(
        request.GET or None,
        show_all=request.user.is_staff or request.user.has_perm('stdweb.view_all_tasks'),
    )

    if not form.is_valid():
        return HttpResponse('Invalid search parameters', status=400)

    coordinates = form.cleaned_data['coordinates']
    extra = form.cleaned_data.get('extra')
    targets_only = form.cleaned_data.get('targets_only')
    radius_arcsec = float(form.cleaned_data.get('radius', 5.0))

    try:
        ra0, dec0, task_data = _find_matching_tasks(
            request.user,
            coordinates,
            extra,
            targets_only,
            form.cleaned_data.get('show_all'),
            radius_arcsec,
        )
    except Exception:
        return HttpResponse('Could not resolve coordinates', status=400)

    if not task_data:
        return HttpResponse('No matching tasks found', status=404)

    # Collect photometry from all tasks
    all_tables = []

    for item in task_data:
        task = item['task']

        files = ['target.vot', 'sub_target.vot']
        if not targets_only:
            files = ['objects.parquet'] + files

        for filename in files:
            obj, dist, metadata = _get_photometry_data(task, ra0, dec0, radius_arcsec, filename=filename)

            if obj is None or len(obj) == 0:
                continue

            obj['distance_arcsec'] = dist * 3600
            _compute_band_column(obj)

            # Build output table with standardized columns
            out = Table()
            out['task_id'] = np.full(len(obj), task.id, dtype=int)
            out['original_name'] = [task.original_name or ''] * len(obj)
            out['source_file'] = [filename] * len(obj)
            out['time'] = [task.config.get('time') or ''] * len(obj)
            out['filter'] = [task.config.get('filter') or ''] * len(obj)
            out['distance_arcsec'] = obj['distance_arcsec']
            out['ra'] = obj['ra']
            out['dec'] = obj['dec']

            if 'mag_calib' in obj.colnames:
                out['mag_calib'] = obj['mag_calib']
            else:
                out['mag_calib'] = np.nan

            if 'mag_calib_err' in obj.colnames:
                out['mag_calib_err'] = obj['mag_calib_err']
            else:
                out['mag_calib_err'] = np.nan

            out['band'] = obj['band'] if 'band' in obj.colnames else ''

            if 'flags' in obj.colnames:
                out['flags'] = obj['flags']
            else:
                out['flags'] = 0

            all_tables.append(out)

    if not all_tables:
        return HttpResponse('No photometry data found', status=404)

    result = vstack(all_tables)

    # Sort by time, task_id, distance
    result.sort(['time', 'task_id', 'distance_arcsec'])

    # Write VOTable to buffer
    vot_buffer = io.BytesIO()
    result.write(vot_buffer, format='votable')
    vot_buffer.seek(0)

    response = FileResponse(vot_buffer, as_attachment=True)
    response['Content-Type'] = 'application/x-votable+xml'
    response['Content-Disposition'] = 'attachment; filename="lightcurve.vot"'

    return response
