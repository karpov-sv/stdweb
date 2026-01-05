from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.template.response import TemplateResponse
from django.template.loader import render_to_string
from django.contrib import messages
from django.contrib.auth.views import redirect_to_login
from django.views.decorators.cache import cache_page
from django.shortcuts import get_object_or_404
from django.db.models import Q

import os
import io
import numpy as np

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table

from stdpipe import astrometry, resolve, photometry

from . import models
from . import forms
from . import utils

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
        # We are guaranteed to have it set?..
        coordinates = form.cleaned_data['coordinates']
        # Extra search params, may be empty
        extra = form.cleaned_data.get('extra')
        targets_only = form.cleaned_data.get('targets_only')

        try:
            coords = resolve.resolve(coordinates)
            ra0, dec0 = coords.ra.deg, coords.dec.deg
            print('radius:', form.cleaned_data.get('radius', 5.0))
            radius_arcsec = float(form.cleaned_data.get('radius', 5.0)) # In arcseconds!
        except:
            coords = None

        if coords is not None:
            # Get all tasks user can access
            tasks = models.Task.objects.all().order_by('-id')

            # Apply permission filtering
            if not form.cleaned_data.get('show_all') or not (request.user.is_staff or request.user.has_perm('stdweb.view_all_tasks')):
                tasks = tasks.filter(user=request.user)

            if extra:
                for token in extra.split():
                    tasks = tasks.filter(
                        Q(original_name__icontains = token) |
                        Q(title__icontains = token) |
                        Q(user__username__icontains = token) |
                        Q(user__first_name__icontains = token) |
                        Q(user__last_name__icontains = token)
                    )


            # Store search coordinates for template display
            context['search_ra'] = ra0
            context['search_dec'] = dec0

            context['search_sr0'] = radius_arcsec

            context['sub_sr'] = max(float(radius_arcsec), 30)/3600

            # Filter tasks by field coverage
            task_data = []

            for task in tasks:
                # TODO: MOC-based filtering
                # Extract field center and radius from task config
                ra, dec, sr = [task.config.get(_) for _ in ['field_ra', 'field_dec', 'field_sr']]

                if ra is not None and dec is not None and sr is not None:
                    # Calculate angular distance between search position and field center
                    dist = astrometry.spherical_distance(ra, dec, ra0, dec0)

                    # Check target info if targets_only is set
                    if targets_only and  dist < sr:
                        targets = task.config.get('targets', [])
                        if not len(targets):
                            dist = np.inf
                        else:
                            for target in targets:
                                if astrometry.spherical_distance(target['ra'], target['dec'], ra0, dec0) > sr:
                                    dist = np.inf
                                    break

                    # Check if search position falls within task's field
                    if dist < sr:
                        # Store task with distance information
                        task_data.append({
                            'task': task,
                            'distance': dist,  # in degrees
                            'distance_arcsec': dist * 3600,  # in arcseconds
                        })

            context['tasks'] = task_data

            context['show_images'] = form.cleaned_data.get('show_images')
            context['targets_only'] = targets_only

            # Display informative message
            # messages.info(
            #     request,
            #     f"Found {len(task_data)} tasks covering position "
            #     f"{ra0:.4f} {dec0:+.4f} (among {len(tasks)} tasks searched)"
            # )

        else:
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

            # Color term
            if 'mag_color_term' in obj.colnames:
                obj['band'] = [photometry.format_color_term(
                    _['mag_color_term'],
                    name=_['mag_filter_name'],
                    color_name=_['mag_color_name']
                ) for _ in obj]
            elif 'mag_filter_name' in obj.colnames:
                obj['band'] = obj['mag_filter_name']
            else:
                obj['band'] = None

        if obj is not None:
            # Render template
            html += render_to_string('lightcurves_photometry_table.html', {
                'measurements': obj,
                'metadata': metadata,
            })

    if not html:
        return HttpResponse(f'<div class="alert alert-info">No detections</div>', status=200)

    return HttpResponse(html)
