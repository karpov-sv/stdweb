from django.http import HttpResponse, HttpResponseForbidden, JsonResponse
from django.template.response import TemplateResponse
from django.contrib import messages
from django.contrib.auth.views import redirect_to_login
from django.views.decorators.cache import cache_page
from django.shortcuts import get_object_or_404

import os
import io

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from astropy.io import fits
from astropy.wcs import WCS

from stdpipe import astrometry, resolve

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
        coords = None

        if form.cleaned_data.get('coordinates'):
            coords = resolve.resolve(form.cleaned_data['coordinates'])
            ra0, dec0 = coords.ra.deg, coords.dec.deg
            radius = form.cleaned_data.get('radius') # In arcseconds!

        if coords:
            # Get all tasks user can access
            tasks = models.Task.objects.all().order_by('-id')

            # Apply permission filtering
            if not form.cleaned_data.get('show_all') or not (request.user.is_staff or request.user.has_perm('stdweb.view_all_tasks')):
                tasks = tasks.filter(user=request.user)

            # Store search coordinates for template display
            context['search_ra'] = ra0
            context['search_dec'] = dec0

            context['search_sr0'] = radius

            if context['search_sr0'] is None:
                context['sub_sr'] = 30/3600
            else:
                context['sub_sr'] = max(float(context['search_sr0']), 30)/3600

            # Filter tasks by field coverage
            task_data = []

            for task in tasks:
                # TODO: MOC-based filtering
                # Extract field center and radius from task config
                ra, dec, sr = [task.config.get(_) for _ in ['field_ra', 'field_dec', 'field_sr']]

                if ra is not None and dec is not None and sr is not None:
                    # Calculate angular distance between search position and field center
                    dist = astrometry.spherical_distance(ra, dec, ra0, dec0)

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

            # Display informative message
            messages.info(
                request,
                f"Found {len(task_data)} tasks covering position "
                f"{ra0:.4f} {dec0:+.4f} (among {len(tasks)} tasks searched)"
            )

        else:
            # Could not parse coordinates
            messages.error(request, "Could not parse coordinates")

    return TemplateResponse(request, 'lightcurves.html', context=context)
