from django.http import HttpResponse, FileResponse, HttpResponseRedirect
from django.template.response import TemplateResponse
from django.urls import reverse
from django.db.models import Q
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.views.decorators.cache import cache_page
from django.conf import settings

import os, glob
import json
import re
import numpy as np
import requests

from astropy.table import Table
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from stdpipe import astrometry

from . import models
from . import forms
from . import processing

# Timeout for requests to SkyPortal, in seconds
SKYPORTAL_TIMEOUT = 60

def skyportal_resolve_source(ra, dec, sr=30/3600, api_token=settings.SKYPORTAL_TOKEN):
    if api_token is not None:
        headers = {'Authorization': f'token {api_token}'}
    else:
        headers = None

    res = requests.get(f'{settings.SKYPORTAL_BASE_URL}/api/sources?ra={ra}&dec={dec}&radius={sr}&group_ids={settings.SKYPORTAL_GROUP_ID}', headers=headers, timeout=SKYPORTAL_TIMEOUT)

    if res:
        json = res.json()
        if json['status'] == 'success':
            for s in json['data']['sources']:
                return s['id']

    return None


def skyportal_get_instruments(api_token=settings.SKYPORTAL_TOKEN):
    if api_token is not None:
        headers = {'Authorization': f'token {api_token}'}
    else:
        headers = None

    try:
        res = requests.get(f'{settings.SKYPORTAL_BASE_URL}/api/instrument', headers=headers, timeout=SKYPORTAL_TIMEOUT)

        if res:
            json = res.json()
            if json['status'] == 'success':
                return json['data']
    except:
        pass

    return []


def skyportal_get_instrument_id_by_name(name, api_token=settings.SKYPORTAL_TOKEN):
    if api_token is not None:
        headers = {'Authorization': f'token {api_token}'}
    else:
        headers = None

    # Network errors propagate so that the caller may tell them from an unknown name
    res = requests.get(f'{settings.SKYPORTAL_BASE_URL}/api/instrument', params={'name': name}, headers=headers, timeout=SKYPORTAL_TIMEOUT)

    if res:
        json = res.json()
        if json.get('status') == 'success' and json.get('data'):
            return json['data'][0]['id']

    return None


def skyportal_upload_photometry(
        sid,
        *,
        mjd,
        mag,
        magerr,
        limit,
        magsys,
        filter,
        instrument,
        ra=None,
        dec=None,
        altdata=None,
        group_ids=[settings.SKYPORTAL_GROUP_ID],
        origin='stdview',
        api_token=settings.SKYPORTAL_TOKEN
):
    if api_token is not None:
        headers = {'Authorization': f'token {api_token}'}
    else:
        headers = None

    payload = {
        'obj_id': sid,
        'instrument_id': instrument,
        'mjd': mjd,
        'mag': mag,
        'magerr': magerr,
        'limiting_mag': limit,
        'magsys': magsys,
        'filter': filter,
        'ra': ra,
        'dec': dec,
        'origin': origin,
        'altdata': altdata,
        'group_ids': group_ids,
    }

    res = requests.put(f'{settings.SKYPORTAL_BASE_URL}/api/photometry', headers=headers, json=payload, timeout=SKYPORTAL_TIMEOUT)

    return res.json()


def skyportal_resolve_task(task):
    # Guess some coordinates relevant for the task
    if 'target_ra' in task.config and 'target_dec' in task.config:
        ra = task.config.get('target_ra')
        dec = task.config.get('target_dec')

    else:
        try:
            filename = f"tasks/{task.id}/image.fits"
            wcsname = f"tasks/{task.id}/image.wcs"

            header = fits.getheader(filename)
            if os.path.exists(wcsname):
                wcs = WCS(fits.getheader(wcsname))
                astrometry.clear_wcs(header)
                header += wcs.to_header(relax=True)

            ra,dec,sr = astrometry.get_frame_center(header=header)

        except:
            raise RuntimeError("Task has no target coordinates")

    # Try resolving using increasing cone search radius
    for sr in [10/3600, 30/3600, 1/60, 10/60, 30/60]:
        sid = skyportal_resolve_source(ra, dec, sr=sr)
        if sid is not None:
            break

    if sid is None:
        raise RuntimeError(f"Cannot resolve SkyPortal source at or around RA={ra:.4f} Dec={dec:.4f}")

    return ra, dec, sid


# filter name -> SkyPortal filter name.
SKYPORTAL_FILTER_MAP = {
    'Umag': 'bessellu',
    'Bmag': 'bessellb',
    'Vmag': 'bessellv',
    'Rmag': 'bessellr',
    'Imag': 'besselli',
    'umag': 'sdssu',
    'gmag': 'sdssg',
    'rmag': 'sdssr',
    'imag': 'sdssi',
    'zmag': 'sdssz',
    'Jmag': '2massj',
    'Hmag': '2massh',
    'Ksmag': '2massks',
    'Gmag': 'gaia::g',
    'BPmag': 'gaia::gbp',
    'RPmag': 'gaia::grp',
}
SKYPORTAL_VEGA_FILTERS = ['Umag', 'Bmag', 'Vmag', 'Rmag', 'Imag', 'Jmag', 'Hmag', 'Ksmag']


def skyportal_prepare_photometry(task, types='best', limit_only=False):
    """
    Get information for uploading photometry to skyportal.

    `types`: 'direct' / 'subtracted' / 'best' (prefer subtracted if available)
    """
    sfiles = []
    if types == 'subtracted' or types == 'best':
        sfiles.append('sub_target.vot')
    if types == 'direct' or types == 'best':
        sfiles.append('target.vot')

    filename = None
    is_diff = False
    for sname in sfiles:
        candidate = os.path.join(task.path(), sname)
        if os.path.exists(candidate):
            filename = candidate
            is_diff = (sname == 'sub_target.vot')
            break

    if filename is None and types == 'subtracted':
        raise RuntimeError('No template-subtracted target measurement')

    try:
        if filename is not None:
            tobj = Table.read(filename)
            row = tobj[0]
            fname = row['mag_filter_name']

            if row['mag_calib_err'] < 1/3 and not limit_only:
                mag = row['mag_calib']
                magerr = row['mag_calib_err']
            else:
                mag = None
                magerr = None
            mag_limit = row['mag_limit']
        else:
            fname = task.config['cat_col_mag']
            mag = None
            magerr = None
            mag_limit = task.config['mag_limit']
    except Exception:
        raise RuntimeError('Cannot load target measurements')

    magsys = 'vega' if fname in SKYPORTAL_VEGA_FILTERS else 'ab'
    fname = SKYPORTAL_FILTER_MAP.get(fname, fname)

    if 'time' in task.config and task.config['time']:
        time = Time(task.config['time'])
    else:
        raise RuntimeError('Cannot get image time')

    return {
        'mjd': time.mjd,
        'filter': fname,
        'magsys': magsys,
        'mag': mag,
        'magerr': magerr,
        'limit': mag_limit,
        'is_diff': is_diff,
    }


@login_required
def skyportal(request):
    context = {}

    instruments = skyportal_get_instruments()
    instruments = [(_['id'], _['name']) for _ in instruments if _['type'].startswith('imag')]
    instruments.sort(key=lambda x: x[1]) # Sort by name
    for i,inst in enumerate(instruments):
        if inst[1] == 'Generic':
            instruments.insert(0, instruments.pop(i))
            break

    form = forms.SkyPortalUploadForm(request.POST, instruments=instruments)
    context['form'] = form

    if request.method == 'POST':
        if form.is_valid() and request.POST.get('action') != 'init':
            action = request.POST.get('action')
            types = request.POST.get('types', 'best')
            limit_only = request.POST.get('limit_only', False)
            context['action'] = action

            instrument = form.cleaned_data.get('instrument')
            ids = form.cleaned_data.get('ids')

            ids = [int(_.strip()) for _ in re.split(r'\W+', ids) if _.isnumeric()]

            # TODO: more accurate handling of ranges?..
            if '-' in form.cleaned_data.get('ids') and len(ids) == 2:
                ids = [int(_) for _ in np.arange(ids[0], ids[1]+1)]

            print(ids)

            tasks = []

            for id in ids:
                ctask = {'id': id}

                try:
                    task = models.Task.objects.get(id=id)
                    if not task:
                        raise RuntimeError('No such task')

                    ctask['name'] = task.original_name
                    ctask['title'] = task.title
                    ctask['ra'], ctask['dec'], ctask['sid'] = skyportal_resolve_task(task)

                    phot = skyportal_prepare_photometry(task, types=types, limit_only=limit_only)

                    ctask['is_diff'] = phot['is_diff']
                    ctask['mjd'] = phot['mjd']
                    ctask['filter'] = phot['filter']
                    ctask['magsys'] = phot['magsys']
                    ctask['mag'] = phot['mag']
                    ctask['magerr'] = phot['magerr']
                    ctask['limit'] = phot['limit']
                    if action == 'upload':
                        res = skyportal_upload_photometry(
                            # 'test_source_stdpipe',
                            ctask['sid'],
                            instrument=instrument,
                            mjd=ctask['mjd'],
                            mag=ctask['mag'],
                            magerr=ctask['magerr'],
                            limit=ctask['limit'],
                            filter=ctask['filter'],
                            magsys=ctask['magsys'],
                            ra=ctask['ra'],
                            dec=ctask['dec'],
                            altdata={'stdview': id},
                        )

                        if res['status'] == 'success':
                            ctask['status'] = 'success'
                        else:
                            ctask['status'] = res['message']

                except BaseException as e:
                    ctask['error'] = str(e)

                    import traceback
                    traceback.print_exc()

                tasks.append(ctask)

                context['tasks'] = tasks

    return TemplateResponse(request, 'skyportal.html', context=context)
