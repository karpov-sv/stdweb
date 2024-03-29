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

from . import models
from . import forms
from . import processing

def skyportal_resolve_source(ra, dec, sr=30/3600, api_token=settings.SKYPORTAL_TOKEN):
    if api_token is not None:
        headers = {'Authorization': f'token {api_token}'}
    else:
        headers = None

    base_url = 'https://skyportal-icare.ijclab.in2p3.fr/api'
    res = requests.get(f'{base_url}/sources?ra={ra}&dec={dec}&radius={sr}&group_ids=3', headers=headers)

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

    base_url = 'https://skyportal-icare.ijclab.in2p3.fr/api'
    res = requests.get(f'{base_url}/instrument', headers=headers)

    if res:
        json = res.json()
        if json['status'] == 'success':
            return json['data']

    return []

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
        group_ids=[3],
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

    base_url = 'https://skyportal-icare.ijclab.in2p3.fr/api'
    res = requests.put(f'{base_url}/photometry', headers=headers, json=payload)

    return res.json()

def skyportal(request):
    context = {}

    instruments = skyportal_get_instruments()
    instruments = [(_['id'], _['name']) for _ in instruments]
    instruments.sort(key=lambda x: x[0])
    for i,inst in enumerate(instruments):
        if inst[1] == 'Generic Instrument':
            instruments.insert(0, instruments.pop(i))
            break

    form = forms.SkyPortalUploadForm(request.POST, instruments=instruments)
    context['form'] = form

    if request.method == 'POST':
        if form.is_valid():
            action = request.POST.get('action')
            context['action'] = action

            instrument = form.cleaned_data.get('instrument')
            ids = form.cleaned_data.get('ids')

            ids = [int(_.strip()) for _ in re.split(r'\W+', ids) if _.isnumeric()]

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
                    ctask['ra'] = task.config.get('target_ra')
                    ctask['dec'] = task.config.get('target_dec')

                    if ctask['ra'] or ctask['dec']:
                        # Try resolving using increasing cone search radius
                        for sr in [10/3600, 30/3600, 1/60, 10/60, 30/60]:
                            sid = skyportal_resolve_source(
                                ctask['ra'],
                                ctask['dec'],
                                sr=sr,
                            )
                            if sid is not None:
                                break

                        if sid is None:
                                raise RuntimeError(f"Cannot resolve SkyPortal source at RA={ctask['ra']:.4f} Dec={ctask['dec']:.4f}")

                    else:
                        raise RuntimeError("Task has no target coordinates")

                    ctask['sid'] = sid

                    filename = f"tasks/{id}/target.vot"
                    try:
                        tobj = Table.read(filename)
                    except:
                        raise RuntimeError('Cannot load target measurements')

                    row = tobj[0]

                    fname = row['mag_filter_name']

                    magsys = 'vega' if fname in ['Umag', 'Bmag', 'Vmag', 'Rmag', 'Imag'] else 'ab'
                    fname = {
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
                    }.get(fname, fname)

                    if 'time' in task.config and task.config['time']:
                        time = Time(task.config['time'])
                    else:

                        raise RuntimeError('Cannot get image time')

                    ctask['mjd'] = time.mjd
                    ctask['filter'] = fname
                    ctask['magsys'] = magsys
                    if row['mag_calib_err'] < 1/5:
                        ctask['mag'] = row['mag_calib']
                        ctask['magerr'] = row['mag_calib_err']
                    else:
                        ctask['mag'] = None
                        ctask['magerr'] = None
                    ctask['limit'] = row['mag_limit']

                    if action == 'upload':
                        res = skyportal_upload_photometry(
                            # 'test_source_stdpipe',
                            sid,
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
