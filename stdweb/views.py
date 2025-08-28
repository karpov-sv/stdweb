from django.http import HttpResponse, FileResponse, HttpResponseRedirect
from django.template.response import TemplateResponse
from django.urls import reverse
from django.conf import settings

from django.contrib import messages

from django.contrib.auth.decorators import login_required

import os, io
import shutil

import mimetypes
import magic
import celery

from astropy.table import Table
from astropy.time import Time
from astropy.io import fits
from astropy.wcs import WCS

# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Circle

from stdpipe import cutouts, plots

from . import forms
from . import models
from . import celery_tasks


def index(request):
    context = {}

    return TemplateResponse(request, 'index.html', context=context)


def sanitize_path(path):
    # Prevent escaping from parent folder
    if not path or os.path.isabs(path):
        path = ''

    return path


def make_breadcrumb(path, base='Root', lastlink=False):
    breadcrumb = []

    while True:
        path1,leaf = os.path.split(path)

        if not leaf:
            break

        if not lastlink and not breadcrumb:
            breadcrumb.append({'name':leaf, 'path':None})
        else:
            breadcrumb.append({'name':leaf, 'path':path})
        path = path1

        if not path:
            break

    breadcrumb.append({'name':base, 'path':'.'})
    breadcrumb.reverse()

    return breadcrumb


@login_required
def list_files(request, path='', base=settings.DATA_PATH):
    context = {}

    path = sanitize_path(path)

    context['path'] = path
    context['breadcrumb'] = make_breadcrumb(path, base="Files")

    fullpath = os.path.join(base, path)

    if os.path.isfile(fullpath):
        # Display a file
        context['mime'] = magic.from_file(filename=fullpath, mime=True)
        context['magic_info'] = magic.from_file(filename=fullpath)
        context['stat'] = os.stat(fullpath)
        context['size'] = context['stat'].st_size
        context['time'] = Time(context['stat'].st_mtime, format='unix')

        context['mode'] = 'download'

        if 'text' in context['mime']:
            try:
                with open(fullpath, 'r') as f:
                    context['contents'] = f.read()
                context['mode'] = 'text'
            except:
                pass

        elif 'fits' in context['mime'] or 'FITS' in context['magic_info'] or os.path.splitext(path)[1].lower().startswith('.fit'):
            context['mode'] = 'fits'

            form = forms.UploadFileForm(filename=path)
            context['form'] = form

            try:
                hdus = fits.open(fullpath) # FIXME: will it be closed?..
                context['fitsfile'] = hdus

                if os.path.splitext(fullpath)[1] == '.cutout':
                # if hdus[0].data is None and hdus[1].name == 'IMAGE' and hdus[2].name == 'TEMPLATE':
                    # Probably FITS with STDPipe cutout, let's try to load it
                    context['cutout'] = cutouts.load_cutout(fullpath)
                    context['mode'] = 'cutout'

                    if context['cutout'] and 'filename' in context['cutout']['meta']:
                        context['cutout_filename'] = os.path.split(context['cutout']['meta']['filename'])[1]

            except:
                import traceback
                traceback.print_exc()
                pass

        elif 'image' in context['mime']:
            context['mode'] = 'image'

        return TemplateResponse(request, 'files.html', context=context)

    elif os.path.isdir(fullpath):
        # List files in directory
        files = []

        for entry in os.scandir(fullpath):
            # Check for broken symlinks
            if not os.path.exists(os.path.join(fullpath, entry.name)):
                continue

            stat = entry.stat()

            elem = {
                'path': os.path.join(path, entry.name),
                'name': entry.name,
                'stat': stat,
                'size': stat.st_size,
                'time': Time(stat.st_mtime, format='unix'),
                'mime': mimetypes.guess_type(entry.name)[0],
                'is_dir': entry.is_dir(),
            }

            if elem['is_dir']:
                elem['type'] = 'dir'
            elif elem['mime'] and 'fits' in elem['mime']:
                elem['type'] = 'fits'
            elif os.path.splitext(entry.name)[1].lower().startswith('.fit'):
                elem['type'] = 'fits'
            elif elem['mime'] and 'image' in elem['mime']:
                elem['type'] = 'image'
            elif elem['mime'] and 'text' in elem['mime']:
                elem['type'] = 'text'
            else:
                elem['type'] = 'file'

            files.append(elem)

        files = sorted(files, key=lambda _: _.get('name'))

        if len(context['breadcrumb']) > 1:
            files = [{'path': os.path.dirname(path), 'name': '..', 'is_dir': True, 'type':'up'}] + files

        context['files'] = files
        context['mode'] = 'list'
        context['form'] = forms.UploadFileForm(filename='*')


    return TemplateResponse(request, 'files.html', context=context)


def download(request, path, attachment=True, base=settings.DATA_PATH):
    path = sanitize_path(path)

    fullpath = os.path.join(base, path)

    if os.path.isfile(fullpath):
        return FileResponse(open(os.path.abspath(fullpath), 'rb'), as_attachment=attachment)
    else:
        return "No such file"


def preview(request, path, width=None, minwidth=256, maxwidth=1024, base=settings.DATA_PATH):
    """
    Preview FITS image as JPEG or PNG
    """
    path = sanitize_path(path)

    fullpath = os.path.join(base, path)

    if not os.path.exists(fullpath):
        return HttpResponse('not found')

    # Custom mask, if relevant
    if path in ['image.fits'] and os.path.exists(os.path.join(base, 'custom_mask.fits')):
        mask = fits.getdata(os.path.join(base, 'custom_mask.fits'), -1) > 0
    else:
        mask = None

    # Optional parameters
    ext = int(request.GET.get('ext', -1))
    fmt = request.GET.get('format', 'jpeg')
    quality = int(request.GET.get('quality', 80))

    width = request.GET.get('width', width)
    if width is not None:
        width = int(width)

    if int(request.GET.get('grid', 0)):
        show_grid = True
    else:
        show_grid = False

    zoom = float(request.GET.get('zoom', 1))

    data = fits.getdata(fullpath, ext)

    figsize = [data.shape[1]*zoom, data.shape[0]*zoom]

    if width is None:
        if figsize[0] < minwidth:
            width = minwidth
        elif figsize[0] > maxwidth:
            width = maxwidth
        else:
            width = figsize[0]

    if width is not None and figsize[0] != width:
        figsize[1] = width*figsize[1]/figsize[0]
        figsize[0] = width

    fig = Figure(dpi=72, figsize=(figsize[0]/72, figsize[1]/72))
    if show_grid:
        dx = 40/figsize[0]
        dy = 20/figsize[1]
        ax = Axes(fig, [dx, dy, 0.99 - 2*dx, 0.99 - dy])
        ax.grid(color='white', alpha=0.3)
    else:
        # No axes, just the image
        ax = Axes(fig, [0., 0., 1., 1.])

    fig.add_axes(ax)

    plots.imshow(data, ax=ax, mask=mask,
                 show_axis=True if show_grid else False,
                 show_colorbar=True if show_grid else False,
                 origin='lower',
                 interpolation='nearest' if data.shape[1]/zoom < 0.5*width else 'bicubic',
                 cmap=request.GET.get('cmap', 'Blues_r'),
                 stretch=request.GET.get('stretch', 'linear'),
                 qq=[float(request.GET.get('qmin', 0.5)), float(request.GET.get('qmax', 99.5))],
                 r0=float(request.GET.get('r0', 0)))

    def get_wcs():
        # Special handling of external WCS solution in .wcs file alongside with image
        wcsname = os.path.splitext(fullpath)[0] + '.wcs'
        if os.path.exists(wcsname):
            header = fits.getheader(wcsname)
        else:
            header = fits.getheader(fullpath, ext)
        wcs = WCS(header)

        return wcs

    if request.GET.get('ra', None) is not None and request.GET.get('dec', None) is not None:
        # Show the position of the object
        ra = [float(_) for _ in request.GET.get('ra').split(',')]
        dec = [float(_) for _ in request.GET.get('dec').split(',')]
        radius = float(request.GET.get('radius', 5.0))

        wcs = get_wcs()

        if wcs is not None and wcs.is_celestial:
            x,y = wcs.all_world2pix(ra, dec, 0)
            for xx,yy in zip(x,y):
                if xx > 0 and xx < data.shape[1] and yy > 0 and yy < data.shape[0]:
                    ax.add_artist(Circle((xx, yy), radius, edgecolor='red', facecolor='none', ls='-', lw=2))

                    for _ in ['2', '3']:
                        radius1 = request.GET.get('radius'+_)
                        if radius1:
                            ax.add_artist(Circle((xx, yy), float(radius1), edgecolor='red', facecolor='none', ls='--', lw=1))


    if request.GET.get('obj'):
        # Overplot list of objects from the file
        objname = os.path.join(os.path.dirname(fullpath), 'objects.vot')
        if os.path.exists(objname):
            obj = Table.read(objname)

            if obj is not None:
                idx = obj['flags'] == 0
                ax.plot(obj['x'][idx], obj['y'][idx], '.', color='red')
                ax.plot(obj['x'][~idx], obj['y'][~idx], '.', color='orange')

    if request.GET.get('cat'):
        # Overplot list of catalogue stars from the file
        catname = os.path.join(os.path.dirname(fullpath), 'cat.vot')
        if os.path.exists(catname):
            cat = Table.read(catname)

            if cat is not None:
                wcs = get_wcs()
                x,y = wcs.all_world2pix(cat['RAJ2000'], cat['DEJ2000'], 0)
                idx = (x > 0) & (x < data.shape[1]) & (y > 0) & (y < data.shape[0])
                ax.plot(x[idx], y[idx], 'o', color='none', mec='brown', ms=10)

    if zoom > 1:
        x0,width = data.shape[1]/2, data.shape[1]
        y0,height = data.shape[0]/2, data.shape[0]

        x0 += float(request.GET.get('dx', 0)) * width/4
        y0 += float(request.GET.get('dy', 0)) * height/4

        ax.set_xlim(x0 - 0.5*width/zoom, x0 + 0.5*width/zoom)
        ax.set_ylim(y0 - 0.5*height/zoom, y0 + 0.5*height/zoom)
    else:
        ax.set_xlim(0, data.shape[1])
        ax.set_ylim(0, data.shape[0])

    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, pil_kwargs={'quality':quality})

    return HttpResponse(buf.getvalue(), content_type='image/%s' % fmt)


def handle_uploaded_file(upload, filename):
    dirname = os.path.dirname(filename)
    try:
        os.makedirs(dirname)
    except OSError:
        pass

    with open(filename, "wb+") as dest:
        for chunk in upload.chunks():
            dest.write(chunk)


def upload_file(request, base=settings.DATA_PATH):
    form = forms.UploadFileForm(request.POST or None, request.FILES or None, filename=request.POST.get('local_file'))

    if request.method == "POST":
        if form.is_valid():
            tasks = []

            if 'file' in request.FILES:
                upload = request.FILES['file']
                task = models.Task(title=form.cleaned_data.get('title'), original_name=upload.name)
                task.user = request.user
                task.save() # to populate task.id

                handle_uploaded_file(upload, os.path.join(task.path(), 'image.fits'))
                messages.success(request, "File uploaded as task " + str(task.id))

                tasks.append(task)
            else:
                files = form.cleaned_data['local_files']
                if not files:
                    files = [form.cleaned_data['local_file']]

                if 'stack_files' in form.data:
                    # Stacking of multiple images as a single task
                    filenames = []

                    for path in files:
                        path = sanitize_path(path)
                        fullpath = os.path.join(base, path)
                        filenames.append(fullpath)

                    task = models.Task(title=form.cleaned_data.get('title'), original_name=os.path.split(path)[-1])
                    task.user = request.user
                    task.save() # to populate task.id

                    os.makedirs(task.path())

                    task.config['stack_filenames'] = filenames

                    messages.success(request, f"Stacking {len(files)} images as task {task.id}")
                    tasks.append(task)

                else:
                    # Normal multi-image processing
                    for path in files:
                        ext = request.POST.get('ext')
                        path = sanitize_path(path)

                        fullpath = os.path.join(base, path)

                        task = models.Task(title=form.cleaned_data.get('title'), original_name=os.path.split(path)[-1])
                        task.user = request.user
                        task.save() # to populate task.id

                        # TODO: merge into handle_uploaded_file?..
                        try:
                            os.makedirs(task.path())
                        except OSError:
                            pass

                        if ext is None:
                            shutil.copyfile(fullpath, os.path.join(task.path(), 'image.fits'))
                            messages.success(request, f"File {path} copied as task " + str(task.id))

                        else:
                            ext = int(ext[0])
                            image = fits.getdata(fullpath, ext)
                            header = fits.getheader(fullpath, ext)
                            fits.writeto(os.path.join(task.path(), 'image.fits'), image, header)
                            messages.success(request, f"Extension {ext} of {path} copied as task " + str(task.id))

                        tasks.append(task)

            for task in tasks:
                # Apply config preset
                if form.cleaned_data.get('preset'):
                    preset = models.Preset.objects.get(id=int(form.cleaned_data.get('preset')))
                    task.config.update(preset.config)
                    if len(tasks) == 1:
                        messages.success(request, "Config updated with preset " + preset.name + " : " + str(preset.config))

                    if preset.files:
                        # Copy preset files into task folder
                        for filename in preset.files.split('\n'):
                            shutil.copy(filename, task.path())
                            if len(tasks) == 1:
                                messages.success(request, filename + " copied into the task")

                task.config['target'] = form.cleaned_data.get('target')

                task.state = 'uploaded'
                task.save()

                # Initiate some processing steps
                todo = []

                if 'stack_filenames' in task.config:
                    todo.append(celery_tasks.task_set_state.subtask(args=[task.id, 'stacking'], immutable=True))
                    todo.append(celery_tasks.task_stacking.subtask(args=[task.id, False], immutable=True))
                    todo.append(celery_tasks.task_break_if_failed.subtask(args=[task.id], immutable=True))
                    todo.append(celery_tasks.task_set_state.subtask(args=[task.id, 'stacking_done'], immutable=True))

                if form.cleaned_data.get('do_inspect'):
                    todo.append(celery_tasks.task_set_state.subtask(args=[task.id, 'inspect'], immutable=True))
                    todo.append(celery_tasks.task_inspect.subtask(args=[task.id, False], immutable=True))
                    todo.append(celery_tasks.task_break_if_failed.subtask(args=[task.id], immutable=True))
                    todo.append(celery_tasks.task_set_state.subtask(args=[task.id, 'inspect_done'], immutable=True))

                if form.cleaned_data.get('do_photometry'):
                    todo.append(celery_tasks.task_set_state.subtask(args=[task.id, 'photometry'], immutable=True))
                    todo.append(celery_tasks.task_photometry.subtask(args=[task.id, False], immutable=True))
                    todo.append(celery_tasks.task_break_if_failed.subtask(args=[task.id], immutable=True))
                    todo.append(celery_tasks.task_set_state.subtask(args=[task.id, 'photometry_done'], immutable=True))

                if form.cleaned_data.get('do_simple_transients'):
                    todo.append(celery_tasks.task_set_state.subtask(args=[task.id, 'transients_simple'], immutable=True))
                    todo.append(celery_tasks.task_transients_simple.subtask(args=[task.id, False], immutable=True))
                    todo.append(celery_tasks.task_break_if_failed.subtask(args=[task.id], immutable=True))
                    todo.append(celery_tasks.task_set_state.subtask(args=[task.id, 'transients_simple_done'], immutable=True))

                if form.cleaned_data.get('do_subtraction'):
                    todo.append(celery_tasks.task_set_state.subtask(args=[task.id, 'subtraction'], immutable=True))
                    todo.append(celery_tasks.task_subtraction.subtask(args=[task.id, False], immutable=True))
                    todo.append(celery_tasks.task_break_if_failed.subtask(args=[task.id], immutable=True))
                    todo.append(celery_tasks.task_set_state.subtask(args=[task.id, 'subtraction_done'], immutable=True))

                if todo:
                    todo.append(celery_tasks.task_finalize.subtask(args=[task.id], immutable=True))

                    task.celery_id = celery.chain(todo).apply_async()
                    task.state = 'running'
                    task.save()

            if len(tasks) > 1:
                return HttpResponseRedirect(reverse('tasks'))
            else:
                return HttpResponseRedirect(reverse('tasks', kwargs={'id': task.id}))

        else:
            messages.error(request, "Error uploading file")

    context = {'form': form}
    return TemplateResponse(request, 'index.html', context=context)


def cutout(request, path, width=None, base=settings.DATA_PATH):
    """
    Preview cutouts FITS image
    """
    path = sanitize_path(path)

    fullpath = os.path.join(base, path)

    if not os.path.exists(fullpath):
        return HttpResponse('not found')

    # Optional parameters
    fmt = request.GET.get('format', 'jpeg')
    quality = int(request.GET.get('quality', 95))

    qq = None
    if 'qmin' in request.GET or 'qmax' in request.GET:
        qq=[float(request.GET.get('qmin', 0.5)), float(request.GET.get('qmax', 99.5))]

    opts = {
        'cmap': request.GET.get('cmap', 'Blues_r'),
        'qq': qq,
        'stretch': request.GET.get('stretch', None),
        'r0': float(request.GET.get('r0', 0)),
    }

    # Load the cutout
    cutout = cutouts.load_cutout(fullpath)

    if request.GET.get('ra', None) is not None and request.GET.get('dec', None) is not None:
        # Show the position of the object
        header = fits.getheader(fullpath, 1)
        wcs = WCS(header)
        # x,y = wcs.all_world2pix(float(request.GET.get('ra')), float(request.GET.get('dec')), 0)
        x,y = wcs.all_world2pix(cutout['meta'].get('ra'), cutout['meta'].get('dec'), 0)
        opts['mark_x'] = x
        opts['mark_y'] = y

        for _ in ['', '2', '3']:
            if request.GET.get('radius'+_, None) is not None:
                opts['mark_r'+_] = float(request.GET.get('radius'+_))

    if 'mag_calib_err' in cutout['meta'] or 'magerr' in cutout['meta']:
        magerr = cutout['meta'].get('mag_calib_err', cutout['meta'].get('magerr'))
        if magerr is not None:
            opts['additional_title'] = f"S/N = {1/magerr:.2f}"

    if request.GET.get('adjust'):
        planes = ['image', 'template', 'filtered', 'convolved', 'diff', 'adjusted', 'footprint', 'mask']
    else:
        planes = ['image', 'template', 'filtered', 'convolved', 'diff', 'footprint', 'mask']

    planes = [_ for _ in planes if _ in cutout]

    figsize = [256*len(planes), 256+40]

    fig = Figure(facecolor='white', dpi=72, figsize=(figsize[0]/72, figsize[1]/72), tight_layout=True)

    plots.plot_cutout(cutout, fig=fig,
                      planes=planes,
                      **opts)

    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, pil_kwargs={'quality':quality})

    return HttpResponse(buf.getvalue(), content_type='image/%s' % fmt)
