from django.http import HttpResponse, FileResponse, HttpResponseRedirect
from django.template.response import TemplateResponse
from django.urls import reverse

from django.contrib import messages

from django.contrib.auth.decorators import login_required

import os, io
import shutil

import mimetypes
import magic

from astropy.time import Time
from astropy.io import fits
from astropy.wcs import WCS

# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Circle

from stdpipe import cutouts, plots

from . import settings
from . import forms
from . import models

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
            elif elem['mime'] and 'image' in elem['mime']:
                elem['type'] = 'image'
            elif elem['mime'] and 'text' in elem['mime']:
                elem['type'] = 'text'
            elif elem['mime'] and 'fits' in elem['mime']:
                elem['type'] = 'fits'
            elif os.path.splitext(entry.name)[1].lower().startswith('.fit'):
                elem['type'] = 'fits'
            else:
                elem['type'] = 'file'

            files.append(elem)

        files = sorted(files, key=lambda _: _.get('name'))

        if len(context['breadcrumb']) > 1:
            files = [{'path': os.path.dirname(path), 'name': '..', 'is_dir': True, 'type':'up'}] + files

        context['files'] = files
        context['mode'] = 'list'

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

    # Optional parameters
    ext = int(request.GET.get('ext', 0))
    fmt = request.GET.get('format', 'jpeg')
    quality = int(request.GET.get('quality', 80))

    width = request.GET.get('width', width)
    if width is not None:
        width = int(width)

    if int(request.GET.get('grid', 0)):
        show_grid = True
    else:
        show_grid = False

    data = fits.getdata(fullpath, ext)

    figsize = [data.shape[1], data.shape[0]]

    if width is None:
        if figsize[0] < minwidth:
            width = minwidth
        elif figsize[0] > maxwidth:
            width = maxwidth

    if width is not None and figsize[0] != width:
        figsize[1] = width*figsize[1]/figsize[0]
        figsize[0] = width

    fig = Figure(dpi=72, figsize=(figsize[0]/72, figsize[1]/72))
    if show_grid:
        dx = 40/figsize[0]
        dy = 20/figsize[1]
        ax = Axes(fig, [dx, dy, 0.99 - dx, 0.99 - dy])
        ax.grid(color='white')
    else:
        # No axes, just the image
        ax = Axes(fig, [0., 0., 1., 1.])

    fig.add_axes(ax)

    plots.imshow(data, ax=ax, show_axis=True if show_grid else False, show_colorbar=False,
                 origin='lower',
                 cmap=request.GET.get('cmap', 'Blues_r'),
                 stretch=request.GET.get('stretch', 'linear'),
                 qq=[float(request.GET.get('qmin', 0.5)), float(request.GET.get('qmax', 99.5))])

    if request.GET.get('ra', None) is not None and request.GET.get('dec', None) is not None:
        # Show the position of the object
        header = fits.getheader(fullpath, ext)
        wcs = WCS(header)
        x,y = wcs.all_world2pix(float(request.GET.get('ra')), float(request.GET.get('dec')), 0)
        ax.add_artist(Circle((x, y), 5.0, edgecolor='red', facecolor='none', ls='-', lw=2))

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


def upload_file(request):
    if request.method == "POST":
        form = forms.UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            upload = request.FILES['file']

            task = models.Task(title=form.cleaned_data.get('title'), original_name=upload.name)
            task.user = request.user
            task.save() # to populate task.id

            handle_uploaded_file(upload, os.path.join(task.path(), 'image.fits'))

            task.state = 'uploaded'
            task.save()

            messages.success(request, "File uploaded as task " + str(task.id))

            return HttpResponseRedirect(reverse('tasks', kwargs={'id': task.id}))

        else:
            messages.error(request, "Error uploading file")

    form = forms.UploadFileForm()
    context = {'form': form}
    return TemplateResponse(request, 'upload.html', context=context)


def reuse_file(request, base=settings.DATA_PATH):
    if request.method == 'POST':
        path = request.POST.get('path')
        path = sanitize_path(path)

        fullpath = os.path.join(base, path)

        task = models.Task(original_name=os.path.split(path)[-1])
        task.user = request.user
        task.save() # to populate task.id

        try:
            os.makedirs(task.path())
        except OSError:
            pass

        shutil.copyfile(fullpath, os.path.join(task.path(), 'image.fits'))

        task.state = 'copied'
        task.save()

        messages.success(request, "File copied as task " + str(task.id))

        return HttpResponseRedirect(reverse('tasks', kwargs={'id': task.id}))

    return HttpResponse('done')


def cutout(request, path, width=None, base=settings.DATA_PATH):
    """
    Preview cutouts FITS image
    """
    path = sanitize_path(path)

    fullpath = os.path.join(base, path)

    if not os.path.exists(fullpath):
        return HttpResponse('not found')

    # Optional parameters
    ext = int(request.GET.get('ext', 0))
    fmt = request.GET.get('format', 'jpeg')
    quality = int(request.GET.get('quality', 95))

    width = request.GET.get('width', width)
    if width is not None:
        width = int(width)

    qq = None
    if 'qmin' in request.GET or 'qmax' in request.GET:
        qq=[float(request.GET.get('qmin', 0.5)), float(request.GET.get('qmax', 99.5))]

    opts = {
        'cmap': request.GET.get('cmap', 'Blues_r'),
        'qq': qq,
        'stretch': request.GET.get('stretch', None),
    }

    if request.GET.get('ra', None) is not None and request.GET.get('dec', None) is not None:
        # Show the position of the object
        header = fits.getheader(fullpath, 1)
        wcs = WCS(header)
        x,y = wcs.all_world2pix(float(request.GET.get('ra')), float(request.GET.get('dec')), 0)
        opts['mark_x'] = x
        opts['mark_y'] = y

    # Load the cutout
    cutout = cutouts.load_cutout(fullpath)

    planes = ['image', 'template', 'convolved', 'diff', 'adjusted', 'footprint', 'mask']
    planes = [_ for _ in planes if _ in cutout]

    figsize = [256*len(planes), 256+40]

    fig = Figure(facecolor='white', dpi=72, figsize=(figsize[0]/72, figsize[1]/72))#, tight_layout=True)

    plots.plot_cutout(cutout, fig=fig,
                      planes=planes,
                      **opts)

    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, pil_kwargs={'quality':quality})

    return HttpResponse(buf.getvalue(), content_type='image/%s' % fmt)
