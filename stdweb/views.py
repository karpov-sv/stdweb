from django.http import HttpResponse, FileResponse
from django.template.response import TemplateResponse

from django.contrib.auth.decorators import login_required

import os, glob, io

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

def index(request):
    context = {}

    return TemplateResponse(request, 'index.html', context=context)

def make_breadcrumb(path, base='ROOT', lastlink=False):
    breadcrumb = []

    while True:
        path1,leaf = os.path.split(path)

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
def list_files(request, path=''):
    context = {}

    path = path or ''

    # Prevent escaping from our data folder
    if os.path.isabs(path):
        path = ''

    base = settings.DATA_PATH

    context['path'] = path
    context['breadcrumb'] = make_breadcrumb(path)

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

                # if hdus[0].data is None and hdus[1].name == 'IMAGE' and hdus[2].name == 'TEMPLATE':
                #     # Probably FITS with STDPipe cutout, let's try to load it
                #     context['cutout'] = cutouts.load_cutout(fullpath)
                #     context['mode'] = 'cutout'

                #     if context['cutout'] and 'filename' in context['cutout']['meta']:
                #         context['cutout_filename'] = os.path.split(context['cutout']['meta']['filename'])[1]

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

        context['files'] = files
        context['mode'] = 'list'

    return TemplateResponse(request, 'files.html', context=context)

def download(request, path, attachment=True):
    base = settings.DATA_PATH

    # Prevent escaping from our data folder
    if os.path.isabs(path):
        path = ''

    fullpath = os.path.join(base, path)

    if os.path.isfile(fullpath):
        return FileResponse(open(os.path.abspath(fullpath), 'rb'), as_attachment=attachment)
    else:
        return "No such file"

def preview(request, path, width=None, minwidth=256, maxwidth=1024):
    """
    Preview FITS image as PNG
    """
    base = settings.DATA_PATH

    # Prevent escaping from our data folder
    if os.path.isabs(path):
        path = ''

    fullpath = os.path.join(base, path)

    # Optional parameters
    ext = int(request.GET.get('ext', 0))
    fmt = request.GET.get('format', 'jpeg')
    quality = int(request.GET.get('quality', 80))

    width = request.GET.get('width', width)
    if width is not None:
        width = int(width)

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

    fig = Figure(facecolor='white', dpi=72, figsize=(figsize[0]/72, figsize[1]/72))
    ax = Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    fig.add_axes(ax)
    plots.imshow(data, ax=ax, show_axis=False, show_colorbar=False,
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
