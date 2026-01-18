from django.http import HttpResponse, FileResponse, HttpResponseRedirect
from django.template.response import TemplateResponse
from django.urls import reverse
from django.conf import settings

from django.contrib import messages

from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required

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

from stdpipe import cutouts, plots, astrometry

from rest_framework.authtoken.models import Token

from . import forms
from . import models
from .action_logging import log_action
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

    fullpath = os.path.join(base, path)

    context['path'] = path
    context['breadcrumb'] = make_breadcrumb(path, base="Files")

    # HACK: silently fall back to .parquet for backward compatibility
    if fullpath.endswith('.vot') and not os.path.exists(fullpath):
        fullpath = os.path.splitext(fullpath)[0] + '.parquet'

    if os.path.isfile(fullpath):
        # Display a file
        context['mime'] = magic.from_file(filename=fullpath, mime=True)
        context['magic_info'] = magic.from_file(filename=fullpath)
        context['stat'] = os.stat(fullpath)
        context['size'] = context['stat'].st_size
        context['time'] = Time(context['stat'].st_mtime, format='unix')

        context['mode'] = 'download'

        if path.endswith('.vot') or path.endswith('.parquet'):
            try:
                context['table'] = Table.read(fullpath)
                context['mode'] = 'table'
            except:
                pass

        elif 'fits' in context['mime'] or 'FITS' in context['magic_info'] or os.path.splitext(path)[1].lower().startswith('.fit') or path.endswith('.fits.gz'):
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

        elif 'text' in context['mime']:
            try:
                with open(fullpath, 'r') as f:
                    context['contents'] = f.read()
                context['mode'] = 'text'
            except:
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

    # Transparently convert Parquet to VOTable if requested
    if fullpath.endswith('.vot') and not os.path.exists(fullpath):
        parquet_path = os.path.splitext(fullpath)[0] + '.parquet'

        if os.path.exists(parquet_path):
            # Load Parquet and convert to VOTable in memory
            try:
                table = Table.read(parquet_path)

                # Write to BytesIO for in-memory conversion
                vot_buffer = io.BytesIO()
                table.write(vot_buffer, format='votable')
                vot_buffer.seek(0)

                # Serve as VOTable with correct MIME type
                response = FileResponse(vot_buffer, as_attachment=attachment)
                response['Content-Type'] = 'application/x-votable+xml'
                if attachment:
                    # Extract filename from path for attachment
                    filename = os.path.basename(path)
                    response['Content-Disposition'] = f'attachment; filename="{filename}"'

                return response
            except Exception as e:
                return HttpResponse(f'Error converting Parquet to VOTable: {str(e)}', status=500)

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

    def get_wcs(get_header=False):
        # Special handling of external WCS solution in .wcs file alongside with image
        wcsname = os.path.splitext(fullpath)[0] + '.wcs'
        if os.path.exists(wcsname):
            header = fits.getheader(wcsname)
        else:
            header = fits.getheader(fullpath, ext)
        wcs = WCS(header)

        if get_header:
            return header
        else:
            return wcs

    # Custom mask, if relevant
    if path in ['image.fits'] and os.path.exists(os.path.join(base, 'custom_mask.fits')):
        mask = fits.getdata(os.path.join(base, 'custom_mask.fits'), -1) > 0
    elif path in ['custom_template.fits'] and os.path.exists(os.path.join(base, 'custom_template_mask.fits')):
        mask = fits.getdata(os.path.join(base, 'custom_template_mask.fits'), -1) > 0
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

    if 'sub_ra' in request.GET:
        sub_ra = float(request.GET.get('sub_ra'))
        sub_dec = float(request.GET.get('sub_dec'))
        sub_size = int(request.GET.get('sub_size', 0))
        sub_sr = float(request.GET.get('sub_sr', 0.01))

        header = get_wcs(get_header=True)
        wcs = WCS(header)
        if wcs is None or not wcs.is_celestial:
            return HttpResponse(status=400, content="Invalid image WCS")

        # Convert RA/Dec to pixel coordinates
        x0, y0 = wcs.all_world2pix(sub_ra, sub_dec, 0)

        if not sub_size:
            sub_size = 2 * sub_sr / astrometry.get_pixscale(wcs=wcs)

        # Check if coordinates are within image bounds
        if x0 < 0 or y0 < 0 or x0 >= data.shape[1] or y0 >= data.shape[0]:
            return HttpResponse(status=400, content="Coordinates outside image bounds")

        data,header = cutouts.crop_image_centered(
            data, x0, y0, sub_size/2,
            header=header
        )
        # New, cropped WCS
        wcs = WCS(header)
    else:
        wcs = None

    figsize = [data.shape[1]*zoom, data.shape[0]*zoom]

    if width is None:
        if figsize[0] < minwidth:
            width = minwidth
        elif figsize[0] > maxwidth:
            width = maxwidth
        else:
            width = figsize[0]

    # Only crop when zoom > 1 AND the zoomed width exceeds maxwidth
    if zoom > 1 and figsize[0] > maxwidth:
        x0,dx0 = data.shape[1]/2, data.shape[1]
        y0,dy0 = data.shape[0]/2, data.shape[0]

        x0 += float(request.GET.get('dx', 0)) * dx0/4
        y0 += float(request.GET.get('dy', 0)) * dy0/4

        xlim = [x0 - 0.5*dx0/zoom, x0 + 0.5*dx0/zoom]
        ylim = [y0 - 0.5*dy0/zoom, y0 + 0.5*dy0/zoom]
    else:
        xlim = [0, data.shape[1]-1]
        ylim = [0, data.shape[0]-1]

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

    plots.imshow(
        data, ax=ax, mask=mask,
        show_axis=True if show_grid else False,
        show_colorbar=True if show_grid else False,
        origin='lower',
        interpolation='nearest' if data.shape[1]/zoom < 0.5*width else 'bicubic',
        cmap=request.GET.get('cmap', 'Blues_r'),
        stretch=request.GET.get('stretch', 'linear'),
        qq=[float(request.GET.get('qmin', 0.5)), float(request.GET.get('qmax', 99.5))],
        r0=float(request.GET.get('r0', 0)),
        # Use fast (approximate) image display
        max_plot_size=1024, xlim=xlim, ylim=ylim, fast=True
    )

    if request.GET.get('ra', None) is not None and request.GET.get('dec', None) is not None:
        # Show the position of the object
        ra = [float(_) for _ in request.GET.get('ra').split(',')]
        dec = [float(_) for _ in request.GET.get('dec').split(',')]
        radius = float(request.GET.get('radius', 5.0))

        if wcs is None:
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
        if path.startswith('sub_'):
             # Transient candidates
            objname = 'candidates.vot'
            use_wcs = True
        elif 'sub_ra' in request.GET or path == 'image_target.fits':
            # Detected objects on cropped image
            objname = 'objects.parquet'
            use_wcs = True
        else:
            objname = 'objects.parquet'
            use_wcs = False
        objname = os.path.join(os.path.dirname(fullpath), objname)
        if os.path.exists(objname):
            obj = Table.read(objname)

            if obj is not None:
                if use_wcs:
                    if wcs is None:
                        wcs = get_wcs()
                    x,y = wcs.all_world2pix(obj['ra'], obj['dec'], 0)
                else:
                    x,y = obj['x'], obj['y']

                idx = obj['flags'] == 0
                ax.plot(x[idx], y[idx], '.', color='red')
                ax.plot(x[~idx], y[~idx], 'x', color='orange')

    if request.GET.get('cat'):
        # Overplot list of catalogue stars from the file
        catname = os.path.join(os.path.dirname(fullpath), 'cat.parquet')
        if os.path.exists(catname):
            cat = Table.read(catname)

            if cat is not None:
                wcs = get_wcs()
                x,y = wcs.all_world2pix(cat['RAJ2000'], cat['DEJ2000'], 0)
                idx = (x > 0) & (x < data.shape[1]) & (y > 0) & (y < data.shape[0])
                ax.plot(x[idx], y[idx], 'o', color='none', mec='brown', ms=10)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

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
            localpaths = {}

            if 'file' in request.FILES:
                upload = request.FILES['file']
                task = models.Task(title=form.cleaned_data.get('title'), original_name=upload.name)
                task.user = request.user
                task.save() # to populate task.id

                handle_uploaded_file(upload, os.path.join(task.path(), 'image.fits'))
                messages.success(request, "File uploaded as task " + str(task.id))

                tasks.append(task)
                source = 'upload'

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
                    for _ in ['stack_method', 'stack_subtract_bg', 'stack_mask_cosmics']:
                        task.config[_] = form.cleaned_data[_]

                    messages.success(request, f"Stacking {len(files)} images as task {task.id}")
                    tasks.append(task)
                    source = 'stack'

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

                        localpaths[task.id] = fullpath

                        tasks.append(task)
                        source = 'local'

            for task in tasks:
                log_details={
                    'original_name': task.original_name,
                    'access': 'web', 'source': source,
                }
                if source == 'local' and task.id in localpaths:
                    log_details['original_path'] = localpaths.get(task.id)
                if source == 'stack' and 'stack_filenames' in task.config:
                    log_details['original_paths'] = task.config.get('stack_filenames')

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

                    log_details['preset'] = preset.name

                task.config['target'] = form.cleaned_data.get('target')

                task.state = 'uploaded'
                task.save()

                # Log task creation
                log_action('task_create', task=task, request=request, details=log_details)

                # Initiate some processing steps
                steps = [
                    'stack' if 'stack_filenames' in task.config else None,
                    'inspect' if form.cleaned_data.get('do_inspect') else None,
                    'photometry' if form.cleaned_data.get('do_photometry') else None,
                    'simple_transients' if form.cleaned_data.get('do_simple_transients') else None,
                    'subtraction' if form.cleaned_data.get('do_subtraction') else None,
                ]
                steps = [s for s in steps if s]  # Filter out None values
                celery_tasks.run_task_steps(task, steps)
                if steps:
                    log_action('processing_start', task=task, request=request,
                               details={'steps': steps, 'access': 'web'})

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


def view_markdown(request, path=''):
    """Render a Markdown file as HTML."""
    import markdown
    from django.http import Http404

    # Allowed root-level files (whitelist)
    ALLOWED_FILES = ['README.md', 'REST_API.md', 'TASK_CONFIG.md', 'CLAUDE.md']

    # Sanitize path
    path = path.strip('/')

    # Determine full path
    base_path = settings.BASE_DIR

    if path in ALLOWED_FILES:
        fullpath = os.path.join(base_path, path)
    elif path.startswith('doc/') and path.endswith('.md'):
        # Allow any .md file in doc/ folder
        fullpath = os.path.join(base_path, path)
    elif path == '' or path == 'README.md':
        # Default to README.md
        path = 'README.md'
        fullpath = os.path.join(base_path, path)
    else:
        raise Http404("File not found")

    # Prevent path traversal
    fullpath = os.path.normpath(fullpath)
    if not fullpath.startswith(str(base_path)):
        raise Http404("Invalid path")

    if not os.path.exists(fullpath):
        raise Http404("File not found")

    # Read and convert markdown
    with open(fullpath, 'r') as f:
        content = f.read()

    html_content = markdown.markdown(
        content,
        extensions=['tables', 'fenced_code', 'toc']
    )

    context = {
        'content': html_content,
        'title': os.path.splitext(os.path.basename(path))[0].replace('_', ' '),
        'filename': path,
    }

    return TemplateResponse(request, 'markdown.html', context=context)


@login_required
def profile(request):
    """User profile page with account info and API token management."""
    token, created = Token.objects.get_or_create(user=request.user)

    if request.method == 'POST':
        if 'regenerate_token' in request.POST:
            token.delete()
            token = Token.objects.create(user=request.user)
            messages.success(request, 'API token regenerated successfully')
            return HttpResponseRedirect(reverse('profile'))

    context = {
        'token': token,
    }

    return TemplateResponse(request, 'profile.html', context=context)


@staff_member_required
def action_log(request, length=20):
    """Simple read-only view of recent action log entries (staff only)."""
    import json
    import humanize

    # Get latest 20 entries
    logs = models.ActionLog.objects.order_by('-timestamp')[:length]

    # Format the logs for display
    formatted_logs = []
    for log in logs:
        formatted_details = None
        if log.details:
            formatted_details = json.dumps(log.details, indent=2, ensure_ascii=False)

        formatted_logs.append({
            'timestamp': log.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'natural_time': humanize.naturaltime(log.timestamp),
            'user': log.user,
            'action': log.get_action_display(),
            'task_id_ref': log.task_id_ref,
            'original_name': log.task.original_name if log.task is not None else None,
            'details': formatted_details,
        })

    context = {
        'logs': formatted_logs,
    }

    return TemplateResponse(request, 'action_log.html', context=context)
