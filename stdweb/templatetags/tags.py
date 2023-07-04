from django import template
from django.template.defaultfilters import stringfilter

import os

from astropy.io import fits

register = template.Library()


@register.simple_tag
def task_file_contents(task, filename):
    path = os.path.join(task.path(), filename)

    contents = ""

    try:
        with open(path, "r") as f:
            contents = f.read()
    except:
        pass

    return contents


@register.simple_tag
def task_fits_header(task, filename):
    path = os.path.join(task.path(), filename)

    contents = ""

    try:
        header = fits.getheader(path)
        contents = header.tostring('\n')
    except:
        context = "Cannot get FITS header from " + filename

    return contents


@register.simple_tag
def make_uuid():
    return str(uuid.uuid1())
