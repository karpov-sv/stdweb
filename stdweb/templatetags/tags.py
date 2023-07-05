from django import template
from django.template.defaultfilters import stringfilter

import os
import uuid

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
        contents = "Cannot get FITS header from " + filename

    return contents


@register.simple_tag
def make_uuid():
    return str(uuid.uuid1())


# Code borrowed from https://stackoverflow.com/a/3715794
@register.tag('make_list')
def make_list(parser, token):
    bits = token.contents.split()
    if len(bits) >= 4 and bits[-2] == "as":
        varname = bits[-1]
        items = bits[1:-2]
        return MakeListNode(items, varname)
    else:
        raise template.TemplateSyntaxError("%r expected format is 'item [item ...] as varname'" % bits[0])


class MakeListNode(template.Node):
    def __init__(self, items, varname):
        self.items = items
        self.varname = varname

    def render(self, context):
        items = map(template.Variable, self.items)
        context[self.varname] = [ i.resolve(context) for i in items ]
        return ""
