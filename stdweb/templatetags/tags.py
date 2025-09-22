from django import template
from django.template.defaultfilters import stringfilter
from django.utils.safestring import mark_safe
from django.utils.html import escape
from django.urls import reverse
from django.conf import settings

import os
import uuid
import re

from functools import partial

from astropy.io import fits
from astropy.table import Table

register = template.Library()

def task_file_link(m, task=None):
    name = m.group(2)
    url = reverse('task_view', kwargs={'id':task.id, 'path':name})

    return r"<a href='" + url + "'>" + name + r"</a>"

@register.simple_tag
def task_file_contents(task, filename, highlight=False):
    path = os.path.join(task.path(), filename)

    contents = ""

    try:
        with open(path, "r") as f:
            contents = f.read()
    except:
        pass

    contents = escape(contents)

    if highlight:
        # Highlight some patterns in the text
        contents = re.sub(r"^(----\s+(.+)+\s+----)$",
                          r"<span class='text-primary'>\1</span>",
                          contents, flags=re.MULTILINE)

        contents = re.sub(r"^(RuntimeError:)(.*)$",
                          r"\1<span class='text-danger fw-bold'>\2</span>",
                          contents, flags=re.MULTILINE)

        contents = re.sub(r"^(\S+Error:)(.*)$",
                          r"\1<span class='text-danger'>\2</span>",
                          contents, flags=re.MULTILINE)

        contents = re.sub(r"^(Warning:)(.*)$",
                          r"\1<span class='text-danger'>\2</span>",
                          contents, flags=re.MULTILINE)

        contents = re.sub(r"\b(file:((\w+/)?\w+\.\w+))\b",
                          partial(task_file_link, task=task),
                          contents, flags=re.MULTILINE)

    return mark_safe(contents)


@register.simple_tag
def task_file_table(task, filename):
    path = os.path.join(task.path(), filename)

    contents = ""

    try:
        table = Table.read(path)

        for col in table.itercols():
            if col.info.dtype.kind == 'f':
                if col.name in ['ra', 'dec', 'RAJ2000', 'DEJ2000']:
                    col.info.format = '.5f'
                elif col.name in ['x', 'y']:
                    col.info.format = '.2f'
                else:
                    col.info.format = '.4g'
            elif col.name in ['flags']:
                col.info.format = '#x'

        contents = "\n".join(table.pformat_all(
            html=True,
            tableclass="table table-sm table-bordered text-center",
            tableid='table_targets',
        ))
    except:
        pass

    return mark_safe(contents)


@register.simple_tag
def task_fits_header(task, filename):
    path = os.path.join(task.path(), filename)

    contents = ""

    try:
        header = fits.getheader(path, -1)
        contents = []

        for card in header.cards:
            cstr = str(card)

            if m := re.match(r'^((HISTORY)|(COMMENT|END))\b(.*)$', cstr):
                contents.append(
                    f"<span class='text-secondary'>{m[1]}</span>"
                    f"<span class='text-info'>{m[4]}</span>"
                )
            elif m := re.match(r'^([^=]+)=(\s*(\'.*?\'|\S+)\s*)(/.*)?$', cstr):
                contents.append(
                    f"<span class='text-primary'>{m[1]}</span>"
                    f"<span class='text-secondary'>=</span>"
                    f"<span class='text-body-emphasis'>{m[2]}</span>"
                    f"<span class='text-info'>{m[4] or ''}</span>"
                )
            elif cstr:
                contents.append(cstr)

        contents = "\n".join(contents)

    except:
        contents = "Cannot get FITS header from " + filename

    return mark_safe(contents)


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


@register.simple_tag
def free_disk_space():
    s = os.statvfs(settings.TASKS_PATH)
    return s.f_bavail*s.f_frsize
