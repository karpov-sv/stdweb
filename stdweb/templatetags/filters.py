from django import template
from django.template.defaultfilters import stringfilter
from django.utils.safestring import mark_safe

import datetime
import uuid
import humanize

import numpy as np

register = template.Library()


@register.filter
def header_to_string(header):
    return header.tostring('\n').strip()


@register.filter
def time_from_unix(unix):
    if unix:
        return datetime.datetime.utcfromtimestamp(int(unix))
    else:
        return None


@register.filter
def user(user):
    """
    Format User object in a readable way
    """
    result = ""

    if user.first_name or user.last_name:
        result = user.first_name + " " + user.last_name
    else:
        result = user.username

    return result


@register.filter
def timestamp(time):
    """
    Format user-readable timestamp
    """

    return time.strftime('%Y-%m-%d %H:%M:%S UTC')


@register.filter
def unix(time):
    """
    Return UNIX timestamp
    """

    return time.timestamp()


@register.filter
def make_uuid(x):
    return str(uuid.uuid1())


@register.filter
def naturalsize(size):
    return humanize.naturalsize(size, binary=True)


@register.filter
def to_sexadecimal(value, plus=False):
    avalue = np.abs(value)
    deg = int(np.floor(avalue))
    min = int(np.floor(60.0*(avalue - deg)))
    sec = 3600.0*(avalue - deg - 1.0*min/60)

    string = '%02d %02d %04.1f' % (deg, min, sec)

    if value < 0:
        string = '-' + string
    elif plus:
        string = '+' + string

    return string


@register.filter
def to_sexadecimal_plus(value):
    return to_sexadecimal(value, plus=True)


@register.filter
def to_sexadecimal_hours(value):
    return to_sexadecimal(value*1.0/15)


@register.filter
def list_extract(value, key):
    return [_[key] for _ in value]


from astropy.table import Table

@register.filter
def show_table(table):
    # FIXME: it repeats the code from tags.py, should be merged together!
    contents = ""

    # Ensure it is a Table
    table = Table(table)

    try:
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
        import traceback
        traceback.print_exc()
        pass

    return mark_safe(contents)


@register.filter
def multiply(value, arg):
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return ''


@register.filter
def subtract(value, arg):
    try:
        return float(value) - float(arg)
    except (ValueError, TypeError):
        return ''


@register.filter
def divide(value, arg):
    try:
        return float(value) / float(arg) if float(arg) != 0 else ''
    except (ValueError, TypeError):
        return ''
