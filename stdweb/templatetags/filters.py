from django import template
from django.template.defaultfilters import stringfilter

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
