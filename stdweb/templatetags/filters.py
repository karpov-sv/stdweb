from django import template
from django.template.defaultfilters import stringfilter

import datetime
import uuid

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
