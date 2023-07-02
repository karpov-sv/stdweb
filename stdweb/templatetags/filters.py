from django import template
from django.template.defaultfilters import stringfilter

import datetime

register = template.Library()

@register.filter
def header_to_string(header):
    return header.tostring('\n')

@register.filter
def time_from_unix(unix):
    if unix:
        return datetime.datetime.utcfromtimestamp(int(unix))
    else:
        return None
