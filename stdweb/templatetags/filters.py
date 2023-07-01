from django import template
from django.template.defaultfilters import stringfilter

register = template.Library()

@register.filter
def header_to_string(header):
    return header.tostring('\n')
