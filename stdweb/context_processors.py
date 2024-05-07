from django.conf import settings


def expose_settings(request):
    return {
        'settings': settings,
    }
