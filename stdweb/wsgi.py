"""
WSGI config for stdweb project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""

import os, sys
import locale

from django.core.wsgi import get_wsgi_application

this_path = os.path.split(os.path.abspath(__file__))[0]
this_uppath = os.path.abspath(os.path.join(this_path, '..'))

sys.path.append(this_uppath)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stdweb.settings')

application = get_wsgi_application()
