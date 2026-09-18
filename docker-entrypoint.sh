#!/bin/sh
set -e

# stdweb's settings.py hardcodes CELERY_BROKER_URL/CELERY_RESULT_BACKEND to
# redis://localhost/ (no env override), so redis, the celery worker, and the
# Django dev server all have to share this one container's network namespace
# -- same layout as local (non-Docker) development.
redis-server --daemonize yes

python manage.py migrate --noinput

celery -A stdweb worker --loglevel=info &

exec python manage.py runserver 0.0.0.0:8000
