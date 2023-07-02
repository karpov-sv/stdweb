#!/bin/sh

watchmedo auto-restart -d stdweb -p '*.py' -- celery -A stdweb worker --loglevel=info
