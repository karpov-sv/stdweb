#!/bin/sh

watchmedo auto-restart -d stdweb -p '*.py' --ignore-patterns="*/.*" -- celery -A stdweb worker --loglevel=info
