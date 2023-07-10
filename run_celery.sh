#!/bin/sh

watchmedo auto-restart -d stdweb -p '*.py' --ignore-patterns="*/.*" -- python3 -m celery -- -A stdweb worker --loglevel=info
