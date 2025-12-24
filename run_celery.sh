#!/bin/sh

# Avoid crashing on newer MacOS
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

watchmedo auto-restart -d stdweb -p '*.py' --ignore-patterns="*/.*" -- python3 -m celery -- -A stdweb worker --loglevel=info
