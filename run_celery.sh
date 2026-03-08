#!/bin/sh

# Avoid crashing on newer MacOS
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Force single-threaded BLAS to avoid segfaults after fork()
# macOS Accelerate uses GCD which is not fork-safe
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

watchmedo auto-restart -d stdweb --recursive -p '**/*.py' -- python3 -m celery -A stdweb worker --loglevel=info
