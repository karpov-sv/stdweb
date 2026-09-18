FROM python:3.12-slim

# SExtractor and SCAMP cover the "inspect" + "photometry" pipeline steps (the
# only ones grandma_tools' comment_watcher.py drives). PSFEx/SWarp/HOTPANTS/
# Astrometry.Net (PSF photometry, stacking, subtraction, blind WCS solving)
# are intentionally omitted here -- see doc/installation.rst for those.
RUN apt-get update && apt-get install -y --no-install-recommends \
        sextractor \
        scamp \
        gcc \
        libcfitsio-dev \
        git \
        redis-server \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# stdpipe is a sibling checkout in local dev (see README); install it inside
# the image the same way so processing.* / stdpipe.* imports resolve.
RUN git clone --depth 1 https://github.com/karpov-sv/stdpipe.git /app/stdpipe \
    && pip install --no-cache-dir -e /app/stdpipe

COPY . .

ENV DATA_PATH=/app/data/ \
    TASKS_PATH=/app/tasks/ \
    STDPIPE_TMPDIR=/tmp/stdpipe
RUN mkdir -p /app/data /app/tasks /tmp/stdpipe \
    && chmod +x docker-entrypoint.sh

EXPOSE 8000
CMD ["./docker-entrypoint.sh"]
