# STDWeb Docker

`STDWeb` is the web version of [STDPipe](https://github.com/karpov-sv/stdpipe), packaged with Docker.

## Prerequisites

- Docker Engine + Docker Compose plugin (`docker compose`)
- At least 8 GB RAM recommended for build/runtime
- Astrometry index files downloaded from [data.astrometry.net](http://data.astrometry.net/)

## Quick Start

1. Clone this repo:

```bash
git clone --depth=1 https://github.com/Astro-Lee/stdweb-docker.git
cd stdweb-docker
```

2. Prepare local mount directories:

```bash
mkdir -p data tasks notebooks index-data
```

3. Put astrometry index files into `./index-data` on host.
- You can refer to `download_index.sh` to fetch index files.
- 
- Container mapping is already configured in `docker-compose.yaml`:
  - `./index-data:/usr/local/astrometry/data/`


4. Build and start:

```bash
docker compose up -d --build
```

5. Check services:

```bash
docker compose ps
docker logs -f stdweb
```

If everything is healthy:
- STDWeb: `http://localhost:8123`
- JupyterLab: `http://localhost:8124`

## First-Time Setup (inside container)

```bash
docker exec -it stdweb bash
```

1. Create admin user:

```bash
/opt/conda3/bin/python manage.py createsuperuser
```

2. Generate and set a production secret key:

```bash
/opt/conda3/bin/python -c "from django.core.management import utils; print(utils.get_random_secret_key())"
```

Edit `/opt/stdweb/.env` and replace:

```bash
SECRET_KEY = 'your django secret key goes here'
```

3. If using reverse proxy / domain, set trusted origins in `stdweb/settings.py`:

```python
CSRF_TRUSTED_ORIGINS = ['https://example.domain.com']
```

Then restart container:

```bash
docker compose restart stdweb
```

## Notes

- `docker-compose.yaml` starts `redis` and `stdweb`.
- `start.sh` launches:
  - JupyterLab (port `8888` in container)
  - Celery worker (with redis broker)
  - Django dev server (port `8000` in container)
- Host port mapping:
  - `8123 -> 8000`
  - `8124 -> 8888`

## Update Existing Deployment

Inside container:

```bash
cd /opt/stdpipe && git pull && /opt/conda3/bin/python -m pip install -e .
cd /opt/stdweb && git pull && /opt/conda3/bin/pip install -r requirements.txt
```

After update:

```bash
docker compose restart stdweb
```
