# STDWeb - Simple Transient Detection for the Web

This is a simple web-based tool for a quick-look photometry and transient detection in astronomical images. It is based on [STDPipe](https://github.com/karpov-sv/stdpipe) library and tries to implement a self-consistent and mostly automatic data analysis workflow from its routines.

It currently allows you to:

- Upload your own FITS files, or analyze some files already on the server
- Do basic pre-processing and masking
- Detect objects in the image and do astrometric calibration, either by blind mathcing or refining existing solution
- Photometrically calibrate the image using one of supported reference catalogues
- Subtract either user-provided or automatically downloaded template images
- Do forced photometry for a specified target in either original or difference image
- Do (experimental) transient detection in difference image

If you want to better understand the routines used for it, please consult [STDPipe documentation](https://stdpipe.readthedocs.io/) and [example notebooks](https://github.com/karpov-sv/stdpipe/tree/master/notebooks), as well as the [paper describing it](https://ojs.cvut.cz/ojs/index.php/ap/article/view/9969)

# Installation

The main requirement for *STDWeb* is obviously *STDPipe* which is available at https://github.com/karpov-sv/stdpipe. Please refer to its documentation for installation of its binary dependencies like *SExtractor*, *HOTPANTS* and *Astrometry.Net*.

Then, deployment of *STDWeb* requires cloning of its code:
```
git clone https://github.com/karpov-sv/stdweb.git
```
and then installing its dependencies (you better do it in a dedicated environment):
```
cd stdweb
pip install -r requirements.txt
```

Another run-time dependence is working *Redis* instance, that may be easily installed like
```
sudo apt install redis-server
```
or
```
conda install redis
```

# Deployment

Running *STDWeb* requires running instance of *Redis* - it may be started either system-wide through normal system means, or just by running 
```
redis-server
```
in a separate terminal.

Then, the actual data processing backend is based on *Celery* and may be started like 
```
python -m celery -A stdweb worker --loglevel=info
```
There is a helper script, `./run_celery.sh`, that runs it in a way so that it is automatically restarted when source files change (e.g. if you are doing some development on the processing code):
The script is based on [watchdog](https://github.com/gorakhargosh/watchdog) package that may be installed separately as 
```
pip install watchdog
```
or 
```
conda install watchdog
```

The rest is deployed like any other *Django* based app (see the [official documentation](https://docs.djangoproject.com/en/5.0/howto/deployment/)), or just by running 
```
python manage.py runserver
```

# Run-time configuration

To initially create a super-user that may access admin panel and do everything else in the app:
```
python manage.py createsuperuser
```

The app expects that all its site-specific settings are set through an `.env` file in the project root. Otherwise, some defaults will be used. 
Example of that file with various options is given below:
```python
SECRET_KEY = 'your django secret key goes here'

DEBUG = True

DATA_PATH = /opt/stdweb/data/
TASKS_PATH = /opt/stdweb/tasks/

# STDPipe settings

STDPIPE_HOTPANTS=/usr/local/bin/hotpants
STDPIPE_SOLVE_FIELD=/usr/local/astrometry/bin/solve-field
```

First line defines your *Django* secret key. You may generate it e.g. like that:
```
python -c "from django.core.management import utils; print(utils.get_random_secret_key())"
```

Second line defines whether the Django [debug mode](https://docs.djangoproject.com/en/5.0/ref/settings/#debug) will be enabled in the app. By default it will be disabled, but if you are doing some debugging - enabling it may significantly help.

Next two lines define the absolute paths for the root folder that will be accessible through the app built-in file browser, and the folder that will hold all files related to data analysis tasks (uploaded images, processing results, caches etc).

The rest of options are completely optional, and may be used to override various paths for binaries that *STDPipe* will run. If not specified, the code will try to locate them in standard paths. 
Complete list of these options may be seen at the very end of `stdweb/settings.py` file, and is given below:
```python
# Settings for STDPipe

# Temporary folder
STDPIPE_TMPDIR = 
# Path to Astrometry.Net executable
STDPIPE_SOLVE_FIELD = 
# Path to Astrometry.Net config
STDPIPE_SOLVE_FIELD_CONFIG = 
# Path to SExtractor executable
STDPIPE_SEXTRACTOR = 
# Path to SCAMP executable
STDPIPE_SCAMP = 
# Path to PSFEx executable
STDPIPE_PSFEX = 
# Path to PSFEx executable
STDPIPE_HOTPANTS = 
# Path to SWarp executable
STDPIPE_SWARP = 
# Path to store PS1 download cache (if not set, use task-local cache)
STDPIPE_PS1_CACHE = 

# SkyPortal API token
SKYPORTAL_TOKEN = 
```

# Referencing & Attribution
STDWeb is being developed by Sergey Karpov, and was originally created as part of [GRANDMA](https://grandma.ijclab.in2p3.fr) project.

If you STDWeb in your work, or use results from STDWeb in a publication, please cite:
> Karpov, S. (2025).
> **STDweb: simple transient detection pipeline for the web**. _Acta Polytechnica_, 65(1), 50-64. 
> https://doi.org/10.14311/AP.2025.65.0050
> 
