from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from django.conf import settings

import os, shutil

from stdweb import models

import argparse
import json

import celery

from ... import celery_tasks
from ...utils import store_kw, parse_kw


class Command(BaseCommand):
    help = 'Manage image processing tasks'

    def add_arguments(self, parser):
        # Named arguments
        parser.add_argument("-l", "--list", action="store_true", dest='do_list', help="List all tasks")
        parser.add_argument("-s", "--sort", action="store", dest='list_sort', default="modified", help="Sort field")

        parser.add_argument("-i", "--import", action="store_true", dest='do_import', help="Import FITS files")

        parser.add_argument("-d", "--delete", action="store_true", dest='do_delete', help="Delete tasks")

        parser.add_argument("--skyportal", action="store_true", dest='do_skyportal', help="Prepare photometry for exporting to SkyPortal")

        # Run specific steps
        parser.add_argument("-r", "--run", action="append", dest='run', default=None, help="Run processing steps for the task")

        # Config values as key-value pairs
        parser.add_argument("-c", "--config", metavar="KEY=VALUE", action=store_kw, nargs=1, dest="config", help="Set initial parameters for the task on importing")

        # Configuration preset
        parser.add_argument('--preset', dest='preset', action='store', help='Apply configuration preset', default=None)

        # Positional arguments
        parser.add_argument("names", nargs="*", type=str)

    def handle(self, *args, **options):
        # Work inside project root
        os.chdir(settings.BASE_DIR)

        if options['do_list']:
            tasks = models.Task.objects.order_by(f"-{options['list_sort']}")

            for task in tasks:
                print(f"{task.id} {task.created.strftime('%Y-%m-%d %H:%M:%S')} - {task.modified.strftime('%Y-%m-%d %H:%M:%S')} - {task.user.username} - {task.state} - {task.original_name}")


        elif options['do_import']:
            for filename in options['names']:
                if os.path.exists(filename):
                    task = models.Task(original_name=os.path.split(filename)[-1])
                    task.user = User.objects.filter(is_staff=True).order_by('id').first() # FIXME: ???
                    task.save() # to populate task.id

                    print(f"{task.original_name} imported as task {task.id}")

                    try:
                        os.makedirs(task.path())
                    except OSError:
                        pass

                    shutil.copyfile(filename, os.path.join(task.path(), 'image.fits'))

                    config = {}

                    # If preset is specified, add it
                    if options['preset']:
                        preset = models.Preset.objects.filter(name=options['preset']).first()
                        if preset is not None:
                            print(f"Applying configuration preset {preset.name}")
                            config.update(preset.config)

                        elif os.path.exists(options['preset']):
                            print(f"Applying configuration preset from file {options['preset']}")
                            config.update(json.loads(file_read(options['preset'])))

                    # Now apply config params from command line
                    if options['config']:
                        config.update(parse_kw(options['config']))

                    task.config.update(config)

                    task.state = 'imported'
                    task.save()

                    if options['run'] is not None:
                        celery_tasks.run_task_steps(task, options['run'])


        elif options['do_delete']:
            for name in options['names']:
                id = int(name)

                print(f"Deleting task {id}")

                task = models.Task.objects.get(id=id)

                if task:
                    task.delete()


        elif options['do_skyportal']:
            from astropy.table import Table
            from stdpipe import cutouts

            # Header
            print(f"mjd,mag,magerr,limiting_mag,magsys,filter")

            for name in options['names']:
                for sname in ['sub_target.vot', 'target.vot']:
                    filename = f"tasks/{name}/{sname}"
                    if os.path.exists(filename):
                        break

                tobj = Table.read(filename)
                cutout = cutouts.load_cutout(filename.replace('.vot', '.cutout'))
                meta = cutout['meta']

                # Data
                for row in tobj:
                    fname = row['mag_filter_name']

                    magsys = 'vega' if fname in ['Umag', 'Bmag', 'Vmag', 'Rmag', 'Imag'] else 'ab'
                    fname = {
                        'Umag': 'bessellu',
                        'Bmag': 'bessellb',
                        'Vmag': 'bessellv',
                        'Rmag': 'bessellr',
                        'Imag': 'besselli',
                        'umag': 'sdssu',
                        'gmag': 'sdssg',
                        'rmag': 'sdssr',
                        'imag': 'sdssi',
                        'zmag': 'sdssz',
                    }.get(fname, fname)

                    print(f"{meta['time'].mjd},{row['mag_calib']:.3f},{row['mag_calib_err']:.3f},{row['mag_limit']:.3f},{magsys},{fname}")


        elif options['run']:
            for name in options['names']:
                id = int(name)
                task = models.Task.objects.get(id=id)

                celery_tasks.run_task_steps(task, options['run'])
