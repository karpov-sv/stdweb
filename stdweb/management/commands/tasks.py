from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from django.conf import settings

import os, shutil

from stdweb import models

import argparse

import celery

from ... import celery_tasks

class store_kw(argparse.Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on append to a dictionary.
    """

    def __call__(self, parser, args, values, option_string=None):
        # try:
        #     d = dict(map(lambda x: x.split('='), values))
        # except ValueError as ex:
        #     raise argparse.ArgumentError(self, f"Could not parse argument \"{values}\" as k1=v1 k2=v2 ... format")

        # setattr(args, self.dest, d)
        assert(len(values) == 1)
        try:
            (k, v) = values[0].split("=", 2)
        except ValueError as ex:
            raise argparse.ArgumentError(self, f"could not parse argument \"{values[0]}\" as k=v format")
        d = getattr(args, self.dest) or {}
        d[k] = v
        setattr(args, self.dest, d)

class Command(BaseCommand):
    help = 'Manage image processing tasks'

    def add_arguments(self, parser):
        # Named arguments
        parser.add_argument("-l", "--list", action="store_true", dest='list', help="List all tasks")
        parser.add_argument("-s", "--sort", action="store", dest='sort', default="modified", help="Sort field")

        parser.add_argument("-i", "--import", action="store_true", dest='import', help="Import FITS files")

        parser.add_argument("-d", "--delete", action="store_true", dest='delete', help="Delete tasks")

        parser.add_argument("--skyportal", action="store_true", dest='skyportal', help="Prepare photometry for exporting to SkyPortal")

        # Run specific steps
        parser.add_argument("-r", "--run", action="store", dest='run', default=None, help="Run processing steps")

        # Config values as key-value pairs
        parser.add_argument("-c", "--config", metavar="KEY=VALUE", action=store_kw, nargs=1, dest="config", help="Initial parameters for the task")

        # Positional arguments
        parser.add_argument("names", nargs="*", type=str)

    def handle(self, *args, **options):
        # Work inside project root
        os.chdir(settings.BASE_DIR)

        if options['list']:
            tasks = models.Task.objects.order_by(f"-{options['sort']}")

            for task in tasks:
                print(f"{task.id} {task.created.strftime('%Y-%m-%d %H:%M:%S')} - {task.modified.strftime('%Y-%m-%d %H:%M:%S')} - {task.user.username} - {task.state} - {task.original_name}")

        elif options['import']:
            for filename in options['names']:
                if os.path.exists(filename):
                    task = models.Task(original_name=os.path.split(filename)[-1])
                    task.user = User.objects.filter(is_staff=True).order_by('id').first() # FIXME: ???
                    task.save() # to populate task.id

                    try:
                        os.makedirs(task.path())
                    except OSError:
                        pass

                    shutil.copyfile(filename, os.path.join(task.path(), 'image.fits'))

                    if options['config']:
                        # Pre-process the config entries
                        config = {}
                        for key in options['config']:
                            value = options['config'][key]

                            # Booleans
                            if value in ['True']:
                                value = True
                            elif value in ['False']:
                                value = False

                            try:
                                config[key] = float(value)
                            except:
                                config[key] = value

                        task.config.update(config)

                    task.state = 'imported'
                    task.save()

                    print(f"{task.original_name} imported as task {task.id}")

                    if options['run'] is not None:
                        run_task(task, options['run'])

        elif options['delete']:
            for name in options['names']:
                id = int(name)

                print(f"Deleting task {id}")

                task = models.Task.objects.get(id=id)

                if task:
                    task.delete()

        elif options['skyportal']:
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

                run_task(task, options['run'])

def run_task(task, run):
    todo = []

    for step in run.split(','):
        print(f"Running {step} for task {task.id}")

        if step == 'inspect':
            todo.append(celery_tasks.task_inspect.subtask(args=[task.id], immutable=True))
        elif step == 'photometry':
            todo.append(celery_tasks.task_photometry.subtask(args=[task.id], immutable=True))
        else:
            print(f"Unknown step: {step}")

    if todo:
        task.celery_id = celery.chain(todo).apply_async()
        task.state = 'running'
        task.save()
