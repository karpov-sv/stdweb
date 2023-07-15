from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from django.conf import settings

import os, shutil

from stdweb import models

class Command(BaseCommand):
    help = 'Manage image processing tasks'

    def add_arguments(self, parser):
        # Named arguments
        parser.add_argument("-l", "--list", action="store_true", dest='list', help="List all tasks")
        parser.add_argument("-s", "--sort", action="store", dest='sort', default="modified", help="Sort field")

        parser.add_argument("-i", "--import", action="store_true", dest='import', help="Import FITS files")

        parser.add_argument("-d", "--delete", action="store_true", dest='delete', help="Delete tasks")

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

                    task.state = 'imported'
                    task.save()

                    print(f"{task.original_name} imported as task {task.id}")

        elif options['delete']:
            for name in options['names']:
                id = int(name)

                print(f"Deleting task {id}")

                task = models.Task.objects.get(id=id)

                if task:
                    task.delete()
