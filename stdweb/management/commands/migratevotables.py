from django.core.management.base import BaseCommand

from stdweb import models
from stdweb import processing

import os
import sys

from astropy.io import fits
from astropy.table import Table

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

class Command(BaseCommand):
    help = 'Migrate task files from VOTable to Parquet'

    def handle(self, *args, **options):
        tasks = models.Task.objects.all()

        # Determine if we should show progress bar
        use_progressbar = tqdm is not None and sys.stdout.isatty()

        # Wrap tasks iterator with tqdm if interactive
        if use_progressbar:
            tasks = tqdm(tasks, desc='Migrating tasks', unit='task')

        total_converted = 0
        total_errors = 0

        for task in tasks:
            converted_files = []

            for filename in ['objects.vot', 'cat.vot']:
                fullname = os.path.join(task.path(), filename)
                newname = os.path.splitext(fullname)[0] + '.parquet'

                if not os.path.exists(fullname) or os.path.exists(newname):
                    continue

                try:
                    table = Table.read(fullname)
                    table.write(newname, overwrite=True)
                    converted_files.append(filename)
                    total_converted += 1

                except Exception as e:
                    total_errors += 1
                    if use_progressbar:
                        tqdm.write(f'Error migrating {filename} in task {task.id}: {str(e)}')
                    else:
                        import traceback
                        traceback.print_exc()
                        print(f"Error upgrading task {task.id}")

            # Update progress bar with what was converted
            if use_progressbar and converted_files:
                tasks.set_postfix({'converted': ', '.join(converted_files)})
            elif not use_progressbar and converted_files:
                print(f"Task {task.id}: converted {', '.join(converted_files)}")

        # Final summary
        if total_converted > 0 or total_errors > 0:
            self.stdout.write(self.style.SUCCESS(
                f'\nMigration complete: {total_converted} files converted, {total_errors} errors'
            ))
