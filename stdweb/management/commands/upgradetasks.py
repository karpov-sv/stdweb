from django.core.management.base import BaseCommand

from stdweb import models
from stdweb import processing

from stdpipe import astrometry

import os
import sys

from astropy.io import fits
from astropy.wcs import WCS
from mocpy import MOC

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

class Command(BaseCommand):
    help = 'Upgrade older tasks'

    def handle(self, *args, **options):
        tasks = models.Task.objects.all()

        # Determine if we should show progress bar
        use_progressbar = tqdm is not None and sys.stdout.isatty()

        # Wrap tasks iterator with tqdm if interactive
        if use_progressbar:
            tasks = tqdm(tasks, desc='Upgrading tasks', unit='task')

        upgraded_count = 0
        error_count = 0

        for task in tasks:
            filename = os.path.join(task.path(), 'image.fits')

            # Missing ra/dec/radius/moc
            if task.ra is None or task.dec is None or task.radius is None or task.moc is None:
                try:
                    header = fits.getheader(filename, -1)

                    processing.fix_header(header)
                    wcs = processing.get_wcs(filename, header=header, verbose=False)

                    if not wcs or not wcs.is_celestial:
                        continue

                    w,h = header['NAXIS1'], header['NAXIS2']
                    ra0,dec0,sr0 = astrometry.get_frame_center(wcs=wcs, shape=(h, w))
                    pixscale = astrometry.get_pixscale(wcs=wcs)

                    task.ra = ra0
                    task.dec = dec0
                    task.radius = sr0

                    if task.config:
                        task.config['pixscale'] = pixscale

                        task.config.pop('field_ra')
                        task.config.pop('field_dec')
                        task.config.pop('field_sr')

                    task.moc = processing.get_moc_for_wcs(wcs, (h, w))

                    task.save()

                    upgraded_count += 1

                    info_msg = f"Field center for task {task.id} is at {ra0:.4f} {dec0:.4f} sr {sr0:.3f} pixscale {pixscale*3600:.2f} arcsec/pix"

                    if use_progressbar:
                        tqdm.write(info_msg)
                        tasks.set_postfix({'upgraded': upgraded_count})
                    else:
                        print(info_msg)

                except Exception as e:
                    error_count += 1
                    if use_progressbar:
                        tqdm.write(f"Error upgrading task {task.id}: {str(e)}")
                    else:
                        import traceback
                        traceback.print_exc()
                        print(f"Error upgrading task {task.id}")

        # Final summary
        if upgraded_count > 0 or error_count > 0:
            self.stdout.write(self.style.SUCCESS(
                f'\nUpgrade complete: {upgraded_count} tasks upgraded, {error_count} errors'
            ))
