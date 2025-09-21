from django.core.management.base import BaseCommand

from stdweb import models
from stdweb import processing

from stdpipe import astrometry

import os

from astropy.io import fits

class Command(BaseCommand):
    help = 'Upgrade older tasks'

    def handle(self, *args, **options):
        tasks = models.Task.objects.all()

        for task in tasks:
            filename = os.path.join(task.path(), 'image.fits')

            # Missing field center coordinates
            if task.config and not task.config.get('pixscale'):
                try:
                    header = fits.getheader(filename, -1)

                    processing.fix_header(header)

                    wcs = processing.get_wcs(filename, header=header, verbose=False)

                    if wcs and wcs.is_celestial:
                        ra0,dec0,sr0 = astrometry.get_frame_center(
                            wcs=wcs, width=header['NAXIS1'], height=header['NAXIS2']
                        )
                        pixscale = astrometry.get_pixscale(wcs=wcs)

                        task.config['field_ra'] = ra0
                        task.config['field_dec'] = dec0
                        task.config['field_sr'] = sr0
                        task.config['pixscale'] = pixscale

                        print(f"Field center for task {task.id} is at {ra0:.4f} {dec0:.4f} sr {sr0:.3f} pixscale {pixscale*3600:.2f} arcsec/pix")

                        task.save()

                except:
                    import traceback
                    traceback.print_exc()
                    print(f"Error upgrading task {task.id}")
