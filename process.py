#!/usr/bin/env python3

import os, glob, shutil
import argparse
import json
import numpy as np

from stdpipe.utils import file_write, file_read

from stdweb import settings # So that we have access to site-specific settings
from stdweb import processing


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


def parse_kw(kw=None):
    config = {}

    if kw is not None:
        for key in kw:
            value = kw[key]

            # Booleans
            if value in ['True']:
                config[key] = True
            elif value in ['False']:
                config[key] = False

            # Everything else
            else:
                try:
                    # Integers
                    config[key] = int(value)
                except:
                    try:
                        # Floats
                        config[key] = float(value)
                    except:
                        config[key] = value

    return config


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        usage="usage: %(prog)s [options] filename ...",
    )

    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Enable verbose output')

    # Processing steps
    parser.add_argument('--cleanup', dest='do_cleanup', action='store_true', help='Run Cleanup task')
    parser.add_argument('--inspect', dest='do_inspect', action='store_true', help='Run Inspect task')
    parser.add_argument('--photometry', dest='do_photometry', action='store_true', help='Run Photometry task')
    parser.add_argument('--simple-transients', dest='do_simple_transients', action='store_true', help='Run Simple Transients Detection task')
    parser.add_argument('--subtract', dest='do_subtraction', action='store_true', help='Run Template Subtraction task')

    # Config values as key-value pairs
    parser.add_argument("-c", "--config", metavar="KEY=VALUE", action=store_kw, nargs=1, dest="config", help="Initial parameters for the task")

    # Explicit shortcuts for most-used config parameters
    parser.add_argument("--gain", dest="gain", type=float, help="Detector gain (e/ADU)")
    parser.add_argument("--saturation", dest="saturation", type=float, help="Saturation level (ADU)")

    parser.add_argument('--preset', dest='preset', action='store', help='Read configuration preset from file', default=None)

    options, filenames = parser.parse_known_args()

    log = print if options.verbose else lambda *args,**kwargs: None

    # Loop over files
    for filename in filenames:
        basepath = os.path.split(filename)[0]
        configname = os.path.join(basepath, '.config')

        config = {}

        # Read existing config
        if os.path.exists(configname):
            log('Reading config from', configname)
            config.update(json.loads(file_read(configname)))

        # Cleanup
        if options.do_cleanup:
            log('Cleaning up', basepath)
            for path in glob.glob(os.path.join(basepath, '*')):
                if os.path.split(path)[-1] != 'image.fits':
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.unlink(path)
            config = {}

        # If preset is specified, add it on top of the config we read
        if options.preset and os.path.exists(options.preset):
            log('Reading preset configuration from', options.preset)
            config.update(json.loads(file_read(options.preset)))

        # Now apply config params from command line
        config.update(parse_kw(options.config))

        # Explicit params override everything else
        if options.gain is not None:
            config['gain'] = options.gain
        if options.saturation is not None:
            config['saturation'] = options.saturation
        log('Config:', config)

        # Inspect
        if options.do_inspect:
            log('\n-- Inspection --')
            processing.inspect_image(filename, config, verbose=options.verbose)

        # Photometry
        if options.do_photometry:
            log('\n-- Photometry --')
            processing.photometry_image(filename, config, verbose=options.verbose)

        # Simple Transients Detection
        if options.do_simple_transients:
            log('\n-- Simple Transients Detection --')
            processing.transients_simple_image(filename, config, verbose=options.verbose)

        # Template Subtraction
        if options.do_subtraction:
            log('\n-- Template Subtraction --')
            processing.subtract_image(filename, config, verbose=options.verbose)

        # Store the config
        log('Writing config to', configname)
        file_write(configname, json.dumps(config, indent=4, sort_keys=False, cls=NumpyEncoder))
