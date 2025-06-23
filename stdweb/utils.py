import argparse


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
            if value in ['True', 'true']:
                config[key] = True
            elif value in ['False', 'false']:
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
