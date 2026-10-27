import argparse
import re


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


# key=value, key!=value, key<value etc, with value either a single word or a quoted string
_KEYWORD_RE = re.compile(r"""(?<!\S)([A-Za-z][A-Za-z0-9_]*)(<=|>=|!=|=|<|>)("[^"]*"|'[^']*'|\S+)(?!\S)""")


def parse_keyword_query(string):
    """
    Split the query string into keyword criteria and remaining free text.

    Returns (text, criteria), where criteria is a list of (key, op, value) tuples.
    Unquoted values are typed like in parse_kw(), quoted ones are kept as strings.
    """
    criteria = []

    for m in _KEYWORD_RE.finditer(string):
        key, op, value = m[1], m[2], m[3]

        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = parse_kw({key: value})[key]

        criteria.append((key, op, value))

    text = " ".join(_KEYWORD_RE.sub(" ", string).split())

    return text, criteria


def resolve_coordinates(string):
    ra,dec,sr = None, None, None

    if len(string):
        # Pair of decimal degrees
        m = re.search(r"^(\d+\.?\d*)\s+([+-]?\d+\.?\d*)(\s+(\d+\.?\d*))?$", string)
        if m:
            ra = float(m[1])
            dec = float(m[2])
            if m[4] is not None:
                sr = float(m[4])

        else:
            # HMS DMS
            m = re.search(
                r"^(\d{1,2})\s+(\d{1,2})\s+(\d{1,2}\.?\d*)\s+([+-])?\s*(\d{1,3})\s+(\d{1,2})\s+(\d{1,2}\.?\d*)(\s+(\d+\.?\d*))?$",
                string,
            ) or re.search(
                r"^(\d{1,2})[:h](\d{1,2})[:m](\d{1,2}\.?\d*)[s]?\s+([+-])?\s*(\d{1,3})[d:](\d{1,2})[m:](\d{1,2}\.?\d*)[s]?(\s+(\d+\.?\d*))?$",
                string,
            )
            if m:
                ra = ( float(m[1]) + float(m[2]) / 60 + float(m[3]) / 3600 ) * 15
                dec = float(m[5]) + float(m[6]) / 60 + float(m[7]) / 3600

                if m[4] == "-":
                    dec *= -1

                if m[9] is not None:
                    sr = float(m[9])

        # TODO: resolve object names?..

    return ra, dec, sr
