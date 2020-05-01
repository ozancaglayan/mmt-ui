import re
import gzip
import lzma
import bz2

from pathlib import Path


def get_image_db(data_folder='data'):
    """Returns a mapping from test sets to ordered image filenames."""
    root = Path(data_folder)
    d = {}
    for test_set in root.glob('*.images'):
        img_names = []
        test_set_name = test_set.name.replace('.images', '')
        with open(test_set) as f:
            for line in f:
                img_names.append(test_set_name + '/' + line.strip())
        d[test_set_name] = img_names
    return d


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def fopen(filename):
    """gzip,bzip2,xz aware file opening function."""

    filename = str(Path(filename).expanduser())
    if filename.endswith('.gz'):
        return gzip.open(filename, 'rt')
    elif filename.endswith('.bz2'):
        return bz2.open(filename, 'rt')
    elif filename.endswith(('.xz', '.lzma')):
        return lzma.open(filename, 'rt')
    else:
        # Plain text
        return open(filename, 'r')
