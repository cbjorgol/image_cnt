from seal_utils import SeaLionData
import os
from collections import namedtuple

SOURCEDIR = os.path.join('..', 'input')

DATADIR = os.path.join('..', 'output')

VERBOSITY = namedtuple('VERBOSITY', ['QUITE', 'NORMAL', 'VERBOSE', 'DEBUG'])(0, 1, 2, 3)

SeaLionCoord = namedtuple('SeaLionCoord', ['tid', 'cls', 'x', 'y'])


# Count sea lion dots and compare to truth from train.csv
sld = SeaLionData(SOURCEDIR, DATADIR, VERBOSITY.NORMAL)
for tid in sld.train_ids[:10]:
    coord = sld.coords(tid)
    sld.save_sea_lion_chunks(coord, '../output', chunksize=128)