from seal_utils import SeaLionData
import os
from collections import namedtuple
import threading

SOURCEDIR = os.path.join('..', 'input')

DATADIR = os.path.join('..', 'cropped_seals')

VERBOSITY = namedtuple('VERBOSITY', ['QUITE', 'NORMAL', 'VERBOSE', 'DEBUG'])(0, 1, 2, 3)

SeaLionCoord = namedtuple('SeaLionCoord', ['tid', 'cls', 'x', 'y'])



def crop_seals(sld, tids):
    # Count sea lion dots and compare to truth from train.csv
    for tid in tids:
        print(threading.current_thread().getName(), tid,)
        coord = sld.coords(tid)
        sld.save_sea_lion_chunks(coord, DATADIR, chunksize=128)


def get_chunks(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def multiprocess_crop_seals(sld, nthreads=1):
    chunks = get_chunks(sld.train_ids, nthreads)

    threads = []
    for i in range(nthreads):
        t = threading.Thread(target=crop_seals, kwargs=dict(sld=sld, tids=chunks[i]))
        threads.append(t)
        t.start()

if __name__ == '__main__':

    for train_dir in ['TrainDotted', 'Train', 'Test']:
        assert os.path.exists(os.path.join(SOURCEDIR, 'TrainDotted'))

    if not os.path.exists(DATADIR):
        os.mkdir(DATADIR)

    sld = SeaLionData(SOURCEDIR, DATADIR, VERBOSITY.QUITE)
    multiprocess_crop_seals(sld=sld, nthreads=30)