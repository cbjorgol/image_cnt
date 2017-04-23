from seal_utils import SeaLionData
import os
from collections import namedtuple
from PIL import Image
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import threading


SOURCEDIR = os.path.join('..', 'input')

DATADIR = os.path.join('..', 'cropped_nonseals')

VERBOSITY = namedtuple('VERBOSITY', ['QUITE', 'NORMAL', 'VERBOSE', 'DEBUG'])(0, 1, 2, 3)

SeaLionCoord = namedtuple('SeaLionCoord', ['tid', 'cls', 'x', 'y'])


def get_noseal_locs(sld, tid, n_images):

    coords = sld.coords(tid)

    y_max = sld.src_img.shape[0]
    x_max = sld.src_img.shape[1]

    noseal_coords = []
    for i in range(n_images):
        y = np.random.randint(64, y_max-64)
        x = np.random.randint(64, x_max-64)


        noseal = True
        for coord in coords:
            if sqrt((y - coord.y)**2 + (x - coord.x)**2) < 36:
                noseal = False

        if noseal:
            noseal_coords.append((x, y))

    return noseal_coords



def show_nonseal_images(noseal_coords, start):
    fig, ax = plt.subplots(5, 5)

    n = start * 5 * 5
    for i in range(5):
        for j in range(5):
            loc = noseal_coords[n]
            img = sld.src_img[loc[1] - 64:loc[1] + 64, loc[0] - 64:loc[0] + 64, :]
            if img.shape[0] != 128 or img.shape[1] != 128:
                print("IMAGE NOT CORRECT SIZE!", img.shape)

            ax[i,j].imshow(img)
            n+=1
    plt.show()


def get_chunks(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def save_noseal_chunks(sld, tid, coords, folder, chunksize=128):

    cls = 6
    last_tid = -1

    for x, y in coords:
        if tid != last_tid:
            img = sld.load_train_image(tid, border=chunksize // 2, mask=True)
            last_tid = tid

        fn = 'chunk_{tid}_{cls}_{x}_{y}_{size}.png'.format(size=chunksize, tid=tid, cls=cls, x=x, y=y)
        fn = os.path.join(folder, fn)
        try:
            Image.fromarray(img[y:y + chunksize, x:x + chunksize, :]).save(fn)
        except:
            print("ERROR SAVING!!!!!", tid, (y+chunksize), (x+chunksize), img.shape)
    return len(coords)

def crop_nonseals(tids):
    sld = SeaLionData(SOURCEDIR, DATADIR, VERBOSITY.QUITE)
    tid = 0
    n_imgs = 0
    for tid in tids:
        noseal_coords = get_noseal_locs(sld, tid, 150)
        n_imgs = save_noseal_chunks(sld, tid, noseal_coords, DATADIR)

        print(threading.current_thread().getName(), tid, "# of images saved: ", n_imgs)

def multiprocess_crop_nonseals(nthreads, **kwargs):

    _tmp_sld = SeaLionData(SOURCEDIR, DATADIR, VERBOSITY.QUITE)
    chunks = get_chunks(_tmp_sld.train_ids, nthreads)

    threads = []
    for i in range(nthreads):
        t = threading.Thread(target=crop_nonseals, kwargs=dict(tids=chunks[i]))
        threads.append(t)
        t.start()

if __name__ == '__main__':

    for train_dir in ['TrainDotted', 'Train', 'Test']:
        assert os.path.exists(os.path.join(SOURCEDIR, 'TrainDotted'))

    if not os.path.exists(DATADIR):
        print("Creating DATADIR:", DATADIR)
        os.mkdir(DATADIR)

    multiprocess_crop_nonseals(nthreads=32)