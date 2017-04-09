from seal_utils import SeaLionData
import os
from collections import namedtuple
from PIL import Image
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt


SOURCEDIR = os.path.join('..', 'input')

DATADIR = os.path.join('..', 'output')

VERBOSITY = namedtuple('VERBOSITY', ['QUITE', 'NORMAL', 'VERBOSE', 'DEBUG'])(0, 1, 2, 3)

SeaLionCoord = namedtuple('SeaLionCoord', ['tid', 'cls', 'x', 'y'])


def get_noseal_locs(sld, tid, n_images):

    coords = sld.coords(tid)

    y_max = sld.src_img.shape[0]
    x_max = sld.src_img.shape[1]

    noseal_coords = []
    for i in range(100):
        y = np.random.randint(64, y_max-64)
        x = np.random.randint(64, x_max-64)


        noseal = True
        for coord in coords:
            if sqrt((y - coord.y)**2 + (x - coord.x)**2) < 48:
                noseal = False

        if noseal:
            noseal_coords.append((x, y))

    return noseal_coords


def save_noseal_chunks(sld, tid, coords, folder, chunksize=128):
    sld._progress('Saving non-seal image chunks...')
    sld._progress('\n', verbosity=VERBOSITY.VERBOSE)

    cls = 6
    last_tid = -1

    for x, y in coords:
        if tid != last_tid:
            img = sld.load_train_image(tid, border=chunksize // 2, mask=True)
            last_tid = tid

        fn = 'chunk_{tid}_{cls}_{x}_{y}_{size}.png'.format(size=chunksize, tid=tid, cls=cls, x=x, y=y)
        fn = os.path.join(folder, fn)
        sld._progress(' Saving ' + fn, end='\n', verbosity=VERBOSITY.VERBOSE)
        Image.fromarray(img[y:y + chunksize, x:x + chunksize, :]).save(fn)
        sld._progress()
    sld._progress('done')


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


# Count sea lion dots and compare to truth from train.csv
sld = SeaLionData(SOURCEDIR, DATADIR, VERBOSITY.NORMAL)

for tid in sld.train_ids:
    noseal_coords = get_noseal_locs(sld, tid, 150)
    save_noseal_chunks(sld, tid, noseal_coords, '../cropped_nonseals')

