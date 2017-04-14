import os
import config as cfg
import shutil
import numpy as np


def migrate_seals(cfg, img_dir, seals, train_size=1., floodgate=None, linelen=120):
    """
    Copy cropped seal images to new directory with one folder for each cropped image 
    
    Parameters
    ----------
    cfg : config.py
        Config module 
    seals : list of str
        List of filepaths to cropped images
    floodgate : int or None
        Limits execution volume for testing
    linelen : int
        Number of items to print to each line

    Returns
    -------
    None
    """
    seal_i = -1
    train_n = 0
    test_n = 0
    for seal_i, seal in enumerate(seals[:floodgate]):
        if train_size < 1.:
            if np.random.rand() > train_size:
                save_dir = cfg.CROP_TEST_DIR
                test_n += 1
            else:
                save_dir = cfg.CROP_TRAIN_DIR
                train_n += 1
        else:
            save_dir = cfg.CROP_TRAIN_DIR
            train_n += 1

        real_img = os.path.join(img_dir, seal)
        seal_type = seal.split('_')[2]
        seal_type_nm = cfg.CLASS_LOOKUP[int(seal_type)]
        seal_type_dir = os.path.join(save_dir, seal_type_nm)

        if not os.path.exists(seal_type_dir):
            os.mkdir(seal_type_dir)

        drop_path = os.path.join(seal_type_dir, seal)

        if not os.path.exists(drop_path):
            shutil.copyfile(real_img, drop_path)
            print(seal_type, end='')
            if seal_i % linelen == linelen - 1:
                print('')

    total_seals = train_n + test_n


    assert seal_i + 1 == total_seals, (seal_i + 1, total_seals)

    msg = '\nTotal Seals: {}\nTrain Seals: {}\nTest Seals: {}\n'
    print(msg.format(total_seals, train_n, test_n))

def great_migration(cfg, remove_existing=False, floodgate=None):
    """
    Copy cropped seal images to new directory with one folder for each cropped image 

    Parameters
    ----------
    cfg : config.py
        Config module 
    remove_existing : bool, optional (default is None)
        Remove existing train/test directories
    floodgate : int or None, optional (default is None)
        Controls how many images are passed through

    Returns
    -------
    None
    """
    np.random.seed(cfg.SEED)

    if remove_existing:
        if os.path.exists(cfg.CROP_TRAIN_DIR):
            shutil.rmtree(cfg.CROP_TRAIN_DIR)

        if os.path.exists(cfg.CROP_TEST_DIR):
            shutil.rmtree(cfg.CROP_TEST_DIR)


    if not os.path.exists(cfg.CROP_TRAIN_DIR):
        os.mkdir(cfg.CROP_TRAIN_DIR)

    if not os.path.exists(cfg.CROP_TEST_DIR):
        os.mkdir(cfg.CROP_TEST_DIR)

    seals = os.listdir(cfg.TRAIN_SEAL_FOLDER_PATH)
    nonseals = os.listdir(cfg.TRAIN_NONSEAL_FOLDER_PATH)

    default_settings = dict(
        cfg=cfg,
        train_size=cfg.TRAIN_PCT,
        floodgate=floodgate,
        linelen=cfg.LINELEN
    )

    print("Migrating real seals...")
    migrate_seals(img_dir=cfg.TRAIN_SEAL_FOLDER_PATH, seals=seals, **default_settings)


    print("Migrating non-seals...")
    migrate_seals(img_dir=cfg.TRAIN_NONSEAL_FOLDER_PATH, seals=nonseals, **default_settings)

    n_train_files = sum([len(files) for r, d, files in os.walk(cfg.CROP_TRAIN_DIR)])

    if cfg.TRAIN_PCT < 1.:
        n_test_files = sum([len(files) for r, d, files in os.walk(cfg.CROP_TEST_DIR)])
    else:
        n_test_files = 0

    n_result_imgs = n_test_files + n_train_files

    if floodgate is None:
        n_original_imgs = sum((len(seals), len(nonseals)))
        assert n_result_imgs == n_original_imgs, (n_result_imgs, n_original_imgs)
    else:
        expected_imgs = floodgate*2
        assert n_result_imgs == expected_imgs, (n_result_imgs, expected_imgs)


if __name__ == '__main__':

    great_migration(cfg, remove_existing=True, floodgate=None)

