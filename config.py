import os


MODEL_PATH = os.path.join('..', 'models')
IMAGE_ROW_SIZE = 128
IMAGE_COLUMN_SIZE = IMAGE_ROW_SIZE
TRAIN_SEAL_FOLDER_PATH = os.path.join('..', 'cropped_seals')
TRAIN_NONSEAL_FOLDER_PATH = os.path.join('..', 'cropped_nonseals')
BATCH_SIZE = 64
CROP_TRAIN_DIR = os.path.join('..', 'train_byclass')
CROP_TEST_DIR = os.path.join('..', 'test_byclass')

CLASS_LOOKUP = {0:'adult_males', 1:'subadult_males', 2:'adult_females',
                3: 'juveniles', 4:'pups', 6:'None'}

TRAIN_PCT = 0.8
LINELEN = 120
SEED = 125431

CROP_CLASSIFIER_WEIGHT_PATH = os.path.join(MODEL_PATH, 'cropped_classifier_weights')
OPTIMAL_WEIGHTS_FILE_RULE = os.path.join(CROP_CLASSIFIER_WEIGHT_PATH,
                                         "epoch_{epoch:03d}-loss_{loss:.5f}-val_loss_{val_loss:.5f}.h5")

PATIENCE = 5
MAXIMUM_EPOCH_NUM = 150

TOTAL_BATCHES = 400