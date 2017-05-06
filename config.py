import os


MODEL_PATH = os.path.join('..', 'models')
IMAGE_ROW_SIZE = 128
IMAGE_COLUMN_SIZE = IMAGE_ROW_SIZE
TRAIN_SEAL_FOLDER_PATH = os.path.join('..', 'cropped_seals')
TRAIN_NONSEAL_FOLDER_PATH = os.path.join('..', 'cropped_nonseals')
BATCH_SIZE = 1028
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

PATIENCE = 8
MAXIMUM_EPOCH_NUM = 250

TOTAL_BATCHES = 50

GEN_OBJ_KWARGS = dict(
    rotation_range=45,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1.0 / 255
)

GEN_KWARGS = dict(
    target_size=(IMAGE_ROW_SIZE, IMAGE_COLUMN_SIZE),
    color_mode='rgb',
    classes=[CLASS_LOOKUP[key] for key in sorted(CLASS_LOOKUP.keys())],
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)