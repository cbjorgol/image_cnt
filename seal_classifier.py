import os
import glob
import pylab
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import config as cfg

def init_model(cfg, target_num=4, FC_block_num=2, FC_feature_dim=512, dropout_ratio=0.5, learning_rate=0.0001):
    # Get the input tensor
    input_tensor = Input(shape=(3, cfg.IMAGE_ROW_SIZE, cfg.IMAGE_COLUMN_SIZE))

    # Convolutional blocks
    pretrained_model = VGG16(include_top=False, weights="imagenet")
    for layer in pretrained_model.layers:
        layer.trainable = False
    output_tensor = pretrained_model(input_tensor)

    # FullyConnected blocks
    output_tensor = Flatten()(output_tensor)
    for _ in range(FC_block_num):
        output_tensor = Dense(FC_feature_dim, activation="relu")(output_tensor)
        output_tensor = BatchNormalization()(output_tensor)
        output_tensor = Dropout(dropout_ratio)(output_tensor)
    output_tensor = Dense(target_num, activation="sigmoid")(output_tensor)

    # Define and compile the model
    model = Model(input_tensor, output_tensor)
    # model.compile(optimizer=Adam(lr=learning_rate), loss="mse")

    model.compile(optimizer=Adam(lr=learning_rate), loss="binary_crossentropy")
    # plot(model, to_file=os.path.join(MODEL_PATH, "model.png"), show_shapes=True, show_layer_names=True)

    return model


class InspectLoss(Callback):
    def __init__(self, cfg):
        super(InspectLoss, self).__init__()

        self.train_loss_list = []
        self.valid_loss_list = []
        self.cfg = cfg

    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get("loss")
        valid_loss = logs.get("val_loss")
        self.train_loss_list.append(train_loss)
        self.valid_loss_list.append(valid_loss)
        epoch_index_array = np.arange(len(self.train_loss_list)) + 1

        pylab.figure()
        pylab.plot(epoch_index_array, self.train_loss_list, "yellowgreen", label="train_loss")
        pylab.plot(epoch_index_array, self.valid_loss_list, "lightskyblue", label="valid_loss")
        pylab.grid()
        pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, ncol=2, mode="expand", borderaxespad=0.)
        pylab.savefig(os.path.join(cfg.MODEL_PATH, "Loss Curve.png"))
        pylab.close()


def fit_model(cfg, train_generator, valid_generator, train_sample_num=1000, valid_sample_num=100):
    model = init_model(cfg, 6, 2, 512, dropout_ratio=0.5, learning_rate=0.0001)

    weights_file_path_list = sorted(glob.glob(os.path.join(cfg.MODEL_PATH, "*.h5")))
    if len(weights_file_path_list) == 0:
        print("Performing the training procedure ...")

        modelcheckpoint_callback = ModelCheckpoint(cfg.OPTIMAL_WEIGHTS_FILE_RULE,
                                                   monitor="val_loss",
                                                   save_best_only=True,
                                                   save_weights_only=True)

        inspectloss_callback = InspectLoss(cfg=cfg)
        earlystopping_callback = EarlyStopping(monitor="val_loss", patience=cfg.PATIENCE)

        model.fit_generator(generator=train_generator,
                            samples_per_epoch=train_sample_num,
                            validation_data=valid_generator,
                            nb_val_samples=valid_sample_num,
                            callbacks=[
                                earlystopping_callback,
                                modelcheckpoint_callback,
                                inspectloss_callback
                            ],
                            nb_epoch=cfg.MAXIMUM_EPOCH_NUM,
                            verbose=1)

        return sorted(glob.glob(os.path.join(cfg.CROP_CLASSIFIER_WEIGHT_PATH, "*.h5")))


if __name__ == '__main__':



    train_generator_object = ImageDataGenerator(**cfg.GEN_OBJ_KWARGS)
    test_generator_object = ImageDataGenerator(**cfg.GEN_OBJ_KWARGS)

    train_generator = train_generator_object.flow_from_directory(directory=cfg.CROP_TRAIN_DIR, **cfg.GEN_KWARGS)
    valid_generator = test_generator_object.flow_from_directory(directory=cfg.CROP_TEST_DIR, **cfg.GEN_KWARGS)


    print('Train Samples:', cfg.TOTAL_BATCHES*cfg.BATCH_SIZE)
    fit_model(cfg, train_generator, valid_generator,
              train_sample_num=cfg.BATCH_SIZE * cfg.TOTAL_BATCHES,
              valid_sample_num=cfg.BATCH_SIZE * cfg.TOTAL_BATCHES)
