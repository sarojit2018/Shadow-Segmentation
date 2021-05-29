from model import shadow_mask_gen
from dataloader import datastreamer
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras import backend as K

from Config import Config

#To be input as args and parsed using arg parser or a config file
#Currently usinf a config class file
batch_size = Config.BATCH_SIZE
num_epochs = Config.NUM_EPOCHS
steps_per_epoch = Config.STEPS_PER_EPOCH
validation_steps = Config.VALIDATION_STEPS
train_from_scratch = Config.TRAIN_FROM_SCRATCH
target_shape = Config.MASK_SHAPE
cvt2gray = Config.CVT2GRAY
rgbbgr_flip = Config.RGBBGR_FLIP
debug = Config.DEBUG

print("Target Shape: " + str(target_shape))

train_datastreamer = datastreamer(input_size = target_shape, batch_size = batch_size, debug = debug, mode = 'train', cvt2gray = cvt2gray, rgbbgr_flip = rgbbgr_flip)
valid_datastreamer = datastreamer(input_size = target_shape, batch_size = batch_size, debug = debug, mode = 'test', cvt2gray = cvt2gray, rgbbgr_flip = rgbbgr_flip)

if cvt2gray:
    input_size = target_shape + (1,)
else:
    input_size = target_shape + (3,)

model = shadow_mask_gen(input_size = input_size)

weights_file = 'shadow_gen.hdf5'

if not train_from_scratch:
    if os.path.exists(weights_file):
        try:
            model.load_weights(weights_file)
            print("INFO: Weights found! Resuming finetuning!!")
        except:
            print("ERR: Some error happened, starting training from beginning")

    else:
        print("INFO: No weights file found, starting training from beginning")


else:
    print("WARN: Starting training from scratch, existing weights will be overwritten.")

print("INFO: Model Summary")
print(model.summary())


model_checkpoint = ModelCheckpoint(weights_file, monitor='val_loss', verbose=1, save_best_only=True)

fully_trained_success_flag = 1


try:
    model.fit(train_datastreamer, steps_per_epoch=steps_per_epoch, epochs=num_epochs, validation_data = valid_datastreamer, validation_steps = validation_steps, callbacks=[model_checkpoint])

except KeyboardInterrupt:
    print("Training Interrupted!! Saving the file trained till now!")
    midway_weights_file = 'midway_trained_' + weights_file
    model.save(midway_weights_file)
    fully_trained_success_flag = 0


if fully_trained_success_flag:
    print("INFO: Successfully trained for entire duration!")
    final_weights_file = 'fully_trained_' + weights_file
    model.save(final_weights_file)
