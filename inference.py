import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import cv2
import matplotlib.pyplot as plt

from Config import Config
from model import shadow_mask_gen

target_shape = Config.MASK_SHAPE
cvt2gray = Config.CVT2GRAY

dst_path = './inference/'

if not os.path.exists(dst_path):
    os.mkdir(dst_path)

src_path = './SBU-shadow/SBU-Test/ShadowImages/'

if cvt2gray:
    input_size = target_shape + (1,)
else:
    input_size = target_shape + (3,)

model = shadow_mask_gen(input_size = input_size)
model.load_weights('fully_trained_shadow_gen.hdf5')


for imgname in os.listdir(src_path):
    img = plt.imread(src_path + imgname)
    img = cv2.resize(img, target_shape, interpolation = cv2.INTER_CUBIC)

    img_max = np.max(img)

    if img_max > 0:
        img = img/img_max

    img_batch = np.reshape(img, (1,) + img.shape)
    pred = np.squeeze(model.predict(img_batch))

    plt.imsave(dst_path + imgname[:-3] + 'png', pred)
