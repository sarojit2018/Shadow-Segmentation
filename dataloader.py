import numpy as np
import cv2
from random import random
from random import shuffle
from random import randint
from random import uniform
from random  import sample
from random import choice
import os
import matplotlib.pyplot as plt

mode_dir = {}

mode_dir['train'] = {}
mode_dir['train'] = 'SBUTrain4KRecoveredSmall/'

mode_dir['test'] = 'SBU-Test/'

rotation_aug = ['CLK_90', 'ACLK_90', 'FLIP', 'SAME']


def datastreamer(input_size = (256, 256), batch_size = 13, debug = False, cvt2gray = False, base_path = './SBU-shadow/', mode = 'train', enable_rot = True, rgbbgr_flip = True):
    img_batch = []
    lbl_batch = []

    if debug:
        print("Mode: " + str(mode))

    if mode not in mode_dir:
        print("Incorrect Mode, please either choose 'train' or 'test'")
        return

    if debug:
        print("Input directory: " + mode_dir[mode])

    dir = mode_dir[mode]
    mode_path  = base_path + dir

    img_path = mode_path + 'ShadowImages/'
    lbl_path = mode_path + 'ShadowMasks/'

    img_list = os.listdir(img_path)

    if debug:
        print("The number of images: " + str(len(img_list)))

    if enable_rot:
        if debug:
            print("Enabling rotations")
    if cvt2gray:
        if debug:
            print("Gray input")

    else:
        if rgbbgr_flip:
            print("RGB and BGR flip randomly")

    print("####################### Starting datastreaming #######################")

    while 1:
        shuffle(img_list)
        img_batch = []
        lbl_batch = []

        for i in range(batch_size):
            img = plt.imread(img_path + img_list[i])
            lbl = plt.imread(lbl_path + img_list[i][:-3] + 'png')

            img = cv2.resize(img, input_size, interpolation = cv2.INTER_CUBIC)
            lbl = cv2.resize(lbl, input_size, interpolation = cv2.INTER_CUBIC)

            if enable_rot:
                rot = choice(rotation_aug)
                flg_rotate = False

                if rot=='CLK_90':
                    if debug:
                        print("Rotation: " + rot)
                    cv2_object = cv2.ROTATE_90_CLOCKWISE
                    flg_rotate = True

                elif rot=='ACLK_90':
                    if debug:
                        print("Rotation: " + rot)
                    cv2_object = cv2.ROTATE_90_COUNTERCLOCKWISE
                    flg_rotate = True

                elif rot=='FLIP':
                    if debug:
                        print("Rotation: " + rot)
                    cv2_object = cv2.ROTATE_180
                    flg_rotate = True

                else:
                    if debug:
                        print("No Rotation")
                    flg_rotate = False


                if flg_rotate:
                    img = cv2.rotate(img, cv2_object)
                    lbl = cv2.rotate(lbl, cv2_object)

            if not cvt2gray:
                if rgbbgr_flip:
                    if uniform(0,1) > 0.5:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        if debug:
                            print("RGB flipped to BGR")


            if cvt2gray:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = np.reshape(img, img.shape + (1,))

            img_max = np.max(img)
            lbl_max = np.max(lbl)

            if img_max > 0:
                img = img/img_max

            if lbl_max > 0:
                lbl = lbl/lbl_max

            lbl[lbl >= 0.5] = 1.0
            lbl[lbl < 0.5] = 0.0
            lbl = np.reshape(lbl, lbl.shape + (1,))



            if debug:
                plt.imshow(img)
                if cvt2gray:
                    plt.title("Gray Input")
                else:
                    plt.title("RGB Input")
                plt.show()

                plt.imshow(lbl)
                plt.title("GT Shadow Mask")
                plt.show()

                print("Input image max: " + str(np.max(img)))
                print("GT label max: " + str(np.max(lbl)))

            img_batch.append(img)
            lbl_batch.append(lbl)

        img_batch = np.array(img_batch)
        lbl_batch = np.array(lbl_batch)

        yield img_batch, lbl_batch
