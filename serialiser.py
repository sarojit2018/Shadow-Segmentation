import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

target_size = (128,128)

def resize_and_save(low, high, path_images, path_masks, path_image_dump, path_label_dump):
    cnt = 0
    print("***************Starting Resizing***************")
    for lv in range(low,high):
        img = path_images + prefix + str(lv+1) + ".jpg"
        mask = path_masks + prefix + str(lv+1) + ".png"
        img = cv2.imread(img)
        mask = cv2.imread(mask)

        image = np.float32(img)
        mask = np.float32(mask)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        bgr = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        bgr = np.float32(bgr)
        mask = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, target_size, interpolation = cv2.INTER_NEAREST)
        mask = cv2.resize(mask, target_size, interpolation = cv2.INTER_NEAREST)

        print("Resizing complete for "+str(lv+1))
        filename = str(cnt+1) + ".png"

        image_dump = path_image_dump + filename
        label_dump = path_label_dump + filename

        plt.imsave(image_dump, img)
        plt.imsave(label_dump, mask)

        cnt += 1


prefix = "lssd"
path_images = "/Users/sarojitauddya/shadow detection/SBU-shadow/SBUTrain4KRecoveredSmall/ShadowImages/"
path_masks = "/Users/sarojitauddya/shadow detection/SBU-shadow/SBUTrain4KRecoveredSmall/ShadowMasks/"


path_image_dump = "/Users/sarojitauddya/shadow detection/shadow_segmentation/train/img/"
path_label_dump = "/Users/sarojitauddya/shadow detection/shadow_segmentation/train/label/"

resize_and_save(0,3977, path_images, path_masks, path_image_dump, path_label_dump)
path_image_dump = "/Users/sarojitauddya/shadow detection/shadow_segmentation/test/img/"
path_label_dump = "/Users/sarojitauddya/shadow detection/shadow_segmentation/test/label/"

resize_and_save(3999, 4107, path_images, path_masks, path_image_dump, path_label_dump)
