# -*- coding:utf-8 -*-
import copy
import os
import random
import numpy as np
from PIL import Image
from nets.segnet import convnet_segnet

if __name__ == "__main__":
    # ---------------------------------------------------#
    # define colors of background and zebra crossing respectively
    # [0,0,0]
    # [0,255,0]
    # ---------------------------------------------------#
    class_colors = [[0, 0, 0], [0, 255, 0]]
    # ---------------------------------------------#
    # size of input image
    # ---------------------------------------------#
    HEIGHT = 416
    WIDTH = 416
    # ---------------------------------------------#
    # background + zebra crossing = 2
    # ---------------------------------------------#
    NCLASSES = 2

    # ---------------------------------------------#
    # load model
    # ---------------------------------------------#
    model = convnet_segnet(n_classes=NCLASSES, input_height=HEIGHT, input_width=WIDTH)
    # --------------------------------------------------#
    #  load weights
    # --------------------------------------------------#
    # model_path = "logs/ep033-loss0.040-val_loss0.037.h5"
    model_path = "logs/ep094-loss0.007-val_loss0.019.h5"
    model.load_weights(model_path)

    # --------------------------------------------------#
    # load test images
    # --------------------------------------------------#
    imgs = os.listdir("./img/")
    for jpg in imgs:
        img = Image.open("./img/" + jpg)

        old_img = copy.deepcopy(img)
        orininal_h = np.array(img).shape[0]
        orininal_w = np.array(img).shape[1]

        # --------------------------------------------------#
        # resize input image to [HEIGHT, WIDTH, 3]
        # --------------------------------------------------#
        img = img.resize((WIDTH, HEIGHT), Image.BICUBIC)
        img = np.array(img) / 255
        img = img.reshape(-1, HEIGHT, WIDTH, 3)

        # --------------------------------------------------#
        # put img into model
        # --------------------------------------------------#
        pr = model.predict(img)[0]
        pr = pr.reshape((int(HEIGHT / 2), int(WIDTH / 2), NCLASSES)).argmax(axis=-1)

        # ------------------------------------------------#
        # plot result mask
        # ------------------------------------------------#
        seg_img = np.zeros((int(HEIGHT / 2), int(WIDTH / 2), 3))
        for c in range(NCLASSES):
            seg_img[:, :, 0] += ((pr[:, :] == c) * class_colors[c][0]).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * class_colors[c][1]).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * class_colors[c][2]).astype('uint8')

        seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h))
        # ------------------------------------------------#
        # blend mask + ori_img (old_img)
        # ------------------------------------------------#
        image = Image.blend(old_img, seg_img, 0.3)

        image.save("./img_out/" + jpg)