# -*- coding:utf-8 -*-
import keras
import numpy as np
import keras.backend as K
from keras.callbacks import (EarlyStopping,ModelCheckpoint,ReduceLROnPlateau,TensorBoard)
from keras.optimizers import Adam
from keras.utils.data_utils import get_file
from keras.activations import *
from keras.layers import *
from PIL import Image
from nets.segnet import convnet_segnet

#-----------------------------------------------------------#
# defines a generator, read imgs and labels in 'datasets2/'
#-----------------------------------------------------------#
def generate_arrays_from_file(lines,batch_size):
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        for _ in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            # -----------------------------------------------------------#
            # read images and normalization
            # -----------------------------------------------------------#
            name = lines[i].split(';')[0]
            img = Image.open('./dataset2/jpg/' + name)
            img = img.resize((WIDTH,HEIGHT),Image.BICUBIC)
            img = np.array(img)/255  # normalization 归一化
            X_train.append(img)
            # -----------------------------------------------------------#
            # read labels and normalization
            # -----------------------------------------------------------#
            name = lines[i].split(';')[1].split()[0]
            label = Image.open('./dataset2/png/' + name)
            label = label.resize((int(WIDTH/2),int(HEIGHT/2)),Image.NEAREST)
            if len(np.shape(label)) == 3:
                label = np.array(label)[:,:,0]
            label = np.reshape(np.array(label),[-1])
            one_hot_label = np.eye(NCLASSES)[np.array(label,np.int32)]
            # transfer 'label' to 'one-hot-label' format
            # 例如他可以将类别总数为6的labels=[1,2,3,0,1,1]的数组转化成
            # 数组[[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,0,0],[0,1,0,0,0,0],[0,1,0,0,0,0]]
            # 这就是所谓的one-hot的形式。
            Y_train.append(one_hot_label)

            i = (i + 1) % n
        yield (np.array(X_train),np.array(Y_train))

if __name__ == "__main__":
    # -----------------------------------------------------------#
    # define height, width, number of classes of the input images
    # -----------------------------------------------------------#
    HEIGHT = 416
    WIDTH = 416
    NCLASSES = 2 # background + zebra crossing

    # -----------------------------------------------------------#
    # download weights of the pretrained model 'vgg16' from website
    # -----------------------------------------------------------#
    log_dir = 'logs/'
    model = convnet_segnet(n_classes=NCLASSES, input_height=HEIGHT, input_width=WIDTH)
    WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models')
    model.load_weights(weights_path, by_name=True)

    # open dataset 'xxx.txt'
    with open("./dataset2/train.txt", "r") as f:
        lines = f.readlines()

    # -----------------------------------------------------------#
    # disrupt the order of images, 10% for val, the left for train
    # -----------------------------------------------------------#
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val

    # -----------------------------------------------------------#
    # training parameter setting
    # checkpoint:details of saved weights;period: the period for saving weights
    # reduce_lr:ways of learning rate reduction
    # early_stopping:if val_loss doesn't change in 10 times, means the model is stable; then stop training.
    # -----------------------------------------------------------#
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=2)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # -----------------------------------------------------------#
    # fine-tune
    # -----------------------------------------------------------#
    trainable_layer = 15
    for i in range(trainable_layer):
        model.layers[i].trainable = False
    print('freeze the first {} layers of total {} layers.'.format(trainable_layer, len(model.layers)))

    if True:
        lr = 1e-3
        batch_size = 4
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=lr),
                      metrics=['accuracy'])

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=50,
                            initial_epoch=0,
                            callbacks=[checkpoint, reduce_lr, early_stopping])

    for i in range(len(model.layers)):
        model.layers[i].trainable = True

    if True:
        lr = 1e-4
        batch_size = 4
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=lr),
                      metrics=['accuracy'])

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=100,
                            initial_epoch=50,
                            callbacks=[checkpoint, reduce_lr, early_stopping])