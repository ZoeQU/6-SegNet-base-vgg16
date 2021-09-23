# -*- coding:utf-8 -*-
import keras
from keras.layers import *
from keras.models import *

from nets.convnet import get_convnet_encoder

def segnet_decoder(f,n_class,n_up=3):
    assert n_up >= 2
    o = f
    # 26,26,512 -> 26,26,512
    o = ZeroPadding2D((1,1))(o)
    o = Conv2D(512,(3,3),padding='valid')(o)
    o = BatchNormalization()(o)
    # upsample,h,w->1/8
    # 26,26,512 -> 52,52,256
    o = UpSampling2D((2,2))(o)
    o = ZeroPadding2D((1,1))(o)
    o = Conv2D(256,(3,3),padding='valid')(o)
    o = BatchNormalization()(o)
    # upsample,h,w->1/4
    # 52,52,256 -> 104,104,128
    for _ in range(n_up -2):
        o = UpSampling2D((2,2))(o)
        o = ZeroPadding2D((1,1))(o)
        o = Conv2D(128,(3,3),padding='valid')(o)
        o = BatchNormalization()(o)
    # upsample,h,w->1/2
    # 104,104,128 -> 208,208,64
    o = UpSampling2D((2,2))(o)
    o = ZeroPadding2D((1,1))(o)
    o = Conv2D(64,(3,3),padding='valid')(o)
    o = BatchNormalization()(o)
    # 208,208,64 -> 208,208,2; 2 means only detect 2 kinds of object in this algorithm
    o = Conv2D(n_class,(3,3),padding='same')(o)
    return o

def _segnet(n_classes, encoder, input_height=416, input_width=416, encoder_level=3):
    # encoder, backbone network
    img_input, levels = encoder(input_height=input_height,input_width=input_width)
    # get hw [f1-f4]
    feat = levels[encoder_level]
    # input feature information to segnet
    o = segnet_decoder(feat, n_classes, n_up=3)
    # reshape
    o = Reshape((int(input_height/2)*int(input_width/2), -1))(o)
    o = Softmax()(o)
    model = Model(img_input,o)
    return model

def convnet_segnet(n_classes, input_height=224, input_width=224, encoder_level=3):
    model = _segnet(n_classes,get_convnet_encoder,input_height=input_height,input_width=input_width,encoder_level=3)
    model.model_name = 'convnet_segnet'
    return model