# -*- coding:utf-8 -*-
#---------------------------------------------#
# check the structure of model
#---------------------------------------------#
from nets.segnet import convnet_segnet

if __name__ == "__main__":
    model = convnet_segnet(2, input_height=416, input_width=416)
    model.summary() # Prints a string summary of the network.