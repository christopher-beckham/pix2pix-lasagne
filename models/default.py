import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.init import *
from lasagne.updates import *
from lasagne.objectives import *
from keras.preprocessing.image import ImageDataGenerator
import os
import sys
#sys.path.append("..") # some important shit we need to import
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time
from nolearn.lasagne.visualize import draw_to_file
import nolearn
from keras_ports import ReduceLROnPlateau
import pickle

def Convolution(layer, f, k=3, s=2, border_mode='same', **kwargs):
    return Conv2DLayer(layer, num_filters=f, filter_size=(k,k), stride=(s,s), pad=border_mode, nonlinearity=linear)

def Deconvolution(layer, f, k=2, s=2, **kwargs):
    return Deconv2DLayer(layer, num_filters=f, filter_size=(k,k), stride=(s,s), nonlinearity=linear)

def concatenate_layers(layers, **kwargs):
    return ConcatLayer(layers, axis=1)

def g_unet(nf=64, act=tanh, num_repeats=0):
    print num_repeats
    def padded_conv(nf, x):
        x = Convolution(x, nf,s=1,k=3)
        x = BatchNormLayer(x)
        x = NonlinearityLayer(x, nonlinearity=leaky_rectify)
        return x
    i = InputLayer((None, 1, 512, 512))
    # in_ch x 512 x 512
    conv1 = Convolution(i, nf)
    conv1 = BatchNormLayer(conv1)
    x = NonlinearityLayer(conv1, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf, x)    
    # nf x 256 x 256
    conv2 = Convolution(x, nf * 2)
    conv2 = BatchNormLayer(conv2)
    x = NonlinearityLayer(conv2, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*2, x)
    # nf*2 x 128 x 128
    conv3 = Convolution(x, nf * 4)
    conv3 = BatchNormLayer(conv3)
    x = NonlinearityLayer(conv3, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*4, x)
    # nf*4 x 64 x 64
    conv4 = Convolution(x, nf * 8)
    conv4 = BatchNormLayer(conv4)
    x = NonlinearityLayer(conv4, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*8 x 32 x 32
    conv5 = Convolution(x, nf * 8)
    conv5 = BatchNormLayer(conv5)
    x = NonlinearityLayer(conv5, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*8 x 16 x 16
    conv6 = Convolution(x, nf * 8)
    conv6 = BatchNormLayer(conv6)
    x = NonlinearityLayer(conv6, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*8 x 8 x 8
    conv7 = Convolution(x, nf * 8)
    conv7 = BatchNormLayer(conv7)
    x = NonlinearityLayer(conv7, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*8 x 4 x 4
    conv8 = Convolution(x, nf * 8)
    conv8 = BatchNormLayer(conv8)
    x = NonlinearityLayer(conv8, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*8 x 2 x 2
    conv9 = Convolution(x, nf * 8, k=2, s=1, border_mode='valid')
    conv9 = BatchNormLayer(conv9)
    x = NonlinearityLayer(conv9, nonlinearity=leaky_rectify)
    # nf*8 x 1 x 1  
    dconv1 = Deconvolution(x, nf * 8,
                           k=2, s=1)
    dconv1 = BatchNormLayer(dconv1)
    x = concatenate_layers([dconv1, conv8])
    x = NonlinearityLayer(x, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*(8 + 8) x 2 x 2
    dconv2 = Deconvolution(x, nf * 8)
    dconv2 = BatchNormLayer(dconv2)
    x = concatenate_layers([dconv2, conv7])
    x = NonlinearityLayer(x, leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*(8 + 8) x 4 x 4
    dconv3 = Deconvolution(x, nf * 8)
    dconv3 = BatchNormLayer(dconv3)
    #dconv3 = DropoutLayer(dconv3, 0.5)
    x = concatenate_layers([dconv3, conv6])
    x = NonlinearityLayer(x, leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*(8 + 8) x 8 x 8
    dconv4 = Deconvolution(x, nf * 8)
    dconv4 = BatchNormLayer(dconv4)
    x = concatenate_layers([dconv4, conv5])
    x = NonlinearityLayer(x, leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*(8 + 8) x 16 x 16
    dconv5 = Deconvolution(x, nf * 8)
    dconv5 = BatchNormLayer(dconv5)
    x = concatenate_layers([dconv5, conv4])
    x = NonlinearityLayer(x, leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*(8 + 8) x 32 x 32
    dconv6 = Deconvolution(x, nf * 4)
    dconv6 = BatchNormLayer(dconv6)
    x = concatenate_layers([dconv6, conv3])
    x = NonlinearityLayer(x, leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*4, x)    
    # nf*(4 + 4) x 64 x 64
    dconv7 = Deconvolution(x, nf * 2)
    dconv7 = BatchNormLayer(dconv7)
    x = concatenate_layers([dconv7, conv2])
    x = NonlinearityLayer(x, leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*2, x)
    # nf*(2 + 2) x 128 x 128
    dconv8 = Deconvolution(x, nf)
    dconv8 = BatchNormLayer(dconv8)
    x = concatenate_layers([dconv8, conv1])
    x = NonlinearityLayer(x, leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf, x)
    # nf*(1 + 1) x 256 x 256
    dconv9 = Deconvolution(x, 3)
    # out_ch x 512 x 512
    #act = 'sigmoid' if is_binary else 'tanh'
    out = NonlinearityLayer(dconv9, act)
    return out

def discriminator(nf=32, act=sigmoid, mul_factor=[1,2,4,8], num_repeats=0, bn=False):
    i_a = InputLayer((None, 1, 512, 512))
    i_b = InputLayer((None, 3, 512, 512))
    i = concatenate_layers([i_a, i_b])
    x = i
    for m in mul_factor:
        for r in range(num_repeats+1):
            x = Convolution(x, nf*m, s=2 if r == 0 else 1)
            x = NonlinearityLayer(x, leaky_rectify)
            if bn:
                x = BatchNormLayer(x)
    x = Convolution(x, 1)
    out = NonlinearityLayer(x, act)
    # 1 x 16 x 16
    return {"inputs": [i_a, i_b], "out":out}

# for debugging

def fake_generator(**kwargs):
    i = InputLayer((None, 1, 512, 512))
    c = Convolution(i, f=3, s=1)
    c = NonlinearityLayer(c, tanh)
    return c

def fake_discriminator(**kwargs):
    i_a = InputLayer((None, 1, 512, 512))
    i_b = InputLayer((None, 3, 512, 512))
    i = concatenate_layers([i_a, i_b])
    c = Convolution(i,1)
    return {"inputs": [i_a, i_b], "out":c}
