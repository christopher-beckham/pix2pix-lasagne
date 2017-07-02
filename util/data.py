"""Auxiliar methods to deal with loading the dataset."""
import os
import random

import numpy as np

from keras.preprocessing.image import ImageDataGenerator

def _get_slices(length, bs):
    slices = []
    b = 0
    while True:
        if b*bs >= length:
            break
        slices.append( slice(b*bs, (b+1)*bs) )
        b += 1
    return slices

def iterate_hdf5(imgen=None, is_a_grayscale=True, is_b_grayscale=False, crop=None, is_uint8=True):
    def _iterate_hdf5(X_arr, y_arr, bs, rnd_state=np.random.RandomState(0)):
        assert X_arr.shape[0] == y_arr.shape[0]
        while True:
            slices = _get_slices(X_arr.shape[0], bs)
            if rnd_state != None:
                rnd_state.shuffle(slices)
            for elem in slices:
                this_X, this_Y = X_arr[elem].astype("float32"), y_arr[elem].astype("float32")
                # TODO: only compatible with theano
                this_X = this_X.swapaxes(3,2).swapaxes(2,1)
                this_Y = this_Y.swapaxes(3,2).swapaxes(2,1)
                # normalise A and B if these are in the range [0,255]
                if is_uint8:
                    this_X = (this_X / 255.0) if is_a_grayscale else (this_X - 127.5) / 127.5
                    this_Y = (this_Y / 255.0) if is_b_grayscale else (this_Y - 127.5) / 127.5
                if crop != None:
                    X_new = np.zeros( this_X.shape[0:2] + (crop,crop), dtype=this_X.dtype )
                    Y_new = np.zeros( this_X.shape[0:2] + (crop,crop), dtype=this_Y.dtype )
                    img_sz = this_X.shape[-1]
                    for i in range(this_X.shape[0]):
                        x_start = np.random.randint(0, img_sz-crop+1)
                        y_start = np.random.randint(0, img_sz-crop+1)
                        X_new[i] = this_X[i, :, y_start:y_start+crop, x_start:x_start+crop]
                        Y_new[i] = this_Y[i, :, y_start:y_start+crop, x_start:x_start+crop]
                    this_X = X_new
                    this_Y = Y_new
                # if we passed an image generator, augment the images
                if imgen != None:
                    seed = rnd_state.randint(0, 100000)
                    this_X = imgen.flow(this_X, None, batch_size=bs, seed=seed).next()
                    this_Y = imgen.flow(this_Y, None, batch_size=bs, seed=seed).next()              
                yield this_X, this_Y
    return _iterate_hdf5

# this just wraps the above functional iterator
class Hdf5Iterator():
    def __init__(self, X, y, bs, imgen, is_a_grayscale, is_b_grayscale, crop, is_uint8=True):
        """
        :X: in our case, the heightmaps
        :y: in our case, the textures
        :bs: batch size
        :imgen: optional image data generator
        :is_a_binary: if the A image is binary, we have to divide
         by 255, otherwise we scale to [-1, 1] using tanh scaling
        :is_b_binary: same as is_a_binary
        """
        assert X.shape[0] == y.shape[0]
        self.fn = iterate_hdf5(imgen, is_a_grayscale, is_b_grayscale, crop, is_uint8)(X, y, bs)
        self.N = X.shape[0]
    def __iter__(self):
        return self
    def next(self):
        return self.fn.next()

class Hdf5DcganIterator():
    def __init__(self, X, bs, imgen, is_binary):
        self.fn = iterate_hdf5(imgen, is_a_binary=is_binary, is_b_binary=False)(X, np.zeros_like(X), bs)
        self.N = X.shape[0]
    def __iter__(self):
        return self
    def next(self):
        from keras_adversarial import gan_targets
        xbatch, _ = self.fn.next()
        ybatch = gan_targets(xbatch.shape[0])
        return  xbatch, ybatch

    
if __name__ == '__main__':

    pass
