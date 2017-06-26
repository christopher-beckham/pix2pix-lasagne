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

#import inspect
#def fn_name():
#    return str(inspect.currentframe()[1][0].f_code.co_name)
#def log(st):
#    print "%s: %s" % (fn_name(), st)

# helper functions

from util.data import TwoImageIterator, iterate_hdf5, Hdf5Iterator
from util.util import MyDict, log, save_weights, load_weights, load_losses, create_expt_dir, convert_to_rgb, compose_imgs

def plot_grid(out_filename, itr, out_fn, is_a_grayscale, is_b_grayscale, N=4):
    plt.figure(figsize=(10, 6))
    for i in range(N*N):
        a, b = itr.next()
        if out_fn != None:
            bp = out_fn(a)
        else:
            bp = b
        img = compose_imgs(a[0], bp[0], is_a_grayscale=is_a_grayscale, is_b_grayscale=is_b_grayscale)
        plt.subplot(N, N, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.savefig(out_filename)
    plt.clf()
    # Make sure all the figures are closed.
    plt.close('all')

    
class Pix2Pix():
    def _print_network(self,l_out):
        for layer in get_all_layers(l_out):
            print layer, layer.output_shape, "" if not hasattr(layer, 'nonlinearity') else layer.nonlinearity
        print "# learnable params:", count_params(layer, trainable=True)
    def __init__(self, gen_fn, disc_fn,
                 gen_params, disc_params,
                 is_a_grayscale, is_b_grayscale,
                 alpha=100, lr=1e-4, opt='adam',
                 reconstruction='l1', reconstruction_only=False, lsgan=False, verbose=True):
        self.is_a_grayscale = is_a_grayscale
        self.is_b_grayscale = is_b_grayscale
        self.verbose = verbose
        l_gen = gen_fn(**gen_params)
        dd_disc = disc_fn(**disc_params)
        if verbose:
            self._print_network(l_gen)
            self._print_network(dd_disc["out"])
        X = T.tensor4('X') # A
        Y = T.tensor4('Y') # B
        # this is the output of the discriminator for real (x,y)
        disc_out_real = get_output(
            dd_disc["out"],
            { dd_disc["inputs"][0]: X, dd_disc["inputs"][1]: Y }
        )
        # this is the output of the discriminator for fake (x, y')
        gen_out = get_output(l_gen, X)
        disc_out_fake = get_output(
            dd_disc["out"],
            { dd_disc["inputs"][0]: X, dd_disc["inputs"][1]: gen_out }
        )
        if lsgan:
            #log("Using LSGAN adversarial loss...")
            adv_loss = squared_error
            # add check that disc out is linear
        else:
            adv_loss = binary_crossentropy
        gen_loss = adv_loss(disc_out_fake, 1.).mean()
        assert reconstruction in ['l1', 'l2']
        if reconstruction == 'l2':
            recon_loss = squared_error(gen_out, Y).mean()
        else:
            recon_loss = T.abs_(gen_out-Y).mean()
        if not reconstruction_only:
            gen_total_loss = gen_loss + alpha*recon_loss
        else:
            #log("GAN disabled, using only pixel-wise reconstruction loss...")
            gen_total_loss = recon_loss
        disc_loss = adv_loss(disc_out_real, 1.).mean() + adv_loss(disc_out_fake, 0.).mean()
        gen_params = get_all_params(l_gen, trainable=True)
        disc_params = get_all_params(dd_disc["out"], trainable=True)
        assert opt in ['adam', 'rmsprop']
        if opt == 'adam':
            opt = adam
        else:
            opt = rmsprop
        from lasagne.utils import floatX
        lr = theano.shared(floatX(lr))
        updates = opt(gen_total_loss, gen_params, learning_rate=lr)
        if not reconstruction_only:
            updates.update(opt(disc_loss, disc_params, learning_rate=lr))
        train_fn = theano.function([X,Y], [gen_loss, recon_loss, disc_loss], updates=updates)
        gen_fn = theano.function([X], gen_out)
        self.train_fn = train_fn
        self.gen_fn = gen_fn
        self.l_gen = l_gen
        self.l_disc = dd_disc["out"]
        self.lr = lr
    def save_model(self, filename):
        with open(filename, "wb") as g:
            pickle.dump( (get_all_param_values(self.l_gen), get_all_param_values(self.l_disc)), g, pickle.HIGHEST_PROTOCOL )
    def load_model(self, filename):
        with open(filename) as g:
            wts = pickle.load(g)
            set_all_param_values(self.l_gen, wts[0])
            set_all_param_values(self.l_disc, wts[1])            
    def train(self, it_train, it_val, batch_size, num_epochs, out_dir, model_dir=None, save_every=10, resume=None, reduce_on_plateau=False):
        header = ["epoch","gen","recon","disc","lr","time"]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if model_dir != None and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        f = open("%s/results.txt" % out_dir, "wb" if resume==None else "a")
        if resume == None:
            f.write(",".join(header)+"\n"); f.flush()
        losses = {'gen':[], 'recon':[], 'disc':[]}
        cb = ReduceLROnPlateau(self.lr,verbose=self.verbose)
        for e in range(num_epochs):
            t0 = time()
            gen_losses = []
            recon_losses = []
            disc_losses = []
            for b in range(it_train.N // batch_size):
                X_batch, Y_batch = it_train.next()
                gen_loss, recon_loss, disc_loss = self.train_fn(X_batch,Y_batch)
                gen_losses.append(gen_loss)
                recon_losses.append(recon_loss)
                disc_losses.append(disc_loss)
            losses['gen'].append(np.mean(gen_losses))
            losses['recon'].append(np.mean(recon_losses))
            if reduce_on_plateau:
                cb.on_epoch_end(np.mean(recon_losses), e+1)
            losses['disc'].append(np.mean(disc_losses))
            out_str = "%i,%f,%f,%f,%f,%f" % (e+1, losses['gen'][-1], losses['recon'][-1], losses['disc'][-1], cb.learning_rate.get_value(), time()-t0)
            print out_str
            f.write("%s\n" % out_str); f.flush()
            plot_grid("%s/out_%i.png" % (out_dir,e+1), it_val, self.gen_fn, is_a_grayscale=self.is_a_grayscale, is_b_grayscale=self.is_b_grayscale)
            if model_dir != None and (e+1) % save_every == 0:
                save_model("%s/%i.model" % (model_dir, e+1))
    def generate_imgs(self, itr, num_batches, out_dir, dont_predict=False):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        from skimage.io import imsave
        ctr = 0
        for n in range(num_batches):
            this_x, this_y = itr.next()
            if dont_predict:
                pred_y = this_y
            else:
                pred_y = gen_fn(this_x)
            for i in range(pred_y.shape[0]):
                img = convert_to_rgb(pred_y[i], is_grayscale=self.is_b_grayscale)
                imsave(fname="%s/%i.texture.png" % (out_dir, ctr), arr=img)
                imsave(fname="%s/%i.hm.png" % (out_dir, ctr), arr=this_x[i][0])
                ctr += 1


def get_iterators(dataset, batch_size, is_a_grayscale, is_b_grayscale, da=True):
    dataset = h5py.File(dataset,"r")
    if da:
        imgen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=360, fill_mode="reflect")
    else:
        imgen = ImageDataGenerator()
    it_train = Hdf5Iterator(dataset['xt'], dataset['yt'], batch_size, imgen, is_a_grayscale=is_a_grayscale, is_b_grayscale=is_b_grayscale)
    it_val = Hdf5Iterator(dataset['xv'], dataset['yv'], batch_size, imgen, is_a_grayscale=is_a_grayscale, is_b_grayscale=is_b_grayscale)
    return it_train, it_val
                
if __name__ == '__main__':
    from models.default import g_unet, discriminator # g-g-g-g g-unittttt
    from models.default import fake_generator, fake_discriminator
    gen_params = {'nf':64, 'act':tanh, 'num_repeats': 0}
    disc_params = {'nf':64, 'bn':True, 'num_repeats': 0, 'act':linear, 'mul_factor':[1,2,4,8]}
    is_a_grayscale, is_b_grayscale = True, False
    md = Pix2Pix(gen_fn=g_unet,
                 disc_fn=discriminator,
                 gen_params=gen_params,
                 disc_params=disc_params,
                 is_a_grayscale=is_a_grayscale,
                 is_b_grayscale=is_b_grayscale,
                 alpha=100, lsgan=True)
    desert_h5 = "/data/lisa/data/cbeckham/textures_v2_brown500.h5"
    bs = 4
    it_train, it_val = get_iterators(desert_h5,
                                     batch_size=bs, is_a_grayscale=is_a_grayscale, is_b_grayscale=is_b_grayscale, da=True)
    md.train(it_train, it_val, bs, 10, "output/test")
