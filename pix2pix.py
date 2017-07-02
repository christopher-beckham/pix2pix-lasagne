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
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time
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
from util.util import convert_to_rgb, compose_imgs

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
                 in_shp, is_a_grayscale, is_b_grayscale,
                 alpha=100, lr=1e-4, opt=adam, opt_args={'learning_rate':1e-3},
                 reconstruction='l1', reconstruction_only=False, lsgan=False, verbose=True):
        self.is_a_grayscale = is_a_grayscale
        self.is_b_grayscale = is_b_grayscale
        self.verbose = verbose
        l_gen = gen_fn(in_shp, is_a_grayscale, is_b_grayscale, **gen_params)
        dd_disc = disc_fn(in_shp, is_a_grayscale, is_b_grayscale, **disc_params)
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
        #from lasagne.utils import floatX
        #lr = theano.shared(floatX(lr))
        updates = opt(gen_total_loss, gen_params, **opt_args)
        if not reconstruction_only:
            updates.update(opt(disc_loss, disc_params, **opt_args))
        train_fn = theano.function([X,Y], [gen_loss, recon_loss, disc_loss], updates=updates)
        loss_fn = theano.function([X,Y], [gen_loss, recon_loss, disc_loss])
        gen_fn = theano.function([X], gen_out)
        self.train_fn = train_fn
        self.loss_fn = loss_fn
        self.gen_fn = gen_fn
        self.l_gen = l_gen
        self.l_disc = dd_disc["out"]
        self.lr = opt_args['learning_rate']
    def save_model(self, filename):
        with open(filename, "wb") as g:
            pickle.dump( (get_all_param_values(self.l_gen), get_all_param_values(self.l_disc)), g, pickle.HIGHEST_PROTOCOL )
    def load_model(self, filename):
        with open(filename) as g:
            wts = pickle.load(g)
            set_all_param_values(self.l_gen, wts[0])
            set_all_param_values(self.l_disc, wts[1])            
    def train(self, it_train, it_val, batch_size, num_epochs, out_dir, model_dir=None, save_every=1, resume=None, reduce_on_plateau=False):
        def _loop(fn, itr):
            gen_losses, recon_losses, disc_losses = [], [], []
            for b in range(itr.N // batch_size):
                X_batch, Y_batch = it_train.next()
                #print X_batch.shape, Y_batch.shape
                gen_loss, recon_loss, disc_loss = fn(X_batch,Y_batch)
                gen_losses.append(gen_loss)
                recon_losses.append(recon_loss)
                disc_losses.append(disc_loss)
            return np.mean(gen_losses), np.mean(recon_losses), np.mean(disc_losses)            
        header = ["epoch","train_gen","train_recon","train_disc","valid_gen","valid_recon","valid_disc","lr","time"]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if model_dir != None and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if self.verbose:
            try:
                from nolearn.lasagne.visualize import draw_to_file
                draw_to_file(get_all_layers(self.l_gen), "%s/gen.png" % out_dir, verbose=True)
                draw_to_file(get_all_layers(self.l_disc), "%s/disc.png" % out_dir, verbose=True)
            except:
                pass
        f = open("%s/results.txt" % out_dir, "wb" if resume==None else "a")
        if resume == None:
            f.write(",".join(header)+"\n"); f.flush()
        else:
            if self.verbose:
                print "loading weights from: %s" % resume
            self.load_model(resume)
        train_losses = {'gen':[], 'recon':[], 'disc':[]}
        valid_losses = {'gen':[], 'recon':[], 'disc':[]}
        cb = ReduceLROnPlateau(self.lr,verbose=self.verbose)
        for e in range(num_epochs):
            t0 = time()
            # training
            a,b,c = _loop(self.train_fn, it_train)
            train_losses['gen'].append(a)
            train_losses['recon'].append(b)
            train_losses['disc'].append(c)
            if reduce_on_plateau:
                cb.on_epoch_end(np.mean(recon_losses), e+1)
            # validation
            a,b,c = _loop(self.loss_fn, it_val)
            valid_losses['gen'].append(a)
            valid_losses['recon'].append(b)
            valid_losses['disc'].append(c)
            out_str = "%i,%f,%f,%f,%f,%f,%f,%f,%f" % \
                      (e+1,
                       train_losses['gen'][-1],
                       train_losses['recon'][-1],
                       train_losses['disc'][-1],
                       valid_losses['gen'][-1],
                       valid_losses['recon'][-1],
                       valid_losses['disc'][-1],
                       cb.learning_rate.get_value(),
                       time()-t0)
            print out_str
            f.write("%s\n" % out_str); f.flush()
            # plot an NxN grid of [A, predict(A)]
            plot_grid("%s/out_%i.png" % (out_dir,e+1), it_val, self.gen_fn, is_a_grayscale=self.is_a_grayscale, is_b_grayscale=self.is_b_grayscale)
            # plot big pictures of predict(A) in the valid set
            self.generate_imgs(it_train, 1, "%s/dump_train" % out_dir)
            self.generate_imgs(it_val, 1, "%s/dump_valid" % out_dir)
            if model_dir != None and e % save_every == 0:
                self.save_model("%s/%i.model" % (model_dir, e+1))
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
                pred_y = self.gen_fn(this_x)
            for i in range(pred_y.shape[0]):
                this_x_processed = convert_to_rgb(this_x[i], is_grayscale=self.is_a_grayscale)
                pred_y_processed = convert_to_rgb(pred_y[i], is_grayscale=self.is_b_grayscale)
                imsave(fname="%s/%i.a.png" % (out_dir, ctr), arr=this_x_processed)
                imsave(fname="%s/%i.b.png" % (out_dir, ctr), arr=pred_y_processed)
                ctr += 1
