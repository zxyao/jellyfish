from pprint import pprint
from random import shuffle
import os
import time
import argparse
import numpy as np

import keras.backend as K
from keras.models import load_model
from keras.optimizers import RMSprop, SGD, Adam

from model import BASIC_D, UNET_G
from util import load_data, read_image, save_fig, minibatch

parser = argparse.ArgumentParser("train")

parser.add_argument('--resume', action='store_true')
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--model-dir', type=str, default='checkpoints')
parser.add_argument('--display-every', type=int, default=10)
parser.add_argument('--nepochs', type=int, default=50)
parser.add_argument('--direction', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--image-size', type=int, default=256)
parser.add_argument('--load-size', type=int, default=286)
parser.add_argument('--lr-d', type=float, default=2e-4)
parser.add_argument('--lr-g', type=float, default=2e-4)
args = parser.parse_args()

pprint(args)

data = args.data
direction = args.direction
trainAB = load_data('datasets/{}/train/*.jpg'.format(data))
valAB = load_data('datasets/{}/val/*.jpg'.format(data))

assert len(trainAB) and len(valAB)

batch_size = args.batch_size
image_size, load_size = args.image_size, args.load_size
nepochs = args.nepochs
lr_d, lr_g = args.lr_d, args.lr_g

model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print("Created directory %s" % model_dir)
if args.resume:
    netD = load_model(os.path.join(model_dir, 'netD.h5'))
    netG = load_model(os.path.join(model_dir, 'netG.h5'))
else:
    nc_in, nc_out = 3, 3
    ngf, ndf = 64, 64
    netD = BASIC_D(nc_in, nc_out, ndf)
    netG = UNET_G(image_size, nc_in, nc_out, ngf)

real_A = netG.input
fake_B = netG.output
netG_generate = K.function([real_A], [fake_B])

def netG_gen(A):
    return np.concatenate([netG_generate([A[i:i+1]])[0] for i in range(A.shape[0])], axis=0)

real_B = netD.inputs[1]
output_D_real = netD([real_A, real_B])
output_D_fake = netD([real_A, fake_B])

loss_fn = lambda output, target : -K.mean(K.log(output+1e-12)*target+K.log(1-output+1e-12)*(1-target))

loss_D_real = loss_fn(output_D_real, K.ones_like(output_D_real))
loss_D_fake = loss_fn(output_D_fake, K.zeros_like(output_D_fake))
loss_G_fake = loss_fn(output_D_fake, K.ones_like(output_D_fake))

loss_L1 = K.mean(K.abs(fake_B-real_B))

loss_D = loss_D_real + loss_D_fake
training_updates = Adam(lr=lr_d, beta_1=0.5).get_updates(netD.trainable_weights,[],loss_D)
netD_train = K.function([real_A, real_B],[loss_D/2], training_updates)

loss_G = loss_G_fake + 100 * loss_L1
training_updates = Adam(lr=lr_g, beta_1=0.5).get_updates(netG.trainable_weights,[], loss_G)
netG_train = K.function([real_A, real_B], [loss_G_fake, loss_L1], training_updates)

t0 = time.time()
gen_iter = 0
errL1 = epoch = errG = 0
errL1_sum = errG_sum = errD_sum = 0

train_batch = minibatch(trainAB, batch_size, load_size, image_size, direction)
val_batch = minibatch(valAB, 6, load_size, image_size, direction)

display_every = args.display_every
while epoch < nepochs: 
    epoch, trainA, trainB = next(train_batch)
    errD,  = netD_train([trainA, trainB])
    errD_sum += errD
    errG, errL1 = netG_train([trainA, trainB])
    errG_sum += errG
    errL1_sum += errL1
    if gen_iter % display_every == 0:
        print('[%d/%d][%d] Loss_D: %f Loss_G: %f loss_L1: %f'
        % (epoch, nepochs, gen_iter, errD_sum/display_every, 
           errG_sum/display_every, errL1_sum/display_every), time.time()-t0)
        _, valA, valB = train_batch.send(6) 
        fakeB = netG_gen(valA)
        save_fig(np.concatenate([valA, valB, fakeB], axis=0), image_size, gen_iter, 'train')
        errL1_sum = errG_sum = errD_sum = 0
        _, valA, valB = next(val_batch)
        fakeB = netG_gen(valA)
        save_fig(np.concatenate([valA, valB, fakeB], axis=0), image_size, gen_iter, 'val')
    netD.save(os.path.join(model_dir, 'netD_epoch%d.h5' % epoch))
    netG.save(os.path.join(model_dir, 'netG_epoch%d.h5' % epoch))
    #print("Saved checkpoint at epoch %d" % epoch)
    gen_iter += 1

_, valA, valB = train_batch.send(6) 
fakeB = netG_gen(valA)
save_fig(np.concatenate([valA, valB, fakeB], axis=0), image_size, gen_iter, 'train')

_, valA, valB = next(val_batch)
fakeB = netG_gen(valA)
save_fig(np.concatenate([valA, valB, fakeB], axis=0), image_size, gen_iter, 'val')
