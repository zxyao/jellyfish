from pprint import pprint
from random import shuffle
import os
import sys
import time
import argparse
import numpy as np

import keras.backend as K
from keras.models import load_model
from keras.optimizers import RMSprop, SGD, Adam

from model import BASIC_D, UNET_G
from util import load_data, read_image, save_fig, minibatch

parser = argparse.ArgumentParser("test")

parser.add_argument('--data', type=str, required=True)
parser.add_argument('--model-dir', type=str, default='checkpoints')
parser.add_argument('--direction', type=int, default=0)
parser.add_argument('--image-size', type=int, default=256)
parser.add_argument('--load-size', type=int, default=286)
args = parser.parse_args()

pprint(args)

data = args.data
direction = args.direction

testAB = load_data('datasets/{}/test/*.jpg'.format(data))

assert len(testAB)

image_size, load_size = args.image_size, args.load_size

model_dir = args.model_dir

netD = load_model(os.path.join(model_dir, 'netD.h5'))
netG = load_model(os.path.join(model_dir, 'netG.h5'))


real_A = netG.input
fake_B = netG.output
netG_generate = K.function([real_A], [fake_B])

def netG_gen(A):
    return np.concatenate([netG_generate([A[i:i+1]])[0] for i in range(A.shape[0])], axis=0)

gen_iter = 0

test_batch = minibatch(testAB, 6, load_size, image_size, direction)

for i in range(3):
    _, testA, testB = next(test_batch)
    fakeB = netG_gen(testA)
    save_fig(np.concatenate([testA, testB, fakeB], axis=0), image_size, gen_iter, 'test')
    gen_iter += 1

