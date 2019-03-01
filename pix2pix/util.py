from PIL import Image
import numpy as np
import glob
from random import randint, shuffle
import os

channel_first = False

def load_data(file_pattern):
    return glob.glob(file_pattern)

def read_image(fn, load_size, image_size, direction=0):
    im = Image.open(fn)
    im = im.resize((load_size*2, load_size), Image.BILINEAR)
    arr = np.array(im).astype(float)
    arr = arr / 255. * 2. - 1.
    w1, w2 = (load_size-image_size)/2, (load_size+image_size)/2
    h1, h2 = w1, w2
    imgA = arr[h1:h2, load_size+w1:load_size+w2, :]
    imgB = arr[h1:h2, w1:w2, :]
    if randint(0, 1):
        imgA = imgA[:, ::-1]
        imgB = imgB[:, ::-1]
    if channel_first:
        imgA = np.moveaxis(imgA, 2, 0)
        imgB = np.moveaxis(imgB, 2, 0)
    if direction == 0:
        return imgA, imgB
    else:
        return imgB, imgA

def save_fig(X, image_size, gen_iter, mode, rows=3):
    if not os.path.exists('results'):
        os.makedirs('results')
        print("Created directory results")
    assert X.shape[0] % rows == 0
    int_X = ((X + 1) / 2 * 255).clip(0, 255).astype('uint8')
    if channel_first:
        int_X = np.moveaxis(int_X.reshape(-1, 3, image_size, image_size), 1, 3)
    else:
        int_X = int_X.reshape(-1, image_size, image_size, 3)
    int_X = int_X.reshape(rows, -1, image_size, image_size, 3).swapaxes(1, 2).reshape(rows*image_size, -1, 3)
    int_X = Image.fromarray(int_X)
    int_X.save("results/%s_result%d.jpg" % (mode, gen_iter))

def minibatch(dataAB, batchsize, load_size, image_size, direction=0):
    length = len(dataAB)
    epoch = i = 0
    tmpsize = None
    while True:
        size = tmpsize if tmpsize else batchsize
        if i + size > length:
            shuffle(dataAB)
            i = 0
            epoch += 1
        dataA = []
        dataB = []
        for j in range(i, i+size):
            imgA, imgB = read_image(dataAB[j], load_size, image_size, direction)
            dataA.append(imgA)
            dataB.append(imgB)
        dataA = np.float32(dataA)
        dataB = np.float32(dataB)
        i += size
        tmpsize = yield epoch, dataA, dataB
