#!/usr/bin/env python
import numpy as np
import lfw
from skimage.transform import resize
import h5py

def to_bc01(b01c):
    return np.transpose(b01c, (0, 3, 1, 2))


def to_b01c(bc01):
    return np.transpose(bc01, (0, 2, 3, 1))


def lfw_imgs(alignment, size, crop):
    imgs, names_idx, names = lfw.LFW(alignment).arrays()
    new_imgs = []
    for img in imgs:
        img = img[crop:-crop, crop:-crop]
        img = resize(img, (size, size, 3), order=3)
        new_imgs.append(img)
    imgs = np.array(new_imgs)
    return imgs


def create_lfw():
    imgs = lfw_imgs(alignment='deepfunneled', size=64, crop=50)

    # Shuffle images
    idxs = np.random.permutation(np.arange(len(imgs)))
    imgs = imgs[idxs]
    imgs = to_bc01(imgs)
    print imgs.shape
    return imgs


if __name__ == '__main__':
    x = create_lfw()
    f = h5py.File('lfw.hdf5', 'w')
    f.create_dataset('lfw', data=x)
    f.close()

