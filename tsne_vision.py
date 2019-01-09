from __future__ import print_function

import os
import os.path as osp
import numpy as np
from PIL import Image as Im
import sys

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# from gen_path import gen_path_list

def gen_path_list(data_folder):
    path_list = []
    # for dir_name in sorted(os.listdir(data_folder)):
    #     if dir_name[0] == '.':
    #         continue
    for filename in sorted(os.listdir(data_folder)):
        if filename[0] == '.':
            continue
        path_list.append(filename)
    return path_list


feature_path = "./temp/market_feat.npy"
label_path  = './temp/market_label.npy'

data_folder = "./temp/tsne"

tsne_path = 'tsne.npy'


vision_img = 'tsne-cuhk03-OS-detected.jpg'



X = np.load(feature_path)
Y = np.load(label_path)

# import pdb
# pdb.set_trace()

# n_train_samples = len(X)
# nsamples = n_train_samples
import pdb
pdb.set_trace()


n_train_samples = len(X)
nsamples = len(Y)


# PCA reduce dimension to accelerate the speed of TSNE

# X_pca = PCA(n_components=32).fit_transform(X)
# X_train = X_pca[:n_train_samples]

X_train = X

# TSNE dimension reduction
X_train_embedded = TSNE(n_components=2, perplexity=4, verbose=2).fit_transform(X_train)
print(type(X_train_embedded))

np.save(tsne_path, X_train_embedded)


# plot figure
tsne = np.load(tsne_path)

#  you can modify here to random sample
tsne = tsne[:nsamples]
tsne -= tsne.min()
expand_factor = 2
tsne *= expand_factor
size = np.ceil(tsne.max()) + 64
size = int(size)
print("printing poster size : (%d, %d)" % (size, size))

# generate images list

path_list = gen_path_list(data_folder)

#  you can modify here to random sample like this
# path_list = path_list[index]

print('there are %05d images in %s' % (len(path_list), data_folder))
images = []
for i, filename in enumerate(path_list):
    img_path = osp.join(data_folder, filename)
    # print(osp.exists(img_path))
    x = Im.open(img_path)
    x = np.array(x)

    # only RGB image is allowed
    if not len(x.shape) == 3:
        x = np.ones((50, 50, 3), dtype=int) * 255
        x = x.astype("uint8")
    images.append(x)
    if i >= nsamples-1:
        break

# poster
bigpic = np.ones((size, size, 3), dtype=int) * 255
for k, (im, coord) in enumerate(zip(images, tsne)):
    print("%.2f %% completed\r" % ((k + 1) * 100. / nsamples))
    sys.stdout.flush()
    x, y = int(coord[0]), int(coord[1])
    im = Im.fromarray(im).resize((50, 50), Im.ANTIALIAS)
    img = np.array(im).astype("uint8")
    bigpic[x:x + 50, y:y + 50, :] = np.array(im).astype("uint8")
    bigpic = bigpic.astype('uint8')

Im.fromarray(bigpic).convert('RGB').save(vision_img)
