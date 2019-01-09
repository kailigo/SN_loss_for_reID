from __future__ import print_function
import os
import os.path as osp
import numpy as np
from PIL import Image as Im
import sys

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# from gen_path import gen_path_list


# feature_path = "represent.npy"
# tsne_path = 'tsne.npy'
# vision_img = 'tsne-cub.jpg'
# X = np.load(feature_path)

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
im_names_path  = './temp/market_im_names.npy'
label_path = './temp/market_label.npy'


tsne_path = 'tsne.npy'
vision_img = 'tsne-market.jpg'
# data_folder = "./temp/tsne"
X = np.load(feature_path)
Y = np.load(label_path)
im_names = np.load(im_names_path)

# import pdb
# pdb.set_trace()

# data_folder = "/Users/wangxun/DataSet/CUB_200_2011/test"

data_folder='/export/reid_datasets/transformed_collection/Market1501/images'
# n_train_samples = len(X)
# nsamples = n_train_samples
n_train_samples = len(im_names)
nsamples = len(im_names)


# PCA reduce dimension to accelerate the speed of TSNE

# X_pca = PCA(n_components=32).fit_transform(X)
# X_train = X_pca[:n_train_samples]
# X_train_embedded = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(X_train)
# print(type(X_train_embedded))
# np.save(tsne_path, X_train_embedded)

# TSNE dimension reduction

# plot figure
tsne = np.load(tsne_path)

# import pdb
# pdb.set_trace()
# idx = np.arange(nsamples)

# np.random.shuffle(idx)
# temp_tsne = [tsne[x] for x in idx]
# temp_im_names = [im_names[x] for x in idx]

# import pdb
# pdb.set_trace()
#
# num = 5000
# tsne = np.array(temp_tsne[:num])
# im_names = np.array(temp_im_names[:num])

keep_ids_num = 400
idx = Y.tolist().index(keep_ids_num)

# import pdb
# pdb.set_trace()

im_names = im_names[:idx]
X = X[:idx]
tsne = tsne[:idx]
nsamples = len(tsne)

vision_img = 'tsne-market_'+str(keep_ids_num)+'.jpg'


#  you can modify here to random sample
tsne = tsne[:nsamples]
tsne -= tsne.min()
expand_factor = 800
tsne *= expand_factor
size = np.ceil(tsne.max()) + 256
size = int(size)

# import pdb
# pdb.set_trace()

print("printing poster size : (%d, %d)" % (size, size))

# generate images list

# path_list = gen_path_list(data_folder)

#  you can modify here to random sample like this
# path_list = path_list[index]

# print('there are %05d images in %s' % (len(path_list), data_folder))
images = []
for i, filename in enumerate(im_names):

    # import pdb
    # pdb.set_trace()

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
    im = Im.fromarray(im).resize((128, 256), Im.ANTIALIAS)
    # im = Im.fromarray(im)

    img = np.array(im).astype("uint8")

    # a = bigpic[x:x + img.shape[0], y:y + img.shape[1], :]
    #
    # print(img.shape[0])
    # print(img.shape[1])
    # print(a.shape[0])
    # print(a.shape[1])
    # print(x)
    # print(y)
    bigpic[x:x + img.shape[0], y:y + img.shape[1], :] = np.array(im).astype("uint8")



    bigpic = bigpic.astype('uint8')

Im.fromarray(bigpic).convert('RGB').save(vision_img)
