from __future__ import print_function

import sys
# sys.path.append('../../')
sys.path.insert(0, '.')
import os

import os.path as osp
import numpy as np
import shutil
import json


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def mkdir_if_nonexist(split_dir):
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)


def move_images_to_dir(image_list, old_dir, old_file_format, new_dir, new_file_format):
    for id in image_list:
        id = id + 1
        old_im_name = old_file_format.format(id)
        new_im_name = new_file_format.format(id)
        old_im_name = osp.join(old_dir, old_im_name)
        new_im_name = osp.join(new_dir, new_im_name)
        shutil.copy(old_im_name, new_im_name)

root = '/export/reid_datasets/transformed_collection/cuhk01'
splits = read_json(osp.join(root, 'splits486.json'))
img_root = osp.join(root, 'images')


for i, split in enumerate(splits):
    trainval = split['trainval']
    query = split['query']
    gallery = split['gallery']

    # split_dir = osp.join(root, 'split_' + str(i))
    split_dir = osp.join(root, 'test486/split_' + str(i))

    mkdir_if_nonexist(split_dir)
    old_dir = img_root

    new_im_dir= osp.join(split_dir, 'bounding_box_train')
    mkdir_if_nonexist(new_im_dir)
    old_file_format = '{:04d}001.png'
    new_file_format = '{:04d}_c0_f0000000.png'
    new_dir = new_im_dir
    move_images_to_dir(trainval, old_dir, old_file_format, new_dir, new_file_format)

    old_file_format = '{:04d}002.png'
    new_file_format = '{:04d}_c0_f0000001.png'
    new_dir = new_im_dir
    move_images_to_dir(trainval, old_dir, old_file_format, new_dir, new_file_format)

    old_file_format = '{:04d}003.png'
    new_file_format = '{:04d}_c1_f0000000.png'
    new_dir = new_im_dir
    move_images_to_dir(trainval, old_dir, old_file_format, new_dir, new_file_format)

    old_file_format = '{:04d}004.png'
    new_file_format = '{:04d}_c1_f0000001.png'
    new_dir = new_im_dir
    move_images_to_dir(trainval, old_dir, old_file_format, new_dir, new_file_format)


    new_im_dir = osp.join(split_dir, 'query')
    mkdir_if_nonexist(new_im_dir)
    old_file_format = '{:04d}001.png'
    new_file_format = '{:04d}_c0_f0000000.png'
    new_dir = new_im_dir
    move_images_to_dir(query, old_dir, old_file_format, new_dir, new_file_format)


    new_im_dir = osp.join(split_dir, 'bounding_box_test')
    mkdir_if_nonexist(new_im_dir)
    old_file_format = '{:04d}003.png'
    new_file_format = '{:04d}_c1_f0000000.png'
    new_dir = new_im_dir
    move_images_to_dir(gallery, old_dir, old_file_format, new_dir, new_file_format)

    # old_file_format = '{:04d}004.png'
    # new_file_format = '{:04d}_c1_f0000001.png'
    # new_dir = new_im_dir
    # move_images_to_dir(gallery, old_dir, old_file_format, new_dir, new_file_format)

    # exit(0)



exit
