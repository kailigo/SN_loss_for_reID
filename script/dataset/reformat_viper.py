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
        old_im_name = old_file_format.format(id)
        new_im_name = new_file_format.format(id)
        old_im_name = osp.join(old_dir, old_im_name)
        new_im_name = osp.join(new_dir, new_im_name)
        shutil.copy(old_im_name, new_im_name)


root = '/export/reid_datasets/transformed_collection/viper'
splits = read_json(osp.join(root, 'splits.json'))
img_root = osp.join(root, 'images')


for i, split in enumerate(splits):
    trainval = split['trainval']
    query = split['query']
    gallery = split['gallery']
    split_dir = osp.join(root, 'split_' + str(i))
    mkdir_if_nonexist(split_dir)
    old_dir = img_root

    new_im_dir= osp.join(split_dir, 'bounding_box_train')
    mkdir_if_nonexist(new_im_dir)
    old_file_format = '{:08d}_00_0000.jpg'
    new_file_format = '{:04d}_c0_f0000000.jpg'
    new_dir = new_im_dir
    move_images_to_dir(trainval, old_dir, old_file_format, new_dir, new_file_format)

    new_im_dir = osp.join(split_dir, 'bounding_box_train')
    mkdir_if_nonexist(new_im_dir)
    old_file_format = '{:08d}_01_0000.jpg'
    new_file_format = '{:04d}_c1_f0000000.jpg'
    new_dir = new_im_dir
    move_images_to_dir(trainval, old_dir, old_file_format, new_dir, new_file_format)

    new_im_dir = osp.join(split_dir, 'query')
    mkdir_if_nonexist(new_im_dir)
    old_file_format = '{:08d}_00_0000.jpg'
    new_file_format = '{:04d}_c0_f0000000.jpg'
    new_dir = new_im_dir
    move_images_to_dir(query, old_dir, old_file_format, new_dir, new_file_format)

    new_im_dir = osp.join(split_dir, 'bounding_box_test')
    mkdir_if_nonexist(new_im_dir)
    old_file_format = '{:08d}_01_0000.jpg'
    new_file_format = '{:04d}_c1_f0000000.jpg'
    new_dir = new_im_dir
    move_images_to_dir(gallery, old_dir, old_file_format, new_dir, new_file_format)


exit


dataset_name = "viper"
archive_basename = ''
fuel_root_path = '/export/reid_datasets/transformed_collection'


fuel_data_path = os.path.join(fuel_root_path, dataset_name)
extracted_dir_path = os.path.join(fuel_data_path, archive_basename)
archive_filepath = extracted_dir_path + ".tgz"
images_dir_path = os.path.join(extracted_dir_path, "images")
label_filepath = os.path.join(extracted_dir_path, "image_class_labels.txt")
image_list_filepath = os.path.join(extracted_dir_path, "images.txt")

id_name_pairs = np.loadtxt(image_list_filepath, np.str)
assert np.array_equal(
    [int(i) for i in id_name_pairs[:, 0].tolist()], range(1, 11789))
id_label_pairs = np.loadtxt(label_filepath, np.str)
assert np.array_equal(
    [int(i) for i in id_label_pairs[:, 0].tolist()], range(1, 11789))
jpg_filenames = id_name_pairs[:, 1].tolist()
class_labels = [int(i) for i in id_label_pairs[:, 1].tolist()]

num_examples = len(jpg_filenames)
num_clases = 200
assert np.array_equal(np.unique(class_labels), range(1, num_clases + 1))


test_head = class_labels.index(101)

train_labels = class_labels[:test_head]
train_file_names = jpg_filenames[:test_head]

test_labels = class_labels[test_head:num_examples]
test_file_names = jpg_filenames[test_head:num_examples]

# new_im_name_tmpl = '{:08d}_{:04d}_{:08d}.jpg'
save_dir = '/export/reid_datasets/CUB_200-2011'
new_im_dir = osp.join(save_dir, 'bounding_box_train')
# may_make_dir(new_im_dir)
if not os.path.exists(new_im_dir):
    os.makedirs(new_im_dir)

file_name_c1 = '{:04d}_c1_f{:07d}.jpg'
file_name_c2 = '{:04d}_c2_f{:07d}.jpg'

ninst = 0
prelabel = 0

# import pdb
# pdb.set_trace()

for i, im in enumerate(train_file_names):
    new_label = train_labels[i]
    nocr = train_labels.count(new_label)

    if new_label != prelabel:
        ninst = 0
    else:
        ninst = ninst + 1


    # train_labels == new_label
    if ninst < (nocr/2):
        new_im_name = file_name_c1.format(train_labels[i], ninst)
    else:
        new_im_name = file_name_c2.format(train_labels[i], ninst)

    print(new_im_name)
    old_im_name = osp.join(images_dir_path, im)
    shutil.copy(old_im_name, osp.join(new_im_dir, new_im_name))
    prelabel = new_label



# new_im_name_tmpl = '{:08d}_{:04d}_{:08d}.jpg'
save_dir = '/export/reid_datasets/CUB_200-2011'
new_im_dir_test = osp.join(save_dir, 'bounding_box_test')
new_im_dir_qry = osp.join(save_dir, 'query')
# new_im_dir = osp.join(save_dir, 'bounding_box_test')
# may_make_dir(new_im_dir)
if not os.path.exists(new_im_dir_test):
    os.makedirs(new_im_dir_test)
if not os.path.exists(new_im_dir_qry):
    os.makedirs(new_im_dir_qry)

# file_name = '{:04d}_c2_f{:07d}.jpg'
file_name_test = '{:04d}_c1_f{:07d}.jpg'
file_name_qry = '{:04d}_c2_f{:07d}.jpg'
ninst = 0
prelabel = 0
for i, im in enumerate(test_file_names):
    # new_im_name = file_name.format(test_labels[i], ninst)
    new_label = test_labels[i]
    if new_label != prelabel:
        ninst = 0
    else:
        ninst = ninst + 1

    if ninst == 0:
        new_im_name = file_name_qry.format(test_labels[i], ninst)
        new_im_dir = new_im_dir_qry
    else:
        new_im_name = file_name_test.format(test_labels[i], ninst)
        new_im_dir = new_im_dir_test

    old_im_name = osp.join(images_dir_path, im)

    shutil.copy(old_im_name, osp.join(new_im_dir, new_im_name))
    prelabel = new_label

exit()


# new_im_name_tmpl = '{:08d}_{:04d}_{:08d}.jpg'
# save_dir = '/export/reid_datasets/CUB_200-2011'
new_im_dir = osp.join(save_dir, 'query')
# new_im_dir = osp.join(save_dir, 'bounding_box_test')
# may_make_dir(new_im_dir)
if not os.path.exists(new_im_dir):
    os.makedirs(new_im_dir)

file_name = '{:04d}_c1_f{:07d}.jpg'
ninst = 0
prelabel = 1
for i, im in enumerate(test_file_names):
    new_im_name = file_name.format(test_labels[i], ninst)
    new_label = test_labels[i]
    if new_label != prelabel:
        ninst = 0
    else:
        ninst = ninst + 1
    old_im_name = osp.join(images_dir_path, im)
    shutil.copy(old_im_name, osp.join(new_im_dir, new_im_name))
    prelabel = new_label


exit()
# import pdb
# pdb.set_trace()



targets = np.array(class_labels, np.int32).reshape(num_examples, 1)
test_head = class_labels.index(101)
split_train, split_test = (0, test_head), (test_head, num_examples)

train_labels = class_labels[:test_head]
test_labels = class_labels[test_head:num_examples]

train_labels[:] = [x - 1 for x in train_labels]
test_labels[:] = [x - 101 for x in test_labels]

train_images = image_all[:test_head]
test_images = image_all[test_head:num_examples]

train_images = np.array(train_images)
test_images = np.array(test_images)
test_labels = np.array(test_labels)
train_labels = np.array(train_labels)