# preprocessing for cuhk03(detected)
# transform raw images to numpy arrays
# 1360 identities in total, 1160 for training, 100 for testing,
# 100 for validation (not used in this work)
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import random
import shutil

def move_images_to_dir(old_image_list, old_dir, new_image_list, new_dir):
    print(len(old_image_list))
    for i in range(len(old_image_list)):
        old_im_name = os.path.join(old_dir, old_image_list[i])
        new_im_name = os.path.join(new_dir, new_image_list[i])
        shutil.copy(old_im_name, new_im_name)


def main_labeled():

    root = '/export/reid_datasets/transformed_collection/CUHK03'
    test_index_all = pickle.load(open(os.path.join(root, 'cuhk03_testsets'), 'rb'))
    test_index = []
    old_dir = os.path.join(root, 'labeled/images/')

    import glob
    # old_name_format = '{:08d}_{:04d}_{:08d}.jpg'
    # new_file_format_c1 = '{:04d}_c0_f0000000.jpg'
    # new_file_format_c2 = '{:04d}_c1_f0000000.jpg'


    for current_seed in range(20):
        old_gal_images = []
        old_prb_images = []
        old_train_images = []

        new_gal_images = []
        new_prb_images = []
        new_train_images = []

        for x, y in test_index_all[current_seed]:
            if x == 1:
                y = 843 + y
            elif x == 2:
                y = 1283 + y
            y = int(y)

            test_index.append(y)
            npp = 0

            temp_old_prb_images = []
            temp_new_prb_images = []
            temp_old_gal_images = []
            temp_new_gal_images = []
            for names in glob.glob(os.path.join(root, 'labeled/images/{:08d}_*'.format(int(y)))):
                name = names.split('/')[-1]
                new_name = '{:04d}_c{:1d}_f{:07d}.jpg'.format(y, int(name[12]), npp)
                if int(name[12]) == 0:
                    temp_old_prb_images.append(name)
                    temp_new_prb_images.append(new_name)
                else:
                    temp_old_gal_images.append(name)
                    temp_new_gal_images.append(new_name)
                    npp = npp + 1

            if temp_old_prb_images == []:
                print('split {:d} only has less than 100 queries'.format(current_seed))
                continue

            old_prb_images.append(temp_old_prb_images[np.random.randint(len(temp_old_prb_images))])
            new_prb_images.append(temp_new_prb_images[np.random.randint(len(temp_new_prb_images))])
            old_gal_images.append(temp_old_gal_images[np.random.randint(len(temp_old_gal_images))])
            new_gal_images.append(temp_new_gal_images[np.random.randint(len(temp_new_gal_images))])

        print(len(old_prb_images))
        print(len(old_gal_images))

        train_index = [x for x in range(1467) if x not in test_index]
        for y in train_index:
            npp = 0
            for names in glob.glob(os.path.join(root, 'labeled/images/{:08d}_*'.format(y))):
                name = names.split('/')[-1]
                old_train_images.append(name)
                new_file_name = '{:04d}_c{:1d}_f{:07d}.jpg'.format(y, int(name[12]), npp)
                new_train_images.append(new_file_name)
                npp = npp + 1

        new_train_dir = os.path.join(root, 'labeled/official_split/'+str(current_seed)+'/bounding_box_train')
        if not os.path.exists(new_train_dir):
            os.makedirs(new_train_dir)
        move_images_to_dir(old_train_images, old_dir, new_train_images, new_train_dir)


        new_gal_dir = os.path.join(root, 'labeled/official_split/'+str(current_seed)+'/bounding_box_test')
        if not os.path.exists(new_gal_dir):
            os.makedirs(new_gal_dir)
        move_images_to_dir(old_gal_images, old_dir, new_gal_images, new_gal_dir)

        new_prb_dir = os.path.join(root, 'labeled/official_split/'+str(current_seed)+'/query')
        if not os.path.exists(new_prb_dir):
            os.makedirs(new_prb_dir)
        move_images_to_dir(old_prb_images, old_dir, new_prb_images, new_prb_dir)


if __name__ == '__main__':

    root = '/export/reid_datasets/transformed_collection/CUHK03'
    test_index_all = pickle.load(open(os.path.join(root, 'cuhk03_testsets'), 'rb'))
    test_index = []
    old_dir = os.path.join(root, 'detected/images/')

    import glob
    # old_name_format = '{:08d}_{:04d}_{:08d}.jpg'
    # new_file_format_c1 = '{:04d}_c0_f0000000.jpg'
    # new_file_format_c2 = '{:04d}_c1_f0000000.jpg'


    for current_seed in range(20):
        old_gal_images = []
        old_prb_images = []
        old_train_images = []

        new_gal_images = []
        new_prb_images = []
        new_train_images = []

        for x, y in test_index_all[current_seed]:
            if x == 1:
                y = 843 + y
            elif x == 2:
                y = 1283 + y
            y = int(y)

            test_index.append(y)
            npp = 0

            temp_old_prb_images = []
            temp_new_prb_images = []
            temp_old_gal_images = []
            temp_new_gal_images = []
            for names in glob.glob(os.path.join(root, 'detected/images/{:08d}_*'.format(int(y)))):
                name = names.split('/')[-1]
                new_name = '{:04d}_c{:1d}_f{:07d}.jpg'.format(y, int(name[12]), npp)
                if int(name[12]) == 0:
                    temp_old_prb_images.append(name)
                    temp_new_prb_images.append(new_name)
                else:
                    temp_old_gal_images.append(name)
                    temp_new_gal_images.append(new_name)
                    npp = npp + 1

            if temp_old_prb_images == []:
                print('split {:d} only has less than 100 queries'.format(current_seed))
                continue

            old_prb_images.append(temp_old_prb_images[np.random.randint(len(temp_old_prb_images))])
            new_prb_images.append(temp_new_prb_images[np.random.randint(len(temp_new_prb_images))])
            old_gal_images.append(temp_old_gal_images[np.random.randint(len(temp_old_gal_images))])
            new_gal_images.append(temp_new_gal_images[np.random.randint(len(temp_new_gal_images))])

        print(len(old_prb_images))
        print(len(old_gal_images))

        train_index = [x for x in range(1467) if x not in test_index]
        for y in train_index:
            npp = 0
            for names in glob.glob(os.path.join(root, 'detected/images/{:08d}_*'.format(y))):
                name = names.split('/')[-1]
                old_train_images.append(name)
                new_file_name = '{:04d}_c{:1d}_f{:07d}.jpg'.format(y, int(name[12]), npp)
                new_train_images.append(new_file_name)
                npp = npp + 1

        new_train_dir = os.path.join(root, 'detected/official_split/'+str(current_seed)+'/bounding_box_train')
        if not os.path.exists(new_train_dir):
            os.makedirs(new_train_dir)
        move_images_to_dir(old_train_images, old_dir, new_train_images, new_train_dir)


        new_gal_dir = os.path.join(root, 'detected/official_split/'+str(current_seed)+'/bounding_box_test')
        if not os.path.exists(new_gal_dir):
            os.makedirs(new_gal_dir)
        move_images_to_dir(old_gal_images, old_dir, new_gal_images, new_gal_dir)

        new_prb_dir = os.path.join(root, 'detected/official_split/'+str(current_seed)+'/query')
        if not os.path.exists(new_prb_dir):
            os.makedirs(new_prb_dir)
        move_images_to_dir(old_prb_images, old_dir, new_prb_images, new_prb_dir)