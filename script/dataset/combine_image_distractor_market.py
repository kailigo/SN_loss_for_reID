import os.path as osp
import numpy as np
import glob
import shutil

image_path = '/export/reid_datasets/Market-1501-v15.09.15/images'
distr_path = '/export/reid_datasets/Market-1501-v15.09.15/distractor/images'
dst_path = '/export/reid_datasets/Market-1501-v15.09.15/temp'

def get_im_names(im_dir, is_distr=False):
  """Get the image names in a dir. Optional to return numpy array, paths."""

  im_paths = glob.glob(osp.join(im_dir, '*.jpg'))
  im_paths.sort()
  # im_names = [osp.basename(path) for path in im_paths]
  good_images = []
  # distractors = []

  # distractors = {'cam1': [], 'cam2': [], 'cam3': [], 'cam4': [], 'cam5': [], 'cam6': []}

  # distractors = [[]] * 6

  distractors = [[] for i in range(6)]

  # import pdb
  # pdb.set_trace()

  for img_path in im_paths:
      name = osp.basename(img_path)
      name_parts = name.split('_')
      if int(name_parts[0]) != 0:
          good_images.append(img_path)
          continue

      if is_distr:
          cam = int(name_parts[1][1])-1
      else:
          cam = int(name_parts[1][3])-1

      # import pdb
      # pdb.set_trace()

      distractors[cam].append(img_path)

  return good_images, distractors


good_images, distractors = get_im_names(image_path)
# import pdb
# pdb.set_trace()

good_images1, distractors1 = get_im_names(distr_path, is_distr=True)

for itr, cam_paths in enumerate(distractors1):
    idx = len(distractors[itr])
    for distr_path in cam_paths:
        # name = osp.basename(distr_path)
        new_name = '00000000_{:04d}_{:08d}.jpg'.format(itr+1, idx)

        print(new_name)

        idx = idx + 1
        # import pdb
        # pdb.set_trace()
        shutil.copy(distr_path, osp.join(dst_path, new_name))










