import numpy as np
import os.path as osp
ospj = osp.join
ospeu = osp.expanduser

from ..utils.utils import load_pickle
from ..utils.dataset_utils import parse_im_name
from .TrainSet import TrainSet
from .TestSet import TestSet


def create_dataset(
    name='market1501',
    part='trainval',
    data_path='/export/reid_datasets/transformed_collection/Market1501',
    **kwargs):
  assert name in ['market1501', 'cuhk03', 'duke', 'combined', 'viper', 'cuhk01', 'cuhk03_OS'], \
    "Unsupported Dataset {}".format(name)

  assert part in ['trainval', 'train', 'val', 'test'], \
    "Unsupported Dataset Part {}".format(part)

  ########################################
  # Specify Directory and Partition File #
  ########################################

  im_dir = ospeu(osp.join(data_path, 'images'))
  partition_file = ospeu(osp.join(data_path, 'partitions.pkl'))



  # if name == 'market1501':
  #   im_dir = ospeu(osp.join(dataset_path, 'images'))
  #   partition_file = ospeu(osp.join(dataset_path, 'partitions.pkl'))
  #
  # elif name == 'cuhk03':
  #   im_type = ['detected', 'labeled'][0]
  #   im_dir = ospeu(ospj('/export/reid_datasets/transformed_collection/CUHK03', im_type, 'images'))
  #   partition_file = ospeu(ospj('/export/reid_datasets/transformed_collection/CUHK03', im_type, 'partitions.pkl'))
  #
  # elif name == 'duke':
  #   im_dir = ospeu('/export/reid_datasets/transformed_collection/DukeMTMC-reID/images')
  #   partition_file = ospeu('/export/reid_datasets/transformed_collection/DukeMTMC-reID/partitions.pkl')
  #
  # elif name == 'viper':
  #   im_dir = ospeu('/export/reid_datasets/transformed_collection/viper/split_0/images')
  #   partition_file = ospeu('/export/reid_datasets/transformed_collection/viper/split_0/partitions.pkl')
  #
  # elif name == 'cuhk01':
  #   im_dir = ospeu('/export/reid_datasets/transformed_collection/cuhk01/test100/split_0/images')
  #   partition_file = ospeu('/export/reid_datasets/transformed_collection/cuhk01/test100/split_0/partitions.pkl')
  #
  # elif name == 'cuhk03_OS':
  #   im_dir = ospeu('/export/reid_datasets/transformed_collection/CUHK03/labeled/official_split/0/images')
  #   partition_file = ospeu('/export/reid_datasets/transformed_collection/CUHK03/labeled/official_split/0/partitions.pkl')
  #
  # elif name == 'combined':
  #   assert part in ['trainval'], \
  #     "Only trainval part of the combined dataset is available now."
  #   im_dir = ospeu('/export/reid_datasets/transformed_collection/combined/trainval_images')
  #   partition_file = ospeu('/export/reid_datasets/transformed_collection/combined/partitions.pkl')


  ##################
  # retrieval #
  ##################
  # elif name == 'CUB':
  #   im_dir = ospeu('/export/retrieval/cub_200_2011/images')
  #   partition_file = ospeu('/export/retrieval/cub_200_2011/partitions.pkl')


  ##################
  # Create Dataset #
  ##################

  # Use standard Market1501 CMC settings for all datasets here.
  cmc_kwargs = dict(separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)

  partitions = load_pickle(partition_file)
  im_names = partitions['{}_im_names'.format(part)]

  if part == 'trainval':
    ids2labels = partitions['trainval_ids2labels']

    ret_set = TrainSet(
      im_dir=im_dir,
      im_names=im_names,
      ids2labels=ids2labels,
      **kwargs)

  elif part == 'train':
    ids2labels = partitions['train_ids2labels']

    ret_set = TrainSet(
      im_dir=im_dir,
      im_names=im_names,
      ids2labels=ids2labels,
      **kwargs)

  elif part == 'val':
    marks = partitions['val_marks']
    kwargs.update(cmc_kwargs)

    ret_set = TestSet(
      im_dir=im_dir,
      im_names=im_names,
      marks=marks,
      **kwargs)

  elif part == 'test':
    marks = partitions['test_marks']
    kwargs.update(cmc_kwargs)

    ret_set = TestSet(
      im_dir=im_dir,
      im_names=im_names,
      marks=marks,
      **kwargs)

  if part in ['trainval', 'train']:
    num_ids = len(ids2labels)
  elif part in ['val', 'test']:
    ids = [parse_im_name(n, 'id') for n in im_names]
    num_ids = len(list(set(ids)))
    num_query = np.sum(np.array(marks) == 0)
    num_gallery = np.sum(np.array(marks) == 1)
    num_multi_query = np.sum(np.array(marks) == 2)

  # Print dataset information
  print('-' * 40)
  print('{} {} set'.format(name, part))
  print('-' * 40)
  print('NO. Images: {}'.format(len(im_names)))
  print('NO. IDs: {}'.format(num_ids))

  try:
    print('NO. Query Images: {}'.format(num_query))
    print('NO. Gallery Images: {}'.format(num_gallery))
    print('NO. Multi-query Images: {}'.format(num_multi_query))
  except:
    pass

  print('-' * 40)

  return ret_set
