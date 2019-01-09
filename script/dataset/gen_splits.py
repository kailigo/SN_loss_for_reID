# Randomly create ten training and test split
import numpy as np
import json
import os.path as osp
import os
import errno


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))

num = 971
splits = []
for _ in range(10):
    pids = np.random.permutation(num).tolist()
    trainval_pids = sorted(pids[:871])
    test_pids = sorted(pids[871:])
    split = {'trainval': trainval_pids,
             'query': test_pids,
             'gallery': test_pids}
    splits.append(split)
write_json(splits, osp.join('/export/reid_datasets/transformed_collection/cuhk01/splits_100.json'))