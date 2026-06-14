
import os
import numpy as np
import cv2
import h5py
from utils.util import *
from PIL import Image
import multiprocessing
from collections import defaultdict
import json


def is_valid_sub_dir(datadir):
    view_types = ["front", "back", "right", "left", "front_right", "front_left", "back_right", "back_left",
                 "top", "front_right_top", "front_left_top", "back_right_top", "back_left_top"]
    if not os.path.exists(datadir):
        return False

    sub_files = os.listdir(datadir)
    # rgb is intentionally not generated; require only what step-3 actually outputs:
    # normals + semantic (cmask/onlymask) + mask per view.
    in_files = [np.sum(['cmask_000_' + viewtype + '.png' in sub_files,
                        'normals_000_' + viewtype + '.png' in sub_files,
                        'onlymask_000_' + viewtype + '.png' in sub_files,
                        'mask_000_' + viewtype + '.png' in sub_files,
                        ]) == 4 for viewtype in view_types]
    if False in in_files:
        return False
    return True



class ParallelRenderDataset2H5Dataset:
    def __init__(self, base_path, paths, start, end, save_dir):
        self.start = start
        self.end = end
        self.base_path = base_path
        self.paths = [pp.split('/')[-1] for pp in paths[start:end]]
        self.save_dir = save_dir
        self.view_types = ["front", "back", "right", "left", "front_right", "front_left", "back_right", "back_left",
                 "top", "front_right_top", "front_left_top", "back_right_top", "back_left_top"]
        # one independent h5 file per chunk -> avoids one huge file; the dataloader reads all *.h5py in the dir
        self.h5py_path = os.path.join(self.save_dir, f'data_{start}_{end}.h5py')
        contents = self.real_all_data()
        if len(contents) >= 1:
            self.merge_to_current_h5py(contents)

    def parallel_read_data(self, current_dir):
        result = dict()
        result['label'] = current_dir
        result['count'] = self.start + self.paths.index(current_dir)

        if is_valid_sub_dir(os.path.join(self.base_path, current_dir)):
            result['isOK'] = True
            for file in os.listdir(os.path.join(self.base_path, current_dir)):
                if file == 'rotate.txt':
                    continue                      # binary (dill) cache, not a text matrix
                if file.endswith('txt'):
                    try:
                        result[file] = np.loadtxt(os.path.join(self.base_path, current_dir, file))
                    except (ValueError, UnicodeDecodeError):
                        continue                  # skip any non-text .txt
                elif file.endswith('png'):
                    data = Image.open(os.path.join(self.base_path, current_dir, file))
                    result[file] = data
        else:
            result['isOK'] = False
        return result

    def real_all_data(self):
        file_paths =  self.paths
        process_num = 8
        if len(file_paths) < process_num:
            process_num = len(file_paths)
        with multiprocessing.Pool(processes=process_num) as pool:
            contents = pool.map(self.parallel_read_data, file_paths)
        contents = [cont for cont in contents if cont['isOK']]
        return contents

    def merge_to_current_h5py(self, contents):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        # each chunk is its own file -> create fresh ('w'); no cross-chunk clearing needed
        dataset = h5py.File(self.h5py_path, 'w')

        summary_dict = defaultdict(list)

        content_keys = list(contents[0].keys())
        for i in range(len(content_keys)):
            key = content_keys[i]
            for cont in contents:
                summary_dict[key].append(cont[key])

        prefix = str(self.start) +'_' +str(self.end)
        for key in summary_dict.keys():
            print(key)
            if key == 'label':
                dataset.create_dataset(prefix + '_' + key, (len(summary_dict[key]), 1), 'S10', summary_dict[key])
            elif key == 'count':
                dataset.create_dataset(prefix + '_' + key,
                                       data=[np.array(count) for count in summary_dict[key]],
                                       compression="gzip", compression_opts=9)
            elif key.endswith("png"):
                dataset.create_dataset(prefix + '_' + key, data=[np.array(img) for img in summary_dict[key]], compression="gzip", compression_opts=9)
            elif key.endswith("txt"):
                dataset.create_dataset(prefix + '_' + key, data=[np.array(img) for img in summary_dict[key]],
                                       compression="gzip", compression_opts=9)

        dataset.close()



def summarize(basepath, paths, start, end, save_dir):
    merge = ParallelRenderDataset2H5Dataset(basepath, paths, start, end, save_dir)



