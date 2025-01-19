import os
import sys

import numpy as np

box_info_path = '/home/user/jiuntian/data/sa-1b_boxes'
ori_size_path = 'extra_info'
new_info_path = '/home/user/jiuntian/data/sa-1b_boxes_sdxl'

npy_files = [sys.argv[1]]
print(npy_files)
assert len(npy_files) == 1

filename = os.path.splitext(npy_files[0].split("/")[-1])[0] + ".npy"
os.makedirs(new_info_path, exist_ok=True)
save_path = os.path.join(new_info_path, filename)
if os.path.exists(save_path):
    print(f"File {save_path} exists, skipping")
    exit()

box_info = np.load(os.path.join(box_info_path, filename), allow_pickle=True)
original_sizes = np.load(os.path.join(ori_size_path, filename), allow_pickle=True)

# indices (iterating indices, sequential), index (image index, follow to file name)
# box_info (from igligen) does not contain all images, thus we have to find and match it
box_info_indices_to_index_map = {i:info[0] for i, info in enumerate(box_info)}
# box_info_index_to_indices_map = {info[0]:i for i, info in enumerate(box_info)}
original_sizes_index_to_indices_map = {info[0]:i for i, info in enumerate(original_sizes)}
# assert len(box_info) == len(original_sizes), f"length of box_info={len(box_info)} and original_sizes={len(original_sizes)} mismatched. at {filename}"

original_size_tuples = np.empty(shape=(len(box_info), 1), dtype=object)
for i in range(len(box_info)):
    # assert row[0] == box_info[i, 0], f'index mismatched, new_info has {row[0]}, box_info has {box_info[i, 0]}'
    im_index = box_info_indices_to_index_map[i]
    ori_size_indices = original_sizes_index_to_indices_map[im_index]
    ori_size_info = original_sizes[ori_size_indices]
    assert ori_size_info[0] == im_index
    original_size = ori_size_info[1] # original_size for index
    original_size_tuples[i, 0] = tuple(original_size)
    
new_box_info = np.concatenate([box_info, original_size_tuples], axis=1)

new_box_info = sorted(new_box_info, key=lambda item: item[0])
    
np.save(save_path, np.array(new_box_info, dtype=object))
print(f"Saved to {save_path}")