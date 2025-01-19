import os
import tqdm
import torch
from PIL import Image
import numpy as np
import tarfile


def stream_tar_contents(tar_file_path, no_transform=False):
    with tarfile.open(tar_file_path, "r:gz") as tar:
        for member in tar:
            if member.isfile():
                if os.path.splitext(member.name)[-1] == ".jpg":
                    file_obj = tar.extractfile(member)
                    if file_obj:

                        img = Image.open(file_obj).convert("RGB")
                        
                        file_obj.close()
                        
                        yield (member.name, img)
                        
class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, tar_files):
        self.tar_files = tar_files
        
    def __iter__(self):
        for tar_file in self.tar_files:
            content = stream_tar_contents(tar_file)
            for image_path, image in content:
                filename = image_path.split("/")[-1]
                index = int(os.path.splitext(filename)[0].replace("sa_", ""))
                original_size = (image.height, image.width)
                try:
                    info = dict(image_path=image_path, index=index, original_size=original_size)
                    yield image, info
                    
                except ValueError as e:
                    print(f"Error: {e}, skipping file {filename} index {index}")
                    continue


def run(dataloader):
    box_info = []

    for ind, (images, info) in enumerate(tqdm.tqdm(dataloader)):
        info['index'] = [info['index']]
        info['original_size'] = [info['original_size']]
        
        for index, original_size in zip(info['index'], info['original_size']):
            box_info.append([index, original_size])

    
    box_info = sorted(box_info, key=lambda item: item[0])
    
    np.save(save_path, np.array(box_info, dtype=object))
    print(f"Saved to {save_path}")


import sys

tar_files = [sys.argv[1]]
print(tar_files)

dataset = Dataset(tar_files)
assert len(dataset.tar_files) == 1
filename = os.path.splitext(dataset.tar_files[0].split("/")[-1])[0] + ".npy"
os.makedirs("extra_info", exist_ok=True)
save_path = "extra_info/" + filename
if os.path.exists(save_path):
    print(f"File {save_path} exists, skipping")
    exit()

dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=1, pin_memory=True)
run(dataloader)
