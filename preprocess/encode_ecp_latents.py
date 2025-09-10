import os
import json
import glob
import torch
import tqdm
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
from typing import Tuple
import argparse
from torch.cuda.amp import autocast

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ECPDataset(torch.utils.data.IterableDataset):
    def __init__(self, root_folder, transform=None, subset=None):
        self.root_folder = root_folder
        self.transform = transform
        self.samples = []

        for class_idx, class_name in enumerate(sorted(os.listdir(root_folder))):
            class_path = os.path.join(root_folder, class_name)
            if not os.path.isdir(class_path):
                continue

            if subset is not None and class_name in subset:
                cls_img_names = subset[class_name]
            else:
                cls_img_names = os.listdir(class_path)

            for fname in cls_img_names:
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(class_path, fname), class_idx, class_name))
                else:
                    print(f"Skipping non-image file: {fname}")

    def __iter__(self):
        for path, cls_id, cls_name in self.samples:
            try:
                image = Image.open(path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                else:
                    image = transforms.ToTensor()(image)
                yield image, {"path": path, "idx": cls_id, "cls": cls_name}
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue


def vae_encode(images, vae):
    latents = vae.encode(images.to(torch.float32)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    return latents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", type=str, required=True, help="Root folder containing image classes")
    parser.add_argument("--latent_root_folder", type=str, required=True, help="Folder to save latents")
    parser.add_argument("--filtered_file", type=str, default=None, help="JSON file with filtered images")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument("--is_qwen_vae", type = bool, default = False) 
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    use_accelerator = device.startswith("cuda")
    
    if use_accelerator:
        from accelerate import Accelerator
        accelerator = Accelerator()
        device = accelerator.device  # accelerator handles multi-GPU automatically
    else:
        accelerator = None

    # Load subset
    subset = None
    if args.filtered_file and os.path.exists(args.filtered_file):
        with open(args.filtered_file, "r") as f:
            subset = json.load(f)

    # Dataset & Dataloader
    dataset = ECPDataset(args.root_folder, subset=subset)
    save_folder = f"latents-{args.resolution}"
    save_dir = os.path.join(args.latent_root_folder, save_folder)
    os.makedirs(save_dir, exist_ok=True)

    # Load VAE
    if not args.is_qwen_vae:
        vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae")
    else:
        vae = AutoencoderKL.from_pretrained("Qwen/Qwen-Image", subfolder= "vae")
    vae.to(device)
    vae.eval()

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True
    )
    if use_accelerator:
        vae = accelerator.prepare(vae)

    # Encode & Save Latents
    for ind, (images, info) in enumerate(tqdm.tqdm(dataloader, total=len(dataset.samples)//args.batch_size + 1)):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.shape[1] != 3:
            images = images.permute(0, 3, 1, 2)

        images = images.to(device, non_blocking=True)
        if use_accelerator:
            with torch.inference_mode(), accelerator.autocast():
                latents = vae_encode(images, vae)
        else:
            # Only use autocast if GPU is available
            if device.startswith("cuda"):
                with torch.inference_mode(), torch.cuda.amp.autocast():
                    latents = vae_encode(images, vae)
            else:
                with torch.inference_mode():
                    latents = vae_encode(images, vae)

        batch_indices = info["idx"] if isinstance(info["idx"], list) else [info["idx"]]
        batch_paths = info["path"] if isinstance(info["path"], list) else [info["path"]]

        for path, idx, latents_item in zip(batch_paths, batch_indices, latents):
            save_path = os.path.join(save_dir, os.path.splitext(os.path.basename(path))[0] + ".npy")
            if os.path.exists(save_path):
                continue
            latents_item = latents_item.to(torch.float16).cpu().numpy()
            np.save(save_path, latents_item)


if __name__ == "__main__":
    main()
