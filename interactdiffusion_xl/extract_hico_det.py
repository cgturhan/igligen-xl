import glob
import itertools
import os
import sys

from PIL import Image
from diffusers import AutoencoderKL
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
import numpy as np
import PIL.Image
import torch

torch.serialization.add_safe_globals([PIL.Image.Image])
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
device = "cuda"
resolution = 1024
min_box_size=0.001

vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path, subfolder="vae",
).to(device)

clip_cache = {}
if os.path.exists("clip_cache_sdxl.pth"):
    clip_cache = torch.load("clip_cache_sdxl.pth", map_location="cpu", weights_only=False)
tokenizer_one = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer_2",
    use_fast=False,
)
text_encoder_one = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder",
).to(device)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder_2",
).to(device)
tokenizers = [tokenizer_one, tokenizer_two]
text_encoders = [text_encoder_one, text_encoder_two]

def center_crop_arr(pil_image: PIL.Image.Image, image_size: int):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    WW, HH = pil_image.size

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)

    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    # at this point, the min of pil_image side is desired image_size
    performed_scale = image_size / min(WW, HH)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2

    info = {"performed_scale": performed_scale, 'crop_y': crop_y, 'crop_x': crop_x, "WW": WW, 'HH': HH}

    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size], info


def transform_image(pil_image: PIL.Image.Image, image_size: int, random_flip: bool):
        arr, info = center_crop_arr(pil_image, image_size)

        info["performed_flip"] = False
        if random_flip:
            arr = arr[:, ::-1]
            info["performed_flip"] = True

        arr = arr.astype(np.float32) / 127.5 - 1
        arr = np.transpose(arr, [2, 0, 1])

        return torch.tensor(arr), info
    

def to_valid(x0, y0, x1, y1, image_size, min_box_size):
    valid = True

    if x0 > image_size or y0 > image_size or x1 < 0 or y1 < 0:
        valid = False  # no way to make this box vide, it is completely cropped out
        return valid, (None, None, None, None)

    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, image_size)
    y1 = min(y1, image_size)

    if (x1 - x0) * (y1 - y0) / (image_size * image_size) < min_box_size:
        valid = False
        return valid, (None, None, None, None)

    return valid, (x0, y0, x1, y1)

 
def recalculate_box_and_verify_if_valid(x, y, w, h, trans_info, image_size, min_box_size):
    """
    x,y,w,h:  the original annotation corresponding to the raw image size.
    trans_info: what resizing and cropping have been applied to the raw image 
    image_size:  what is the final image size  
    """

    x0 = x * trans_info["performed_scale"] - trans_info['crop_x']
    y0 = y * trans_info["performed_scale"] - trans_info['crop_y']
    x1 = (x + w) * trans_info["performed_scale"] - trans_info['crop_x']
    y1 = (y + h) * trans_info["performed_scale"] - trans_info['crop_y']

    # at this point, box annotation has been recalculated based on scaling and cropping
    # but some point may fall off the image_size region (e.g., negative value), thus we 
    # need to clamp them into 0-image_size. But if all points falling outsize of image 
    # region, then we will consider this is an invalid box. 
    valid, (x0, y0, x1, y1) = to_valid(x0, y0, x1, y1, image_size, min_box_size)

    if valid:
        # we also perform random flip. 
        # Here boxes are valid, and are based on image_size 
        if trans_info["performed_flip"]:
            x0, x1 = image_size - x1, image_size - x0

    return valid, (x0, y0, x1, y1)

    
@torch.no_grad()
def vae_encode(images):
    latents = vae.encode(images.to(torch.float32)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    return latents

def encode_text(text, tokenizers, text_encoders):
    if text in clip_cache:
        return clip_cache[text]
    gligen_pooler_embeds = []
    
    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                text,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=False,
                return_dict=True,
            )
            
            if isinstance(text_encoder, CLIPTextModel):
                gligen_pooler_embeds.append(prompt_embeds.pooler_output)
            elif isinstance(text_encoder, CLIPTextModelWithProjection):
                gligen_pooler_embeds.append(prompt_embeds.text_embeds)
    
    gligen_embeds = torch.concat(gligen_pooler_embeds, dim=-1)
    clip_cache[text] = gligen_embeds
    torch.save(clip_cache, 'clip_cache_sdxl.pth')
    return gligen_embeds

def extract_sdxl_clip_embed(embed_file, flip: bool):
    old_embed = torch.load(embed_file, map_location='cpu', weights_only=False)
    
    new_embed = {}
    new_embed['file_name'] = old_embed['file_name']
    new_embed['anno_id'] = old_embed['anno_id']
    new_embed['data_id'] = old_embed['data_id']
    new_embed['caption'] = old_embed['caption']
    new_embed['width'] = old_embed['width']
    new_embed['height'] = old_embed['height']
    
    # image need to process
    image_tensor, trans_info = transform_image(old_embed['image'], image_size=resolution, random_flip=flip)
    image_tensor = image_tensor.to(device, non_blocking=True)[None]
    new_embed['latent'] = vae_encode(image_tensor).cpu()  # torch.Size([1, 4, 64, 64])
    new_embed['trans_info'] = trans_info
    # hois need to process
    new_hois = []
    for hoi in old_embed['hois']:
        _new_hoi = {}
        _new_hoi['subject_xywh'] = hoi['subject_xywh']
        _new_hoi['object_xywh'] = hoi['object_xywh']
        
        s_x, s_y, s_w, s_h = _new_hoi['subject_xywh']
        s_valid, (s_x0, s_y0, s_x1, s_y1) = recalculate_box_and_verify_if_valid(s_x, s_y, s_w, s_h, trans_info,
                                                                                resolution, min_box_size)
        o_x, o_y, o_w, o_h = _new_hoi['object_xywh']
        o_valid, (o_x0, o_y0, o_x1, o_y1) = recalculate_box_and_verify_if_valid(o_x, o_y, o_w, o_h, trans_info,
                                                                                resolution, min_box_size)
        if s_valid and o_valid:
            _new_hoi['action'] = hoi['action']
            _new_hoi['subject'] = hoi['subject']
            _new_hoi['object'] = hoi['object']
            
            _new_hoi['area'] = (s_x1 - s_x0) * (s_y1 - s_y0) + (o_x1 - o_x0) * (o_y1 - o_y0)  # area = subject + object
            _new_hoi['subject_box'] = torch.tensor([s_x0, s_y0, s_x1, s_y1]) / resolution
            _new_hoi['object_box'] = torch.tensor([o_x0, o_y0, o_x1, o_y1]) / resolution
            # torch.Size([1, 2048])
            _new_hoi['subject_text_embedding_before'] = encode_text(hoi['subject'], tokenizers, text_encoders)
            _new_hoi['action_text_embedding_before'] = encode_text(hoi['action'], tokenizers, text_encoders)
            _new_hoi['object_text_embedding_before'] = encode_text(hoi['object'], tokenizers, text_encoders)
            new_hois.append(_new_hoi)
    new_embed['hois'] = new_hois
    
    return new_embed

if __name__ == "__main__":
    source_dir = sys.argv[1]
    target_dir = sys.argv[2]
    print(f"source: {source_dir}\ntarget: {target_dir}")
    
    source_files = glob.glob(os.path.join(source_dir, 'embed_*.clip.pt'))

    flips = [False, True]
    for file_path, flip in tqdm(itertools.product(source_files, flips), total=len(source_files) * len(flips)):
        filename = os.path.splitext(file_path.split("/")[-1])[0] + (f".flip" if flip else "") + ".pt"
        os.makedirs(target_dir, exist_ok=True)
        save_path = os.path.join(target_dir, filename)
        if os.path.exists(save_path):
            print(f"File {filename} exitsts, skipping")
            continue
        
        new_embed = extract_sdxl_clip_embed(file_path, flip=flip)
        torch.save(new_embed, save_path)