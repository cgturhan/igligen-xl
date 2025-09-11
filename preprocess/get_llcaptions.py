import os
import json
import argparse
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from accelerate import Accelerator
from transformers import BitsAndBytesConfig, AutoProcessor, LlavaForConditionalGeneration
import glob

import os
import json
import argparse
from tqdm import tqdm
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration

class ImagePathDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        return path

@torch.no_grad()
def batch2caption(image_paths, llava_model, processor, convo, device):
    images = [Image.open(path).convert("RGB") for path in image_paths]
    convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)

    inputs = processor(
    text=[convo_string]*len(images),
    images=images,
    return_tensors="pt"
    ).to(device)
    
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    generate_ids = llava_model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    for i, gen_ids in enumerate(generate_ids):
        caption = processor.tokenizer.decode(
            gen_ids[inputs['input_ids'].shape[1]:],  # trim prompt
            skip_special_tokens=True
        ).strip()
    return caption

def process_city(city_name, data_path, city_filtered_files, batch_size=8, num_workers=4):
    city_captions = {}
    image_paths = [os.path.join(data_path, city_name, fname) for fname in city_filtered_files]
    dataset = ImagePathDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    for batch_paths in tqdm(dataloader, desc=f"Processing {city_name}"):
        batch_captions = batch2caption(batch_paths, llava_model, processor, convo, device)
        for p, c in zip(batch_paths, batch_captions):
            key = os.path.basename(p)
            city_captions[key] = c.lower()  # overwrite if duplicate

    return city_captions

def save_captions_asjson(out_folder, city_name, captions):
    os.makedirs(out_folder, exist_ok=True)
    path = os.path.join(out_folder, f"{city_name if city_name else 'all'}_captions.json")
    with open(path, "w") as f:
        json.dump(captions, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", type=str, required=True, help="Folder containing images")
    parser.add_argument("--caption_root_folder", type=str, required=True, help="Folder to save captions")
    parser.add_argument("--filtered_file", type=str, default=None, help="JSON file with filtered images")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sub_folder", type=str, default=None, help="Optional subfolder")
    parser.add_argument("--model_dir", type = str, default = None)
    parser.add_argument("--is_qnt", type =bool , default = True)
    args = parser.parse_args()

    os.makedirs(args.caption_root_folder, exist_ok=True)

    if args.filtered_file and os.path.exists(args.filtered_file):
        with open(args.filtered_file, "r") as f:
            filtered_files = json.load(f)
    else:
        # fallback: use all images
        filtered_files = {}
        for fname in os.listdir(args.root_folder):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                filtered_files[fname] = [fname]

    PROMPT = "Write a long descriptive caption for this image in a formal tone."
    MODEL_NAME = args.model_dir or "fancyfeast/llama-joycaption-beta-one-hf-llava"

    convo = [
        {"role": "system", "content": "You are a helpful image captioner."},
        {"role": "user", "content": PROMPT},
    ]

    qnt_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["vision_tower", "multi_modal_projector"]
    )
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    if args.is_qnt:
        llava_model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            quantization_config=qnt_config,
            torch_dtype="auto"
        )
    else:
        llava_model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype="auto"
        )
    llava_model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    accelerator = Accelerator()
    llava_model = accelerator.prepare(llava_model)
    if args.sub_folder != None:
        save_path = os.path.join(args.caption_root_folder, f"{args.sub_folder}_captions.json")
    else:
        save_path = os.path.join(args.caption_root_folder, "all_captions.json")

    for subfolder, image_fnames in filtered_files.items():
        if args.sub_folder and subfolder != args.sub_folder:
            continue

        print(f"Processing {subfolder} with {len(image_fnames)} images")

        if os.path.exists(save_path):
            print(f"{save_path} exists, skipping")
            continue
        
        dataset = ImagePathDataset([os.path.join(args.root_folder, subfolder, fname) for fname in image_fnames])
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

        all_captions = {}

        for image_paths in tqdm(dataloader, total=len(dataset)//args.batch_size + 1):
            # Generate captions
            captions = batch2caption(image_paths, llava_model, processor, convo, device)
            all_captions.update(dict(zip(subfolder, captions)))

        # Save to JSON
        with open(save_path, "w") as f:
            json.dump(all_captions, f)
        print(f"Saved captions to {save_path}")


if __name__ == "__main__":
    main()
