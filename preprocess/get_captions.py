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

MODEL_NAME = "fancyfeast/llama-joycaption-beta-one-hf-llava"
PROMPT = "Write a long descriptive caption for this image in a formal tone."

qnt_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)
llava_model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map="auto",   # auto GPU placement
    quantization_config=qnt_config,
    torch_dtype="auto",
)

# Accelerator for multi-GPU / device placement
accelerator = Accelerator()
llava_model = accelerator.prepare(llava_model)
llava_model.eval()
device = accelerator.device

convo = [
    {"role": "system", "content": "You are a helpful image captioner."},
    {"role": "user", "content": PROMPT},
]

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
    ).to(accelerator.device)
    
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    generate_ids = llava_model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )[0]

    generate_ids = generate_ids[inputs["input_ids"].shape[1]:]
    captions = processor.tokenizer.decode(generate_ids, skip_special_tokens=True)
    return [cap.strip() for cap in captions.split("\n") if cap.strip()]

def process_city(city_name, data_path, city_filtered_files, batch_size=8, num_workers=4):
    city_captions = {}
    image_paths = [os.path.join(data_path, city_name, fname) for fname in city_filtered_files]
    dataset = ImagePathDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    for batch_paths in tqdm(dataloader, desc=f"Processing {city_name}"):
        batch_captions = batch2caption(batch_paths, llava_model, processor, convo, device)
        city_captions.update({os.path.basename(p): c.lower() for p, c in zip(batch_paths, batch_captions)})

    return city_captions

def save_captions_asjson(out_folder, city_name, captions):
    os.makedirs(out_folder, exist_ok=True)
    path = os.path.join(out_folder, f"{city_name if city_name else 'all'}_captions.json")
    with open(path, "w") as f:
        json.dump(captions, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", type=str, required=True)
    parser.add_argument("--caption_root_folder", type=str, required=True)
    parser.add_argument("--filtered_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--sub_folder", type=str, default=None)
    args = parser.parse_args()

    with open(args.filtered_file, "r") as f:
        filtered_files = json.load(f)

    data_path = args.root_folder
    batch_size = args.batch_size
    num_workers = args.num_workers
    city_name = args.sub_folder

    if city_name:
        city_filtered_files = filtered_files.get(city_name, [])
        captions = process_city(city_name, data_path, city_filtered_files, batch_size, num_workers)
        save_captions_asjson(args.caption_root_folder, city_name, captions)
    else:
        all_captions = {}
        for city_name, city_filtered_files in filtered_files.items():
            city_caps = process_city(city_name, data_path, city_filtered_files, batch_size, num_workers)
            all_captions.update(city_caps)
        save_captions_asjson(args.caption_root_folder, None, all_captions)

if __name__ == "__main__":
    main()



