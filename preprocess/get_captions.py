import os
import json
import argparse
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from accelerate import Accelerator
from transformers import BitsAndBytesConfig, AutoProcessor, LlavaForConditionalGeneration

class ImagePathDataset(Dataset):
    def __init__(self, root_folder, subset, cityname=None):
        self.root_folder = root_folder
        self.image_fnames = []

        for class_idx, class_name in enumerate(sorted(os.listdir(root_folder))):
            class_path = os.path.join(root_folder, class_name)
            if not os.path.isdir(class_path):
                continue

            if cityname is not None and class_name in subset:
                cls_img_names = subset[class_name]
            else:
                cls_img_names = os.listdir(class_path)

            for fname in cls_img_names:
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_fnames.append(os.path.join(class_path, fname))
                else:
                    print(f"Skipping non-image file: {fname}")                

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        image_path = self.image_fnames[idx]
        return image_path

def image2batch_caption(image_paths,llava_model, processor, convo, device, dtype=torch.bfloat16):
    """
    Generate captions for a batch of images.
    """
    images = [Image.open(path).convert("RGB") for path in image_paths]
    convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    assert isinstance(convo_string, str)

    inputs = processor(
        text=[convo_string] * len(images),
        images=images,
        return_tensors="pt"
    ).to(device)
    
    inputs['pixel_values'] = inputs['pixel_values'].to(dtype)

    with torch.inference_mode(), torch.autocast(device_type=device, dtype=dtype):
        outputs = llava_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            use_cache=True
        )
    
    # Trim prompt tokens and decode
    gen_ids = outputs[:, inputs['input_ids'].shape[1]:]
    captions = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    return [caption.strip() for caption in captions]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", type=str, required=True, help="Folder containing images")
    parser.add_argument("--caption_root_folder", type=str, required=True, help="Folder to save captions")
    parser.add_argument("--filtered_file", type=str, default=None, help="JSON file with filtered images")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sub_folder", type=str, default=None, help="Optional subfolder")
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
    MODEL_NAME = "fancyfeast/llama-joycaption-beta-one-hf-llava"
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
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=qnt_config,
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

        dataset = ImagePathDataset(args.root_folder, filtered_files, subfolder)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

        all_captions = {}
        for image_paths in tqdm(dataloader):
            # Generate captions
            captions = image2batch_caption(image_paths, llava_model, processor, convo, device)
            all_captions.update(dict(zip(subfolder, captions)))

        # Save to JSON
        with open(save_path, "w") as f:
            json.dump(all_captions, f)
        print(f"Saved captions to {save_path}")


if __name__ == "__main__":
    main()
