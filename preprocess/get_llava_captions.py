import os
import json
import argparse
from tqdm import tqdm
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig
from accelerate import Accelerator
from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration

# PromptPROMPT = "Write a long descriptive caption for this image in a formal tone."
MODEL_NAME = "fancyfeast/llama-joycaption-beta-one-hf-llava"
PROMPT = "Write a long descriptive caption for this image in a formal tone."

qnt_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
)

# Load processor + model
processor = AutoProcessor.from_pretrained(MODEL_NAME)
llava_model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map="auto",   # auto GPU placement
    quantization_config=qnt_config,
    torch_dtype="auto",
)
device = "cuda" if torch.cuda.is_available() else "cpu"
accelerator = Accelerator()
llava_model = accelerator.prepare(llava_model)

llava_model.eval()


convo = [
    {"role": "system", "content": "You are a helpful image captioner."},
    {"role": "user", "content": PROMPT},
]

def image2caption(image_path, llava_model=llava_model, processor=processor, convo=convo):
    with torch.no_grad():
        image = Image.open(image_path).convert("RGB")
        convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to("cuda")
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        generate_ids = llava_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )[0]

        # Trim prompt tokens
        generate_ids = generate_ids[inputs["input_ids"].shape[1]:]
        caption = processor.tokenizer.decode(generate_ids, skip_special_tokens=True)
        return caption.strip()

def batch2caption(image_paths, llava_model=llava_model, processor=processor, convo=convo):
    with torch.no_grad():
        images = [Image.open(path).convert("RGB") for path in image_paths]
        convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)

        inputs = processor(text=[convo_string], images=images, return_tensors="pt").to("cuda")
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
        return [cap.strip() for cap in captions.split("\n") if cap.strip()]  # safer split


def save_captions_asjson(out_folder, city_name, captions):
    os.makedirs(out_folder, exist_ok=True)
    caption_path = os.path.join(out_folder, f"{city_name if city_name else 'all'}_captions.json")
    with open(caption_path, "w") as f:
        json.dump(captions, f, indent=2)


def process_city(city_name, data_path, city_filtered_files, batch_size=4):
    city_captions = {}

    for i in tqdm(range(0, len(city_filtered_files), batch_size), desc=f"Processing {city_name}"):
        batch_names = city_filtered_files[i:i + batch_size]
        batch_paths = [os.path.join(data_path, city_name, fname) for fname in batch_names]

        batch_captions = batch2caption(
            batch_paths,
            llava_model=llava_model,
            processor=processor,
            convo=convo,
        )

        city_captions.update({fname: cap.lower() for fname, cap in zip(batch_names, batch_captions)})

    return city_captions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", type=str, required=True)
    parser.add_argument("--caption_root_folder", type=str, required=True)
    parser.add_argument("--filtered_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sub_folder", type=str, default=None)
    args = parser.parse_args()

    with open(args.filtered_file, "r") as f:
        filtered_files = json.load(f)

    data_path = args.root_folder
    batch_size = args.batch_size
    city_name = args.sub_folder if args.sub_folder else None 

    if city_name:  
        city_filtered_files = filtered_files.get(city_name, [])
        captions = process_city(city_name, data_path, city_filtered_files, batch_size) 
        save_captions_asjson(args.caption_root_folder, city_name, captions)

    else:  # all cities
        all_captions = {}
        for city_name, city_filtered_files in filtered_files.items():
            all_captions.append(process_city(city_name, data_path, city_filtered_files, batch_size)) 
        save_captions_asjson(args.caption_root_folder, None, all_captions)

if __name__ == "__main__":
    main()
