import os
import json
import argparse
from tqdm import tqdm
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig
from accelerate import Accelerator
from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration
import json

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
        inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to(accelerator.device)
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

        inputs = processor(text=[convo_string]*len(images), images=images, return_tensors="pt").to(accelerator.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        generate_ids = llava_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False #True,
            #temperature=0.6,
            #top_p=0.9,
        )
        captions = [
            processor.tokenizer.decode(
                gen_ids[inputs['input_ids'].shape[1]:],  # trim prompt tokens
                skip_special_tokens=True
            ).strip()
            for gen_ids in generate_ids
        ]
        return captions
        
def batch2caption_wmeta(image_paths, meta_lookup, llava_model=llava_model, processor=processor, convo_temp=convo):
    batch_convo_string = []
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with torch.no_grad():
        for img_path in image_paths:
            img_name = os.path.basename(img_path)  
            cls_name = os.path.basename(os.path.dirname(img_path)) 
            fname = f"{cls_name}/{img_name}"
            if fname not in meta_lookup:
                print(f"Warning: no metadata for {fname}")
                continue  # skip or handle missing metadata
                
            m = meta_lookup[fname]

            # Build prompt using metadata
            #cls_name = m.get("cls_name", "an european city")
            prompt = f"Write a caption for given image. The description should begin with 'A photo of' and describe that '{m['text']}' objects as foreground and end with 'in {cls_name}' including details about the background, the surroundings, and the weather not mentioning about image quality as maximum 2 sentences."

            #prompt = 
            #    f"Describe image that contains '{m['text']}' for describing foreground in a sentence"
            #    f"which starts with 'A photo of' ends with 'in {cls_name}' and describes also background with weather"
            #)

            #prompt = f"Describe image that contains '{m['text']}' in a sentence which starts with a photo of .. and ends with .. in {m['file_name'].split('/')[0]}."
    
            # Copy and update conversation
            convo = convo_temp.copy()
            convo[-1]["content"] = prompt  # Replace the user prompt
    
            convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            
            batch_convo_string.append(convo_string)
        
        images = [Image.open(path).convert("RGB") for path in image_paths]       
        inputs = processor(text=list(batch_convo_string), images=images, padding=True, truncation=True, return_tensors="pt").to(accelerator.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        generate_ids = llava_model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False#,
            #temperature=0.6,
            #top_p=0.9,
        )

        captions = [
            processor.tokenizer.decode(
                gen_ids[inputs['input_ids'].shape[1]:],  # trim prompt tokens
                skip_special_tokens=True
            ).strip()
            for gen_ids in generate_ids
        ]
        return captions

def save_captions_asjson(out_folder, city_name, captions, split="train"):
    os.makedirs(out_folder, exist_ok=True)
    caption_path = os.path.join(out_folder, f"{city_name if city_name else 'all'}_{split}captions.json")
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

def process_city_wmeta(city_name, data_path, city_filtered_files, meta, batch_size=4):
    city_captions = {}

    for i in tqdm(range(0, len(city_filtered_files), batch_size), desc=f"Processing {city_name}"):
        batch_names = city_filtered_files[i:i + batch_size]
        batch_paths = [os.path.join(data_path, city_name, fname) for fname in batch_names]

        batch_captions = batch2caption_wmeta(
            batch_paths,
            meta,
            llava_model=llava_model,
            processor=processor,
            convo_temp=convo,
        )

        city_captions.update({fname: cap.lower() for fname, cap in zip(batch_names, batch_captions)})

    return city_captions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", type=str, required=True)
    parser.add_argument("--caption_root_folder", type=str, required=True)
    parser.add_argument("--filtered_file", type=str, required=False)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sub_folder", type=str, default=None)
    parser.add_argument("--metafile", type=str, default=None) #metadata_boxphrases.jsonl"
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

  
    os.makedirs(args.caption_root_folder, exist_ok=True)
    
    data_path = args.root_folder
    batch_size = args.batch_size
    city_name = args.sub_folder if args.sub_folder else None 
    
    if args.metafile:
        meta = []
        with open(args.metafile, "r") as f:
            for line in f:
                meta.append(json.loads(line))
        
        # meta is a list of dicts, each with "file_name" and other info
        meta_lookup = {m["file_name"]: m for m in meta}

    if args.filtered_file:
        with open(args.filtered_file, "r") as f:
            filtered_files = json.load(f)
        if city_name:  
            city_filtered_files = filtered_files.get(city_name, [])
            if meta_lookup:
                captions = process_city_wmeta(city_name, data_path, city_filtered_files, meta_lookup, batch_size) 
            else:
                captions = process_city(city_name, data_path, city_filtered_files, batch_size) 
            save_captions_asjson(args.caption_root_folder, city_name, captions)
        else:  # all cities
            all_captions = {}
            for city_name, city_filtered_files in filtered_files.items():
                if meta_lookup:
                    all_captions.append(process_city_wmeta(city_name, data_path, city_filtered_files, meta_lookup, batch_size) 
                else:
                    all_captions.append(process_city(city_name, data_path, city_filtered_files, batch_size)) 
            save_captions_asjson(args.caption_root_folder, None, all_captions, split)
    else:
        for city in os.listdir(input_dir):
            if city_name is not None and city_name != city:
                continue
            else:
                if not any(f.startswith(f"{city}_") for f in os.listdir(out_dir)): 
                    city_list = os.listdir(os.path.join(data_path, city))
                    if meta_lookup:
                        captions = process_city_wmeta(city, data_path, city_list, meta_lookup, batch_size) 
                    else:
                        captions = process_city(city, data_path, city_list, batch_size) 
                    save_captions_asjson(args.caption_root_folder, city, captions, split)
                else:
                    print(f"Skip {city}")

if __name__ == "__main__":
    main()
