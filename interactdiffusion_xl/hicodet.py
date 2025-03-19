import glob
import os
import random
import torch

class HICODetDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 path,
                 prob_use_caption=1,
                 max_boxes_per_data=30,
                 embedding_len=768,
                 ):
        super().__init__()
        self.path = path
        self.embedding_len = embedding_len
        self.prob_use_caption = prob_use_caption
        self.max_boxes_per_data = max_boxes_per_data
        
        self.files = glob.glob(os.path.join(self.path, 'embed_*.clip*.pt'))
        assert len(self.files) > 0, f'No file found at {self.dataset_path}!'
        
    def get_item(self, index):
        item = torch.load(self.files[index], map_location="cpu", weights_only=False)
        return item
    
    def __getitem__(self, index):
        if self.max_boxes_per_data > 99:
            assert False, "Are you sure setting such large number of boxes per image?"
        
        raw_item = self.get_item(index)
        
        out = {}
        # -------------------- id and latent ------------------- #
        out['id'] = raw_item['data_id'] 
        out['latents'] = raw_item['latent']
        trans_info = raw_item['trans_info']
        out['crop_top_lefts'] = [trans_info['crop_y'], trans_info['crop_x']]
        out['original_sizes'] = [trans_info['HH'], trans_info['WW']]
        
        # -------------------- grounding token ------------------- #
        areas = [hoi['area'] for hoi in raw_item['hois']]
        wanted_idxs = torch.tensor(areas).sort(descending=True)[1]
        wanted_idxs = wanted_idxs[0:self.max_boxes_per_data]
        
        subject_boxes = torch.zeros(self.max_boxes_per_data, 4)
        object_boxes = torch.zeros(self.max_boxes_per_data, 4)
        masks = torch.zeros(self.max_boxes_per_data)
        subject_text_embeddings = torch.zeros(self.max_boxes_per_data, self.embedding_len)
        object_text_embeddings = torch.zeros(self.max_boxes_per_data, self.embedding_len)
        action_text_embeddings = torch.zeros(self.max_boxes_per_data, self.embedding_len)
        
        for i, idx in enumerate(wanted_idxs):
            subject_boxes[i] = raw_item['hois'][idx]['subject_box']
            object_boxes[i] = raw_item['hois'][idx]['object_box']
            masks[i] = 1
            subject_text_embeddings[i] = raw_item['hois'][idx]['subject_text_embedding_before'][0]
            object_text_embeddings[i] = raw_item['hois'][idx]['object_text_embedding_before'][0]
            action_text_embeddings[i] = raw_item['hois'][idx]['action_text_embedding_before'][0]
        
        text_masks = masks
        
        out["subject_boxes"] = subject_boxes
        out["object_boxes"] = object_boxes
        out["masks"] = masks  # indicating how many valid objects for this image-text data
        # out["image_masks"] = image_masks  # indicating how many objects still there after random dropping applied
        out["text_masks"] = text_masks  # indicating how many objects still there after random dropping applied
        out["subject_text_embeddings"] = subject_text_embeddings
        out["object_text_embeddings"] = object_text_embeddings
        out["action_text_embeddings"] = action_text_embeddings
        
        # -------------------- caption ------------------- #
        if random.uniform(0, 1) < self.prob_use_caption or len(wanted_idxs) == 0:
            out["caption"] = raw_item["caption"]
        else:
            out["caption"] = ""
            
        return out
    
    def __len__(self):
        return len(self.files)
    
