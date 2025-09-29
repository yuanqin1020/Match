
import logging
import torch
import torch.nn as nn
import data_handler as handler
from transformers import BertTokenizer, BertModel
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from torch_geometric.data import Data


import os
from datetime import datetime
import pickle


def attr_encoder(root):
    cache_dir = root + "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir, local_files_only=True)
    model = BertModel.from_pretrained("bert-base-uncased", cache_dir=cache_dir, local_files_only=True,
                                      output_hidden_states=True)
    model.eval()
    print("BERT model load complete.")

    return model, tokenizer

def sent_embed(sentence, model, tokenizer, device):
    inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])

    sentence_vector = outputs.last_hidden_state.mean(dim=1)

    return sentence_vector


def sam_encoder(device, model_type, ckp_path):
    sam = sam_model_registry[model_type](checkpoint=ckp_path).to(device=device)
    sam.eval()
    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=32,
                                               # points_per_batch=58,
                                               pred_iou_thresh=0.9,
                                               stability_score_thresh=0.92,
                                               crop_n_layers=1,
                                               crop_n_points_downscale_factor=2,
                                               min_mask_region_area=100)
    print("SAM model load complete.")

    return mask_generator


def attr_feature(entity_index, model, tokenizer):
    vertex_labels = entity_index.keys()
    num_vertices = len(entity_index)

    embedding = nn.Embedding(num_vertices, 768)

    vertex_embeddings = torch.zeros(num_vertices, 768)

    for i, label in enumerate(vertex_labels):
        marked_label = "[CLS] " + label + " [SEP]"
        tokenized_label = tokenizer.tokenize(marked_label)
        indexed_label = tokenizer.convert_tokens_to_ids(tokenized_label)
        label_tensor = torch.tensor([indexed_label])
        with torch.no_grad():
            outputs = model(label_tensor)
            hidden_states = outputs[2]

        label_representation = hidden_states[-1][0][0]

        vertex_embeddings[i] = label_representation

    vertex_embeddings = torch.tensor(vertex_embeddings).to(torch.float)

    return vertex_embeddings / vertex_embeddings.norm(dim=-1, keepdim=True)


import csv
def read_caption_dict(file_path):
    dict = {}
    start = datetime.now()

    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        
        # Skip the header row
        next(csv_reader)
        
        # Iterate over each line in the CSV file
        for line in csv_reader:
            id_value, class_value = line[0], line[1]
            dict[id_value] = class_value

    print(f"caption read takes: {datetime.now() - start}.")
    
    return dict

def get_img_caption(file, mode, dataset):
    import json
    path = file + mode + "_" + dataset + "_with_guidance.json"
    print(f"Read path: {path}")

    result = {}
    with open(path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                image_path = data['image_path'].split("/")[-1].split(".")[0]
                captions = data['caption']
                scores = data['scores']
                max_score_index = scores.index(max(scores))
                max_score_caption = captions[max_score_index]
                result[image_path] = max_score_caption
            except Exception as e:
                print(f"caption error : {e}")
                continue
            
            if len(result) == 1:
                print(f"img_path: {image_path}, caption: {max_score_caption}")

    return result


def get_scene_graph(mask_generator, image_path, iou_threshold, 
                    vis_merge, crop, image_encoder, clip_preprocess, device, captions=None):

    logging.info(image_path)

    imageid = str(image_path.split("/")[-1].split(".")[0])

    start = datetime.now()

    pixel_embs, edge_index = handler.img_to_graph(mask_generator, image_path, vis_merge, crop, 
                                                  image_encoder, clip_preprocess, device, iou_threshold)

    obj_embs = pixel_embs.clone().detach().to(torch.float)
    obj_embs = obj_embs / obj_embs.norm(dim=-1, keepdim=True)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()

    scene_graph = Data(x=obj_embs, edge_index=edge_index, y="")

    if captions != None and imageid in captions:
        caption = captions[imageid]
    else:
        caption = ""
    scene_graph.y = caption

    print(f"Number of {len(pixel_embs)} masks feature extraction takes: {datetime.now() - start}.")

    return scene_graph


def check_disk_space(path, threshold):
    stat = os.statvfs(path)
    free_space = stat.f_frsize * stat.f_bavail / (1024*1024*1024)  # 转换为GB
    return free_space > threshold


def get_scene_batches(mask_generator, images, iou_threshold, vis_merge, crop, image_encoder, clip_preprocess, device, captions=None):
    scenes = {}

    for image in images:
        scene_graph = get_scene_graph(mask_generator, image, iou_threshold, vis_merge, crop, image_encoder, clip_preprocess, device, captions=None)
        if scene_graph.edge_index.numel() != 0:
            scenes[image] = scene_graph

    return scenes

def kg_x_load(root, mode, kg, id2attr, clip_model, device):
    import clip

    pkl = root + mode + "_kg.pkl"
    if os.path.exists(pkl):
        kg = torch.load(pkl)
        print(f"Load kg text encodes complete from : {pkl}")
    else:
        with torch.no_grad():
            tokens = [clip.tokenize(id2attr[i][:77]).to(device) for i in range(len(id2attr))]
            x = [clip_model.encode_text(tokens[i]).to(device) for i in range(len(id2attr))]
            x = torch.cat(x, dim=0)
            kg.x = x / x.norm(dim=-1, keepdim=True)
            torch.save(kg, pkl)
    
    return kg
