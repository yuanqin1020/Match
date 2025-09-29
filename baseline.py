import csv
import torch
import torch.nn.functional as F
import logging

import clip
import torch
import time
from evaluate import *


def img_feature_extract(imgs, device):
    start = time.time()
    model, _ = clip.load("ViT-B/32", device)
    # text_input = torch.cat([clip.tokenize(str(c))[:77] for c in classes]).to(args.device)
    # with torch.no_grad():
    #     text_features = model.encode_text(text_input)
    # text_features /= text_features.norm(dim=-1, keepdim=True)


    image_features = []
    with torch.no_grad():
        features = model.encode_image(imgs.to(device))
        # print(f"features:{features.shape}")
        image_features.append(features)

        image_features = torch.cat(image_features)
        image_features /= image_features.norm(dim=-1, keepdim=True)


def base_method(root, vertices, imgs, train_ground, ent_dict, tops, device, dataset):
    start = time.time()
    model, preprocess = clip.load("ViT-B/32", device)
    text_input = torch.cat([clip.tokenize(str(c))[:77] for c in vertices]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    image_features = []
    with torch.no_grad():
        for img in imgs:
            img = img.to(device)
            features = model.encode_image(img.to(device))
            image_features.append(features)

        image_features = torch.cat(image_features)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    
    logits = 100.0 * image_features @ text_features.T

    ents_index = logits.argmax(dim=1)
    scene_ents = [vertices[ind] for ind in ents_index]

    attr_encoder, tokenizer = extractor.attr_encoder(root)
    pred_scene_ents, truth_scene_ents = retrieval_prediction(logits, imgs, scene_ents, train_ground, ent_dict, tops, attr_encoder, tokenizer, device)
    hits_at_k, mrr = calculate_hits_mrr(truth_scene_ents, pred_scene_ents, device, dataset)
    
    print(f"hits@k: {hits_at_k}, mrr: {mrr}")
