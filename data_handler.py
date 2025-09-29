
import os
import numpy as np
import pandas as pd
import cv2
import torch
import utils
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans

cache_dir=".../bert-base-uncased"


def read_str_triples(file):
    triples = set()
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            params = line.strip('\n').split(',')
            assert len(params) == 3
            h = str(params[0])
            r = str(params[1])
            t = str(params[2])
            triples.add((h, r, t))
        f.close()
    return triples

def get_triples_id(triples):
    ent_dict = dict()
    rel_dict = dict()
    ent = 0
    rel = 0
    for item in triples:
        if item[0] not in ent_dict.keys():
            ent_dict[item[0]] = ent
            ent += 1
        if item[2] not in ent_dict.keys():
            ent_dict[item[2]] = ent
            ent += 1
        if item[1] not in rel_dict.keys():
            rel_dict[item[1]] = rel
            rel += 1

    triple_id= set()
    for item in triples:
        triple_id.add((ent_dict.get(item[0]), rel_dict.get(item[1]), ent_dict.get(item[2])))

    return ent_dict, rel_dict, triple_id


def read_img_list(folder):
    img_paths = [os.path.join(folder, file) for file in os.listdir(folder)]
    print(f"image number: {len(img_paths)}")

    return img_paths

def bbox_overlap(bbox_1, bbox_2):
    ## (X, Y, W, H)
    x1 = bbox_1[0] + bbox_1[2]
    y1 = bbox_1[1] - bbox_1[3]  # bottom right
    x2 = bbox_2[0] + bbox_2[2]
    y2 = bbox_2[1] - bbox_2[3]  # bottom right

    if bbox_1[0] > bbox_2[0] + bbox_2[2]:
        return 0
    if bbox_1[1] > bbox_2[1] + bbox_2[3]:
        return 0
    if bbox_1[0] + bbox_1[2] < bbox_2[0]:
        return 0
    if bbox_1[1] + bbox_1[3] < bbox_2[1]:
        return 0

    colInt = abs(min(bbox_1[0] + bbox_1[2], bbox_2[0] + bbox_2[2]) - max(bbox_1[0], bbox_2[0]))
    rowInt = abs(min(bbox_1[1] + bbox_1[3], bbox_2[2] + bbox_2[3]) - max(bbox_1[1], bbox_2[1]))
    overlap_area = colInt * rowInt
    area_1 = bbox_1[2] * bbox_1[3]
    area_2 = bbox_2[2] * bbox_2[3]

    return overlap_area / float(area_1 + area_2 - overlap_area) # IOU


# 检查bounding boxes是否有重叠
def boundingbox_overlap(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    if x1 > x2 + w2 or x2 > x1 + w1:
        return False
    if y1 > y2 + h2 or y2 > y1 + h1:
        return False

    return True



def images_graphs(device, model_type, image_root, ckp_path):
    image_paths = [os.path.join(image_root, file) for file in os.listdir(image_root)]

    entity2id = dict() # {entity: id}
    graphs_embs_dict = dict()
    graphs_matrix_dict = dict()
    for path in image_paths:
        pixel_embs, matrix = img_to_graph(device, model_type, path, ckp_path)
        # for i in range(len(pixel_embs)):
        #     entity2id[len(entity2id) + i] = pixel_embs[i]

        graphs_embs_dict[path] = pixel_embs
        graphs_matrix_dict[path] = matrix

    return graphs_embs_dict, graphs_matrix_dict,

def finetune_to_scenegraph():

    pass

def merge_bounding_boxes(bboxes):
    if len(bboxes) == 0:
        return None

    min_x = min([bbox[0] for bbox in bboxes])
    min_y = min([bbox[1] for bbox in bboxes])
    max_x = max([bbox[0] + bbox[2] for bbox in bboxes])
    max_y = max([bbox[1] + bbox[3] for bbox in bboxes])
    merged_bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
    return merged_bbox

def get_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    box1 = [box1[0], box1[1] - box1[3], box1[0] + box1[2], box1[1]]
    box2 = [box2[0], box2[1] - box2[3], box2[0] + box2[2], box2[1]]

    # Calculate the intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate the union area
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union = area1 + area2 - intersection

    # Calculate the IoU
    iou = intersection / union

    return iou



def img_cluster_to_graph(mask_generator, image_path, iou_threshold, n_clusters):
    """
        cluster the fully-connected graph into several subgraphs and
        share the subgroup features for object pairs within the subgraph,
        then a factorized connection graph is obtained by treating each subgraph as a node
    """
    x = torch.tensor([], dtype=torch.float)
    edge_index = torch.tensor([], dtype=torch.long)

    try:
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # dict_keys(['segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box', 'embed'])
        masks = mask_generator.generate(image_rgb)
        print(f"masks numbers: {len(masks)}")
    except Exception as e:
        print(f"encounter error: {image_path}")
        print(e)
        return x, edge_index

    bboxes = torch.tensor([masks[i]['bbox'] for i in range(len(masks))])
    mask_embeds = torch.tensor([masks[i]['embed'] for i in range(len(masks))])

    mask_embeds_flat = mask_embeds.view(len(masks), -1)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(mask_embeds_flat)
    labels = kmeans.labels_

    cluster_dict = {}
    for cluster_label in range(n_clusters):
        print(f"cluster {cluster_label}")
        cluster_indices = np.where(labels == cluster_label)[0]
        cluster_masks = [mask_embeds[i] for i in cluster_indices]
        cluster_bboxes = merge_bounding_boxes([bboxes[i] for i in cluster_indices])

        if len(cluster_masks) == 0: continue
        cluster_mean_embed = torch.mean(torch.stack(cluster_masks), dim=0)

        cluster_dict[cluster_label] = {
            'masks': cluster_masks,
            'bboxes': cluster_bboxes,
            'mean_embed': cluster_mean_embed
        }
    print(f"cluster_dict: {len(cluster_dict)}")

    edge_index = []
    for cluster_label in cluster_dict.keys():
        cluster_info = cluster_dict[cluster_label]
        masks = cluster_info['masks']
        bboxes = cluster_info['bboxes']
        mean_embed = cluster_info['mean_embed']

        x = torch.cat((x, mean_embed), dim=0)

        for cluster_label2 in cluster_dict.keys():
            if cluster_label >= cluster_label2:
                continue

            cluster_info2 = cluster_dict[cluster_label2]
            masks2 = cluster_info2['masks']
            bboxes2 = cluster_info2['bboxes']
            mean_embed2 = cluster_info2['mean_embed']

            if boundingbox_overlap(bboxes, bboxes2):
                edge_index.append([cluster_label, cluster_label2])

    print(f"x : {len(x)}, edge_index : {edge_index}")


    return x, edge_index


def nms_refine_masks(masks, iou_threshold):
    """
        Non-Maximum Suppression (NMS) on the boxes according to their intersection-over-union (IoU)
        and iteratively removes lower scoring boxes
    """
    bboxes = torch.tensor([masks[i]['bbox'] for i in range(len(masks))])
    embeds = torch.tensor([masks[i]['embed'] for i in range(len(masks))])
    scores = torch.tensor([masks[i]['stability_score'] for i in range(len(masks))])

    sorted_indices = np.argsort(-scores)
    refined_masks = []

    print(f"sorted_indices {sorted_indices}")
    while len(sorted_indices) > 0:
        current_index = sorted_indices[0]
        refined_masks.append(embeds[current_index])
        ious = [get_iou(bboxes[current_index], bboxes[i]) for i in sorted_indices[1:]]
        indices_to_remove = np.where(np.array(ious) > iou_threshold)[0] + 1
        sorted_indices = np.delete(sorted_indices, indices_to_remove)

    return refined_masks


def img_to_graph(mask_generator, image_path, vis_merge, crop, image_encoder, clip_preprocess, device, iou_threshold):
    x = torch.tensor([], dtype=torch.float)
    edge_index = torch.tensor([], dtype=torch.long)

    try:
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        masks = mask_generator.generate(image_rgb)
        # print(f"masks numbers: {len(masks)}")
    except Exception as e:
        print(f"encounter error: {image_path}")
        print(e)
        return x, edge_index

    if vis_merge == True:
        pixel_embs, edge_index = visual_instance_merge(masks)
        print(f"Number of masks before: {len(masks)}, after: {len(pixel_embs)}")
    else:
        pixel_embs, edge_index = get_mask_embeddings(masks)
    
    if crop == True:
        pixel_embs, edge_index = get_crop_embeddings(masks, image_rgb, image_encoder, clip_preprocess, device)
    
    pixel_embs = torch.cat(pixel_embs, dim=0)

    return pixel_embs, edge_index

def visual_instance_merge(masks, threshold_area=0.05):
    refine_masks = masks

    grouped_masks = {}
    best_masks = []
    for i in range(len(refine_masks)):
        mask_i = refine_masks[i] 
        is_overlapped = False
        
        for j in range(len(refine_masks)):
            mask_j = refine_masks[j]
            if i != j:
                iou = get_iou(mask_i['bbox'], mask_j['bbox'])
                if iou > 0.5:  # 可根据具体情况调整重叠程度的阈值
                    if i not in grouped_masks:
                        grouped_masks[i] = []
                    grouped_masks[i].append(j)
                    is_overlapped = True

        if not is_overlapped:
            best_masks.append(mask_i)

    for i, group in grouped_masks.items():
        max_area = refine_masks[i]['area']
        best_mask = refine_masks[i]
        for j in group:
            if refine_masks[j]['area'] > max_area:
                max_area = refine_masks[j]['area']
                best_mask = refine_masks[j]
        best_masks.append(best_mask)

    edge_index = []
    pixel_embs = []
    for i in range(len(best_masks)):  # todo 跳过背景
        pixel_embs.append(best_masks[i]['embed'])

    return pixel_embs, edge_index

def get_crop_embeddings(refine_masks, image_rgb, image_encoder, preprocess, device):
    from PIL import Image

    features = []
    edge_index = []

    for i in range(len(refine_masks)):
        mask = refine_masks[i]
        crop_box = mask['crop_box']
        x, y, w, h = map(int, crop_box)
        crop_image = image_rgb[y:y+h, x:x+w]

        pil_image = Image.fromarray(crop_image)
        preprocessed_image = preprocess(pil_image).unsqueeze(0).to(device)
        preprocessed_image = preprocessed_image.type(torch.half)

        # Step 3: Pass the preprocessed images through the image encoder
        with torch.no_grad():
            image_features = image_encoder(preprocessed_image)
        image_features = image_features.clone().detach().float()
        features.append(image_features)

        for j in range(len(refine_masks)):
            if i == j: continue
            if get_iou(refine_masks[i]['bbox'], refine_masks[j]['bbox']) > 0:
                edge_index.append([i, j])

    return features, edge_index


def get_mask_embeddings(refine_masks):
    ## create scene graph for refined masks
    pixel_embs = list()
    edge_index = []
    for i in range(len(refine_masks)):  # todo 跳过背景
        pixel_embs.append(torch.tensor(refine_masks[i]['embed']))
        for j in range(len(refine_masks)):
            if i == j: continue
            if get_iou(refine_masks[i]['bbox'], refine_masks[j]['bbox']) > 0:
                edge_index.append([i, j])

    return pixel_embs, edge_index



def get_attr_embeds(entity_index):
    # Define vertex labels and number of vertices
    vertex_labels = entity_index.keys()
    num_vertices = len(entity_index)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir, local_files_only=True)
    model = BertModel.from_pretrained("bert-base-uncased", cache_dir=cache_dir, local_files_only=True, output_hidden_states=True)
    model.eval()
    embedding = nn.Embedding(num_vertices, 768)
    print(f"Number of kg vertices: {num_vertices}, vertex embedding shape: [{num_vertices}, 768]")
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

def sentence_to_graph():

    pass
