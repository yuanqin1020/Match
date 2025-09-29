import csv
import torch
import torch.nn.functional as F
import logging

import feature_extractor as extractor

def map_img(img_path, data, ent2txt):
    task = data.strip().split("/")[-1]
    class_id = 0

    if task == "FB15k":
        # ent2txt: /m/0145m  Afrobeat
        # print(f"img_path: {img_path}")
        ent = img_path.strip().split("/")[-2].replace("m.", "m/")
        class_name = ent2txt["/" + ent].lower()
    elif task == "WN18":
        ent = img_path.strip().split("/")[-2]
        class_name = ent2txt[ent[1:]].lower()

    return class_name, class_id

def true_targets(image_paths, data, ent2txt=None):
    true_img_class = {}

    for i in range(len(image_paths)):
        name, id = map_img(image_paths[i], data, ent2txt)
        true_img_class[image_paths[i]] = name

    return true_img_class


def retrieve_image_labels(mats, imgs, classes, epoch_pred, attr_encoder, tokenizer, device):
    pred_img_class = {}
    predictions = {}

    for i in range(len(imgs)):  # image
        predicted, indices = torch.topk(mats[i], 3, dim=0)
        unique_indices = torch.unique(indices)
        # print(f"{indices}; {predicted}; {unique_indices}")

        top_k_txts = [classes[i][index.item()] for index in unique_indices] # class
        top_k_tensors = [extractor.sent_embed(txt, attr_encoder, tokenizer, device) for txt in top_k_txts]
        predictions[imgs[i]] = top_k_txts
        pred_img_class[imgs[i]] = top_k_tensors
        epoch_pred[imgs[i]] = top_k_tensors

    # print(f"predictions retrieval: {predictions}")

    return pred_img_class, epoch_pred


def retrieval_prediction(mats, imageids, secne_ents, train_ground, ent_dict, tops, attr_encoder, tokenizer, device):
    pred_scene_ents = {}

    for i in range(len(mats)):
        _, indices = torch.topk(mats[i], tops, dim=0, largest=True)
        imageid = imageids[i].split("/")[-1].split(".")[0]

        for ind in indices:
            # ents = torch.tensor([ent_dict[ent] for ent in secne_ents[ind]])
            # print(f"secne_ents[i][ind]: {len(secne_ents[i][ind])}, {secne_ents[i][ind].shape}")
            ents = secne_ents[i][ind.item()]
            # print(f"prediction: {imageid}, {ents}")

            avg_ents = [extractor.sent_embed(ent, attr_encoder, tokenizer, device) for ent in ents] 
            avg_ents = torch.stack(avg_ents)
            avg_ents = torch.mean(avg_ents, dim=0, keepdim=True).squeeze(0)
            # print(f"ents: {ents.shape}")
            if imageid not in pred_scene_ents:
                pred_scene_ents[imageid] = []
            pred_scene_ents[imageid].append(avg_ents)
    
    truth_scene_ents = {}
    for i in range(len(mats)):
        imageid = imageids[i].split("/")[-1].split(".")[0]
        if imageid not in train_ground:
            print(f"Imageid {imageid} not in ground truth")
            continue
        else:
            truths = train_ground[imageid]
            truths = [ent_dict[ent] for ent in truths]
            # print(f"truth: {imageid}, {truths}")

            avg_ents = [extractor.sent_embed(ent, attr_encoder, tokenizer, device) for ent in truths]
            avg_ents = torch.stack(avg_ents)
            # print(f"avg_ents 1: {avg_ents.shape}")
            avg_ents = torch.mean(avg_ents, dim=0, keepdim=True).squeeze(0)
            # print(f"avg_ents 2: {avg_ents.shape}")
            truth_scene_ents[imageid] = avg_ents

    return pred_scene_ents, truth_scene_ents



def obtain_ground_truth(task_root, mode):
    image_ents = task_root + "/" + mode + "-image-ents.txt"
    ent_des = task_root + "/ent-descriptions.csv"

    ent_dict = {}
    with open(ent_des, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            ent_dict[row[0]] = row[1]

    image_ent_dict = {}
    instances = 0
    with open(image_ents, 'r') as txtfile:
        for line in txtfile:
            image_id, ent_list = line.strip().split(':')
            ent_list = [ent_id for ent_id in eval(ent_list)]
            image_ent_dict[image_id] = ent_list
            instances += len(ent_list)
    
    print(f"Number of instances: {instances}")

    return image_ent_dict, ent_dict

def calculate_hits_mrr(ground_truths, predictions, device, task, threshold, attr_encoder=None, tokenizer=None, k_values = [1, 3, 5, 10]):
    print(f"Testing task is {task}")
    hits = {k: 0 for k in k_values}
    mrr = 0
    
    for key in predictions.keys():

        if key not in ground_truths.keys():
            continue
        
        prediction = predictions[key]
        ground_truth = ground_truths[key]
        if task in ["FB15k", "WN18"]:
            ground_truth = extractor.sent_embed(ground_truth, attr_encoder, tokenizer, device)

        ground_truth = torch.tensor(ground_truth).to(device)

        for k in k_values:
            rank = -1
            top_k_predictions = prediction[:k]

            # print(f"top_k_predictions: {top_k_predictions[0].shape}, ground_truth: {ground_truth.shape}")
            for item in top_k_predictions:
                item = torch.tensor(item).to(device)
                simi = F.cosine_similarity(item, ground_truth)
                # print(f"simi: {simi}")
                if simi > threshold:
                    hits[k] += 1

                    for i, item in enumerate(top_k_predictions):
                        if F.cosine_similarity(item, ground_truth) > threshold:
                            rank = i + 1
                            break
                    
                    mrr += 1 / rank
                
                    break


    total_keys = len(ground_truths.keys() & predictions.keys())

    hits_at_k = {k: round(hits[k] / total_keys * 100.0, 5) for k in k_values}
    mean_reciprocal_rank = round(mrr / total_keys / len(hits), 5)

    return hits_at_k, mean_reciprocal_rank


def calculate_hits_mrr_global(ground_truths, predictions, k_values = [1, 3, 5, 10]):
    hits = {k: 0 for k in k_values}
    mrr = 0

    print(f"ground truth: {len(ground_truths)}")
    print(f"predict: {len(predictions)}")

    for key in predictions.keys():

        if key not in ground_truths.keys():
            print(f"{key} not in ground truth")
            continue
        ground_truth = ground_truths[key]
        prediction = predictions[key]

        for k in k_values:
            rank = -1
            top_k_predictions = prediction[:k]

            if any(item in ground_truth for item in top_k_predictions):
                hits[k] += 1

                for i, item in enumerate(top_k_predictions):
                    if item in ground_truth:
                        rank = i + 1
                        break

                mrr += 1 / rank

    hits_at_k = {k: round(hits[k] / len(predictions) * 100.0, 5) for k in k_values}
    mean_reciprocal_rank = round(mrr / len(predictions) / len(hits), 5)

    return hits_at_k, mean_reciprocal_rank
