import logging
import sys
import torch
import torch.nn as nn
import subgraph_sampling as sampler
import torch.nn.functional as F
import utils
from tqdm import tqdm
import torch_geometric as tg
import torch.optim as optim
import feature_extractor as extractor
import data_loader as loader
from torch_geometric.data import DataLoader
import clip
    

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime
from evaluate import *

class CrossAttentionGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads):
        super(CrossAttentionGAT, self).__init__()
        self.hidden_dim = hidden_dim
        self.gat1 = tg.nn.GATConv(in_dim, hidden_dim, heads=num_heads, concat=False)
        self.gat2 = tg.nn.GATConv(in_dim, hidden_dim, heads=num_heads, concat=False)
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())

    def forward(self, graph1, graph2):
        # Compute initial representations for each graph
        if graph1.edge_index.numel() == 0 or graph1.edge_index.shape[0] == 0\
                or graph2.edge_index.numel() == 0 or graph2.edge_index.shape[0] == 0:
            emb1 = self.proj(graph1.x)
            emb2 = self.proj(graph2.x)

        else:
            emb1 = self.gat1(graph1.x, graph1.edge_index)
            emb2 = self.gat2(graph2.x, graph2.edge_index)

        attention_scores = torch.matmul(emb1, emb2.transpose(0, 1))
        att_weights_1 = torch.softmax(attention_scores, dim=1)
        att_weights_2 = torch.softmax(attention_scores, dim=0)
        cross_att_1 = torch.matmul(att_weights_2, emb2)
        cross_att_2 = torch.matmul(att_weights_1.transpose(0, 1), emb1)

        return cross_att_1, cross_att_2
    



class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphEncoder, self).__init__()
        self.conv1 = tg.nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = tg.nn.GCNConv(hidden_dim, output_dim)
        self.pool = tg.nn.global_mean_pool
        self.proj = nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU())


    def forward(self, x, edge_index, batch):
        if edge_index.numel() == 0 or edge_index.shape[0] == 0:
            x = self.proj(x)
        elif edge_index.shape[1] == 0:
            x = self.conv1(x, edge_index)
        else:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            x = self.pool(x, batch) # global mean pooling, shape [num_graphs, output_dim]
        return x


class GraphMatchModel(nn.Module):
    def __init__(self, g_encoder, scene_encoder, attr_encoder, tokenizer, temperature, args, lin_dim, lout_dim, device):
        super(GraphMatchModel, self).__init__()
        self.temperature = temperature
        self.sampling_num = args.sampling_num
        self.tops = args.tops
        self.hops = args.hops
        self.sampling_bound = args.sampling_bound
        self.device = device
        self.lin_dim = lin_dim
        self.lout_dim = lout_dim
        self.image_reduction = nn.Sequential(nn.Linear(256*64*64, lout_dim),
                                             nn.ReLU())
        self.text_reduction = nn.Sequential(nn.Linear(768, lout_dim),
                                            nn.ReLU())
        self.linear = nn.Linear(self.lin_dim, self.lout_dim)  # [128, 256]
        self.loss_fn = nn.BCELoss()
        # self.loss_fn = nn.BCEWithLogitsLoss()

        self.crop = args.crop
        self.imp_sample = args.imp_sample

        self.attr_encoder = attr_encoder
        self.tokenizer = tokenizer
        self.scene_encoder = scene_encoder
        self.g_encoder = g_encoder
        self.iou_threshold = args.iou_threshold
        self.pos_num = 1

    def forward_nonsample(self, kg, id2attr, scene, clip_model):
        scene = scene.to(self.device)
        graph = kg.clone().to(self.device)

        scene.x = scene.x.float()
        graph.x = graph.x.float()
        y = torch.mean(scene.x, dim=0)
        kg_emb = self.g_encoder(kg.x[id], kg.edge_index) # kg.edge_index[:, kg.edge_index[0] == id]
        kg_emb = torch.norm(kg_emb, dim=1, keepdim=True)
        sim_score = torch.matmul(scene.x, kg_emb.t())

        matches = []
        classes = []
        top_values, top_indices = torch.topk(sim_score, self.tops, dim=1, largest=True)
        match_degree = torch.mean(sim_score[top_indices], dim=1)
        matches.append(match_degree)
        classes.append(top_indices)

        if self.crop == True:
            with torch.no_grad():
                sub_first_txt = [clip_model.encode_text(clip.tokenize(cl[:77]).to(self.device)).to(self.device) for cl in classes]
                tokens = clip.tokenize(scene.y[:77]).to(self.device)
                scene_txt = clip_model.encode_text(tokens).to(self.device)
        else:
            sub_first_txt = [extractor.sent_embed(classes, self.attr_encoder, self.tokenizer, self.device) for cl in classes]
            scene_txt = extractor.sent_embed(scene.y, self.attr_encoder, self.tokenizer, self.device)
        
        return matches, classes, y, scene_txt, sub_first_txt


    def forward(self, kg, id2attr, node_pair_rel, scene, g_type, clip_model, explore, refine, rf_clip, sampling_type):
        scene = scene.to(self.device)
        graph = kg.clone().to(self.device)

        scene.x = scene.x.float()
        graph.x = graph.x.float()

        # print(f"scene shape: {scene.x.shape}, graph shape: {graph.x.shape}")

        if explore == True:
            sampling_num = self.sampling_num
        else:
            # random(+sage), around(-e)
            sampling_num = len(graph.x)
        sim_score = torch.matmul(scene.x, graph.x.t())

        sorted_indices = torch.argsort(sim_score, dim=1, descending=True)
        top_indices = []
        for i in sorted_indices.view(-1):
            if i.item() not in top_indices and len(top_indices) < sampling_num:
                top_indices.append(i.item())

            if len(top_indices) == sampling_num:
                break

        indices = torch.tensor(top_indices)

        max_sim_values, _ = torch.max(sim_score, dim=1)
        max_row_id = torch.argmax(max_sim_values).item()
        y = scene.x[max_row_id].unsqueeze(0)
        x_y_sim = torch.cosine_similarity(y, graph.x)
        subgraphs = sampler.get_sampling_subgraph(graph, id2attr, node_pair_rel, x_y_sim, indices, sampling_type,
                                                int(self.sampling_bound), self.hops)
        
        if g_type == "GCN":
            scene_emb = self.g_encoder(scene.x, scene.edge_index, scene.batch)

        sub_first_txt = torch.tensor([]).to(self.device)
        sub_second_txt = torch.tensor([]).to(self.device)
        sub_embs = torch.tensor([]).to(self.device)
        scene_embs = torch.tensor([]).to(self.device)
        matches = []
        sub_ents = []
        classes = []

        if refine == True:
            for start, subgraph in subgraphs.items():
                classes.append(id2attr[start.item()])
                subgraph = subgraph.to(self.device)
                
                if g_type == "GCN":
                    sub_emb = self.g_encoder(subgraph.x, subgraph.edge_index, subgraph.batch)

                elif g_type == "C-GAT":
                    sub_emb, scene_emb = self.g_encoder(subgraph, scene)
                    scene_embs = torch.cat((scene_embs, scene_emb), dim=0)

                match_degree = self.matching_score(scene_emb, sub_emb)
                matches.append(match_degree)
                sub_embs = torch.cat((sub_embs, sub_emb), dim=0)
                sub_ents.append(subgraph.y["ents"])

                if self.crop == True:
                    with torch.no_grad():
                        sub_first_txt = torch.cat((sub_first_txt, 
                            clip_model.encode_text(clip.tokenize(subgraph.y["1"][:77]).to(self.device)).to(self.device)), dim=0)
                        sub_second_txt = torch.cat((sub_second_txt, 
                            clip_model.encode_text(clip.tokenize(subgraph.y["2"][:77]).to(self.device)).to(self.device)), dim=0)
                else:
                    sub_first_txt = torch.cat((sub_first_txt, 
                            extractor.sent_embed(subgraph.y["1"], self.attr_encoder, self.tokenizer, self.device)), dim=0)
                    sub_second_txt = torch.cat((sub_second_txt, 
                            extractor.sent_embed(subgraph.y["2"], self.attr_encoder, self.tokenizer, self.device)), dim=0)
            
            match = torch.tensor(matches)
            top_values, top_indices = torch.topk(match, self.tops, dim=0, largest=True)
            top_values = top_values.unsqueeze(1).to(self.device)
            top_indices = top_indices.to(self.device)

            scene_emb = scene_embs[top_indices] if g_type == "C-GAT" else scene_emb
            if self.crop == True:
                with torch.no_grad():
                    tokens = clip.tokenize(scene.y[:77]).to(self.device)
                    scene_txt = clip_model.encode_text(tokens).to(self.device)
            else:
                scene_txt = extractor.sent_embed(scene.y, self.attr_encoder, self.tokenizer, self.device)

        else:
            scene_emb = torch.mean(scene.x, dim=0, keepdim=True).to(self.device)
            
            for start, subgraph in subgraphs.items():
                classes.append(id2attr[start.item()])
                if rf_clip == True:
                    sub_emb = graph.x[start.item()]
                sub_emb = torch.mean(subgraph.x, dim=0, keepdim=True).to(self.device)

                match_degree = self.matching_score(scene_emb, sub_emb)
                matches.append(match_degree)
                sub_embs = torch.cat((sub_embs, sub_emb), dim=0)
                sub_ents.append(subgraph.y["ents"])

                sub_first_txt = torch.cat((sub_first_txt, 
                        extractor.sent_embed(subgraph.y["1"], self.attr_encoder, self.tokenizer, self.device)), dim=0)
                sub_second_txt = torch.cat((sub_second_txt, 
                        extractor.sent_embed(subgraph.y["2"], self.attr_encoder, self.tokenizer, self.device)), dim=0)

            match = torch.tensor(matches)
            top_values, top_indices = torch.topk(match, self.tops, dim=0, largest=True)
            top_values = top_values.unsqueeze(1).to(self.device)
            top_indices = top_indices.to(self.device)

            scene_txt = extractor.sent_embed(scene.y, self.attr_encoder, self.tokenizer, self.device)

        top_sub = sub_embs[top_indices]
        top_sub_first = sub_first_txt[top_indices]
        top_sub_second = sub_second_txt[top_indices]
        top_sub_ents = [sub_ents[ind] for ind in top_indices]
        classes_str = [classes[ind]for ind in top_indices]

        
        return top_values, top_sub, scene_emb, scene_txt, top_sub_first, top_sub_second, top_sub_ents, classes_str

    
    def matching_score(self, sub_emb, sce_emb):
        C = torch.matmul(sce_emb, sub_emb.t())

        max_similarity, _ = torch.max(C, dim=1) 
        match_degree = torch.mean(max_similarity)  

        return match_degree

    def dist_loss(self, mats, scene_txts, sub_first, sub_second, global_weight=0.2, margin=0.1):
        loss = 0

        fv_simi = mats
        fs_simi_1 = torch.cosine_similarity(scene_txts, sub_first, dim=1).unsqueeze(1)

        f_simi = fv_simi + fs_simi_1 * global_weight

        top_k_values, top_k_indices = torch.topk(f_simi, k=3, dim=1)
        positive_mask = torch.zeros_like(f_simi)
        positive_mask.scatter_(1, top_k_indices[:, 0].unsqueeze(1), 1)
        negative_mask = 1 - positive_mask
        positive_logits = torch.sum(f_simi * positive_mask, dim=1)
        negative_logits = torch.sum(f_simi * negative_mask, dim=1)

        loss = torch.mean(torch.max(torch.zeros_like(positive_logits), margin - positive_logits + negative_logits))

        loss.requires_grad = True

        return loss


def train(gpu, kg, id2attr, node_pair_rel, images, captions, args, train_ground, ent_dict, lin_dim=512, lout_dim=512):
    
    rank = args.nr * args.gpus + gpu
    task = args.data.strip().split("/")[-1]
    scene_encoder = extractor.sam_encoder(rank, args.sam_type, args.root + args.sam_ckp)
    
    clip_model, clip_preprocess = clip.load("ViT-B/32", rank)
    kg = extractor.kg_x_load(args.root + args.data + args.kg_dir, "train", kg, id2attr, clip_model, gpu)

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.gpus * args.nodes,
        rank=rank)
    torch.cuda.set_device(gpu)
    
    if args.g_type == "GCN":
        g_encoder = GraphEncoder(input_dim=128, hidden_dim=128, output_dim=128)
    elif args.g_type == "C-GAT":
        g_encoder = CrossAttentionGAT(in_dim=512, hidden_dim=512, num_heads=8)
    
    attr_encoder, tokenizer = extractor.attr_encoder(args.root)

    mm = GraphMatchModel(g_encoder, scene_encoder, attr_encoder, tokenizer, temperature=0.1, args=args, 
                         lin_dim=lin_dim, lout_dim=lout_dim, device=rank).to(gpu)
    
    print("Turning off gradients in bert and clip encoders")
    for param in mm.attr_encoder.parameters():
        param.requires_grad = False

    enabled = set()
    for name, param in mm.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    print(f"Parameters to be updated: {enabled}")

    mm1 = DDP(mm, device_ids=[gpu], find_unused_parameters=True)
    mm1.train()

    split = int(len(images) / 2)
    scene_loader = DataLoader(images[: split - 1] if gpu == 0 else images[split: len(images)- 1],
                              batch_size=args.batch_size, shuffle=True) 

    print(f"gpu {rank} - each gpu processes image number: {split}")
    
    optimizer = optim.Adam(mm.parameters(), lr=args.learning_rate)
    for epoch in range(args.num_epochs):
        print(f"rank: {rank}, epoch: {epoch}")
        epoch_loss = 0

        ind = 0
        batch = 0
        epoch_scene_ents = {}
        epoch_img_truth = {}
        epoch_pred = {}

        for img_list in scene_loader: ## batch
            optimizer.zero_grad()
            scenes = extractor.get_scene_batches(scene_encoder, img_list, args.iou_threshold, args.vis_merge, args.crop, clip_model.visual, clip_preprocess, gpu, captions)
            print(f"Processing {len(scenes)} images in {batch}-th batch.")
            start = datetime.now()

            if args.imp_sample == True:
                mats = []
                sub_embs = []
                # scene_embs = torch.tensor([]).to(gpu)
                scene_txts = []
                sub_first = []
                sub_second = []
                scene_ents = []
                classes = []
                for path, scene in scenes.items():
                    mat_value, sub_emb, scene_emb, scene_txt, sub_first_txt, sub_second_txt, ents, classes_str = mm(kg, id2attr, 
                                node_pair_rel, scene, args.g_type, clip_model, args.explore, args.refine, args.rf_clip, args.sampling_type)

                    mats.append(mat_value)
                    sub_embs.append(sub_emb)
                    scene_txts.append(scene_txt)
                    sub_first.append(sub_first_txt)
                    sub_second.append(sub_second_txt)
                    scene_ents.append(ents) # (batch_size)
                    classes.append(classes_str)

                mats = torch.stack(mats).to(gpu) # (batch_size, tops, 1)
                sub_embs = torch.stack(sub_embs).to(gpu) # (batch_size, tops, t_dim)
                scene_txts = torch.stack(scene_txts).to(gpu) # (batch_size, 1, v_dim)
                sub_first = torch.stack(sub_first).to(gpu) # (batch_size, tops, t_dim)
                sub_second = torch.stack(sub_second).to(gpu) # (batch_size, tops, t_dim)

                ind += len(scenes)

                loss = mm.dist_loss(mats, scene_txts, sub_first, sub_second)

            else:
                mats = []
                sub_embs = []
                # scene_embs = torch.tensor([]).to(gpu)
                scene_txts = []
                sub_first = []
                sub_second = []
                classess = []
                for path, scene in scenes.items():
                    matches, classes, y, scene_txt, sub_first_txt = mm(kg, id2attr, scene, clip_model)

                    mats.append(matches)
                    classess.append(classess)
                    scene_txts.append(scene_txt)
                    sub_first.append(sub_first_txt)

            
            torch.autograd.set_detect_anomaly(True)
            loss.backward(retain_graph=True)
            optimizer.step()

            if task in ['openImages']:
                pred_scene_ents, truth_scene_ents = retrieval_prediction(
                                mats, list(scenes.keys()), scene_ents, train_ground, ent_dict, args.tops, attr_encoder, tokenizer, gpu)
                epoch_scene_ents.update(pred_scene_ents)
                epoch_img_truth.update(truth_scene_ents)
                hits_at_k, mrr = calculate_hits_mrr(truth_scene_ents, pred_scene_ents, gpu, task, args.threshold)
            else:
                pred_img_class, epoch_pred = retrieve_image_labels(mats, list(scenes.keys()), classes, epoch_pred, attr_encoder, tokenizer, gpu)
                hits_at_k, mrr = calculate_hits_mrr(train_ground, pred_img_class, gpu, task, args.threshold, attr_encoder, tokenizer)

            if batch == 0: epoch_time = datetime.now() - start
            else: epoch_time += (datetime.now() - start)
            print(f"gpu {rank} - batch {batch}, loss: {loss.item()}, hits@k: {hits_at_k}, mrr: {mrr}, takes {datetime.now() - start} sec")

            batch += 1
            epoch_loss += loss.item()
            torch.cuda.empty_cache()

        start = datetime.now()
        if task in ['openImages']:
            hits_at_k, mrr = calculate_hits_mrr(epoch_img_truth, epoch_scene_ents, gpu, task, args.threshold)
        else:
            # hits_at_k, mrr = calculate_hits_mrr_global(train_ground, epoch_pred)
            hits_at_k, mrr = calculate_hits_mrr(train_ground, epoch_pred, gpu, task, args.threshold, attr_encoder, tokenizer)
        epoch_time += (datetime.now() - start)

        epoch_loss = epoch_loss / ind
        print(f"gpu {rank} - epoch {epoch}, {datetime.now()}, loss: {epoch_loss:.4f}, hits@k: {hits_at_k}, mrr: {mrr}, take {epoch_time} Sec")
        torch.save(mm.state_dict(), "./output/model" + "_" + task + "_" + args.g_type + ".pth")
        print(f"rank: {rank}, {epoch}-th complete")

    suff = ''
    if args.explore == True:
        suff = suff + "_e" 
    if args.refine == True:
        suff = suff + "_r" 
    torch.save(mm.state_dict(), "./output/model" + "_" + task + "_" + args.g_type + suff + ".pth")

    dist.destroy_process_group() 


def test(kg, id2attr, node_pair_rel, images, args, test_ground, ent_dict, lin_dim=512, lout_dim=512):
    scene_encoder = extractor.sam_encoder(args.device, args.sam_type, args.root + args.sam_ckp)
    attr_encoder, tokenizer = extractor.attr_encoder(args.root)
    clip_model, clip_preprocess = clip.load("ViT-B/32", args.device)
    kg = extractor.kg_x_load(args.root + args.data + args.kg_dir, "test", kg, id2attr, clip_model, args.device)
    image_encoder = clip_model.visual
    
    if args.g_type == "GCN":
        g_encoder = GraphEncoder(input_dim=128, hidden_dim=128, output_dim=128)
    elif args.g_type == "C-GAT":
        g_encoder = CrossAttentionGAT(in_dim=512, hidden_dim=512, num_heads=8)

    start = datetime.now()
    task = args.data.strip().split("/")[-1]
    model = GraphMatchModel(g_encoder, scene_encoder, attr_encoder, tokenizer, temperature=0.1, args=args, 
                            lin_dim=lin_dim, lout_dim=lout_dim, device=args.device).to(args.device)
    
    suff = ''
    if args.explore == True:
        suff = suff + "_e" 
    if args.refine == True:
        suff = suff + "_r" 
    model.load_state_dict(torch.load("./output/model" + "_" + task + "_" + args.g_type + suff + ".pth"))
    model.eval()

    scene_loader = DataLoader(images, batch_size=args.batch_size, shuffle=True) 

    scene_ents = []
    with torch.no_grad():
        for img_list in scene_loader: ## batch
            iter = 0
            scenes = extractor.get_scene_batches(scene_encoder, img_list, args.iou_threshold, args.vis_merge,  args.crop, 
                                                 image_encoder, clip_preprocess, args.device)
            start = datetime.now()

            mats = []
            for path, scene in scenes.items():
                mat_value, _, _, _, _, _, ents, classes_str = model(kg, id2attr, node_pair_rel, scene, args.g_type, 
                                                                    clip_model, args.explore, args.refine, args.rf_clip, args.sampling_type)
                # print(f"mat_value: {mat_value.shape}, ents: {len(ents)}")

                mats.append(mat_value)
                scene_ents.append(ents)
            
            if iter == 0: time = datetime.now() - start
            else: time += (datetime.now() - start)
            iter += 1

        mats = torch.stack(mats).to(args.device)

    start = datetime.now()
    if task in ['openImages']:
        pred_scene_ents, truth_scene_ents = retrieval_prediction(
                        mats, list(scenes.keys()), scene_ents, test_ground, ent_dict, args.tops, attr_encoder, tokenizer, args.device)
        hits_at_k, mrr = calculate_hits_mrr(truth_scene_ents, pred_scene_ents, args.device, task, args.threshold)
    else:
        epoch_pred = {}
        pred_img_class, epoch_pred = retrieve_image_labels(mats, list(scenes.keys()), classes_str, epoch_pred, attr_encoder, tokenizer, args.device)
        # hits_at_k, mrr = calculate_hits_mrr_global(train_ground, pred_img_class)
        hits_at_k, mrr = calculate_hits_mrr(test_ground, pred_img_class, args.device, task, args.threshold, attr_encoder, tokenizer)
    
    time += (datetime.now() - start)

    print(f"hits@k: {hits_at_k}, mrr: {mrr}, takes {time} sec")




def calculate_hits_and_mrr(predicted_indices, true_index):
    hits = [1 if true_index in predicted_indices[:k] else 0 for k in range(1, len(predicted_indices) + 1)]
    mrr = 1 / (predicted_indices.index(true_index) + 1) if true_index in predicted_indices else 0

    return hits, mrr


def compute_metrics(prediction, label):
    prediction = prediction.numpy()
    label = label.numpy()

    true_positives = ((prediction == 1) & (label == 1)).sum()
    false_positives = ((prediction == 1) & (label == 0)).sum()
    false_negatives = ((prediction == 0) & (label == 1)).sum()

    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return precision, recall, f1


def test_data_load(kg, file):
    ids = []
    matches = []

    with open(file, "r") as file:
        for line in file:
            line = line.strip().split(", ")

            id = int(line[0])
            strs = []
            for i in range(1, len(line)):
                ver = get_vertex_id(line[i], kg)
                if ver != -1:
                    strs.append(ver)

            matches.append(strs)
            ids.append(id)

    return ids, matches



def get_vertex_id(str_value, graph):

    for i in range(graph.num_nodes):
        if graph.x[i] == str_value:
            return i
    return -1

