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

# Define the GAT model
class CrossAttentionGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads):
        super(CrossAttentionGAT, self).__init__()
        self.hidden_dim = hidden_dim
        self.gat1 = tg.nn.GATConv(in_dim, hidden_dim, heads=num_heads, concat=False)
        self.gat2 = tg.nn.GATConv(in_dim, hidden_dim, heads=num_heads, concat=False)
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        # self.linear = nn.Linear(1024, 128)

    def forward(self, graph1, graph2):
        # Compute initial representations for each graph
        if graph1.edge_index.numel() == 0 or graph1.edge_index.shape[0] == 0\
                or graph2.edge_index.numel() == 0 or graph2.edge_index.shape[0] == 0:
            emb1 = self.proj(graph1.x)
            emb2 = self.proj(graph2.x)

        else:
            emb1 = self.gat1(graph1.x, graph1.edge_index)
            emb2 = self.gat2(graph2.x, graph2.edge_index)
        
        # print(f"emb1: {emb1.shape}, emb2: {emb2.shape}")

        attention_scores = torch.matmul(emb1, emb2.transpose(0, 1))
        att_weights_1 = torch.softmax(attention_scores, dim=1)
        att_weights_2 = torch.softmax(attention_scores, dim=0)
        cross_att_1 = torch.matmul(att_weights_2, emb2)
        cross_att_2 = torch.matmul(att_weights_1.transpose(0, 1), emb1)

        # cross_att_1 = self.linear(cross_att_1)
        # cross_att_2 = self.linear(cross_att_2)

        # print(f"cross_att_1: {cross_att_1.shape}, cross_att_2: {cross_att_2.shape}")

        # cross_att_1 = torch.mean(cross_att_1, dim=0).unsqueeze(0)
        # cross_att_2 = torch.mean(cross_att_2, dim=0).unsqueeze(0)

        return cross_att_1, cross_att_2
    



class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphEncoder, self).__init__()
        self.conv1 = tg.nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = tg.nn.GCNConv(hidden_dim, output_dim)
        self.pool = tg.nn.global_mean_pool
        self.proj = nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU())


    def forward(self, x, edge_index, batch):
        # x: node features, shape [num_nodes, input_dim]
        # edge_index: edge indices, shape [2, num_edges]
        if edge_index.numel() == 0 or edge_index.shape[0] == 0:
            x = self.proj(x)
        elif edge_index.shape[1] == 0:
            # Special case: graph with only a single node
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
        # self.device = args.device
        self.device = device
        self.lin_dim = lin_dim
        self.lout_dim = lout_dim
        # 定义图像和文本的降维层
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
        # self.kg_vector = extractor.attr_feature(kg.x, self.attr_encoder, self.tokenizer)
        # self.kg = kg


    def forward_nonsample(self, kg, id2attr, scene, clip_model):
        scene = scene.to(self.device)
        graph = kg.clone().to(self.device)

        scene.x = scene.x.float()
        graph.x = graph.x.float()

        # sim_score = torch.matmul(scene.x, graph.x.t())
        y = torch.mean(scene.x, dim=0)
        # for id, attr in id2attr.item():
        kg_emb = self.g_encoder(kg.x[id], kg.edge_index) # kg.edge_index[:, kg.edge_index[0] == id]
        kg_emb = torch.norm(kg_emb, dim=1, keepdim=True)
        sim_score = torch.matmul(scene.x, kg_emb.t())

        matches = []
        classes = []
        top_values, top_indices = torch.topk(sim_score, self.tops, dim=1, largest=True)
        match_degree = torch.mean(sim_score[top_indices], dim=1)
        matches.append(match_degree)
        classes.append(top_indices)


        # max_similarity, max_indices = torch.max(sim_score, dim=1)  # 每行的最大相似度
        # match_degree = torch.mean(max_similarity)  # 平均匹配度
        # common_id = torch.mode(max_indices).mode.item()
        # matches = match_degree
        # classes = id2attr[common_id]

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

        # 对图像和文本进行特征空间映射
        # print(f"scene shape: {scene.x.shape}, graph shape: {graph.x.shape}")

        # img_vec = self.image_reduction(scene.x.view(scene.x.shape[0], -1)).type(torch.FloatTensor)  # [10, 1048576] -> [10, 128]
        # print(f"scene shape: {scene.x.shape}, scene reduction: {img_vec.shape}")
        # text_vec = self.text_reduction(graph.x.to(self.device))
        # print(f"graph shape: {graph.x.shape}, graph reduction shape: {text_vec.shape}")
        # scene.x = img_vec.to(self.device)
        # graph.x = text_vec.to(self.device)
        # x2 = text_vec

        scene.x = scene.x.float()
        graph.x = graph.x.float()

        # print(f"scene shape: {scene.x.shape}, graph shape: {graph.x.shape}")

        if explore == True:
            sampling_num = self.sampling_num
        else:
            # random(+sage), around(-e)
            sampling_num = len(graph.x)

        # 1.选择与anchor相似度最高的前top个节点作为采样的起点，并去除重复节点
        sim_score = torch.matmul(scene.x, graph.x.t())
        # values, indices = torch.topk(sim_score, sampling_num, dim=1, largest=True)
        # # print(f"gpu {self.device} - anchor: {anchors}, values: {values}, index: {indices}")
        # _, unique_indices = torch.unique(indices, dim=1, return_inverse=True)
        # indices = unique_indices.squeeze()

        sorted_indices = torch.argsort(sim_score, dim=1, descending=True)
        top_indices = []
        for i in sorted_indices.view(-1):
            if i.item() not in top_indices and len(top_indices) < sampling_num:
                top_indices.append(i.item())

            if len(top_indices) == sampling_num:
                break

        indices = torch.tensor(top_indices)

        # print(f"gpu {self.device} - unique index : {indices.shape}")

        # 2.计算知识图谱节点与场景图均值特征向量y的相似度，并基于相似度采样子图
        # y = torch.mean(scene.x, dim=0)
        # max_index = torch.argmax(sim_score, dim=0)
        max_sim_values, _ = torch.max(sim_score, dim=1)
        max_row_id = torch.argmax(max_sim_values).item()
        y = scene.x[max_row_id].unsqueeze(0)
        x_y_sim = torch.cosine_similarity(y, graph.x)
        # print(f"scene x: {scene.x.shape}, graph x: {graph.x.shape}, sim_score: {sim_score.shape}, y: {y.shape}, x_y_sim: {x_y_sim.shape}")
        # x_y_sim = F.softmax(x_y_sim, dim=0)
        # nodes = set((node for node in index) for index in indices)
        subgraphs = sampler.get_sampling_subgraph(graph, id2attr, node_pair_rel, x_y_sim, indices, sampling_type,
                                                int(self.sampling_bound), self.hops)
        
        if g_type == "GCN":
            scene_emb = self.g_encoder(scene.x, scene.edge_index, scene.batch)

        # 3.采样子图进行嵌入计算和匹配度计算
        sub_first_txt = torch.tensor([]).to(self.device)
        sub_second_txt = torch.tensor([]).to(self.device)
        sub_embs = torch.tensor([]).to(self.device)
        scene_embs = torch.tensor([]).to(self.device)
        matches = []
        sub_ents = []
        classes = []

        if refine == True:
            # print(f"Number of subgraphs: {len(subgraphs)}")
            for start, subgraph in subgraphs.items():
                classes.append(id2attr[start.item()])
                subgraph = subgraph.to(self.device)
                
                if g_type == "GCN":
                    sub_emb = self.g_encoder(subgraph.x, subgraph.edge_index, subgraph.batch)
                    # print(f"sub_emb: {sub_emb.shape}, sce_emb: {scene_emb.shape}")

                elif g_type == "C-GAT":
                    # print(f"subgraph.x: {subgraph.x.shape}, scene.x: {scene.x.shape}")
                    sub_emb, scene_emb = self.g_encoder(subgraph, scene)
                    # print(f"sub_emb: {sub_emb.shape}, sce_emb: {scene_emb.shape}")
                    scene_embs = torch.cat((scene_embs, scene_emb), dim=0)
                
                # print(f"sub_emb: {sub_emb.shape}, sce_emb: {scene_emb.shape}")

                match_degree = self.matching_score(scene_emb, sub_emb)
                matches.append(match_degree)
                sub_embs = torch.cat((sub_embs, sub_emb), dim=0)
                # avg_ents = torch.stack([extractor.sent_embed(ent, self.attr_encoder, self.tokenizer, self.device) for ent in subgraph.y["ents"]])
                # avg_ents = torch.mean(avg_ents, dim=0, keepdim=True).squeeze(0)
                # print(f"avg_ents: {avg_ents.shape}")
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
            
            # 对每个scene返回最大匹配度的子图
            match = torch.tensor(matches)
            # print(f"match: {match.shape}")
            # mat_value, mat_ind = torch.max(match, dim=0)
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
            
            # 对每个scene返回最大匹配度的子图
            match = torch.tensor(matches)
            # print(f"match: {match.shape}")
            top_values, top_indices = torch.topk(match, self.tops, dim=0, largest=True)
            top_values = top_values.unsqueeze(1).to(self.device)
            top_indices = top_indices.to(self.device)

            scene_txt = extractor.sent_embed(scene.y, self.attr_encoder, self.tokenizer, self.device)

        # print(f"sub_embs: {sub_embs.shape}, sub_first_txt: {sub_first_txt.shape}, sub_second_txt: {sub_second_txt.shape}, sub_ents; {len(sub_ents)}")
        
        top_sub = sub_embs[top_indices]
        top_sub_first = sub_first_txt[top_indices]
        top_sub_second = sub_second_txt[top_indices]
        top_sub_ents = [sub_ents[ind] for ind in top_indices]
        classes_str = [classes[ind]for ind in top_indices]

        
        return top_values, top_sub, scene_emb, scene_txt, top_sub_first, top_sub_second, top_sub_ents, classes_str

    
    def matching_score(self, sub_emb, sce_emb):
        # 计算scene-graph matching score in definition
        # C = torch.matmul(sub_emb, sce_emb.t()) 
        # C = C.t() 
        # normA = torch.norm(sce_emb, dim=1, keepdim=True)  # 计算A的范数
        # normB = torch.norm(sub_emb, dim=1, keepdim=True)  # 计算B的范数
        C = torch.matmul(sce_emb, sub_emb.t())  # 相似度矩阵归一化

        max_similarity, _ = torch.max(C, dim=1)  # 每行的最大相似度
        match_degree = torch.mean(max_similarity)  # 平均匹配度

        return match_degree

    def dist_loss(self, mats, scene_txts, sub_first, sub_second, global_weight=0.2, margin=0.1):
        loss = 0

        fv_simi = mats
        fs_simi_1 = torch.cosine_similarity(scene_txts, sub_first, dim=1).unsqueeze(1)

        f_simi = fv_simi + fs_simi_1 * global_weight
        # print(f"f_simi: {f_simi}")

        top_k_values, top_k_indices = torch.topk(f_simi, k=3, dim=1)
        positive_mask = torch.zeros_like(f_simi)
        positive_mask.scatter_(1, top_k_indices[:, 0].unsqueeze(1), 1)
        negative_mask = 1 - positive_mask

        # 将正样本和负样本分别与相似度矩阵相乘，得到正负样本对应的相似度
        positive_logits = torch.sum(f_simi * positive_mask, dim=1)
        negative_logits = torch.sum(f_simi * negative_mask, dim=1)

        loss = torch.mean(torch.max(torch.zeros_like(positive_logits), margin - positive_logits + negative_logits))

        loss.requires_grad = True

        return loss




#args=(knowledge_graph, id2attr, node_pair_rel, train_imgs, captions, args, train_ground, ent_dict)
def train(gpu, kg, id2attr, node_pair_rel, images, captions, args, train_ground, ent_dict, lin_dim=512, lout_dim=512):
    
    rank = args.nr * args.gpus + gpu
    task = args.data.strip().split("/")[-1]
    scene_encoder = extractor.sam_encoder(rank, args.sam_type, args.root + args.sam_ckp)
    
    clip_model, clip_preprocess = clip.load("ViT-B/32", rank)
    kg = extractor.kg_x_load(args.root + args.data + args.kg_dir, "train", kg, id2attr, clip_model, gpu)
    # with torch.no_grad():
    #     tokens = [clip.tokenize(id2attr[i][:77]).to(gpu) for i in range(len(id2attr))]
    #     x = [clip_model.encode_text(tokens[i]).to(gpu) for i in range(len(id2attr))]
    #     x = torch.cat(x, dim=0)
    #     kg.x = x / x.norm(dim=-1, keepdim=True)

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
    # for param in mm.clip_model.parameters():
    #     param.requires_grad = False

    enabled = set()
    for name, param in mm.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    print(f"Parameters to be updated: {enabled}")

    # warp the model
    mm1 = DDP(mm, device_ids=[gpu], find_unused_parameters=True)
    mm1.train()
    # print(f"gpu {rank} - model: {mm}")
    # print(f'gpu {rank} - model: {next(mm.parameters()).device}')

    split = int(len(images) / 2)
    scene_loader = DataLoader(images[: split - 1] if gpu == 0 else images[split: len(images)- 1],
                              batch_size=args.batch_size, shuffle=True) # 在每个epoch开始时对数据进行随机打乱

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
                    # print(f"mat_value: {mat_value.shape}, scene_txt: {scene_txt.shape},sub_first_txt: {sub_first_txt.shape}, 
                    #         sub_second_txt: {sub_second_txt.shape}, ents: {len(ents)}")

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
                # print(f"mats: {mats.shape}, scene_txts: {scene_txts.shape},sub_first: {sub_first.shape}, sub_second: {sub_second.shape}, secne_ents: {len(secne_ents)}")

                ind += len(scenes)
                # print(f"batch-{batch} for {len(mats)} images")

                loss = mm.dist_loss(mats, scene_txts, sub_first, sub_second)
                # print(f"loss: {loss}")
                # loss = mm.margin_loss(zs, scene_embs, pos=0.7, neg=0.4)

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

            # predict of matching pairs and evaluate
            if task in ['openImages']:
                pred_scene_ents, truth_scene_ents = retrieval_prediction(
                                mats, list(scenes.keys()), scene_ents, train_ground, ent_dict, args.tops, attr_encoder, tokenizer, gpu)
                epoch_scene_ents.update(pred_scene_ents)
                epoch_img_truth.update(truth_scene_ents)
                hits_at_k, mrr = calculate_hits_mrr(truth_scene_ents, pred_scene_ents, gpu, task, args.threshold)
            else:
                pred_img_class, epoch_pred = retrieve_image_labels(mats, list(scenes.keys()), classes, epoch_pred, attr_encoder, tokenizer, gpu)
                # hits_at_k, mrr = calculate_hits_mrr_global(train_ground, pred_img_class)
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

        # dist.barrier() # 同步参数

    suff = ''
    if args.explore == True:
        suff = suff + "_e" 
    if args.refine == True:
        suff = suff + "_r" 
    torch.save(mm.state_dict(), "./output/model" + "_" + task + "_" + args.g_type + suff + ".pth")

    dist.destroy_process_group() # 释放空间


# train(gpu, kg, id2attr, node_pair_rel, images, captions, args, train_ground, ent_dict, lin_dim=512, lout_dim=512)
def test(kg, id2attr, node_pair_rel, images, args, test_ground, ent_dict, lin_dim=512, lout_dim=512):
    """
        Hits@K(命中率)指的是在前 K 个推荐结果中，有多少个是用户实际感兴趣的项目
        MRR(平均倒数排名)是指在所有用户的推荐结果中，计算用户感兴趣的项目在排序列表中的倒数排名的平均值
    """

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

    scene_loader = DataLoader(images, batch_size=args.batch_size, shuffle=True) # 在每个epoch开始时对数据进行随机打乱

    # mats = torch.tensor([]).to(args.device)
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
    # predict of matching pairs and evaluate
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
    # 将预测结果和标签转换为 numpy 数组
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
    # data每一行包含image以及匹配的20个vertex label
    ids = []
    matches = []

    # 读取 txt 文件
    with open(file, "r") as file:
        for line in file:
            # 拆分每一行的内容
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


"""
def train(gpu, kg, id2attr, node_pair_rel, images, args, lin_dim=128, lout_dim=128):
    optimizer = optim.Adam(mm.parameters(), lr=args.learning_rate)
    for epoch in range(args.num_epochs):
        epoch_loss = 0

        ind = 0
        batch = 0
        for img_list in scene_loader:  ## batch
            ind += 1
            start = datetime.now()

            scenes = extractor.get_scene_batches(scene_encoder, img_list, args.iou_threshold, args.img_dir)
            zs = torch.tensor([]).to(gpu)
            scene_embs = torch.tensor([]).to(gpu)
            for path, scene in scenes.items():
                z, scene_emb = mm(kg, id2attr, node_pair_rel, scene,args.g_type)
                zs = torch.cat((zs, z), dim=0)
                scene_embs = torch.cat((scene_embs, scene_emb), dim=0)
                if torch.numel(scene_emb) == 0:
                    print(f"null encoded scene feature: {path}")
                del scene

            if torch.numel(scene_embs) == 0: continue

            ind += len(scene_embs)
            print(f"{len(img_list)} images, batch-{batch} for {len(scene_embs)} images")

            loss = mm.margin_loss(zs, scene_embs, pos=0.7, neg=0.4)
            # 梯度累积
            loss = loss / steps  # 将损失值除以累积步数
            loss.backward()

            if ind % steps == 0:
                # 更新参数
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            if ind % steps == 0 or ind == len(scene_loader):
                print(f"gpu {rank} - batch {batch}, loss: {loss.item()}, takes {datetime.now() - start} sec")

            del scene_embs, zs

            batch += 1
            epoch_loss += loss.item()
"""

"""
    def margin_loss(self, z, scene_emb, pos = 0.5, neg = 0.3):
        print(f"z shape: {z.shape}")
        print(f"scene shape: {scene_emb.shape}")
        similarities = torch.mm(scene_emb, torch.transpose(z, 0, 1))  # shape [1, z.shape[0]]
        similarity_matrix = torch.sigmoid(similarities)
        # 创建标签矩阵，相似度大于阈值a的为1，小于阈值b的为-1
        labels = torch.where(similarity_matrix > pos, torch.tensor(1),
                             torch.where(similarity_matrix < neg, torch.tensor(-1), torch.tensor(0)))
        print(f"margin similarity_matrix: {similarity_matrix}")
        print(f"margin labels: {labels}")

        # 使用MarginRankingLoss计算margin loss
        target = torch.zeros_like(labels)  # 创建目标值张量，全为零
        loss = F.margin_ranking_loss(similarity_matrix.view(-1), labels.view(-1), target.view(-1), margin=0.5)

        return loss
"""

"""
def raw_train(scene_encoder, kg, scene_loader, args, lin_dim, lout_dim, g_type):
    if g_type == "GCN":
        g_encoder = GraphEncoder(input_dim=128, hidden_dim=128, output_dim=128)
    elif g_type == "C-GAT":
        g_encoder = CrossAttentionGAT(in_dim=128, hidden_dim=128, num_heads=8)

    mm = GraphMatchModel(g_encoder, scene_encoder, temperature=0.1, args=args, lin_dim=lin_dim, lout_dim=lout_dim).to(args.device)
    mm.train()
    print(f"model: {mm}")
    print(f'model: {next(mm.parameters()).device}')

    steps = 8  # 每8个批次更新一次参数

    optimizer = optim.Adam(mm.parameters(), lr=args.learning_rate)
    for epoch in range(args.num_epochs):
        epoch_loss = 0

        ind = 0
        for img_list in scene_loader: ## batch
            ind += 1

            scenes = extractor.get_scene_batches(scene_encoder, img_list, args.iou_threshold)
            zs = torch.tensor([]).to(args.device)
            scene_embs = torch.tensor([]).to(args.device)
            for scene in scenes:
                z, scene_emb = mm(kg, scene, g_type)
                zs = torch.cat((zs, z), dim=0)
                scene_embs = torch.cat((scene_embs, scene_emb), dim=0)
                del scene

            if torch.numel(zs) == 0: continue
            loss = mm.margin_loss(zs, scene_embs, pos=0.7, neg=0.4)

            # 梯度累积
            loss = loss / steps  # 将损失值除以累积步数
            loss.backward()

            if ind % steps == 0:
                # 更新参数
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            if ind % steps == 0 or ind == len(scene_loader) :
                print(f"Batch {ind}/{len(scene_loader)}, Loss: {loss.item()}")

            del scene_embs, scenes, zs

            epoch_loss += loss.item()
            # epoch_loss += batch_loss

        # 在最后一个批次之后，确保更新最后一次参数
        if ind % steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss = epoch_loss / len(scene_loader)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

    # torch.save(mm.state_dict(), "model.pth")
"""

"""
def train_1(scene_encoder, kg, scene_loader, args, lin_dim, lout_dim, g_type):
    if g_type == "GCN":
        g_encoder = GraphEncoder(input_dim=768, hidden_dim=128, output_dim=128)
    elif g_type == "C-GAT":
        g_encoder = CrossAttentionGAT(in_dim=768, hidden_dim=128, num_heads=8)

    mm = GraphMatchModel(g_encoder, scene_encoder, temperature=0.1, args=args, lin_dim=lin_dim, lout_dim=lout_dim).to(args.device)
    mm.train()
    print(f"model: {mm}")
    print(f'model: {next(mm.parameters()).device}')

    ind = 0
    optimizer = optim.Adam(mm.parameters(), lr=args.learning_rate)
    for epoch in range(args.num_epochs):
        epoch_loss = 0

        for img_list in scene_loader: ## batch
            batch_loss = 0

            scenes = extractor.get_scene_batches(scene_encoder, img_list, args.iou_threshold)

            zs = torch.tensor([]).to(args.device)
            scene_embs = torch.tensor([]).to(args.device)
            for scene in scenes:
                z, scene_emb = mm(kg, scene, g_type)
                zs = torch.cat((zs, z), dim=0)
                scene_embs = torch.cat((scene_embs, scene_emb), dim=0)
                del scene

            loss = mm.margin_loss(zs, scene_embs, pos=0.7, neg=0.4)

            batch_loss += loss.item()

            # 梯度归零
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            # backpropagate and update the parameters
            loss.backward()

            optimizer.step()

            batch_loss = batch_loss / len(scenes)  # average over batch
            print(f"Batch loss: {batch_loss:.4f}")

            del scene_embs, scenes, zs


            epoch_loss += batch_loss
        epoch_loss = epoch_loss / len(scene_loader)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

    # torch.save(mm.state_dict(), "model.pth")
"""

"""
    def compute_loss(self, z, scene_emb):
        print(f"z length: {len(z)}")
        similarities = torch.mm(scene_emb, z.t()) # shape [1, z.shape[0]]
        similarities = torch.sigmoid(similarities)
        _, top_indices = torch.topk(similarities, k=1, dim=1, largest=True) # todo k
        pos_pairs = []
        neg_pairs = []
        for i in range(len(scene_emb)):
            pos_pairs.append((i, top_indices[i][0])) # todo k=3, top_indices[i][1], top_indices[i][2]
            for j in range(len(z)):
                if j not in top_indices[i]:
                    neg_pairs.append((i, j))

        print(f"pos_pairs: {len(pos_pairs)}, neg_pairs: {len(neg_pairs)}")
        pos_pairs = torch.tensor(pos_pairs)
        neg_pairs = torch.tensor(neg_pairs)
        num_neg_pairs = len(neg_pairs)
        labels = torch.cat([torch.ones(self.pos_num), torch.zeros(num_neg_pairs)])
        pos_sim = similarities[pos_pairs[:, 0].unsqueeze(1), pos_pairs[:, 1:]] # shape [1, 3]
        neg_sim = similarities[neg_pairs[:, 0], neg_pairs[:, 1]] # shape [z.shape[0] - 3]
        print(f"pos_sim: {pos_sim.shape}, neg_sim: {neg_sim.shape}， labels: {labels.shape}")
        neg_sim = neg_sim.expand(pos_sim.shape[0], -1)
        labels = labels.expand(pos_sim.shape[0], -1).to(torch.float32).to(self.device)
        similarities = torch.cat([pos_sim, neg_sim], dim=1)
        # print(f"similarities: {similarities}, labels: {labels}")
        print(f"similarities: {similarities.shape}, labels: {labels.shape}")
        loss = self.loss_fn(similarities, labels)

        return loss
"""



