import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
import dgl
import torch_geometric as tg
import copy
from copy import deepcopy



class ProjectedDotProductSimilarity(nn.Module):
    def __init__(self, tensor_1_dim, tensor_2_dim, projected_dim, reuse_weight=False, bias=False, activation=None):
        super(ProjectedDotProductSimilarity, self).__init__()
        self.reuse_weight = reuse_weight
        self.projecting_weight_1 = nn.Parameter(torch.Tensor(tensor_1_dim, projected_dim))
        if self.reuse_weight:
            if tensor_1_dim != tensor_2_dim:
                raise ValueError('if reuse_weight=True, tensor_1_dim must equal tensor_2_dim')
        else:
            self.projecting_weight_2 = nn.Parameter(torch.Tensor(tensor_2_dim, projected_dim))
        self.bias = nn.Parameter(torch.Tensor(1)) if bias else None
        self.activation = activation

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.projecting_weight_1)
        if not self.reuse_weight:
            nn.init.xavier_uniform_(self.projecting_weight_2)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, tensor_1, tensor_2):
        print(f"tensor_1: {tensor_1.shape}")
        print(f"tensor_2: {tensor_2.shape}")
        projected_tensor_1 = torch.matmul(tensor_1.view(-1, tensor_1.shape[-3], tensor_1.shape[-2], tensor_1.shape[-1]), self.projecting_weight_1)
        if self.reuse_weight:
            projected_tensor_2 = torch.matmul(tensor_2.view(-1, tensor_2.shape[-1]), self.projecting_weight_1)
        else:
            projected_tensor_2 = torch.matmul(tensor_2, self.projecting_weight_2)
        result = (projected_tensor_1.view(-1, projected_tensor_1.shape[-1]) * projected_tensor_2).sum(dim=-1)
        if self.bias is not None:
            result = result + self.bias
        if self.activation is not None:
            result = self.activation(result)
        return result





def find_top_k_subgraphs(scene, kg, k, n=5):
    print(f"kg dim: {kg.x.shape}")
    print(f"scene dim: {scene.x.shape}")
    flattened_embedding = scene.x.view(-1, scene.x.shape[2]*scene.x.shape[3]*scene.x.shape[4])

    similarity = torch.nn.functional.cosine_similarity(flattened_embedding, kg.x, dim=1)
    print(f"similarity: {similarity}")

    top_k_indices = torch.topk(similarity, k).indices
    print(f"indices: {top_k_indices}")

    subgraphs = set()
    for index in top_k_indices:
        subgraph = []
        for i in range(n):
            subgraph.append(scene.x[i, 0, :, index // (64*64), (index % (64*64)) // 64])
            print(f"subgraph: {subgraph}")
        subgraphs.add(Data(x=torch.stack(subgraph, dim=0)))

    return subgraphs



def sample_induced_graph(scene, kg, n = 10, k=3):

    linear = nn.Linear(scene.x.shape[2] * scene.x.shape[3] * scene.x.shape[4], kg.x.shape[1])  # [1048576, 768]
    x1 = scene.x.view(scene.x.shape[0], -1)  # [129, 1048576]
    x1 = linear(x1)  # [129, 768]
    x1 = x1.type(torch.FloatTensor)
    x2 = kg.x  # [14256, 768]
    print(f"x1 dim: {x1.shape}")
    print(f"x2 dim: {x2.shape}")
    sim_score = x1 @ x2.T  # [129, 768] * [768, 14256] = [129, 14256]
    values, indices = torch.topk(sim_score, n, largest=True)
    print(f"values: {values}, values shape: {values.shape}")
    print(f"indices: {indices}, index shape: {indices.shape}")

    subgraphs = []
    for index in indices:
        for num in index: # range(n)
            sub = dgl.khop_out_subgraph(kg, num, k) 
            subgraph = tg.utils.from_networkx(sub.to_networkx())
            subgraphs.append(subgraph)

    return subgraphs


def sample_subgraph(kg, scene, mode, n = 10, k=3):

    subgraphs = []
    if mode == "induced":
        subgraphs.extend(sample_induced_graph(scene, kg, n, k))
    print(f"subgraph size: {len(subgraphs)}")
    return subgraphs

import numpy as np


def drop_nodes(data):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num / 10)

    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    idx_dict = {idx_nondrop[n]:n for n in list(range(node_num - drop_num))}

    edge_index = data.edge_index.cpu().numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index

    return data


def permute_edges(data):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num / 10)

    edge_index = data.edge_index.transpose(0, 1).numpy()

    idx_add = np.random.choice(node_num, (permute_num, 2))
    edge_index = edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)]
    data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data


def subgraph(data):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * 0.2)

    edge_index = data.edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub
    idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}

    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index

    return data

def mask_nodes(data):
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num / 10)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim)), dtype=torch.float32)

    return data


def sub_augmentation(subgraph, operation):
    if operation == "node_drop":
        data_aug = drop_nodes(deepcopy(subgraph))
    if operation == "edge_perturb":
        data_aug = permute_edges(deepcopy(subgraph))
    if operation == "attr_mask":
        data_aug = mask_nodes(deepcopy(subgraph))
    if operation == "subgraph":
        data_aug = subgraph(deepcopy(subgraph))

    return data_aug


def k_hop_subgraph(kg, node_id, hops, device):
    nodes = [node_id]
    for i in range(hops):
        neighbors = tg.utils.to_networkx(kg).subgraph(nodes).edges()
        neighbors = [n[1] for n in neighbors]
        nodes.extend([n for n in neighbors if n not in nodes])

    nodes = torch.tensor(nodes).to(device)
    subgraph = kg.subgraph(nodes)

    return subgraph.to(device)

def get_khop_subgraph(graph, node, k):
  
    x = graph.x[node.item()].unsqueeze(0)  
    edge_index = [[], []]

    depth = {node: 0} 
    queue = [(node, 0)]

    node_mapping = dict()  
    node_mapping[node] = 0

    while queue:
        current_node, current_depth = queue.pop(0)

        if current_depth == k:
            break

        neighbors = graph.edge_index[1][graph.edge_index[0] == current_node]
        for neighbor in neighbors:
            neighbor = neighbor.item()
            if neighbor not in depth:
                depth[neighbor] = current_depth + 1
                queue.append((neighbor, current_depth + 1))

          
                new_node_id = len(node_mapping)
                node_mapping[neighbor] = new_node_id
                x = torch.cat([x, graph.x[neighbor].unsqueeze(0)], dim=0)
                edge_index[0].append(current_node)
                edge_index[1].append(neighbor)

    edge_index[0] = [node_mapping[node_id] for node_id in edge_index[0]]
    edge_index[1] = [node_mapping[node_id] for node_id in edge_index[1]]

    subgraph = Data(x=torch.tensor(x), edge_index=torch.tensor(edge_index))

    return subgraph


def around_sampling_subgraph(kg, start, sample_num):
    start_node = start
    queue = [start_node]
    visited = set()
    visited.add(start_node)
    subgraph = {start_node: kg.edge_index[:, kg.edge_index[0] == start_node]}

    while len(queue) > 0 and len(subgraph) < sample_num:
        node = queue.pop(0)

        neighbors = kg.edge_index[1, kg.edge_index[0] == node].tolist()
        neighbor_scores = [1 / len(neighbors) for _ in neighbors]
        sorted_neighbors = [x for _, x in sorted(zip(neighbor_scores, neighbors), reverse=True)]
        for neighbor in sorted_neighbors:
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
                subgraph[neighbor] = kg.edge_index[:, kg.edge_index[0] == neighbor]

    edge_index = torch.cat([subgraph[node] for node in subgraph], dim=1)
    filtered_indices = torch.tensor([edge[1] in subgraph.keys() for edge in edge_index.T], dtype=torch.bool)
    edge_index = edge_index[:, filtered_indices]

    x = torch.tensor([[node] for node in subgraph.keys()])
    data = tg.data.Data(x=x, edge_index=edge_index)

    x_sub = [kg.x[x[i][0].item()].tolist() for i in range(x.shape[0])]
    x_sub = torch.tensor(x_sub)

    dict = {}
    for i in range(x.shape[0]):
        dict[x[i][0].item()] = i

    edge_index2 = []
    for i in range(len(edge_index)):
        row = []
        for j in range(len(edge_index[i])):
            item = edge_index[i][j].item()
            # print(f"{item}, {dict[item]}")
            row.append(dict[item])
        edge_index2.append(row)
    edge_index2 = torch.tensor(edge_index2, dtype=torch.long)

    data2 = tg.data.Data(x=x_sub, edge_index=edge_index2)

    node_attr_dict = {v: k for k, v in dict.items()}

    return data2, node_attr_dict

def compute_similarity(node1, node2):
    return torch.dot(node1, node2) / (torch.norm(node1) * torch.norm(node2))


def monotone_submodular_function(subgraph):
    semantic_info = torch.sum(
        torch.stack([1 / (compute_similarity(node1, node2) + 1e-6) for node1 in subgraph for node2 in subgraph]))
    return semantic_info


def importance_sampling_subgraph(graph, x_y_sim, start_node, num_steps, sample_size, type):
    visited_nodes = set()
    queue = [start_node]
    edge_index = []
    x = [start_node]
    visited_nodes.add(start_node)
    step = 0

    while len(queue) > 0 and len(x) < sample_size and step < num_steps:
        current_node = queue.pop(0)
        neighbors = graph.edge_index[1][graph.edge_index[0] == current_node]
        neigh_feat = graph.x[neighbors]

        if len(neighbors) == 0:
            continue

        probabilities = compute_probabilities(neighbors, current_node, x_y_sim, x, edge_index, type, neigh_feat)  
        selected_node = max(probabilities, key=probabilities.get) 

        if selected_node not in visited_nodes:
            selected_node = selected_node.item()
            x.append(selected_node) 
            visited_nodes.add(selected_node)
            queue.append(selected_node)
            edge_index.append((current_node, selected_node))  

        step += 1  

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    subgraph = tg.data.Data(x=torch.tensor(graph.x[x]), edge_index=edge_index)  

    del visited_nodes, queue

    node_mapping = {}
    new_node_index = 0
    for i in x:
        node_mapping[i] = new_node_index
        new_node_index += 1

    new_x = torch.tensor([node_mapping[i] for i in x])
    new_edge_index = torch.tensor(
        [[node_mapping[edge[0].item()], node_mapping[edge[1].item()]] for edge in edge_index.t()]).t().contiguous()

    x = torch.tensor(graph.x[x])
    data2 = tg.data.Data(x=x, edge_index=new_edge_index)

    node_attr_dict = {v: k for k, v in node_mapping.items()}

    return data2, node_attr_dict


def compute_graph_entropy(x, edge, x_y_sim):

    entropy = 1
    if x is None:
        return entropy
    for v in x:
        prob_v = torch.exp(x_y_sim[v])
        prob_neigh = 1
        for v1, v2 in edge:
            if v1 == v:
                prob_neigh += torch.exp(x_y_sim[v2])
        prob = torch.tensor(prob_v / prob_neigh)
        entropy -= prob * torch.log2(prob)

    return entropy


def compute_probabilities(neighbors, node, x_y_sim, x, edge_index, type, neigh_feat):
    probability = {}
    if type == "importance":
        base = compute_graph_entropy(x, edge_index, x_y_sim)
        for neigh in neighbors:
            # print(f"for neighbor: {neigh}")
            cand = x[:]
            cand_x = cand.append(neigh)
            c_e = edge_index[:]
            cand_edge = c_e.append((node, neigh))
            entropy = compute_graph_entropy(cand_x, cand_edge, x_y_sim)
            probability[neigh] = entropy - base

    elif type == "random":
        for n in neighbors:
            probability[n] = 1 / len(neighbors)


    return probability


def get_sampling_subgraph(kg, id2attr, node_pair_rel, x_y_sim, indices, type, sample_num, step):
    # print(f"Sampling {len(indices)} number of subgraphs")

    subgraphs = {}
    node_mapping = {}
    for index in indices:
        start_node = index.item()

        subgraph = tg.data.Data()
        if type == "around":
            subgraph, node_mapping = around_sampling_subgraph(kg, start_node, sample_num)
        elif type == "random" or type == "importance":
            subgraph, node_mapping = importance_sampling_subgraph(kg, x_y_sim, start_node, sample_num, step, type)

        subgraph1 = copy.copy(subgraph)

        ents = get_sub_vertices(subgraph1, node_mapping, start_node, id2attr)

        first_order_seq, second_order_seq = generate_sequences(subgraph1, node_mapping, start_node, id2attr, node_pair_rel)
        subgraph.y = {"1": first_order_seq, "2": second_order_seq, "ents": ents}
        subgraphs[index] = subgraph

    return subgraphs


def get_sub_vertices(subgraph, node_mapping, start, id2attr):
    ents = []
    new_start = [key for key, value in node_mapping.items() if value == start][0]
    start_attr = id2attr[start]
    ents.append(start_attr)

    if subgraph.edge_index.size == 0 or subgraph.edge_index.shape[0] == 0:
        return ents

    neighbor_indices = subgraph.edge_index[1][subgraph.edge_index[0] == new_start]
    for neighbor in neighbor_indices:
        neighbor = neighbor.item()
        neighbor_attr = id2attr[node_mapping[neighbor]]
        ents.append(neighbor_attr)

        second_neighs = subgraph.edge_index[1][subgraph.edge_index[0] == new_start]
        for second_neigh in second_neighs:
            if second_neigh != neighbor and second_neigh != start:
                second_neigh = second_neigh.item()
                second_neigh_attr = id2attr[node_mapping[second_neigh]]
                ents.append(second_neigh_attr)

    return ents


def generate_sequences(subgraph, node_mapping, start, id2attr, node_pair_rel):
    new_start = [key for key, value in node_mapping.items() if value == start][0]
    start_attr = id2attr[start]

    if subgraph.edge_index.size == 0 or subgraph.edge_index.shape[0] == 0:
        return start_attr, start_attr

    neighbor_indices = subgraph.edge_index[1][subgraph.edge_index[0] == new_start]

    first_order_seq = start_attr + " "
    for neighbor in neighbor_indices:
        neighbor = neighbor.item()
        neighbor_attr = id2attr[node_mapping[neighbor]]

        edge_label = node_pair_rel[(start, node_mapping[neighbor])]

        first_order_seq += edge_label + " " + neighbor_attr

    return first_order_seq, second_order_seq

