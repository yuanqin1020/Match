
import torch
import torch.nn as nn
from torch_geometric.data import Data, InMemoryDataset, DataLoader
import data_handler as handler
import subgraph_sampling as sampler
import torch_geometric as tg
import os
from datetime import datetime



def read_str_triples(file, task):
    triples = list()
    if task in ['openImages']: sepa = ","
    else: sepa = "\t"

    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            # print(line)
            params = line.strip('\n').split(sepa)
            assert len(params) == 3
            h = str(params[0])
            r = str(params[1])
            t = str(params[2])
            triples.append((h, r, t))
        f.close()
    return triples


def triples_to_edges(triples):

    entity_index = {}
    relation_index = {}
    node_pair_relation = {}

    edge_index = []
    edge_attr = []

    for i, row in enumerate(triples):
        subject = row[0]
        predicate = row[1]
        object = row[2]

        if subject not in entity_index:
            entity_index[subject] = len(entity_index)
        if object not in entity_index:
            entity_index[object] = len(entity_index)

        if predicate not in relation_index:
            relation_index[predicate] = len(relation_index)

        edge_index.append([entity_index[subject], entity_index[object]])
        edge_attr.append(relation_index[predicate])

        node_pair_relation[(entity_index[subject], entity_index[object])] = predicate

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)

    entities = [v for _, v in entity_index.items()]

    return edge_index, edge_attr, entity_index, node_pair_relation, entities

def embeddings_to_features(embeddings_df, entity_index):
    dim = len(embeddings_df[0])

    num_nodes = len(entity_index)
    x = torch.zeros(num_nodes, dim)

    for i, row in embeddings_df.iterrows():

        entity = row["entity"]
        vector = torch.tensor(row)
        if entity in entity_index:
          index = entity_index[entity]
          x[index] = vector

    return x

def read_visual_feat(dir):
    ent2vis = torch.load(dir + '/visual_features_ent_sorted.pt')
    print(f"ent2vis: {ent2vis['parking_meter.n.01'].shape}")

    vis_label = []
    for key, value in ent2vis.items():
        for feat in value:
            vis_label.append({feat: key})

    return vis_label

def get_jpg_files(task_path, img_dir):
    folder_path = task_path + img_dir
    print(f"Read file: {folder_path}")
    jpg_files = []
    ent = 0

    for root, dirs, files in os.walk(folder_path):
        ent += 1

        for file in files:
            file_path = os.path.join(root, file)
     
            _, extension = os.path.splitext(file_path)
            if extension.lower() == ".jpg" or extension.lower() == ".jpeg":
                jpg_files.append(file_path)
            if extension.lower() == ".pkl":
                os.remove(file_path)

    return jpg_files


def get_kg_data(file, task, mode):
    if task in ['openImages']:
        file = file + "triplets"
    else:
        file = file + mode + "_triplets"
    start = datetime.now()
    data = file + ".txt"
    pkl = file + ".pkl"
    dict = {}

    triples = read_str_triples(data, task)  # A list of triples (subject, predicate, object)
    print(f"Number of triplets: {len(triples)}")
    edge_index, edge_attr, entity2id, relation2id, entities = triples_to_edges(triples) # [h_id, t_id], [r_id], {entity: id}, {relation: id}

    # entity embeddings of shape [num_entities, num_features]
    if os.path.exists(pkl):
        dict = torch.load(pkl)
        knowledge_graph = Data(x=dict["x"], edge_index=dict["edge_index"])
        # with open(pkl, 'rb') as f:
        #     data = pickle.load(f)
        #     knowledge_graph = Data.from_dict(data)
        print(f"load pkl kg takes: {datetime.now() - start}.")

    else:
        entity_embeddings = handler.get_attr_embeds(entity2id)
        dict["x"] = entity_embeddings
        dict["edge_index"] = edge_index
        knowledge_graph = Data(x=entity_embeddings, edge_index=edge_index, y=list(entity2id.keys()))
        print(f"kg read takes: {datetime.now() - start}.")

        dump = datetime.now()
        torch.save(dict, pkl)
        print(f"kg write out takes: {datetime.now() - dump}.")

    return entity2id, entities, knowledge_graph, relation2id

def get_ent_mapping(path, scala):
    entity2text = {}
    ent_file = 'entity2text.txt' if scala == "" else 'entity2text_1' + scala + ".txt"
    print(ent_file)
    with open(path + ent_file, 'r') as f:
        for line in f:
            entity_id, entity_name = line.strip().lower().split('\t')
            entity2text[entity_id] = entity_name
            if len(entity2text) == 1:
                print(entity2text)
    
    return entity2text


def generate_pairs(kg, sgs, num_pairs):

    positive_pairs = []
    negative_pairs = []
    for i in range(num_pairs):
        for sg in sgs:
            for sub in sampler.sample_subgraph(kg, sg, mode="induced", n=10, k=3):
                positive_pairs.append((sub, A.FeatureMasking(pf=0.1).augment(sub)))
                positive_pairs.append((sub, A.EdgeRemoving(pe=0.1).augment(sub)))

                num_nodes = torch.randint(0, len(kg.x), (1,))
                subset = torch.randperm(kg.num_nodes)[:num_nodes]
                negative_subgraph = kg.subgraph(subset)
                negative_pairs.append((sub, negative_subgraph))

    return positive_pairs, negative_pairs


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

class GraphPairDataset(torch.utils.data.Dataset):
    def __init__(self, scene_graphs, knowledge_graph, num_negatives, tops, hops):
        # scene_graphs: a list of scene graphs, each represented by a tg.data.Data object
        # knowledge_graph: a knowledge graph, represented by a tg.data.Data object
        # num_negatives: the number of negative subgraphs to sample for each scene graph
        super(GraphPairDataset).__init__()
        self.scene_graphs = scene_graphs
        self.knowledge_graph = knowledge_graph
        self.num_negatives = num_negatives
        self.tops = tops
        self.hops = hops

    def __len__(self):
        return len(self.scene_graphs)

    def __getitem__(self, index):
        # get the positive graph pair
        scene_graph = self.scene_graphs[index]
        subgraph = sampler.sample_subgraph(self.knowledge_graph, scene_graph, mode="induced", n=self.tops, k=self.hops)
        neg_subgraphs = []
        positive_pairs = []
        for sub in subgraph:
            # sample positive and negative subgraphs
            for _ in range(self.num_negatives):
                # randomly removing some nodes and edges from the knowledge graph
                print(f"sub.num_nodes: {sub.num_nodes}, sub.num_edges: {sub.num_edges}")
                num_nodes = torch.randint(1, sub.num_nodes, (1,)).item()
                num_edges = torch.randint(1, sub.num_edges, (1,)).item()
                node_mask = torch.randperm(sub.num_nodes)[:num_nodes]
                edge_mask = torch.randperm(sub.num_edges)[:num_edges]
                print(f"num_nodes: {num_nodes}, num_edges: {num_edges}, node_mask: {node_mask}, edge_mask: {edge_mask}")
                aug_subgraph = tg.data.Data(
                    x=sub.x[node_mask],
                    edge_index=sub.edge_index[:, edge_mask],
                    edge_attr=sub.edge_attr[edge_mask]
                )

                positive_pairs.append((sub, aug_subgraph))
                neg_subgraphs.append(self.knowledge_graph[torch.randint(1, self.knowledge_graph.num_nodes, (1,)).item()])

        return [scene_graph, subgraph] + neg_subgraphs



class KGDataset(InMemoryDataset):
    def __init__(self, root: str,
                 transform = None,
                 pre_transform = None):
        super(KGDataset, self).__init__(root, transform, pre_transform)

        self.file = root + "kg_triples_text.txt"
        self.raw_data = read_str_triples(self.file)  # A list of triples (subject, predicate, object)
        # [h_id, t_id], [r_id], {entity: id}, {relation: id}
        edge_index, edge_attr, self.entity2id, self.relation2id = triples_to_edges(self.raw_data)

        # entity embeddings of shape [num_entities, num_features]
        self.entity_embeddings = handler.get_attr_embeds(self.entity2id)
        # relation embeddings of shape [num_relations, num_features]
        self.relation_embeddings = handler.get_attr_embeds(self.relation2id)

        # self.processed_paths = root + "sub_graph.txt"


    @property
    def raw_file_names(self):
        return self.file

    @property
    def processed_file_names(self):
        return ["processed_paths"]

    def download(self):
        pass

    def process(self):
        """
        生成 Data 对象列表，并保存在 self.data_list 属性中
        文件包含所有采样的子图数据，每个图占一行，格式为：label node1 node2 ... edge1 edge2 ...
        """
        data_list = []
        with open('graphs.txt') as f:
            for line in f:

                items = line.strip().split()
                y = int(items[0])
                num_nodes = 100
                x = torch.randn(num_nodes, 16)
                edge_index = []
                for i in range(1, len(items), 2):
                    edge_index.append([int(items[i]), int(items[i+1])])
                edge_index = torch.tensor(edge_index).t()
                data = Data(x=x, edge_index=edge_index, y=y)
                data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'



class ImageDataset(InMemoryDataset):
    def __init__(self, root: str, model_type, ckp_path, device,
                 transform = None,
                 pre_transform = None):
        super(ImageDataset, self).__init__(root, transform, pre_transform)
        self.device = device
        self.model_type = model_type
        self.ckp_path = ckp_path


    @property
    def raw_file_names(self):
        return self.file

    @property
    def processed_file_names(self):
        return ["processed_paths"]

    def download(self):
        pass

    def process(self):
        """
        生成 Data 对象列表，并保存在 self.data_list 属性中
        文件包含所有转换的图数据，每个图占一行，格式为：label node1 node2 ... edge1 edge2 ...
        """
        data_list = []
        graphs_embs_dict, graphs_matrix_dict = handler.images_graphs(self.device, self.model_type, self.root, self.ckp_path)
        for img, graph in graphs_embs_dict.iterrows:
            x = torch.tensor(graph.values)
            edge_index = list()
            matrix = graphs_matrix_dict[img]
            for i in range(len(matrix)):
                for j in matrix[i]:
                    if matrix[i][j] == 1:
                        edge_index.append([i, j])

            edge_index = torch.tensor(edge_index).t()
            data = Data(x=x, edge_index=edge_index)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
