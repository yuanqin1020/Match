
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


# 将三元组转换为边索引和边属性
def triples_to_edges(triples):
    # 字典存储实体和关系的索引
    entity_index = {}
    relation_index = {}
    node_pair_relation = {}

    # 列表存储边索引和边属性
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

        # 将边索引和边属性添加到列表中
        edge_index.append([entity_index[subject], entity_index[object]])
        edge_attr.append(relation_index[predicate])

        node_pair_relation[(entity_index[subject], entity_index[object])] = predicate

    # 将列表转换为张量
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)

    entities = [v for _, v in entity_index.items()]

    return edge_index, edge_attr, entity_index, node_pair_relation, entities


# 将嵌入矩阵转换为节点特征
def embeddings_to_features(embeddings_df, entity_index):
    # 获取嵌入的维度
    dim = len(embeddings_df[0])

    # 创建一个零张量，用于存储节点特征
    num_nodes = len(entity_index)
    x = torch.zeros(num_nodes, dim)

    # 遍历嵌入的每一行
    for i, row in embeddings_df.iterrows():
        # 获取实体和向量
        entity = row["entity"]
        vector = torch.tensor(row)
        # 如果实体在索引中，就将向量赋值给对应的位置
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


# 接受一个文件夹路径作为参数，返回该文件夹下所有后缀为jpg的文件路径的列表
def get_jpg_files(task_path, img_dir):
    folder_path = task_path + img_dir
    print(f"Read file: {folder_path}")
    # 创建一个空列表，用于存储文件路径
    jpg_files = []
    ent = 0
    # 遍历文件夹下的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        ent += 1
        # if ent > 1000: break

        # 遍历每个文件
        for file in files:
            # 获取文件的绝对路径
            file_path = os.path.join(root, file)
            # 获取文件的后缀名
            _, extension = os.path.splitext(file_path)
            # 如果后缀名为.jpg，将文件路径添加到列表中
            if extension.lower() == ".jpg" or extension.lower() == ".jpeg":
                jpg_files.append(file_path)
            if extension.lower() == ".pkl":
                os.remove(file_path)

    print(f"Number of images: {len(jpg_files)}")

    # 返回文件路径列表
    return jpg_files


def get_kg_data(file, task, mode):
    """
    Define the knowledge graph and scene graph as Data objects
    Each Data object has x (node features), edge_index (edge list), and y (node labels)
    Assume the node features are 16-dimensional and the node labels are binary
    Assume the knowledge graph has 100 nodes and the scene graph has 50 nodes
    Assume the edge list is a tensor of shape [2, num_edges] where each column is a pair of node indices
    """
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
    # 读取entity2text.txt文件
    entity2text = {}
    # /m/0145m        Afrobeat
    ent_file = 'entity2text.txt' if scala == "" else 'entity2text_1' + scala + ".txt"
    print(ent_file)
    with open(path + ent_file, 'r') as f:
        for line in f:
            entity_id, entity_name = line.strip().lower().split('\t')
            entity2text[entity_id] = entity_name
            if len(entity2text) == 1:
                print(entity2text)
    
    return entity2text



# def get_kg_list(file, hops):
#     """
#         load a set of subgraphs of kg
#         each data object is a subgraph induced by a vertex
#     """
#     start = datetime.now()
#     triples = read_str_triples(file)  # A list of triples (subject, predicate, object)
#     edge_index, edge_attr, entity2id, relation2id = triples_to_edges(triples)  # [h_id, t_id], [r_id], {entity: id}, {relation: id}
#
#     entity_embeddings = handler.get_attr_embeds(entity2id)
#     knowledge_graph = Data(x=entity_embeddings, edge_index=edge_index)
#
#     subgraphs = []
#     for node_id in range(len(knowledge_graph.x)):
#         # sub = dgl.khop_out_subgraph(knowledge_graph, node, hops, relabel_nodes=False) # True: remove the isolated nodes and relabel nodes in the extracted subgraph
#         # subgraph = Data(x=knowledge_graph.x[sub[0]], edge_index=sub[1])
#
#         subgraph = get_khop_subgraph(knowledge_graph, node_id, hops)
#         subgraphs.append(subgraph)
#
#     print(f"kg load takes: {datetime.now() - start}.")
#
#     return subgraphs



def generate_pairs(kg, sgs, num_pairs):
    """
    generate positive and negative pairs of graphs
    # positive pair: a subgraph of the kg and a scene graph
    # negative pair: two random subgraphs of the knowledge graph

    positive pair: two views of a subgraph in the kg
    negative pair: two random subgraphs of the knowledge graph
    :param num_pairs: [kg subgraph, sg]
    :return:
    """
    positive_pairs = []
    negative_pairs = []
    # Loop over the number of pairs
    for i in range(num_pairs):
        for sg in sgs:
            for sub in sampler.sample_subgraph(kg, sg, mode="induced", n=10, k=3):
                positive_pairs.append((sub, A.FeatureMasking(pf=0.1).augment(sub)))
                positive_pairs.append((sub, A.EdgeRemoving(pe=0.1).augment(sub)))

                # Sample a random number of nodes for the subgraph
                num_nodes = torch.randint(0, len(kg.x), (1,))
                # Sample another random subset of node indices for the negative subgraph
                subset = torch.randperm(kg.num_nodes)[:num_nodes]
                # Extract the negative subgraph from the knowledge graph
                negative_subgraph = kg.subgraph(subset)
                # Add the subgraph and the negative subgraph as a negative pair
                negative_pairs.append((sub, negative_subgraph))

    return positive_pairs, negative_pairs


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

# Define a data loader that provides batches of positive and negative graph pairs
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

        # return the graph pair and the negative subgraphs as a list of tg.data.Data objects
        return [scene_graph, subgraph] + neg_subgraphs



class KGDataset(InMemoryDataset):
    def __init__(self, root: str,
                 transform = None,
                 pre_transform = None):
        # root 是存储数据的根目录，transform 和 pre_transform 是可选的数据转换函数
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
                # 读取一行数据，分割成列表
                items = line.strip().split()
                # 第一个元素是图的标签
                y = int(items[0])
                # 假设节点数是已知的，为 100
                num_nodes = 100
                # 假设节点特征是随机生成的，维度为 16
                x = torch.randn(num_nodes, 16)
                # 剩余的元素是边，每两个元素表示一条边，格式为：source target
                edge_index = []
                for i in range(1, len(items), 2):
                    # 将边的两个端点添加到 edge_index 中
                    edge_index.append([int(items[i]), int(items[i+1])])
                # 将 edge_index 转换为 torch 张量，形状为 [2, num_edges]
                edge_index = torch.tensor(edge_index).t()
                # 创建一个 Data 对象，保存图的属性
                data = Data(x=x, edge_index=edge_index, y=y)
                # 将 Data 对象添加到 data_list 中
                data_list.append(data)

        # 如果有 pre_transform，对 data_list 中的每个 Data 对象进行转换
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        # 将 data_list 转换为一个大的 Data 对象，并保存在处理后的文件中
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

        # 将 data_list 转换为一个大的 Data 对象，并保存在处理后的文件中
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
