import os
import sys
import logging
import torch
import argparse
import numpy as np
from datetime import datetime
import data_loader as loader
from torch_geometric.data import DataLoader, Data
import feature_extractor as extractor
import model_paral as model
from evaluate import obtain_ground_truth
import clip

import torch.multiprocessing as mp

from data_loader import *
from evaluate import *


os.environ["NCCL_SOCKET_NTHREADS"] = "3600"

logging.basicConfig(filename='out.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _setup_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='bert', type=str, help="The name of bert.")
    parser.add_argument('--bert_name', default='bert-base-uncased', type=str, help="Pretrained language model name, bert-base or bert-large")
    parser.add_argument('--sam_type', type=str, default="vit_b")
    parser.add_argument('--root', type=str, default=".../Match")
    parser.add_argument('--sam_ckp', default="/openai/sam_vit_b_01ec64.pth")
    parser.add_argument('--device', default='cuda:1', type=str, help="cuda or cpu")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="threshold of intersection-over-union of bounding bosex")

    # Basic arguments
    parser.add_argument('--data', type=str, default="/dataset/openImages", help="openImages, FB15k, WN18")
    parser.add_argument('--kg_dir', type=str, default="/kg/")
    parser.add_argument('--img_dir', type=str, default="/images/")
    parser.add_argument('--annotate', type=str, default="/image-ents.txt")

    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument('--num_epochs', default=1, type=int, help="Training epochs")
    parser.add_argument('--batch_size', default=32, type=int, help="batch size")
    parser.add_argument("--tops", type=int, default=30)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--step", type=int, default=6)

    parser.add_argument("--learning_rate", type=float, default=0.00005)
    parser.add_argument('--g_type', type=str, default="C-GAT", help="GIN, C-GAT, GCN, SAGE")
    parser.add_argument('--sampling_bound', type=float, default=15)
    parser.add_argument("--sampling_num", type=int, default=30)
    parser.add_argument('--sampling_type', type=str, default="random", help="around, random, importance") # random(+sage), around(-e)
    parser.add_argument("--threshold", type=float, default=0.78)
    parser.add_argument("--exp_num", type=int, default=3000)
    parser.add_argument("--test_num", type=int, default=150)

    # ========== optimization parameters
    parser.add_argument("--t5", type=bool, default=True, help="capation guided by flant5")
    parser.add_argument("--vis_merge", type=bool, default=False, help="cvisual instance merge")
    parser.add_argument("--crop", type=bool, default=True, help="")
    parser.add_argument("--imp_sample", type=bool, default=True, help="")
    parser.add_argument("--explore", type=bool, default=False, help="exploration")
    parser.add_argument("--refine", type=bool, default=True, help="refinement")
    parser.add_argument("--rf_clip", type=bool, default=True, help="replace refinement with clip")

    # ========== Disturbited parameters
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N', help='number of data loading workers (default: 2)')
    parser.add_argument('-g', '--gpus', default=2, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N', help='number of total epochs to run')

    args = parser.parse_args()
    # args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'

    return parser.parse_args()


def _main():
    args = _setup_parser()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    task_path = args.root + args.data
    dataset = args.data.strip().split("/")[-1]

    path = task_path + args.kg_dir
    print(f"data path: {path}")

    start = datetime.now()

    if dataset in ['FB15k', 'WN18']:
        imgs = loader.get_jpg_files(task_path, args.img_dir)
        train_imgs = imgs[: args.exp_num]
        test_imgs = imgs[args.exp_num: args.exp_num + args.test_num]
        print(f"Number of images: {len(imgs)}, training examples: {len(train_imgs)}, test examples: {len(test_imgs)}")

        if args.t5 == True:
            captions = extractor.get_img_caption(args.root + "/cache/guidance/", "", dataset)
        else:
            captions = extractor.read_caption_dict(task_path + args.img_dir + "annotations/train_caption.csv")

        entity2id, entities, knowledge_graph, node_pair_rel = loader.get_kg_data(path, dataset, args.mode)
        id2attr = {v: k for k, v in entity2id.items()}
        print(f"Number of vertices: {knowledge_graph.x.shape}, number of edges: {len(node_pair_rel)}")

        entity2text = get_ent_mapping(path, "")
        ent_dict = None

        if args.mode == "train":
            true_img_class = true_targets(train_imgs, args.data, entity2text)
            
            mp.spawn(model.train, nprocs=args.gpus, 
                     args=(knowledge_graph, id2attr, node_pair_rel, train_imgs, captions, args, true_img_class, ent_dict))

        print(f"Number of test examples: {len(test_imgs)}")
        test_ground = true_targets(test_imgs, args.data, entity2text)
        model.test(knowledge_graph, id2attr, node_pair_rel, test_imgs, args, test_ground, ent_dict, lin_dim=512, lout_dim=512)
    
    else: # openImages
        
        train_imgs = loader.get_jpg_files(task_path, args.img_dir + args.mode)[: args.exp_num]
        print(f"image: {train_imgs[0]}")
        print(f"Number of training examples: {len(train_imgs)}")

        entity2id, entities, knowledge_graph, node_pair_rel = loader.get_kg_data(path, dataset, args.mode)
        id2attr = {v: k for k, v in entity2id.items()}
        print(f"Number of vertices: {knowledge_graph.x.shape}, number of edges: {len(node_pair_rel)}")

        if args.mode == "train":
            train_ground, ent_dict = obtain_ground_truth(args.root + args.data, args.mode)

            if args.t5 == True:
                captions = extractor.get_img_caption(args.root + "/cache/guidance/", "train", dataset)
            else:
                captions = extractor.read_caption_dict(task_path + args.img_dir + "annotations/train_caption.csv")
            
            mp.spawn(model.train, nprocs=args.gpus, args=(knowledge_graph, id2attr, node_pair_rel, train_imgs, captions, args, train_ground, ent_dict))


        test_imgs = loader.get_jpg_files(task_path, args.img_dir + "test")[ : args.test_num]
        print(f"Number of test examples: {len(test_imgs)}")
        test_ground, ent_dict = obtain_ground_truth(args.root + args.data, "test")
        model.test(knowledge_graph, id2attr, node_pair_rel, test_imgs, args, test_ground, ent_dict, lin_dim=512, lout_dim=512)

    print(f'complete task: {args.data}')



if __name__ == "__main__":

    _main()

