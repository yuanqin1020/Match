
# CrossETR: A Semantic-Driven Framework for Entity Matching Across Images and Graph

CrossETR is a novel semantic-driven entity matching framework (namely ) that follows an exploration-then-refinement paradigm. A candidate exploration policy is proposed to boost the training efficiency, which explores candidate pairs according to entity correlations and captures structural semantics by adaptive sampling the most informative neighborhood subgraphs. Subsequently, the cross-modal entity representations are refined to break modality heterogeneity to support unsupervised matching prediction. 

For more technical details, see [CrossETR: A Semantic-Driven Framework for Entity Matching Across Images and Graph](https://ieeexplore.ieee.org/document/11113064).

![Overall architecture of CrossETR framework. ](./CrossETR.jpg)

## Dependencies and Installion
scikit-learn==1.3.2
torch==2.1.0+cu118
torch-cluster==1.6.3+pt21cu118
torch-geometric==2.6.1
torch-scatter==2.1.2+pt21cu118
torch-sparse==0.6.18+pt21cu118
torchaudio==2.1.0+cu118
torchvision==0.16.0+cu118
tqdm==4.67.1
transformers==4.24.0

We recommend creating a new conda environment to install the dependencies:
conda env remove --name promptdi
conda create -y -n promptdi python=3.9
conda activate promptdi
pip install transformers==4.24.0

## Datasets

We use real-world benchmark datasets from [WN18-IMG](https://github.com/wangmengsd/RSME?tab=readme-ov-file), [FB15K-237-IMG](https://github.com/mniepert/mmkb) and [OpenImages](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xu_Scene_Graph_Generation_CVPR_2017_paper.pdf).
- WN18-IMG is an extended dataset of WN18 with 10 images for each entity, where WN18 is a knowledge graph originally extracted from WordNet. 
- FB15K-237-IMG consists of a subset of the large-scale knowledge graph Freebase and a set of images where each entity is associated with 10 images.
- OpenImages is a large-scale dataset providing a large number of examples for object bounding boxes and object segmentation.

## Quick Start

To train and evaluate with CrossETR.

```
python main_paral.py [<args>] [-h | --help]
```

e.g.

```
python ./main_paral.py --data /dataset/FB15k --sampling_type random
```
- Task parameters: bert_name, data_name, k (the proportion of training data used)
- Pre-training parameters: gm (GIN, GAT, SAGE), gm_mode, gm_lr, gm_in(768), gm_out(128), gm_agg(mean, sum, max"), gm_batch, gm_epochs, gm_hidden(256), gm_layer(2), hops, gm_temperature, dropout(0.1)
- Promoting parameters: task(sm, map, em"), prompting(agg, weighted, linear, weighted-sum, weighted-sum-p, att), p_num(10), update_pretrain, lr(0.00001), w, sc, batch_size, num_epochs

- Basic arguments:
    model_name, bert_name, sam_type, root, sam_ckp, device, iou_threshold, data, kg_dir, img_dir, annotate
- optimization parameters:
    t5, vis_merge, crop, imp_sample, explore, refine, rf_clip
- Disturbited parameters:
    n, g, nr, epochs


## Download the models
We use bert-base-uncased as the pretrained language model in all experiments.

You can download the pre-trained checkpoint from [Google-BERT](https://huggingface.co/google-bert/bert-base-uncased) manually.

The pre-trained [Segment Anything](https://github.com/facebookresearch/segment-anything) checkpoint can be download from [vit_b](https://huggingface.co/facebook/sam-vit-base)

More detailed results can be consulted by following [Quick Start](#quick-start).

## Citation
If you find our work useful, please kindly cite the following paper:

```
@inproceedings{DBLP:conf/icde/YuanWQYW25,
  author       = {Qin Yuan and
                  Zhenyu Wen and
                  Jiaxu Qian and
                  Ye Yuan and
                  Guoren Wang},
  title        = {CrossETR: A Semantic-Driven Framework for Entity Matching Across Images and Graph},
  booktitle    = {41st {IEEE} International Conference on Data Engineering, {ICDE} 2025,
                  Hong Kong, May 19-23, 2025},
  pages        = {641--654},
  publisher    = {{IEEE}},
  year         = {2025},
  doi          = {10.1109/ICDE65448.2025.00054}
}
```