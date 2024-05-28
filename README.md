# Bridge-Tune: Measuring Task Similarity and Its Implication in Fine-Tuning Graph Neural Networks

## About

This repo is the official code for AAAI-24 "Measuring Task Similarity and Its Implication in Fine-Tuning Graph Neural Networks".
<span id='introduction'/>

## Brief Introduction 
Not all downstream tasks can effectively benefit from a graph pre-trained model. In light of
this, we propose a novel fine-tuning strategy called Bridge-Tune.

-  Instead of directly fine-tuning a pre-trained model, Bridge-Tune takes an intermediate step that bridges
the pre-training and downstream tasks and refines the model representations.
- The traditional fine-tuning easily falls  into a suboptimal point in the downstream task. In comparison,  the pre-trained model refinement step helps find a better  starting point for fine-tuning and so Bridge-Tune potentially
builds a better model for the downstream task.

For more technical details, kindly refer to the following links:

<a href='https://ojs.aaai.org/index.php/AAAI/article/view/29156'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> 
<a href='https://underline.io/lecture/93719-measuring-task-similarity-and-its-implication-in-fine-tuning-graph-neural-networks-video'><img src='https://img.shields.io/static/v1?label=Video/Poster&message=underline&color=blue'></a> 


## Getting Started

<span id='all_catelogue'/>

### Table of Contents:
* <a href='#File structure'>1. File structure</a>
* <a href='#Environment dependencies'>2. Environment dependencies </a>
* <a href='#Usage'>3. Usage: How to run the code </a>
  * <a href='#Training Bridge-Tune'>3.1. Fine-tuning via Bridge-Tune </a>
  * <a href='#Evaluating model'>3.2. Evaluating the fine-tuned models</a>


<span id='File structure'/>

##  1. File Structure <a href='#all_catelogue'>[Back to Top]</a>

```
.
├── README.md
├── data
│   ├── DD242
│   │   ├── DD242.edges
│   │   ├── DD242.node_labels
│   │   └── readme.html
│   ├── DD68
│   │   ├── DD68.edges
│   │   ├── DD68.node_labels
│   │   └── readme.html
│   ├── DD687
│   │   ├── DD687.edges
│   │   ├── DD687.node_labels
│   │   └── readme.html
│   ├── hindex
│   │   ├── aminer_hindex_rand1_5000.edgelist
│   │   ├── aminer_hindex_rand1_5000.nodelabel
│   │   ├── aminer_hindex_rand20intop200_5000.edgelist
│   │   ├── aminer_hindex_rand20intop200_5000.nodelabel
│   │   ├── aminer_hindex_top1_5000.edgelist
│   │   └── aminer_hindex_top1_5000.nodelabel
│   ├── panther
│   └── struc2vec
│       ├── barbell.edgelist
│       ├── brazil-airports.edgelist
│       ├── brazil-airports.nodelabel
│       ├── europe-airports.edgelist
│       ├── europe-airports.nodelabel
│       ├── facebook348.edgelist
│       ├── karate-mirrored.edgelist
│       ├── usa-airports.edgelist
│       └── usa-airports.nodelabel
├── dataset.json
├── gcc
│   ├── __init__.py
│   ├── contrastive
│   │   ├── __init__.py
│   │   ├── criterions.py
│   │   └── memory_moco.py
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── data_util.py
│   │   └── graph_dataset.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── emb
│   │   │   ├── __init__.py
│   │   │   └── from_numpy.py
│   │   ├── gat.py
│   │   ├── gcn.py
│   │   ├── gin.py
│   │   ├── graph_encoder.py
│   │   └── mpnn.py
│   ├── tasks
│   │   ├── __init__.py
│   │   └── node_classification.py
│   └── utils
│       └── misc.py
├── generate.py
├── requirements.txt
├── saved
│   └── Pretrain_moco_True_dgl_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_hid_64_samples_2000_nce_t_0.07_nce_k_16384_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_16_pos_32_momentum_0.999
│       └── current.pth
├── scripts
│   ├── download.py
│   ├── evaluate.sh
│   └── node_classification
│       ├── baseline.sh
│       └── ours.sh
├── splits
│   ├── d1d5bdd41805e6a6eb0fdf335ebbfb7e.zip
│   └── f54ddc32338ab0eac9511aaa355b666a.zip
├── train_bridge.py
└── utils
    ├── __init__.py
    ├── dataset.py
    ├── pgnn.py
    ├── signac_tools.py
    └── sparsegraph
        ├── __init__.py
        ├── io.py
        └── preprocess.py
```

*****

Below, we will specifically explain the meaning of important file folders to help the user better understand the file structure.

`data`: contains the data of "DD242, DD68, DD687, usa_airport, brazil_airport, europe_airport".

`splits`: **need to unzip**, contains the split data of "cornell, wisconsin".

`scripts`: contains all the scripts for running code.

`gcc&utils`: contains the code of models.

`saved`: contains the pre-trained graph obtained from [GCC](https://github.com/THUDM/GCC). The specific path for pre-trained model is located in `saved/Pretrain_moco_True_dgl_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_hid_64_samples_2000_nce_t_0.07_nce_k_16384_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_16_pos_32_momentum_0.999/`


<span id='Environment dependencies'/>


## 2. Environment dependencies <a href='#all_catelogue'>[Back to Top]</a>

The script has been tested running under Python 3.7.10, with the following packages installed (along with their dependencies):

- [PyTorch](https://pytorch.org/). Version >=1.4 required. You can find instructions to install from source [here](https://pytorch.org/get-started/previous-versions/).
- [DGL](https://www.dgl.ai/). 0.5 > Version >=0.4.3 required. You can find instructions to install from source [here](https://www.dgl.ai/pages/start.html).
- [rdkit](https://anaconda.org/conda-forge/rdkit). Version = 2019.09.2 required. It can be easily installed with 
			```conda install -c conda-forge rdkit=2019.09.2```
- [Other Python modules](https://pypi.python.org). Some other Python module dependencies are listed in requirements.txt, which can be easily installed with pip:

	`pip install -r requirements.txt`

In addition, CUDA 10.0 has been used in our project. Although not all dependencies are mentioned in the installation instruction links above, you can find most of the libraries in the package repository of a regular Linux distribution.



<span id='Usage'/>

## 3. Usage: How to run the code  <a href='#all_catelogue'>[Back to Top]</a>
Bridge-Tune paradigm consists of two stages: (1) Fine-tuning via Bridge-Tune (2) Evaluating the fine-tuned model.
<span id='Training Bridge-Tune'/>

### 3.1. Fine-tuning via Bridge-Tune 

To conduct Bridge-Tune, you can execute `train_bridge.py` as follows:

```bash
python train_bridge.py \
  --resume <pre-trained model file> \
  --dataset <downstream dataset>
  --reg-coeff <coefficient for Bridge-Tune Loss> \
  --model-path <fine-tuned model saved file> \
  --gpu <gpu id> \
  --epochs <epoch number> \
  --bridge
```

For more detail, the help information of the main script `train_bridge.py` can be obtained by executing the following command.

```bash
python train_bridge.py -h

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of training epochs (default:50)
  --optimizer {sgd,adam,adagrad}  optimizer (default:adam)
  --learning_rate LEARNING_RATE  learning rate (default:0.005)
  --resume PATH         path for pre-trained model (default: GCC)
  --dataset {usa_airport,brazil_airport,europe_airport,h-index, DD242, DD68, DD687, cornell, wisconsin}
  --reg-coeff REG_COEFF  coefficient for Bridge-Tune (default:10)
  --hidden-size HIDDEN_SIZE  (default:64)
  --model-path MODEL_PATH    path to save fine-tuned model (default:saved)
  --gpu GPU              GPU id to use.
  --bridge              whether to conduct bridge-tune
```

**Demo:**	

```bash
python train_bridge.py \
  --dataset usa_airport \
  --reg-coeff 10 \
  --model-path saved \
  --gpu 0 \
  --epochs 50 \
  --bridge
```

Note: No need to specify the pre-trained model path here; the default pre-trained model is provided. If a user requires a specific model, they should add the `--resume`.


<span id='Evaluating model'/>

### 3.2. Evaluating the fine-tuned model

`generate.py` file helps generate embeddings on a specific dataset. The help information of the main script `generate.py` can be obtained by executing the following command.

```bash
python generate.py -h

optional arguments:
  --load-path LOAD_PATH
  --dataset Dataset
  --gpu GPU  GPU id to use.
```
The embedding will be used for evaluation in node classification. The script `evaluate.sh` is available to simplify the evaluation process as follows: 

```
bash evaluate.sh <model_path> <name> <dataset> <gpu id>
```
Here, `<saved_path>` refers to the main directory for finetuning, and `<name>` is the name of specific model directory.

**Demo:**
Here is the demo instruction, after the user has trained using the demo provided above.
```
bash scripts/evaluate.sh saved path_bridge10_usa_airport usa_airport 0
```

## Contact
If you have any questions about the code or the paper, feel free to contact me.
Email: renh2@zju.edu.cn

## Cite
If you find this work helpful, please cite

```
@article{huang2024measuring,
  title={Measuring Task Similarity and Its Implication in Fine-Tuning Graph Neural Networks},
  author={Huang, Renhong and Xu, Jiarong and Jiang, Xin and Pan, Chenglu and Yang, Zhiming and Wang, Chunping and Yang, Yang},
  booktitle={AAAI},
  volume={38},
  number={11}, 
  pages={12617-12625},
  year={2024}
}
```

## Acknowledgements
Part of this code is inspired by Qiu et al.'s [GCC: Graph Contrastive Coding](https://github.com/THUDM/GCC).
