# Bridge-Tune



## About

This repo is the official code for AAAI-24 "Measuring Task Similarity and Its Implication in Fine-Tuning Graph Neural Networks"

## Dependencies
The script has been tested running under Python 3.7.10, with the following packages installed (along with their dependencies):

- [PyTorch](https://pytorch.org/). Version >=1.4 required. You can find instructions to install from source [here](https://pytorch.org/get-started/previous-versions/).
- [DGL](https://www.dgl.ai/). 0.5 > Version >=0.4.3 required. You can find instructions to install from source [here](https://www.dgl.ai/pages/start.html).
- [rdkit](https://anaconda.org/conda-forge/rdkit). Version = 2019.09.2 required. It can be easily installed with 
			```conda install -c conda-forge rdkit=2019.09.2```
- [Other Python modules](https://pypi.python.org). Some other Python module dependencies are listed in requirements.txt, which can be easily installed with pip:

	`pip install -r requirements.txt`

In addition, CUDA 10.0 has been used in our project. Although not all dependencies are mentioned in the installation instruction links above, you can find most of the libraries in the package repository of a regular Linux distribution.


## File folders

`data`: contains the data of "DD242, DD68, DD687, usa_airport, brazil_airport, europe_airport".

`splits`: **need to unzipped**, contains the split data of "cora, pubmed, cornell and wisconsin".

`scripts`: contains all the scripts for running code.

`gcc&utils`: contains the code of models.

`saved`: contains the pre-trained graph obtained from [GCC](https://github.com/THUDM/GCC). The specific path for pre-trained model is located in `saved/Pretrain_moco_True_dgl_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_hid_64_samples_2000_nce_t_0.07_nce_k_16384_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_16_pos_32_momentum_0.999/`

## Usage: How to run the code
We divide it into two steps (1) Finetuning (2) Evaluating the performance finetune model.

### 1. Fine-tuning via Bridge Tune

```bash
python train_bridge.py \
  --resume <pre-trained model file> \
  --dataset <downstream dataset>
  --reg-coeff <coefficient for Bridge-Tune Loss> \
  --model-path <fine-tune model saved file> \
  --gpu <gpu id> \
  --epochs <epoch number> \
  --bridge
```

For more detail, the help information of the main script `train_bridge.py` can be obtain by executing the following command.

```bash
python train_bridge.py -h

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of training epochs (default:30)
  --optimizer {sgd,adam,adagrad}
                        optimizer (default:adam)
  --learning_rate LEARNING_RATE  learning rate (default:0.005)
  --resume PATH         path for pre-trained model (default: GCC)
  --dataset {usa_airport,brazil_airport,europe_airport,h-index, texas,DD242,cornell,wisconsin,citeseer}
  --reg-coeff REG_COEFF  coefficient for Bridge-Tune (default:10)
  --hidden-size HIDDEN_SIZE  (default:64)
  --model-path MODEL_PATH    path to save finetune model (default:saved)
  --finetune            whether to conduct finetune
  --gpu GPU [GPU ...]   GPU id to use.
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

### 2. Evaluating

`generate.py` file helps generate embeddings on a specific dataset. The help information of the main script `generate.py` can be obtain by executing the following command.

```bash
python generate.py -h

optional arguments:
  --load-path LOAD_PATH
  --dataset Dataset
  --gpu GPU  GPU id to use.
```
The embedding will be used for evaluation in node classification. The script `evaluate.sh` are available to simplify the evaluation process as follows: 

```
bash evaluate.sh <saved_path> <model_path> <dataset> <cuda>
```
Here, `<saved_path>` refers to the main directory for finetuning, and `<model_path>` is the address of the subdirectory within it.

**Demo:**
Here is the demo instruction, after the user has trained using the demo provided above.
```
bash scripts/evaluate.sh saved path_bridge10_usa_airport usa_airport 0
```


## Acknowledgements
Part of this code is inspired by Qiu et al.'s [GCC: Graph Contrastive Coding](https://github.com/THUDM/GCC).


## Contact
If you have any question about the code or the paper, feel free to contact me.
Email: renh2@zju.edu.cn

## Cite
If you find this work helpful, please cite

```
@article{huang2024measuring,
  title={Measuring Task Similarity and Its Implication in Fine-Tuning Graph Neural Networks},
  author={Huang, Renhong and Xu, Jiarong and Jiang, Xin and Pan, Chenglu and Yang, Zhiming and Wang, Chunping and Yang, Yang},
  booktitle={AAAI},
  year={2024}
}

```