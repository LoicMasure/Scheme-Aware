# Scheme-Aware-Architectures
This Git repo contains some source code used in the paper  [Don’t Learn What You Already Know Scheme-Aware Modeling for Profiling Side-Channel Analysis against Masking](https://tches.iacr.org/index.php/TCHES/article/view/9946), by  Loïc Masure (UCLouvain), Valence Cristiani (CEA, Leti), Maxime Lecomte(CEA, Leti), and François-Xavier Standaert (UCLouvain).

The repository contains the source codes of the Scheme-Aware models, along with some simulation scripts.

The scripts work on **Python 3.8** with a **NVIDIA RTX A6000** of 48GB running on **Cuda 11.7**. The deep learning is implemented with **Torch 1.13.1**.
The CUDA implementation of the Walsh-Hadamard transform is the one released by Anna Thomas, Albert Gu, Tri Dao, Atri Rudra, Christopher Ré in [Learning Compressed Transforms with Low Displacement Rank](https://proceedings.neurips.cc/paper/2018/hash/8e621619d71d0ae5ef4e631ad586334f-Abstract.html). The corresponding repository is [here](https://github.com/HazyResearch/structured-nets).


## Setup

The whole python environment can be installed in a two-stage procedure. First, install the virtual environment, as follows.
```
python3 -m venv env
source env/bin/activate
pip3 install requirements.txt
```

Second, install the library for computing the Walsh-Hadamard transform on the GPU.
```
cd hadamard_cuda
python3 setup.py install
# The following command is to test whether the installation went well.
python hadamard.py
```


## Experiment 1
This script aims at simulating the leakage of a secret masked with Boolean or arithmetical masking.
Here is a starting example:
```
python3 src/experiment_1.py --n_bits 8 --n_draws 10000 -normalize_probs --num_epochs 1000 --learning_rate 1e-4 --batch_size 10000
```
The script stores the results of the different trainings as text files in a common folder. These raw data may then be post-processed. You should see that among all the model tested, the white-box model (denoted by 'WB' in the logs) and the scheme-aware model (denoted by 'WH' in the logs) have the lowest validation losses.

### Options
```
usage: experiment_1 [-h] [--n_bits N_BITS] [--n_targets N_TARGETS] [--leakage_model LEAKAGE_MODEL]
                    [--sigma SIGMA] [--n_draws N_DRAWS] [--n_val N_VAL] [--seed SEED]
                    [--n_hidden N_HIDDEN] [-normalize_probs] [--num_epochs NUM_EPOCHS]
                    [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE]
                    [--num_threads NUM_THREADS] [--res_dir RES_DIR] [-debug]
```
```
optional arguments:
  -h, --help            show this help message and exit
  --n_bits N_BITS       The number of bits on which the random variables are drawn.
  --n_targets N_TARGETS
                        The number of target variables masked with the same masks.
  --leakage_model LEAKAGE_MODEL
                        The leakage model assumed for the shares
  --scheme SCHEME       The masking scheme to simulate
  --order ORDER         The masking order, i.e., #shares - 1
  --sigma SIGMA         The gaussian noise parameter.
  --n_draws N_DRAWS     The number of random draws for the simulations.
  --n_val N_VAL         The number of draws for the validation set.
  --seed SEED           To manage the reproducibility of the experiments.
  --n_hidden N_HIDDEN   The number of neurons in the hidden layer
  -normalize_probs      Whether normalizing the inputs of convolution as probabilities.
  -share_weights        Whether the branches share the same weights.
  --num_epochs NUM_EPOCHS
                        The number of epochs when training the model.
  --learning_rate LEARNING_RATE
                        The learning rate for Adam optimizer.
  --batch_size BATCH_SIZE
                        The batch size for the datasets.
  --num_threads NUM_THREADS
                        The number of threads when loading the data.
  --res_dir RES_DIR     The directory to store the results (logs, data).
  -debug                If activated, all the deterministic options are removed.
```
For now, the simulator supports the Hamming weight, the LSB, and the ID leakage models.

## Experiment affine
We also provide a script simulating the leakage of ASCADv2, where the masking scheme is affine, and where the multiplicative share leaks according to the field multiplication lookup table that is generated during the pre-computation, in order to better reflect a true leakage.

### Options
```
usage: experiment_affine [-h] [--n_bits N_BITS] [--n_targets N_TARGETS] [--leakage_model LEAKAGE_MODEL] [--sigma SIGMA] [--n_draws N_DRAWS] [--n_val N_VAL] [--seed SEED] [--n_hidden N_HIDDEN] [-normalize_probs]
                    [--num_epochs NUM_EPOCHS] [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE] [--num_threads NUM_THREADS] [--res_dir RES_DIR] [-debug]

optional arguments:
  -h, --help            show this help message and exit
  --n_bits N_BITS       The number of bits on which the random variables are drawn.
  --n_targets N_TARGETS
                        The number of target variables masked with the same masks.
  --leakage_model LEAKAGE_MODEL
                        The leakage model assumed for the shares.
  --sigma SIGMA         The gaussian noise parameter.
  --n_draws N_DRAWS     The number of random draws for the simulations.
  --n_val N_VAL         The number of draws for the validation set.
  --seed SEED           To manage the reproducibility of the experiments.
  --n_hidden N_HIDDEN   The number of neurons in the hidden layer
  -normalize_probs      Whether normalizing the inputs of convolution as probabilities.
  --num_epochs NUM_EPOCHS
                        The number of epochs when training the model.
  --learning_rate LEARNING_RATE
                        The learning rate for Adam optimizer.
  --batch_size BATCH_SIZE
                        The batch size for the datasets.
  --num_threads NUM_THREADS
                        The number of threads when loading the data.
  --res_dir RES_DIR     The directory to store the results (logs, data).
  -debug                If activated, all the deterministic options are removed.
```

### How to cite this work
```
@article{Masure_Cristiani_Lecomte_Standaert_2022, 
  title={Don’t Learn What You Already Know: Scheme-Aware Modeling for Profiling Side-Channel Analysis against Masking}, 
  volume={2023}, url={https://tches.iacr.org/index.php/TCHES/article/view/9946}, 
  DOI={10.46586/tches.v2023.i1.32-59}, 
  number={1}, 
  journal={IACR Transactions on Cryptographic Hardware and Embedded Systems}, 
  author={Masure, Loïc and Cristiani, Valence and Lecomte, Maxime and Standaert, François-Xavier}, 
  year={2022}, 
  month={Nov.}, 
  pages={32–59} 
}
```
