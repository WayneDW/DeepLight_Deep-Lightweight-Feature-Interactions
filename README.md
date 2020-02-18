# Sparse_DeepFwFM

Deploying the end-to-end deep factorization machines has a critical issue in prediction latency. We study the acceleration of the prediction by conducting structural pruning, which ends up with 46X speed-ups without sacrifice of the state-of-the-art performance on Criteo dataset.

Please refer to the [arXiv paper](https://arxiv.org/pdf/2002.06987.pdf) if you are interested. 


## Environment

1. Python2.7

2. PyTorch

3. Pandas

4. Sklearn


## How to run the dense models

The folder already has a tiny dataset to test. You can run the following models through

LR: logistic regression
```bash
$ python main_all.py -use_fm 0 -use_fwfm 0 -use_deep 0 -use_lw 0 -use_logit 1 > ./logs/all_logistic_regression
```

FM: factorization machine

```bash
$ python main_all.py -use_fm 1 -use_fwfm 0 -use_deep 0 -use_lw 0 > ./logs/all_fm_vanilla
```

FwFM: field weighted factorization machine

```bash
$ python main_all.py -use_fm 0 -use_fwfm 1 -use_deep 0 -use_lw 0 > ./logs/all_fwfm_vanilla
```

DeepFM: deep factorization machine

```bash
$ python main_all.py -use_fm 1 -use_fwfm 0 -use_deep 1 -use_lw 0 > ./logs/all_deepfm_vanilla
```

NFM: factorization machine

```bash
$ python NFM.py > ./logs/all_nfm
```

xDeepFM: extreme factorization machine

You may try the link here https://github.com/Leavingseason/xDeepFM


## How to conduct strctural pruning


The default code gives 0.8123 AUC if apply 90% sparsity on the DNN component and the field matrix R and apply 40% (90%x0.444) on the embeddings.

```bash
python main_all.py -l2 6e-7 -n_epochs 10 -warm 2 -prune 1 -sparse 0.90  -prune_deep 1 -prune_fm 1 -prune_r 1 -use_fwlw 1 -emb_r 0.444 -emb_corr 1. > ./logs/deepfwfm_l2_6e_7_prune_all_and_r_warm_2_sparse_0.90_emb_r_0.444_emb_corr_1
```



## Preprocess full dataset

The Criteo dataset has 2-class labels with 22 categorical features and 11 numerical features.

To download the full dataset, you can use the link below
http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/

Unzip the raw data and save it in ./data/large folder:
>> tar xvzf dac.tar.gz

Move to the data folder and process the raw data.
```bash
$ python preprocess.py
```

When the dataset is ready, you need to change the files in main_all.py as follows
```py
#result_dict = data_preprocess.read_data('./data/tiny_train_input.csv', './data/category_emb', criteo_num_feat_dim, feature_dim_start=0, dim=39)
#test_dict = data_preprocess.read_data('./data/tiny_test_input.csv', './data/category_emb', criteo_num_feat_dim, feature_dim_start=0, dim=39)
result_dict = data_preprocess.read_data('./data/large/train.csv', './data/large/criteo_feature_map', criteo_num_feat_dim, feature_dim_start=1, dim=39)
test_dict = data_preprocess.read_data('./data/large/valid.csv', './data/large/criteo_feature_map', criteo_num_feat_dim, feature_dim_start=1, dim=39)
```





## How to analyze the prediction latency

You need to download this repo: https://github.com/uestla/Sparse-Matrix before you start.

After the setup, you can change the directory in line-23 of the cpp file to your local dir.

```bash
cd latency
g++ criteo_latency.cpp  -o criteo.out
```


To avoid setting the environment, you can also consider to test the compiled file directly.

```bash
./criteo.out
```



## Acknowledgement

https://github.com/nzc/dnn_ctr
