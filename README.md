# Sparse_DeepFwFM


## Environment

1. Python2.7

2. PyTorch

3. Pandas

4. Sklearn

#### Preprocess data

The Criteo dataset has a single 2-class label with 22 categorical features and 11 numerical features.

The folder already has a tiny dataset to test. 

To download the full dataset, you can use the link below
http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/

Unzip the raw data and save it in ./data/large folder:
>> tar xvzf dac.tar.gz

Move to the data folder and process the raw data.
```bash
$ python preprocess.py  
```

#### 



```py
#result_dict = data_preprocess.read_data('./data/tiny_train_input.csv', './data/category_emb', criteo_num_feat_dim, feature_dim_start=0, dim=39)
#test_dict = data_preprocess.read_data('./data/tiny_test_input.csv', './data/category_emb', criteo_num_feat_dim, feature_dim_start=0, dim=39)
result_dict = data_preprocess.read_data('./data/large/train.csv', './data/large/criteo_feature_map', criteo_num_feat_dim, feature_dim_start=1, dim=39)
test_dict = data_preprocess.read_data('./data/large/valid.csv', './data/large/criteo_feature_map', criteo_num_feat_dim, feature_dim_start=1, dim=39)
```





#### How to analyze the prediction latency

You need to download this repo: https://github.com/uestla/Sparse-Matrix before you start
After the setup, you can change the directory in line-23 of the cpp file to your local dir.

```cpp
cd latency
g++ oath_latency.cpp
./a.out
```
