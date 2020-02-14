#!/usr/bin/env python

import random
import argparse

import numpy as np

from model import DeepFMs
from utils import data_preprocess

import torch


parser = argparse.ArgumentParser(description='Hyperparameter tuning')
parser.add_argument('-c', default='DeepFwFM', type=str, help='Models: FM, DeepFwFM ...')
parser.add_argument('-use_cuda', default='1', type=int, help='Use CUDA or not')
parser.add_argument('-gpu', default=0, type=int, help='GPU id')
parser.add_argument('-n_epochs', default=8, type=int, help='Number of epochs')
parser.add_argument('-numerical', default=13, type=int, help='Numerical features, 13 for Criteo')
parser.add_argument('-use_multi', default='0', type=int, help='Use multiple CUDAs')
parser.add_argument('-use_logit', default=0, type=int, help='Use Logistic regression')
parser.add_argument('-use_fm', default=0, type=int, help='Use FM module or not')
parser.add_argument('-use_fwlw', default=0, type=int, help='If to include FwFM linear weights or not')
parser.add_argument('-use_lw', default=1, type=int, help='If to include FM linear weights or not')
parser.add_argument('-use_ffm', default=0, type=int, help='Use FFM module or not')
parser.add_argument('-use_fwfm', default=1, type=int, help='Use FwFM module or not')
parser.add_argument('-use_deep', default=1, type=int, help='Use Deep module or not')
parser.add_argument('-num_deeps', default=1, type=int, help='Number of deep networks')
parser.add_argument('-deep_nodes', default=400, type=int, help='Nodes in each layer')
parser.add_argument('-h_depth', default=3, type=int, help='Deep layers')
parser.add_argument('-prune', default=0, type=int, help='Prune model or not')
parser.add_argument('-prune_r', default=0, type=int, help='Prune r')
parser.add_argument('-prune_deep', default=1, type=int, help='Prune Deep component')
parser.add_argument('-prune_fm', default=1, type=int, help='Prune FM component')
parser.add_argument('-emb_r', default=1., type=float, help='Sparse FM ratio over Sparse Deep ratio')
parser.add_argument('-emb_corr', default=1., type=float, help='Sparse Corr ratio over Sparse Deep ratio')
parser.add_argument('-sparse', default=0.9, type=float, help='Sparse rate')
parser.add_argument('-warm', default=10, type=float, help='Warm up epochs before pruning')
parser.add_argument('-ensemble', default=0, type=int, help='Ensemble models or not')
parser.add_argument('-embedding_size', default=10, type=int, help='Embedding size')
parser.add_argument('-batch_size', default=2048, type=int, help='Batch size')
parser.add_argument('-random_seed', default=0, type=int, help='Random seed')
parser.add_argument('-learning_rate', default= 0.001, type=float, help='Learning rate')
parser.add_argument('-momentum', default= 0, type=float, help='Momentum')
parser.add_argument('-l2', default=3e-7, type=float, help='L2 penalty')
pars = parser.parse_args()

print(pars)
np.random.seed(pars.random_seed)
random.seed(pars.random_seed)
torch.manual_seed(pars.random_seed)
torch.cuda.manual_seed(pars.random_seed)

save_model_name = './saved_models/' + pars.c + '_l2_' + str(pars.l2) + '_sparse_' + str(pars.sparse) + '_seed_' + str(pars.random_seed)

criteo_num_feat_dim = set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
result_dict = data_preprocess.read_data('./data/tiny_train_input.csv', './data/category_emb', criteo_num_feat_dim, feature_dim_start=0, dim=39)
test_dict = data_preprocess.read_data('./data/tiny_test_input.csv', './data/category_emb', criteo_num_feat_dim, feature_dim_start=0, dim=39)
#result_dict = data_preprocess.read_data('./data/large/train.csv', './data/large/criteo_feature_map', criteo_num_feat_dim, feature_dim_start=1, dim=39)
#test_dict = data_preprocess.read_data('./data/large/valid.csv', './data/large/criteo_feature_map', criteo_num_feat_dim, feature_dim_start=1, dim=39)

with torch.cuda.device(pars.gpu):
    model = DeepFMs.DeepFMs(field_size=39,feature_sizes=result_dict['feature_sizes'], embedding_size=pars.embedding_size, n_epochs=pars.n_epochs, \
            verbose=True, use_cuda=pars.use_cuda, use_fm=pars.use_fm, use_fwfm=pars.use_fwfm, use_ffm=pars.use_ffm, use_deep=pars.use_deep, \
            batch_size=pars.batch_size, learning_rate=pars.learning_rate, weight_decay=pars.l2, momentum=pars.momentum, sparse=pars.sparse, warm=pars.warm, \
            h_depth=pars.h_depth, deep_nodes=pars.deep_nodes, num_deeps=pars.num_deeps, numerical=pars.numerical, use_lw=pars.use_lw, use_fwlw=pars.use_fwlw, \
            use_logit=pars.use_logit, random_seed=pars.random_seed)
    if pars.use_cuda:
        model = model.cuda()
        model.fit(result_dict['index'], result_dict['value'], result_dict['label'], test_dict['index'], test_dict['value'], test_dict['label'], \
                prune=pars.prune, prune_fm=pars.prune_fm, prune_r=pars.prune_r, prune_deep=pars.prune_deep, save_path=save_model_name, emb_r=pars.emb_r, emb_corr=pars.emb_corr)
