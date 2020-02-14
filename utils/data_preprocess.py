"""
Created on June 23, 2019
@author: Wei Deng
"""

import sys
import math
import argparse
import csv, math, os

# criteo feature dimension starts from 0, total feature dimensions 39
# oath dataset starts from 1
def load_category_index(file_path, feature_dim_start=0, dim=39):
    f = open(file_path,'r')
    cate_dict = []
    for i in range(dim):
        cate_dict.append({})
    for line in f:
        datas = line.strip().split(',')
        cate_dict[int(datas[0]) - feature_dim_start][datas[1]] = int(datas[2])
    return cate_dict


def read_data(file_path, emb_file, num_list, feature_dim_start=0, dim=39):
    result = {'label':[], 'value': [], 'index':[],'feature_sizes':[]}
    cate_dict = load_category_index(emb_file, feature_dim_start, dim)
    # the left part is numerical features and the right is categorical features
    result['feature_sizes'] = [1] * len(num_list)
    for num, item in enumerate(cate_dict):
        if num + 1 not in num_list:
            result['feature_sizes'].append(len(item) + 1)

    f = open(file_path,'r')
    for line in f:
        datas = line.strip().split(',')
        result['label'].append(int(datas[0]))
        
        indexs = [int(item) for i, item in enumerate(datas) if i not in num_list and i != 0]
        values = [int(item) for i, item in enumerate(datas) if i in num_list]
        result['index'].append(indexs)
        result['value'].append(values)
    return result

