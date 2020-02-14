"""
Preprocess Criteo raw data

Created by Wei Deng (deng106@purdue.edu) on Jun.13, 2019


First, you can download the raw dataset dac.tar.gz using the link below
http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/

Unzip the raw data:
>> tar xvzf dac.tar.gz

Criteo data has 13 numerical fields and 26 category fields. 

train.txt: The 1st column is label, and the rest are features.
test.txt: All the column are features. We don't use it.

In the training dataset

Step 1, randomly split the training data into training and valid dataset

Step 1, cnt_freq_train: count the frequency of features in each field; 
        ignore data that has less than 40 columns (it consists of 47% of the whole dataset)

Step 2, ignore_long_tail: set the long-tail features with frequency less than a threshold as 0; \
        generate the feature-map index, the columns are: field index, unique feature, mapped index

In the valid dataset

Map the known feature to existing index and set the unknown as 0.

"""

import random, math, os
random.seed(0)

def random_split(inputs, output1, valid):
    fout1 = open(output1, 'w')
    fout2 = open(valid, 'w')
    for line in open(inputs):
        if random.uniform(0, 1) < 0.9:
            fout1.write(line)
        else:
            fout2.write(line)
    fout1.close()
    fout2.close()

# https://github.com/WayneDW/AutoInt/blob/master/Dataprocess/Criteo/scale.py
def scale(x):
    if x == '':
        return '0'
    elif float(x) > 2:
        return str(int(math.log(float(x))**2))
    else:
        return x

def cnt_freq_train(inputs):
    count_freq = []
    for i in range(40):
        count_freq.append({})
    for idx, line in enumerate(open(inputs)):
        line = line.replace('\n', '').split('\t')
        if idx % 1000000 == 0 and idx > 0:
            print idx
        for i in range(1, 40):
            if i < 14:
                #line[i] = project_numeric(line[i])
                line[i] = scale(line[i])
            if line[i] not in count_freq[i]:
                count_freq[i][line[i]] = 0
            count_freq[i][line[i]] += 1
    return count_freq


def generate_feature_map_and_train_csv(inputs, train_csv, file_feature_map, freq_dict, threshold=4):
    feature_map = []
    for i in range(40):
        feature_map.append({})
    fout = open(train_csv, 'w')
    for idx, line in enumerate(open(inputs)):
        line = line.replace('\n', '').split('\t')
        if idx % 1000000 == 0 and idx > 0:
            print idx
        output_line = [line[0]]
        for i in range(1, 40):
            # map numerical features
            if i < 14:
                #line[i] = project_numeric(line[i])
                line[i] = scale(line[i])
                output_line.append(line[i])
            # handle categorical features
            elif freq_dict[i][line[i]] < threshold:
                output_line.append('0')
            elif line[i] in feature_map[i]:
                output_line.append(feature_map[i][line[i]])
            else:
                output_line.append(str(len(feature_map[i]) + 1))
                feature_map[i][line[i]] = str(len(feature_map[i]) + 1)
        output_line = ','.join(output_line)
        fout.write(output_line + '\n')

    # write feature_map file
    f_map = open(file_feature_map, 'w')
    for i in range(1, 40):
        #only_one_zero_index = True
        for feature in feature_map[i]:
            #if feature_map[i][feature] == '0' and only_one_zero_index == False:
            #    continue
            f_map.write(str(i) + ',' + feature + ',' + feature_map[i][feature] + '\n')
            #if only_one_zero_index == True and feature_map[i][feature] == '0':
            #    only_one_zero_index = False
    return feature_map

def generate_valid_csv(inputs, valid_csv, feature_map):
    fout = open(valid_csv, 'w')
    for idx, line in enumerate(open(inputs)):
        line = line.replace('\n', '').split('\t')
        output_line = [line[0]]
        for i in range(1, 40):
            if i < 14:
                #line[i] = project_numeric(line[i])
                line[i] = scale(line[i])
                output_line.append(line[i])
            elif line[i] in feature_map[i]:
                output_line.append(feature_map[i][line[i]])
            else:
                output_line.append('0')
        output_line = ','.join(output_line)
        fout.write(output_line + '\n')

print('Split the orignal dataset into train and valid dataset.')
random_split('train.txt', 'train1.txt', 'valid.txt')
#print('Count the frequency.')
#freq_dict = cnt_freq_train('train1.txt')

# Not the best way, follow xdeepfm
freq_dict = cnt_freq_train('train.txt')


print('Generate the feature map and impute the training dataset.')
feature_map = generate_feature_map_and_train_csv('train1.txt', 'train.csv', 'criteo_feature_map', freq_dict, threshold=8)
print('Impute the valid dataset.')
generate_valid_csv('valid.txt', 'valid.csv', feature_map)
print('Delete unnecessary files')
os.system('rm train1.txt valid.txt')

def get_feature_size(fname):
    cnts = [0] * 40
    mins = [1] * 40
    maxs = [1] * 40
    dicts = []
    for i in range(40):
        dicts.append(set())
    for line in open(fname):
        line = line.strip().split(',')
        for i in range(40):
            if line[i] not in dicts[i]:
                cnts[i] += 1
                dicts[i].add(line[i])
            try:
                mins[i] = min(mins[i], float(line[i]))
                maxs[i] = max(maxs[i], float(line[i]))
            except:
                print line
    print cnts
    print mins
    print maxs

#get_feature_size('train_shuffle.csv')
