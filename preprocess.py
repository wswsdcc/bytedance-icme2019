import json
import pandas as pd
import numpy as np
import scipy.sparse as ss

video_file = "./input/track2_video_features.txt"

train_data = pd.read_csv('./input/track2_train_item_id.txt', sep='\t', names=['item_id'])
test_data = pd.read_csv('./input/track2_test_item_id.txt', sep='\t', names=['item_id'])

def process_audio_feature():
    '''
    Find out the audio features of items in train data and test data,
    and arrange them in the same order as the item_ids are in 'final_track2_xxx'.

    The order in which the item_id appear has been sorted out(see in 
    './input/track2_train_item_id.txt' and './input/track2_test_item_id.txt').
    '''
    audio_file = "./input/track2_audio_features.txt"
    train_audio_out = "./input/track2_train_audio_features.txt"
    test_audio_out = "./input/track2_test_audio_features.txt"
    
    feature_dict = {}
    i = 0
    j = 0

    with open(audio_file,'r') as ff:
        for line in ff:
            print(i)
            l = json.loads(line)
            feature_dict[l["item_id"]] = l["audio_feature_128_dim"]
            i += 1
    ff.close()

    with open(train_audio_out,'w') as train_out_file:
        for row in train_data.itertuples():
            print(j)
            item_id = getattr(row,'item_id')
            feature = feature_dict.get(item_id,[0 for m in range(128)])
            for n in feature:
                train_out_file.write(str(n)+' ')
            train_out_file.write('\n')
            j += 1
    train_out_file.close()

def process_title_feature():
    '''
    Arrange the title features of items in the same order as the item_ids are in
    'final_track2_xxx'.

    To convert dict into matrix, I use 'coo_matrix' in 'scipy.sparse' because the
    word frequency matrix is very sparse, and 'coo_matrix' can handle it easily.

    The order in which the item_id appear has been sorted out(see in 
    './input/track2_train_item_id.txt' and './input/track2_test_item_id.txt').
    '''
    title_file = "./input/track2_title.txt"
    train_title_out = "./input/track2_train_title_features.txt"
    test_title_out = "./input/track2_test_title_features.txt"
    
    item_id = []
    feature = []

    with open(title_file,'r') as ff:
        for i,line in enumerate(ff):
            print(i)
            l = json.loads(line)
            item_id.append(l["item_id"])
            feature.append(l["title_features"])

    title_data = pd.DataFrame({'item_id':item_id,'title_features':feature})

    train_title_data = train_data.merge(title_data, how='left', on='item_id')
    test_title_data = test_data.merge(title_data, how='left', on='item_id')

    values = []
    rows = []
    cols = []

    for row in train_title_data.itertuples():
        index = getattr(row,'Index')
        print(index)
        title_features = getattr(row,'title_features')
        for k in title_features.keys():
            rows.append(index)
            cols.append(int(k))
            values.append(title_features[k])
    
    sparse_train = ss.coo_matrix((values, (rows, cols)))
    
    values = []
    rows = []
    cols = []

    for row in test_title_data.itertuples():
        index = getattr(row,'Index')
        print(index)
        title_features = getattr(row,'title_features')
        for k in title_features.keys():
            rows.append(index)
            cols.append(int(k))
            values.append(title_features[k])
    
    sparse_test = ss.coo_matrix((values, (rows, cols)))
    
    print(sparse_train.shape)
    print(sparse_test.shape)
    return sparseM_train, sparse_test

    #test_title_data['title_features'].to_csv(test_title_out,index=None)

process_title_feature()