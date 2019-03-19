import json
import pandas as pd
import numpy as np
import scipy.sparse as ss

video_file = "../track2_video_features.txt"
train_data = pd.read_csv('./track2_train_item_id.txt', sep='\t', names=['item_id'])
test_data = pd.read_csv('./track2_test_item_id.txt', sep='\t', names=['item_id'])

def process_audio_feature():
    audio_file = "../track2_audio_features.txt"
    train_audio_out = "./track2_train_audio_features.txt"
    test_audio_out = "./track2_test_audio_features.txt"
    
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

def process_title_feature(title_file):
    train_title_out = "./track2_train_title_features.txt"
    test_title_out = "./track2_test_title_features.txt"
    padding_len = 35
    
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

    with open(train_title_out,'w') as train_out_file:
        for row in train_title_data.itertuples():
            index = getattr(row,'Index')
            print(index)
            title_features = getattr(row,'title_features')
            padding = []
            # item has title_features dict
            if type(title_features).__name__ == 'dict':
                padding = [int(k) for k in title_features.keys()]
                if len(padding) > padding_len:
                    padding = padding[:padding_len]
                else:
                    padding += [0] * (padding_len - len(padding))
            # item has no title_features dict
            else:
                padding = [0] * padding_len
            for i in padding:
                train_out_file.write(str(i)+' ')
            train_out_file.write('\n')
    train_out_file.close()
    
    with open(test_title_out,'w') as test_out_file:
        for row in test_title_data.itertuples():
            index = getattr(row,'Index')
            print(index)
            title_features = getattr(row,'title_features')
            padding = []
            # item has title_features dict
            if type(title_features).__name__ == 'dict':
                padding = [int(k) for k in title_features.keys()]
                if len(padding) > padding_len:
                    padding = padding[:padding_len]
                else:
                    padding += [0] * (padding_len - len(padding))
            # item has no title_features dict
            else:
                padding = [0] * padding_len
            for i in padding:
                test_out_file.write(str(i)+' ')
            test_out_file.write('\n')
    test_out_file.close()

def process_face_attrs(face_file):
    train_face_out = "./track2_train_face_features.txt"
    test_face_out = "./track2_test_face_features.txt"
    
    item_ids = []
    men = []
    women = []
    beauties = []
    position_0 = []
    position_1 = []
    position_2 = []
    position_3 = []

    with open(face_file,'r') as ff:
        for i,line in enumerate(ff):
            print(i)
            l = json.loads(line)
            item_ids.append(l["item_id"])
            face_attrs = l["face_attrs"]
            man = 0
            woman = 0
            beauty = 0
            position = np.array([0, 0, 0, 0])
            for item in face_attrs:
                if item["gender"] == 1:
                    man += 1
                else:
                    woman += 1
                beauty += item["beauty"]
                position = np.array(item["relative_position"]) + position
            men.append(int(man))
            women.append(int(woman))
            people = man + woman
            if people == 0:
                beauties.append(int(0))
                position_0.append(int(0))
                position_1.append(int(0))
                position_2.append(int(0))
                position_3.append(int(0))
            else:
                beauties.append(round(beauty / people, 8))
                position_0.append(round(position[0] / people, 8))
                position_1.append(round(position[1] / people, 8))
                position_2.append(round(position[2] / people, 8))
                position_3.append(round(position[3] / people, 8))

    face_data = pd.DataFrame({'item_id': item_ids, 'man': men, 'woman': women, 'avg_beauty': beauties,
                              'position_0': position_0, 'position_1': position_1, 'position_2': position_2, 'position_3': position_3})

    train_face_data = train_data.merge(face_data, how='left', on='item_id')
    test_face_data = test_data.merge(face_data, how='left', on='item_id')

    train_face_data.to_csv(train_face_out, sep=',', header=None, index=None)
    test_face_data.to_csv(test_face_out, sep=',', header=None, index=None) 

process_face_attrs("../input/track2_face_attrs.txt")