import pandas as pd
import numpy as np
# import sys
from deepctr import SingleFeat, VarLenFeat
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from model import xDeepFM_MTL
from helper import reduce_mem_usage, reduce_mem_usage_with_fillna

loss_weights = [0.7, 0.3, ]  # [0.7,0.3]任务权重可以调下试试
VALIDATION_FRAC = 0.2  # 用做线下验证数据比例
chunkSize = 1024*64

# log_file = open('log_3_11.txt', 'w')
# sys.stdout = log_file

def read_chunk(reader, chunkSize):
    chunk = reader.get_chunk(chunkSize)
    # reduce_mem_usage_with_fillna(chunk)
    return chunk

if __name__ == "__main__":
    data = pd.read_csv('./input/final_track2_train.txt', sep='\t', names=['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'did', 'creat_time', 'video_duration'])
    audio = pd.read_csv('./input/track2_train_audio_features.txt', iterator=True, sep=' ', header=None)
    test_data = pd.read_csv('./input/final_track2_test_no_anwser.txt', sep='\t', names=['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'did', 'creat_time', 'video_duration'])
    test_audio = pd.read_csv('./input/track2_test_audio_features.txt', iterator=True, sep=' ', header=None)
    
    # reduce_mem_usage(data)
    # reduce_mem_usage(test_data)
    train_size = data.shape[0]
    data = data.append(test_data)

    sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'music_id', 'did', ]
    dense_features = ['video_duration']  # 'creat_time',
    target = ['finish', 'like']

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    sparse_feature_list = [SingleFeat(feat, data[feat].nunique()) for feat in sparse_features]
    dense_feature_list = [SingleFeat(feat, 0) for feat in dense_features]
    sequence_feature = [VarLenFeat('audio', 128, 128, 'mean')]

    model = xDeepFM_MTL({"sparse": sparse_feature_list, "dense": dense_feature_list, "sequence": sequence_feature})
    model.compile("adagrad", "binary_crossentropy", loss_weights=loss_weights,)

    train_flag = 0
    while True:
        try:
            audio_chunk = read_chunk(audio, chunkSize)
            audio_chunk.drop([audio_chunk.columns[-1]], axis=1, inplace=True)
            audio_chunk = audio_chunk.values
            # audio_chunk = pad_sequences(audio_chunk, maxlen=128, padding='post',)
            chunk_size = audio_chunk.shape[0]
            train_chunk = data.iloc[train_flag : chunk_size + train_flag]
            train_flag += chunk_size
            train_model_input = [train_chunk[feat.name].values for feat in sparse_feature_list] + \
                [train_chunk[feat.name].values for feat in dense_feature_list] + [audio_chunk]
            train_labels = [train_chunk[target[0]].values, train_chunk[target[1]].values]
            model.fit(train_model_input, train_labels, batch_size=4096, epochs=1, verbose=1)
        except StopIteration:
            # print("Iteration is stopped.")
            break

    test_flag = train_size
    pred_chunks = []
    while True:
        try:
            audio_chunk = read_chunk(test_audio, chunkSize)
            audio_chunk.drop([audio_chunk.columns[-1]], axis=1, inplace=True)
            audio_chunk = audio_chunk.values
            #audio_chunk = pad_sequences(audio_chunk, maxlen=128, padding='post',)
            chunk_size = audio_chunk.shape[0]
            test_chunk = data.iloc[test_flag : chunk_size + test_flag]
            test_flag += chunk_size
            test_model_input = [test_chunk[feat.name].values for feat in sparse_feature_list] + \
                [test_chunk[feat.name].values for feat in dense_feature_list] + [audio_chunk]
            pred_chunk = model.predict(test_model_input, batch_size=4096)
            pred_chunks.append(pred_chunk)
        except StopIteration:
            # print("Iteration is stopped.")
            break

    pred_ans = np.concatenate(pred_chunks, axis = 1)
    result = test_data[['uid', 'item_id', 'finish', 'like']].copy()
    result.rename(columns={'finish': 'finish_probability', 'like': 'like_probability'}, inplace=True)
    result['finish_probability'] = pred_ans[0]
    result['like_probability'] = pred_ans[1]
    result[['uid', 'item_id', 'finish_probability', 'like_probability']].to_csv('result.csv', index=None, float_format='%.6f')