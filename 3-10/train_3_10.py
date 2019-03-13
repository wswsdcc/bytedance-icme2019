import pandas as pd
from deepctr import SingleFeat, VarLenFeat
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from model import xDeepFM_MTL
from helper import reduce_mem_usage, reduce_mem_usage_with_fillna

ONLINE_FLAG = True
loss_weights = [0.7, 0.3, ]  # [0.7,0.3]任务权重可以调下试试
VALIDATION_FRAC = 0.2  # 用做线下验证数据比例
chunkSize = 1024*128

def read_chunk(reader, chunkSize):
    chunks = []
    while True:
        try:
            chunk = reader.get_chunk(chunkSize)
            reduce_mem_usage_with_fillna(chunk)
            chunks.append(chunk)
        except StopIteration:
            # print("Iteration is stopped.")
            break
    df = pd.concat(chunks, ignore_index=True)

    return df

if __name__ == "__main__":
    data = pd.read_csv('../input/final_track2_train.txt', sep='\t', names=[
        'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'did', 'creat_time', 'video_duration'])
    reduce_mem_usage(data)
    print("read train data done")
    audio_reader = pd.read_csv('../input/track2_train_audio_features.txt', iterator=True, sep=' ', header=None)
    audio_data = read_chunk(audio_reader, chunkSize)

    print("read train audio data done")
    if ONLINE_FLAG:
        test_data = pd.read_csv('../input/final_track2_test_no_anwser.txt', sep='\t', names=[
                                'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'did', 'creat_time', 'video_duration'])
        reduce_mem_usage(test_data)
        print("read test data done")
        test_audio_reader = pd.read_csv('../input/track2_test_audio_features.txt', iterator=True, sep=' ', header=None)
        test_audio_data = read_chunk(test_audio_reader, chunkSize)
        print("read test audio data done")
        audio_data = audio_data.append(test_audio_data)
        audio_data.drop([audio_data.columns[-1]], axis=1, inplace=True)
        audio_data = audio_data.values

        train_size = data.shape[0]
        data = data.append(test_data)
    else:
        train_size = int(data.shape[0]*(1-VALIDATION_FRAC))

    sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'music_id', 'did', ]
    dense_features = ['video_duration']  # 'creat_time',

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0,)

    target = ['finish', 'like']

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    ## preprocess the sequence feature
    # audio_data = pad_sequences(audio_data, maxlen=128, padding='post',)

    sparse_feature_list = [SingleFeat(feat, data[feat].nunique())
                           for feat in sparse_features]
    dense_feature_list = [SingleFeat(feat, 0)
                          for feat in dense_features]
    sequence_feature = [VarLenFeat('audio', 128, 128, 'max')]

    train = data.iloc[:train_size]
    test = data.iloc[train_size:]

    train_model_input = [train[feat.name].values for feat in sparse_feature_list] + \
        [train[feat.name].values for feat in dense_feature_list] + [audio_data[:train_size]]
    test_model_input = [test[feat.name].values for feat in sparse_feature_list] + \
        [test[feat.name].values for feat in dense_feature_list] + [audio_data[train_size:]]

    train_labels = [train[target[0]].values, train[target[1]].values]
    test_labels = [test[target[0]].values, test[target[1]].values]

    model = xDeepFM_MTL({"sparse": sparse_feature_list,
                         "dense": dense_feature_list,
                         "sequence": sequence_feature})
    model.compile("adagrad", "binary_crossentropy", loss_weights=loss_weights,)

    if ONLINE_FLAG:
        history = model.fit(train_model_input, train_labels,
                            batch_size=4096, epochs=1, verbose=1)
        pred_ans = model.predict(test_model_input, batch_size=2**14)

    else:
        history = model.fit(train_model_input, train_labels,
                            batch_size=4096, epochs=1, verbose=1, validation_data=(test_model_input, test_labels))

    if ONLINE_FLAG:
        result = test_data[['uid', 'item_id', 'finish', 'like']].copy()
        result.rename(columns={'finish': 'finish_probability',
                               'like': 'like_probability'}, inplace=True)
        result['finish_probability'] = pred_ans[0]
        result['like_probability'] = pred_ans[1]
        result[['uid', 'item_id', 'finish_probability', 'like_probability']].to_csv(
            'result.csv', index=None, float_format='%.6f')
