import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
from model_2_28 import myAutoInt
from deepctr import SingleFeat

loss_weights = [0.7, 0.3, ]

data = pd.read_csv('../input/final_track2_train.txt', sep='\t', names=[
        'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'did', 'creat_time', 'video_duration'])
test_data = pd.read_csv('../input/final_track2_test_no_anwser.txt', sep='\t', names=[
        'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'did', 'creat_time', 'video_duration'])

train_size = data.shape[0]
data = data.append(test_data)

sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'music_id', 'did', ]
dense_features = ['video_duration', ]

data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0,)

target = ['finish', 'like']

for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])

sparse_feature_list = [SingleFeat(feat, data[feat].nunique()) for feat in sparse_features]
dense_feature_list = [SingleFeat(feat, 0) for feat in dense_features]

train = data.iloc[:train_size]
test = data.iloc[train_size:]

train_model_input = [train[feat.name].values for feat in sparse_feature_list] + [train[feat.name].values for feat in dense_feature_list]
test_model_input = [test[feat.name].values for feat in sparse_feature_list] + [test[feat.name].values for feat in dense_feature_list]

train_labels = [train[target[0]].values, train[target[1]].values]
test_labels = [test[target[0]].values, test[target[1]].values]

model = myAutoInt({"sparse": sparse_feature_list, "dense": dense_feature_list})
model.compile("adagrad", "binary_crossentropy", loss_weights=loss_weights,)

history = model.fit(train_model_input, train_labels, batch_size=4096, epochs=1, verbose=1)
pred_ans = model.predict(test_model_input, batch_size=4096)

result = test_data[['uid', 'item_id', 'finish', 'like']].copy()
result.rename(columns={'finish': 'finish_probability',
                        'like': 'like_probability'}, inplace=True)
result['finish_probability'] = pred_ans[0]
result['like_probability'] = pred_ans[1]
result[['uid', 'item_id', 'finish_probability', 'like_probability']].to_csv(
    'result_2_28.csv', index=None, float_format='%.6f')