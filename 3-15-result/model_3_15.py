import tensorflow as tf
from tensorflow.python.keras.layers import Input, Embedding, LSTM, Dense, GlobalAveragePooling1D
from deepctr.input_embedding import preprocess_input_embedding
from deepctr.layers.core import MLP, PredictionLayer
from deepctr.layers.interaction import CIN
from deepctr.layers.utils import concat_fun
from deepctr.utils import check_feature_config_dict


def MTL_with_Title(feature_dim_dict, embedding_size=8, hidden_size=(256, 256), cin_layer_size=(256, 256,) ,cin_split_half=True, task_net_size=(128,), l2_reg_linear=0.00001, l2_reg_embedding=0.00001, seed=1024, ):
    check_feature_config_dict(feature_dim_dict)
    if len(task_net_size) < 1:
        raise ValueError('task_net_size must be at least one layer')
    
    # xDeepFM Model

    deep_emb_list, linear_logit, inputs_list = preprocess_input_embedding(feature_dim_dict, embedding_size, l2_reg_embedding, l2_reg_linear, 0.0001, seed)

    fm_input = concat_fun(deep_emb_list, axis=1)

    if len(cin_layer_size) > 0:
        exFM_out = CIN(cin_layer_size, 'relu', cin_split_half, seed)(fm_input)
        exFM_logit = tf.keras.layers.Dense(1, activation=None, )(exFM_out)

    deep_input = tf.keras.layers.Flatten()(fm_input)
    deep_out = MLP(hidden_size)(deep_input)

    finish_out = MLP(task_net_size)(deep_out)
    finish_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(finish_out)

    like_out = MLP(task_net_size)(deep_out)
    like_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(like_out)

    # Add Title Features

    title_input = Input(shape = (35,), dtype = 'int32', name = 'title_input')
    title_embedding = Embedding(output_dim = 32, input_dim = 134545, input_length = 35)(title_input)
    lstm_out = LSTM(units = 32, return_sequences = True)(title_embedding)
    avg_out = GlobalAveragePooling1D()(lstm_out)
    dense1 = Dense(32, activation='relu')(avg_out)
    dense2 = Dense(1, activation='relu')(dense1)

    # 
    
    finish_logit = tf.keras.layers.add([linear_logit, finish_logit, exFM_logit, dense2])
    like_logit = tf.keras.layers.add([linear_logit, like_logit, exFM_logit, dense2])

    output_finish = PredictionLayer('sigmoid', name='finish')(finish_logit)
    output_like = PredictionLayer('sigmoid', name='like')(like_logit)
    print(str(inputs_list))
    inputs_list.append(title_input)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=[output_finish, output_like])
    return model
