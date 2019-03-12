import tensorflow as tf
from deepctr.input_embedding import preprocess_input_embedding
from deepctr.layers.core import PredictionLayer, MLP
from deepctr.layers.interaction import InteractingLayer
from deepctr.utils import check_feature_config_dict
from deepctr.layers.utils import concat_fun

def myAutoInt(feature_dim_dict, embedding_size=8, att_layer_num=3, att_embedding_size=8, att_head_num=4, att_res=True, hidden_size=(256, 256), activation='relu',
            l2_reg_deep=0, l2_reg_embedding=1e-5, use_bn=False, keep_prob=1.0, init_std=0.0001, seed=1024,
            final_activation='sigmoid',):
    if len(hidden_size) <= 0 and att_layer_num <= 0:
        raise ValueError("Either hidden_layer or att_layer_num must > 0")
    check_feature_config_dict(feature_dim_dict)

    deep_emb_list, _, inputs_list = preprocess_input_embedding(feature_dim_dict, embedding_size, l2_reg_embedding, 0, init_std, seed, False)
    att_input = concat_fun(deep_emb_list, axis=1)

    for _ in range(att_layer_num):
        att_input = InteractingLayer(att_embedding_size, att_head_num, att_res)(att_input)
    att_output = tf.keras.layers.Flatten()(att_input)

    deep_input = tf.keras.layers.Flatten()(concat_fun(deep_emb_list))
    deep_out = MLP(hidden_size, activation, l2_reg_deep, keep_prob, use_bn, seed)(deep_input)

    finish_out = tf.keras.layers.Concatenate()([att_output, deep_out])
    finish_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(finish_out)

    like_out = tf.keras.layers.Concatenate()([att_output, deep_out])
    like_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(like_out)

    output_finish = PredictionLayer(final_activation, name='finish')(finish_logit)
    output_like = PredictionLayer(final_activation, name='like')(like_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=[output_finish, output_like])

    return model