import tensorflow as tf
from tensorflow.python.keras.layers import Input, Embedding, Flatten
from tensorflow.python.keras.initializers import RandomNormal, Zeros
from tensorflow.python.keras.regularizers import l2
from deepctr.layers.sequence import WeightedSequenceLayer, SequencePoolingLayer
from deepctr.layers.utils import concat_func, Linear, add_func
from deepctr.layers.core import DNN, PredictionLayer
from deepctr.layers.interaction import FM


def deepFM(nums_dict, embedding_dim_dict, tag_history_max_len=50, dnn_hidden_units=(128, 128),
           l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
           dnn_activation='relu', dnn_use_bn=False, task='binary'):
    # 输入层
    gender = Input(shape=(1,), name="gender", dtype="int32")
    age = Input(shape=(1,), name="age", dtype="int32")
    province = Input(shape=(1,), name="province", dtype="int32")
    city = Input(shape=(1,), name="city", dtype="int32")
    tagid_history = Input(shape=(tag_history_max_len,), name="tagid_history", dtype="int32")
    tagid_weight_input = Input(shape=(tag_history_max_len,), name="tagid_weight", dtype="float32")
    tagid_history_len = Input(shape=(1,), name="tagid_history_len", dtype="float32")
    # 嵌入层，包括lr+deep的嵌入
    gender_lr_emb = Embedding(nums_dict['gender'], 1,
                              embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
                              embeddings_regularizer=l2(l2_reg_linear),
                              name="gender_lr_emb")(gender)  # (B, 1, 1)
    age_lr_emb = Embedding(nums_dict['age'], 1,
                           embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
                           embeddings_regularizer=l2(l2_reg_linear),
                           name="age_lr_emb")(age)  # (B, 1, 1)
    province_lr_emb = Embedding(nums_dict['province'], 1,
                                embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
                                embeddings_regularizer=l2(l2_reg_linear),
                                name="province_lr_emb")(province)  # (B, 1, 1)
    city_lr_emb = Embedding(nums_dict['city'], 1,
                            embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
                            embeddings_regularizer=l2(l2_reg_linear),
                            name="city_lr_emb")(city)  # (B, 1, 1)
    tagid_lr_emb = Embedding(nums_dict['tagid_history'], 1,
                             embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
                             embeddings_regularizer=l2(l2_reg_linear),
                             name="tagid_lr_emb",
                             mask_zero=True)(tagid_history)  # (B, max_len, 1)
    gender_emb = Embedding(nums_dict['gender'], embedding_dim_dict['gender'],
                           embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
                           embeddings_regularizer=l2(l2_reg_embedding),
                           name="gender_emb")(gender)  # (B, 1, d)
    age_emb = Embedding(nums_dict['age'], embedding_dim_dict['age'],
                        embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
                        embeddings_regularizer=l2(l2_reg_embedding),
                        name="age_emb")(age)  # (B, 1, d)
    province_emb = Embedding(nums_dict['province'], embedding_dim_dict['province'],
                             embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
                             embeddings_regularizer=l2(l2_reg_embedding),
                             name="province_emb")(province)  # (B, 1, d)
    city_emb = Embedding(nums_dict['city'], embedding_dim_dict['city'],
                         embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
                         embeddings_regularizer=l2(l2_reg_embedding),
                         name="city_emb")(city)  # (B, 1, d)
    tagid_emb = Embedding(nums_dict['tagid_history'], embedding_dim_dict['tagid_history'],
                          embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
                          embeddings_regularizer=l2(l2_reg_embedding),
                          name="tagid_emb",
                          mask_zero=True)(tagid_history)  # (B, max_len, d)
    # 对tagid_history进行加权
    tagid_weight = tf.expand_dims(tagid_weight_input, axis=-1)  # (B, max_len, 1)
    tagid_lr_emb = WeightedSequenceLayer(weight_normalization=False) \
        ([tagid_lr_emb, tagid_history_len, tagid_weight])  # (B, max_len,1)
    tagid_lr_emb = SequencePoolingLayer(mode='sum')([tagid_lr_emb, tagid_history_len])  # (B, 1, 1)

    tagid_emb = WeightedSequenceLayer(weight_normalization=False) \
        ([tagid_emb, tagid_history_len, tagid_weight])  # (B, max_len, d)
    tagid_emb = SequencePoolingLayer(mode='sum')([tagid_emb, tagid_history_len])  # (B, 1, d)
    # lr
    lr_emb_concat = Flatten()(concat_func([gender_lr_emb, age_lr_emb, province_lr_emb, city_lr_emb, tagid_lr_emb]))
    lr_output = Linear(mode=0, seed=seed)(lr_emb_concat)
    # deep
    dnn_emb_concat = Flatten()(concat_func([gender_emb, age_emb, province_emb, city_emb, tagid_emb]))
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_emb_concat)
    deep_output = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(dnn_output)
    # fm
    fm_output = FM()(concat_func([gender_emb, age_emb, province_emb, city_emb, tagid_emb], axis=1))

    all_output = add_func([lr_output, fm_output, deep_output])
    output = PredictionLayer(task)(all_output)
    model = tf.keras.models.Model(
        inputs=[gender, age, province, city, tagid_history, tagid_weight_input, tagid_history_len], outputs=output)
    return model
