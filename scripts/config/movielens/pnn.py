# -*- coding: utf-8 -*-
# @Time    : 2021/7/15 下午4:12
# @Author  : islander
# @File    : movielens.py
# @Software: PyCharm


from collections import OrderedDict

EMB_DIM = 18
DEFAULT_SEQ_LEN = 100
MAX_NUM_CATEGORIES = 10
_FEA_CONFIG = {
    # 用户，0 被用作填充
    'user_id': {
        'shape': (1,),
        'category': 'emb_vec',
        'emb_shape': (138494, EMB_DIM),
    },
    # 待评价电影，0 被用作不合法数据填充
    'movie_id': {
        'shape': (1,),
        'category': 'emb_tgt',
        'emb_shape': (131263, EMB_DIM // 2)
    },
    # 待评价电影的类别，20 被用作不合法数据填充
    'category_ids': {
        'shape': (MAX_NUM_CATEGORIES,),
        'category': 'emb_tgt',
        'emb_shape': (21, EMB_DIM // 2),
        'default_val': 20,
        'emb_process': 'mean_skip_padding'
    },
    # 历史喜欢过的电影序列
    'movie_id_seq_pos': {
        'shape': (DEFAULT_SEQ_LEN, 1),
        'category': 'emb_seq',
        'emb_shape': (131263, EMB_DIM // 2),
        'seq_name': 'pos_rating'
    },
    # 历史喜欢过的电影类别序列
    'category_ids_seq_pos': {
        'shape': (DEFAULT_SEQ_LEN, MAX_NUM_CATEGORIES),
        'category': 'emb_seq',
        'emb_shape': (21, EMB_DIM // 2),
        'seq_name': 'pos_rating',
        'default_val': 20,
        'emb_process': 'mean_skip_padding'
    },
    # 历史不喜欢过的电影序列
    'movie_id_seq_neg': {
        'shape': (DEFAULT_SEQ_LEN, 1),
        'category': 'emb_seq',
        'emb_shape': (131263, EMB_DIM // 2),
        'seq_name': 'neg_rating'
    },
    # 历史不喜欢过的电影类别序列
    'category_ids_seq_neg': {
        'shape': (DEFAULT_SEQ_LEN, MAX_NUM_CATEGORIES),
        'category': 'emb_seq',
        'emb_shape': (21, EMB_DIM // 2),
        'seq_name': 'neg_rating',
        'default_val': 20,
        'emb_process': 'mean_skip_padding'
    },
}


# data file columns should follow this order: label,*_fea_config_order
_fea_config_order = ('user_id', 'movie_id', 'category_ids', 'movie_id_seq_pos', 'category_ids_seq_pos', 'movie_id_seq_neg', 'category_ids_seq_neg')

# although python>=3.7 dict is ordered by default, to compat lower python version, use an OrderDict object here
FEA_CONFIG = OrderedDict()
for key in _fea_config_order:
    FEA_CONFIG[key] = _FEA_CONFIG[key]


"""Configuration of which inputs share the embedding

format: embedding_name -> (input1, input2, ...)
"""
SHARED_EMB_CONFIG = {
    'movie_emb': ('movie_id', 'movie_id_seq_neg', 'movie_id_seq_pos'),
    'category_emb': ('category_ids', 'category_ids_seq_pos', 'category_ids_seq_neg')
}

if not isinstance(FEA_CONFIG, OrderedDict):
    raise AssertionError('FEA_CONFIG not an OrderedDict object')
