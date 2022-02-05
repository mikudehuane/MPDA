# -*- coding: utf-8 -*-
# @Time    : 2021/8/28 下午12:49
# @Author  : islander
# @File    : din.py
# @Software: PyCharm


from collections import OrderedDict

EMB_DIM = 9
DEFAULT_SEQ_LEN = 50
MAX_NUM_CATEGORIES = 6
USER_EMB_DIM = 192404
ITEM_EMB_DIM = 63002
CATEGORY_EMB_DIM = 1362
_FEA_CONFIG = {
    # 用户，0 被用作填充
    'user_id': {
        'shape': (1,),
        'category': 'emb_vec',
        'emb_shape': (USER_EMB_DIM, EMB_DIM),
    },
    # 待评价商品，0 被用作不合法数据填充
    'item_id': {
        'shape': (1,),
        'category': 'emb_tgt',
        'emb_shape': (ITEM_EMB_DIM, EMB_DIM)
    },
    # 待评价电影的类别，20 被用作不合法数据填充
    'category_ids': {
        'shape': (MAX_NUM_CATEGORIES,),
        'category': 'emb_tgt',
        'emb_shape': (CATEGORY_EMB_DIM, EMB_DIM),
        'emb_process': 'mean_skip_padding'
    },
    # 历史喜欢过的电影序列
    'item_id_seq': {
        'shape': (DEFAULT_SEQ_LEN, 1),
        'category': 'emb_seq',
        'emb_shape': (ITEM_EMB_DIM, EMB_DIM),
        'seq_name': 'pos_rating'
    },
    # 历史喜欢过的电影类别序列
    'category_ids_seq': {
        'shape': (DEFAULT_SEQ_LEN, MAX_NUM_CATEGORIES),
        'category': 'emb_seq',
        'emb_shape': (CATEGORY_EMB_DIM, EMB_DIM),
        'seq_name': 'pos_rating',
        'emb_process': 'mean_skip_padding'
    },
}


# data file columns should follow this order: label,*_fea_config_order
_fea_config_order = ('user_id', 'item_id', 'category_ids', 'item_id_seq', 'category_ids_seq')

# although python>=3.7 dict is ordered by default, to compat lower python version, use an OrderDict object here
FEA_CONFIG = OrderedDict()
for key in _fea_config_order:
    FEA_CONFIG[key] = _FEA_CONFIG[key]


"""Configuration of which inputs share the embedding

format: embedding_name -> (input1, input2, ...)
"""
SHARED_EMB_CONFIG = {
    'item_emb': ('item_id', 'item_id_seq'),
    'category_emb': ('category_ids', 'category_ids_seq')
}

if not isinstance(FEA_CONFIG, OrderedDict):
    raise AssertionError('FEA_CONFIG not an OrderedDict object')

