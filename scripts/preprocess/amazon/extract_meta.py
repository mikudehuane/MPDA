# -*- coding: utf-8 -*-
# @Time    : 2021/8/28 下午12:10
# @Author  : islander
# @File    : extract_meta.py
# @Software: PyCharm
import csv

import _init_paths2

import json

from tensorflow.python.platform import gfile

import project_path
import os.path as osp
import gutils


def main():
    data_fd = osp.join(project_path.data_fd, 'amazon', 'Electronics_5')

    output_mapping_fp = osp.join(data_fd, 'processed', 'item2category.csv')
    output_category_fp = osp.join(data_fd, 'processed', 'category.csv')
    input_fp = osp.join(data_fd, 'meta.txt')
    item_mapping_fp = osp.join(data_fd, 'id_map.json')

    # noinspection PyTypeChecker
    item_id_dict = json.load(gfile.GFile(item_mapping_fp))
    item_id_dict = item_id_dict['item']

    category_id_dict = dict()

    with gfile.GFile(input_fp) as input_f:
        with gfile.GFile(output_mapping_fp, 'w') as output_f:
            output_f = csv.writer(output_f)

            for meta_idx, meta in enumerate(input_f):
                meta = json.loads(meta)
                item_id = meta['asin']
                if item_id in item_id_dict:  # 跳过没有映射的商品，因为没有出现在数据集
                    item_id = item_id_dict[item_id]
                    categories = meta['category']

                    for category_idx, category in enumerate(categories):
                        category_id = gutils.get_id_and_update_dict(category_id_dict, category)
                        categories[category_idx] = str(category_id)

                    output_f.writerow([item_id, '|'.join(categories)])

                if meta_idx % 1000 == 0:
                    print('{} rows processed'.format((meta_idx + 1)))

    with gfile.GFile(output_category_fp, 'w') as output_f:
        output_f = csv.writer(output_f)
        for category, category_id in category_id_dict.items():
            output_f.writerow([category_id, category])


if __name__ == '__main__':
    main()
