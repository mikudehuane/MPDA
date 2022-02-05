# -*- coding: utf-8 -*-
# @Time    : 2021/7/19 下午3:27
# @Author  : islander
# @File    : train_utils.py
# @Software: PyCharm

import os.path as osp


def get_estimator_config(args, checkpoint_fd):  # 获取 estimator 的 RunConfig
    import tensorflow as tf

    distribution_dict = {
        'ParameterServerStrategy': tf.contrib.distribute.ParameterServerStrategy(num_gpus_per_worker=1),
        'OneDeviceStrategy': tf.contrib.distribute.OneDeviceStrategy(device=args.device),
    }
    # 获取 estimator 的 RunConfig，具体细节请参考 tensorflow 的 doc
    distribution = distribution_dict[args.distribute]
    session_config = tf.ConfigProto(allow_soft_placement=True,  # 允许自行判断放置的设备
                                    # log_device_placement=True,  # 打印每个变量所在的设备到日志
                                    gpu_options=tf.GPUOptions(allow_growth=True))  # 若不设定该参数，会占用全部显存
    estimator_config = tf.estimator.RunConfig(
        model_dir=checkpoint_fd,
        tf_random_seed=args.tf_random_seed,
        save_summary_steps=args.save_summary_steps,
        save_checkpoints_steps=args.save_checkpoint_steps,
        save_checkpoints_secs=args.save_checkpoint_secs,
        keep_checkpoint_max=args.keep_checkpoint_max,
        keep_checkpoint_every_n_hours=args.keep_checkpoint_every_n_hours,
        log_step_count_steps=args.log_step_count_steps,
        session_config=session_config,
        train_distribute=distribution,
        eval_distribute=distribution,
    )
    return estimator_config


def get_amazon_input_fn(data_fd, mapping_fp, *,
                        seed_plus,
                        require='dataset', fea_config, slice_index=0, slice_count=1,
                        shuffle=False, shuffle_cache_size=10000, batch_size=32, log_responsible_fns=None):
    import data
    from tensorflow.python.platform import gfile
    import model
    from gutils import get_array_slice

    if isinstance(mapping_fp, str):
        item2categories = data.load_category_mapping(mapping_fp)
    else:
        item2categories = mapping_fp

    # 取分片对应的文件
    if gfile.IsDirectory(data_fd):
        fns = gfile.ListDirectory(data_fd)
        fns = get_array_slice(fns, slice_count=slice_count)
        fns = fns[slice_index]
        if log_responsible_fns is not None:
            for fn in fns:
                log_responsible_fns.write(fn + '\n')
        user_ids = [int(fn.rsplit('.', 1)[0]) for fn in fns]
        data_reader = data.amazon.DataReaderMultiple(
            [osp.join(data_fd, fn) for fn in fns], kernel_reader_class=data.amazon.DataReaderNegSampling,
            kwargs_list=[{'seed': user_id + seed_plus} for user_id in user_ids],
            item2categories=item2categories, config=fea_config)
    else:  # 单个文件
        data_fp = data_fd
        data_fd, data_fn = osp.split(data_fd)
        user_id, ext = data_fn.rsplit('.', 1)
        user_id = int(user_id)
        assert slice_index == 0 and slice_count == 1
        data_reader = data.amazon.DataReaderNegSampling(data_fp, item2categories, config=fea_config, seed=user_id + seed_plus)

    if shuffle:
        data_reader = model.ShuffleReader(data_reader, cache_size=shuffle_cache_size)
    generator = model.din.DataGenerator(data_reader, config=fea_config, batch_size=batch_size)
    if require == 'dataset':
        input_fn = model.din.DataInputFn(generator, data_gen_kwargs={'reset': True})
        return input_fn
    else:
        return generator


def get_movielens_input_fn(data_fd, mapping_fp, *,
                           movie_genome_fp=None,
                           require='dataset', fea_config, slice_index=0, slice_count=1,
                           shuffle=False, shuffle_cache_size=10000, batch_size=32, log_responsible_fns=None):
    """获取 movielens 数据集对 din 模型的输入函数（用于 estimator）

    Args:
        movie_genome_fp: 电影 TAG 数据的的文件路径
        require: dataset（提供给 estimator）或者 generator（普通的生成器）
        log_responsible_fns: 记录负责的文件名列表的流输出对象
        slice_count: 把数据集切成 $() 片
        slice_index: 从数据集切片中取第 $() 片
        batch_size: 训练批大小
        fea_config: 特征配置字典
        mapping_fp (Union[str, Dict]): 电影类别映射文件的路径，或已经解析好的字典
        data_fd: 数据所在目录，或者直接给定数据文件的路径
        shuffle: 是否 shuffle 数据集
        shuffle_cache_size: shuffle 数据集时的缓存大小

    Returns:
        用于 estimator 的 InputFn 对象
    """
    import data
    from tensorflow.python.platform import gfile
    import model
    from gutils import get_array_slice

    if isinstance(mapping_fp, str):
        movie2categories = data.movielens.utils.load_category_mapping(mapping_fp)
    else:
        movie2categories = mapping_fp

    if movie_genome_fp is None:
        movie_genomes = None
    elif isinstance(movie_genome_fp, str):
        print('loading genomes')
        movie_genomes = data.movielens.utils.load_genome(movie_genome_fp)
        print('genomes loaded')
    else:  # 直接传入了加载好的字典对象
        movie_genomes = movie_genome_fp

    # 取分片对应的文件
    if gfile.IsDirectory(data_fd):
        fns = gfile.ListDirectory(data_fd)
        fns = get_array_slice(fns, slice_count=slice_count)
        fns = fns[slice_index]
        if log_responsible_fns is not None:
            for fn in fns:
                log_responsible_fns.write(fn + '\n')
            log_responsible_fns.flush()
        data_reader = data.movielens.DataReaderMultiple(
            [osp.join(data_fd, fn) for fn in fns], movie_genomes=movie_genomes,
            movie2categories=movie2categories, config=fea_config)
    else:  # 单个文件
        assert slice_index == 0 and slice_count == 1
        data_reader = data.movielens.DataReader(data_fd, movie2categories, config=fea_config,
                                                movie_genomes=movie_genomes)

    if shuffle:
        data_reader = model.ShuffleReader(data_reader, cache_size=shuffle_cache_size)
    generator = model.din.DataGenerator(data_reader, config=fea_config, batch_size=batch_size)
    if require == 'dataset':
        input_fn = model.din.DataInputFn(generator, data_gen_kwargs={'reset': True})
        return input_fn
    else:
        return generator


get_movielens_din_input_fn = get_movielens_input_fn
