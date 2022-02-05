# -*- coding: utf-8 -*-
# @Time    : 2021/8/6 下午4:44
# @Author  : islander
# @File    : select.py
# @Software: PyCharm
import log
import model


def get_metric_values(net, input_fn, metric_names, graph_to_eval='train'):
    """获取评估的结果（计算出来的 metric 值）

    Notes: 评估前会进行如下初始化操作：
        - 重置 input_fn 的 reader 指针
        - 变更为评估图并从 graph_to_eval 加载参数
    """
    input_fn.reader.seek(0)
    net.switch_graph('eval')
    net.load_from(graph_to_eval)

    eval_res = model.evaluate_by_net(net, input_fn=input_fn)
    metric_values = log.metrics.get_metrics(eval_res, metrics=metric_names)
    return metric_values
