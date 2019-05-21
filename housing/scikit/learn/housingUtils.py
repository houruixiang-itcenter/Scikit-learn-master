#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/4 下午9:41
# @Author  : Aries
# @Site    : 
# @File    : housingUtils.py
# @Software: PyCharm
import logging
import os
import tarfile

# from six.moves import urllib
# from sklearn.externals.six.moves.urllib import request

import pandas as pd
from six.moves import urllib
from sklearn.preprocessing import LabelBinarizer

DOWNLOAD_ROOT = "https://raw.githubusercontent.housing/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


# todo 获取数据
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# todo 加载数据
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# todo 创建测试集   --- 通常情况下是数据集的20%   然后将其放置导一边
import numpy as np


# todo params  1.mldata 数据集   2.test_ratio 测试比率  需要注意的是 这里的20-80对应的数据集是随机的
def spilt_train_test(data, test_ratio):
    # 随机返回一个数列  eg:参数为10  就随机排列10以内的自然数 然后返回
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    # todo 这里的data.iloc就是数据的截断  即20%是测试数据  80%是训练数据
    # todo 需要区分的是  当索引是String类型时候 使用data.loc  反之使用 mldata.iloc
    # todo train_indices/test_indices  是一个索引的数组  通过索引
    return data.iloc[train_indices], data.iloc[test_indices]


# todo  spilt_train_test这个function  是每次生成随机索引然后进行处理 那么我们面临一个问题   每次训练集和测试集虽然len是固定的 但是数据不具备唯一性
# todo 解决方式1: np.random.permutation 在生成随机索引之前 先np.random.seed(42) 这样可以控制随机的索引一致
# todo 解决方式2: 就是第一次运行完后保存测试集和训练集  这样下次运行直接加载即可
# todo 但是方式1和2  有一个弊端 那就是更新新数据时候 会中断  这样还是要数据大洗牌  然后进行重新分配训练集和测试集
# todo 那么问题来了我们该如何处理呢,就是通过hash值来做判断  每一个实例对应一个hash的唯一标识
import hashlib


def test_set_check(identifier, test_radio, hash):
    logging.info('id_', identifier)
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_radio


def unique_spilt_train_test_by(data, test_radio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    # 测试数据获取 --- from function-test_set_check
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_radio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


