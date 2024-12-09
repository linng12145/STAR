import pandas as pd
import numpy as np
import os
import random


# 添加一个新的列 固定删除点的比率为0.1
def drop_ratio_01(df):
    return np.full(len(df), 0.1)

# 添加一个新的列 固定删除点的比率为0.2
def drop_ratio_02(df):
    return np.full(len(df), 0.2)


# 添加一个新的列 删除点的比率
def drop_ratio(df):
    choices = [0.2, 0.3, 0.4, 0.5, 0.6]
    probabilities = [1 / len(choices)] * len(choices)  # 每个选项被选中的概率相等

    return np.random.choice(choices, size=len(df), p=probabilities)


# 添加一个新的列 删除点的段数
def tagging_num(df):
    choices = [1, 2, 3, 4]
    probabilities = [1 / len(choices)] * len(choices)  # 每个选项被选中的概率相等

    return np.random.choice(choices, size=len(df), p=probabilities)


# 确定删除点的个数
def delete_num_exact_division(tagging_num, delete_num):
    # 查找不大于delete_num，且能被tagging_num或tagging_num+1整除的最大的数
    for i in range(delete_num, 0, -1):
        if i % tagging_num == 0 or i % (tagging_num + 1) == 0:
            return i


# 设置tagging_labels列中的删除位置
def set_values_to_one(tagging_label, random_numbers):
    # 遍历随机下标列表
    for index in random_numbers:
        # 将对应位置的值设置为 1
        tagging_label[index] = 1

    return tagging_label


# 设置num_labels列中的删除位置
def set_values_to_delete(delete_num, tagging_num, num_label, random_numbers):
    # 确定删除点在各段的分配方式
    if delete_num % tagging_num == 0 and delete_num % (tagging_num + 1) == 0:
        choices = [tagging_num, tagging_num + 1]
        probabilities = [1 / len(choices)] * len(choices)  # 每个选项被选中的概率相等
        delete_denominator = np.random.choice(choices, p=probabilities)
    elif delete_num % tagging_num == 0:
        delete_denominator = tagging_num
    else:
        delete_denominator = tagging_num + 1

    # 每份分配删点数量
    every_delete = int(delete_num / delete_denominator)

    # print(random_numbers)
    # print(delete_num)
    # print(delete_denominator)
    # print(every_delete)
    # print(num_label)

    # 遍历随机下标列表
    for index in random_numbers:
        # 将对应位置的值设置为 1
        num_label[index] = every_delete

    if delete_denominator == tagging_num + 1:
        # 从 random_numbers 中随机选择一个数字
        selected_number = random.choice(random_numbers)
        num_label[selected_number] += every_delete

    return num_label


# 生成tagging_labels列
def tagging_labels(df):
    tagging_labels = []
    delete_nums = []
    num_labels = []
    for index, trip in df.iterrows():
        trip_length = trip['trip_length']
        drop_ratio = trip['drop_ratio']
        tagging_num = trip['tagging_num']
        delete_num = int(trip_length * drop_ratio)
        # print(trip_length, drop_ratio, tagging_num, delete_num)
        # print(delete_num)

        if delete_num <= tagging_num:
            df.at[index, 'tagging_num'] = delete_num
            tagging_num = delete_num

        delete_num = delete_num_exact_division(tagging_num, delete_num)

        delete_nums.append(delete_num)

        # 生成单个列表
        rest_num = trip_length - delete_num
        tagging_label = [0] * rest_num

        random_numbers = random.sample(range(0, rest_num - 1), tagging_num)
        tagging_label = set_values_to_one(tagging_label, random_numbers)
        tagging_labels.append(tagging_label)

        num_label = set_values_to_delete(delete_num, tagging_num, tagging_label, random_numbers)
        num_labels.append(num_label)
    # print(tagging_labels)

    return delete_nums, tagging_labels, num_labels


def dataset_sparse(trips, num_labels):
    # 将每个部分转换为列表
    trips_sparse = []
    for trip, num_label in zip(trips, num_labels):
        trip_sparse = []
        # 按照逗号分割每一部分，并转换为相应的数据类型
        trip_point = trip.split(';')
        num_label_pos = 0
        skip_counter = 0
        for loc in trip_point:
            if skip_counter > 0:
                skip_counter -= 1
                continue
            # print(trips)
            # print("trips{}".format(len(trip)))
            # print("num_label_pos:{} num_label:{}".format(num_label_pos, num_label))
            if num_label[num_label_pos] != 0:
                # loc跳过num_label_point个
                skip_counter = num_label[num_label_pos]

            num_label_pos += 1

            idx, lon, lat, cog, sog, time = loc.split(',')
            trip_sparse.append([int(idx), float(lon), float(lat), float(cog), float(sog), int(time)])
        trips_sparse.append(trip_sparse)
    return trips_sparse


def delete_grid_trip_new(df):
    # mask = (df['trips_new'] == '0')
    # df.drop(df[mask].index, inplace=True)
    df = df[df['trips_new'] != '0']
    return df


def save_file(df, output_path, new_filename):
    # 保存处理后的数据集
    output_path = os.path.join(output_path, new_filename)
    df.to_csv(output_path, index=False)


def trips_drop(data_format, data_name):
    data_path = os.path.join('../data', 'AIS', data_name)
    if data_format == 'csv':
        df = pd.read_csv(os.path.join(data_path, 'trips_new_cleaned_'+ data_name +'.csv'))
        print('trips_drop read finish')

        # 删除trip_new中网格不存在的轨迹
        df = delete_grid_trip_new(df)
        print("trips zero delete finish")

        df['drop_ratio'] = drop_ratio_01(df)
        print('trips drop ratio finish')

        df['tagging_num'] = tagging_num(df)
        print('trips num_labels finish')

        df['delete_nums'], df['tagging_labels'], df['num_labels'] = tagging_labels(df)
        print('trips tagging labels finish')

        df['trips_sparse'] = dataset_sparse(df['trips_new'], df['num_labels'])
        print('trips sparse labels finish')

        save_file(df, data_path, 'trips_drop_cleaned_'+ data_name +'.csv')
        print('finish')

# trips_drop('csv', 'AIS_z')
