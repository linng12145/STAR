import pandas as pd
import os
from pyproj import Transformer
from constants import *
from geopy.distance import geodesic, distance
import math


def trip(df):
    # 初始化一个变量用于存储递增的 count
    point_count = 0
    count = 0

    print("len(df):{}".format(len(df)))
    progress= 0

    # 用于存储所有的 count 结果
    count_results = []

    # 对每个 mmsi 分组
    for mmsi, group in df.groupby('MMSI'):
        # 初始化一个变量用于存储上一个 BaseDateTime
        last_base_date_time = None
        # print(df.head())

        for index, point in group.iterrows():
            # 获取当前组的第一个 BaseDateTime
            base_date_time = point['BaseDateTime']

            if progress % 50000 == 0:
                print("{} / {}".format(progress, len(df)))
            progress = progress + 1

            # 如果这是第一次遇到这个 mmsi，或者 BaseDateTime 大于 200，重置 incremental_count
            if last_base_date_time is None :
                last_base_date_time = base_date_time
                # df.at[index, 'COUNT'] = count
                count_results.append((index, count))
                continue

            if (base_date_time - last_base_date_time) >= time_max or (base_date_time - last_base_date_time) <= time_min or point_count >= trip_count_max :
                count += 1
                point_count = 0

            # 更新 'COUNT' 列
            # df.at[index, 'COUNT'] = count

            # 存储 'COUNT' 值
            count_results.append((index, count))

            point_count += 1

            last_base_date_time = base_date_time

        count += 1

    # 将结果一次性写入 DataFrame
    df.loc[[idx for idx, _ in count_results], 'COUNT'] = [cnt for _, cnt in count_results]
    return df


def save_file(df, output_path, new_filename):
    # 保存处理后的数据集
    output_path = os.path.join(output_path, new_filename)
    df.to_csv(output_path, index=True)


def trip_count(data_format, data_name):
    data_path = os.path.join('../data', 'AIS', data_name)
    if data_format == 'csv':
        df = pd.read_csv(os.path.join(data_path, 'grid_delete_cleaned_'+ data_name +'.csv'))
        print('trip_count read finish')

        # 只保留部分
        df = df[['MMSI', 'BaseDateTime', 'LAT', 'LON', 'COG', 'SOG', 'GRID']]

        df['COUNT'] = -1
        # trip(df)
        df = trip(df)
        save_file(df, data_path, 'count_'+ data_name +'.csv')
        print('trip count finish')

        df = df.groupby('COUNT').filter(lambda x: x['COUNT'].count() >= trip_count_min)
        save_file(df, data_path, 'delete_count_'+ data_name +'.csv')
        print('delete trip count min finish')

        print('finish')

        # 显示 DataFrame 的前几行以确认数据是否正确加载
        # print(df.head())

# trip_count('csv', 'AIS_z')
