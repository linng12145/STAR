import numpy as np
import pandas as pd
import os
from pyproj import Transformer
import pickle
from constants import *
from joblib import Parallel, delayed
import math


def compute_distances(group, group_name):
    last_point = None
    delete_index = []

    for idx, row in group.iterrows():
        if last_point is None:
            last_point = (row['LAT'], row['LON'])
            continue

        point = (row['LAT'], row['LON'])
        dis = math.sqrt((point[0] - last_point[0]) ** 2 + (point[1] - last_point[1]) ** 2)

        if dis <= distance_min:
            delete_index.append(idx - 1)

        last_point = point

    # print(f"Processed group: {group_name}")
    return delete_index

def trips_diff(df):
    print(f"trip length: {len(df)}")

    grouped = df.groupby('MMSI')

    # 并行计算各组的删除索引
    delete_indices = Parallel(n_jobs=-1)(
        delayed(compute_distances)(group, mmsi) for mmsi, group in grouped
    )

    # 将所有删除索引合并为一个列表
    delete_indices = [idx for sublist in delete_indices for idx in sublist]

    # 删除指定索引的行
    df.drop(index=delete_indices, inplace=True)

    return df


def wgs84_to_utm(northing, easting):
    transformer = Transformer.from_crs("epsg:4326", "epsg:4575")
    # transformer = Transformer.from_crs("epsg:4326", "epsg:3086")
    lat, lon = transformer.transform(northing, easting)
    return lat, lon


# 给轨迹点加上网格号
def trip_grids(df):

    print("df length:{}".format(len(df)))

    lat_min = df['LAT'].min()
    lon_min = df['LON'].min()
    lat_max = df['LAT'].max()
    lon_max = df['LON'].max()

    lat_count = math.ceil((lat_max - lat_min) / grid_side)
    lon_count = math.ceil((lon_max - lon_min) / grid_side)

    # print("lat_min:{}, lon_min:{}, lat_max:{}, lon_max:{}".format(lat_min, lon_min, lat_max, lon_max))

    print("lat_count:{}, lon_count:{}".format(lat_count, lon_count))

    # 计算纬度和经度方向上的索引
    lat_indices = np.floor_divide((df['LAT'] - lat_min).values, grid_side).astype(int)
    lon_indices = np.floor_divide((df['LON'] - lon_min).values, grid_side).astype(int)

    # 计算网格ID
    grid_ids = lon_indices + lon_count * lat_indices

    df['GRID'] = grid_ids

    # 创建满足条件的网格字典
    grids_dict = {}
    unique, counts = np.unique(grid_ids, return_counts=True)
    for grid_id, count in zip(unique, counts):
        if count < grid_weight_min:
            continue

        lat_index = grid_id // lon_count
        lon_index = grid_id % lon_count

        lat_min_val = lat_min + lat_index * grid_side
        lat_max_val = lat_min_val + grid_side
        lon_min_val = lon_min + lon_index * grid_side
        lon_max_val = lon_min_val + grid_side
        center_lat = (lat_min_val + lat_max_val) / 2
        center_lon = (lon_min_val + lon_max_val) / 2

        grids_dict[grid_id] = (center_lon, center_lat)

    return df, grids_dict


# 创建字典
def create_dict(grids):
    grids_dict = {}

    for i in range(len(grids)):
        if grids[i]['weight'] < grid_weight_min:
            continue

        grid_id = i
        center_lon = grids[i]['center_lon']
        center_lat = grids[i]['center_lat']
        grids_dict[grid_id] = (center_lon, center_lat)


    # for i, grid in enumerate(grids):
    #     print(f"Index: {i}, Grid: {grid}")
    #     break
    #
    # # 使用字典推导式来加速字典创建
    # grids_dict = {
    #     i: (grid['center_lon'], grid['center_lat'])
    #     for i, grid in enumerate(grids)
    #     if grid['weight'] >= grid_weight_min
    # }

    return grids_dict


def save_file(df, output_path, new_filename):
    # 保存处理后的数据集
    output_path = os.path.join(output_path, new_filename)
    df.to_csv(output_path, index=True)


def meshing(data_format, data_name):
    data_path = os.path.join('../data', 'AIS', data_name)
    if data_format == 'csv':
        df = pd.read_csv(os.path.join(data_path, 'cleaned_'+ data_name +'.csv'))
        print('meshing read finish')

        # print("len(df:{})".format(len(df)))
        # df = df[df['Status'] != 1]
        # print("删除状态为1的轨迹点 end\n")
        # print("len(df:{})".format(len(df)))

        # 只保留部分
        df = df[['MMSI', 'BaseDateTime', 'LAT', 'LON', 'COG', 'SOG']]

        # 将 AIS 数据按 MMSI 和时间升幂排序
        df.sort_values(by=['MMSI', 'BaseDateTime'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        print("将 AIS 数据按 MMSI 和时间升幂排序 end\n")

        # 将时间字符串转换为 datetime 对象
        # 将 datetime 对象转换为 Unix 时间戳（秒）
        df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
        df['BaseDateTime'] = df['BaseDateTime'].apply(lambda x: x.timestamp())

        # 将 经纬度 对象转换 （米）
        df['LAT'], df['LON'] = wgs84_to_utm(df['LAT'], df['LON'])
        print('data trans finish')

        trips_diff(df)
        save_file(df, data_path, 'diff_dis_' + data_name + '.csv')
        print('delete distance finish')
        print("len(df):{}".format(len(df)))

        # 给轨迹点加上网格号 同时创建字典
        df, grids_dict = trip_grids(df)

        # 保存到pickle文件
        pickle.dump(grids_dict, open(os.path.join(data_path, 'grids_'+ data_name +'.pickle'), 'wb'))
        # 读取
        grids_AIS_EAST = pickle.load(open(os.path.join(data_path, 'grids_'+ data_name +'.pickle'), 'rb'))
        open(os.path.join(data_path, 'grids_'+ data_name +'.txt'), 'w').write(f"{grids_AIS_EAST}\n")
        print('create dict finish')

        print("len(grids_dict:{})".format(len(grids_dict)))
        print("len(df):{}".format(len(df)))

        df = df[df['GRID'].isin(grids_dict)]

        print("len(df:{})".format(len(df)))
        print('delete not in gird finish')
        save_file(df, data_path, 'grid_delete_cleaned_' + data_name + '.csv')

        print('finish')

        # 显示 DataFrame 的前几行以确认数据是否正确加载
        # print(df.head())


# meshing('csv', 'AIS_z')


