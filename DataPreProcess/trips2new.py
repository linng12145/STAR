from collections import Counter
import numpy as np
import pandas as pd
import os
import pickle
from constants import *


def trips_to_new(trips, grids_AIS_EAST):
    # 将每个部分转换为列表
    trips_new_list = []

    # trips_new = pd.DataFrame(columns=['trips_new'])

    # 创建一个空字典用于存储数据和对应的自然数
    center = {}
    # 创建一个空字典用于存储新pickle
    grids2center = {}
    # 创建一个计数器，用于生成连续的自然数
    counter = 0

    print(f"trip length{len(trips)}")
    i = 0

    for trip in trips:
        if i % 10000 == 0:
            print("{}/{}".format(i, len(trips)))
        i += 1

        trip_new = []
        # 按照逗号分割每一部分，并转换为相应的数据类型
        trip_point = trip.split(';')

        # 提取第一列
        first_column = [part.split(',')[0] for part in trip_point]


        # if len(set(first_column)) < trip_grid_set_min * len(first_column):
        #     trips_new_list.append('0')
        #     continue


        # if Counter(first_column).most_common(1)[0][1] >= trip_grid_max * len(first_column):
        #     trips_new_list.append('0')
        #     continue


        # 是否跳过字典转换
        # skip = 0
        # skip = any(int(number) not in grids_AIS_EAST for number in first_column)

        for loc in trip_point:
            idx, lon, lat, cog, sog, time = loc.split(',')

            # print(idx)

            # if skip == 1:
            #     # 添加到结果DataFrame
            #     trip_new = '0'
            #     break

            if idx not in center:
                center[idx] = counter
                new_idx = counter
                grids2center[counter] = grids_AIS_EAST[int(idx)]
                counter += 1

            else:
                new_idx = center[idx]

            str_data = f"{new_idx},{lon},{lat}, {cog}, {sog}, {time}"

            # 将每组数据合并，并用分号分隔
            trip_new.append(str_data)

        # 合并每组数据，并用分号分隔
        combined_str = ';'.join(trip_new)

        trips_new_list.append(combined_str)

        # 添加到结果DataFrame
        # trips_new = pd.concat([trips_new, pd.DataFrame({'trips_new': [combined_str]})], ignore_index=True)

    # 将结果列表转换为 DataFrame
    trips_new_df = pd.DataFrame({'trips_new': trips_new_list})

    return trips_new_df['trips_new'], grids2center



def save_file(df, output_path, new_filename):
    # 保存处理后的数据集
    output_path = os.path.join(output_path, new_filename)
    df.to_csv(output_path, index=False)

def trips2new(data_format, data_name):
    data_path = os.path.join('../data', 'AIS', data_name)
    if data_format == 'csv':
        df = pd.read_csv(os.path.join(data_path, 'trips_cleaned_'+ data_name +'.csv'))
        print('trips2new read finish')

        # 读取
        grids_AIS_EAST = pickle.load(open(os.path.join(data_path, 'grids_'+ data_name +'.pickle'), 'rb'))

        # 添加一个新的列 'trips_new'
        df['trips_new'], grids2cneter = trips_to_new(df['trips'], grids_AIS_EAST)

        print("len(grids2cneter):{}".format(len(grids2cneter)))

        # 保存到pickle文件
        pickle.dump(grids2cneter, open(os.path.join(data_path, 'grid2center_'+ data_name +'.pickle'), 'wb'))
        # 读取
        grid2center_AIS_EAST = pickle.load(open(os.path.join(data_path, 'grid2center_'+ data_name +'.pickle'), 'rb'))
        open(os.path.join(data_path, 'grid2center_'+ data_name +'.txt'), 'w').write(f"{grid2center_AIS_EAST}\n")
        print("len(df):{}".format(len(df)))
        print('create new dict finish')

        save_file(df, data_path, 'trips_new_cleaned_'+ data_name +'.csv')

        print('trips new finish')

        print('finish')

# trips2new('csv', 'AIS_z')