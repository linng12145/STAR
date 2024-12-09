import os
import pandas as pd

import numpy as np
from collections import Counter
import math
import pickle

from datetime import datetime
from geopy.distance import geodesic

# 相似判定中的分组mmsi数量
mmsi_count_has_sim = 0
mmsi_count = 0

# 通过经纬度计算出连续两点间的实际距离𝐷𝑖𝑠
# 其中𝑅代表地球半径，(𝑥1 ,𝑦1 )和(𝑥2 ,𝑦2 )分别代表两个点之间的经纬度坐标
# 𝐷𝑖𝑠 = 𝑅 × 2000 × arcsin√𝐷，其中D的公式为
# 𝐷 = {sin[0.5 × (𝑥2 − 𝑥1)]}^2 + cos𝑥1 × cos𝑥2 × {sin[0.5 × (𝑦2 − 𝑦1)]}^2
# 单位：米
def calculate_distance(data):
    # print(data)
    x1 = np.radians(data['LAT'].shift(1))
    y1 = np.radians(data['LON'].shift(1))
    x2 = np.radians(data['LAT'])
    y2 = np.radians(data['LON'])
    earth_r = 6371.393
    d = (np.sin(((x2 - x1) / 2))) ** 2 + np.cos(x1) * np.cos(x2) * ((np.sin(((y2 - y1) / 2))) ** 2)
    distance = 2000 * earth_r * np.arcsin(np.sqrt(d))

    # 两个点之间的理论距离
    time = pd.to_datetime(data['BaseDateTime'], format='%Y-%m-%dT%H:%M:%S').diff().dt.total_seconds()
    max_distance = time * 51.2 * 1852 / 3600

    data['Distance'] = distance.fillna(0)
    data['MaxDistance'] = max_distance.fillna(0)

    return data


# 数值型数据的相似判断
def similar_number(num1, num2):
    if pd.isna(num1) and pd.isna(num2):
        return 1
    if pd.isna(num1) or pd.isna(num2):
        return 0
    if max(num1, num2) == 0:
        if num1 == num2:
            return 1
        else:
            return 0
    sim = 1 - abs((num1 - num2) / max(num1, num2))
    return sim


# 字符型数据的相似判断
# 余弦相似度算法
def similar_string(str1, str2):
    if pd.isna(str1) and pd.isna(str2):
        return 1
    if pd.isna(str1) or pd.isna(str2):
        return 0
    vector1 = Counter(list(str1))
    vector2 = Counter(list(str2))

    shared = set(vector1.keys()) & set(vector2.keys())
    numerator = sum([vector1[x] * vector2[x] for x in shared])

    sum1 = sum([vector1[x] ** 2 for x in vector1.keys()])
    sum2 = sum([vector2[x] ** 2 for x in vector2.keys()])
    denominator = math.sqrt(sum1 * sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


# 布尔型数据的相似判断
# 如船舶的 AIS 设备种类表示为 A 类和 B 类，为了方便计算，将 A 类记为 0，B 类记为 1
def similar_bool(bool1, bool2):
    if pd.isna(bool1) and pd.isna(bool2):
        return 1
    if pd.isna(bool1) or pd.isna(bool2):
        return 0
    return bool1 == bool2


# 数据的相似判断
def similar(wei, row1, row2):
    # sim_mmsi = similar_number(row1['MMSI'], row2['MMSI'])
    sim_time = similar_string(row1['BaseDateTime'], row2['BaseDateTime'])
    sim_lat = similar_number(row1['LAT'], row2['LAT'])
    sim_lon = similar_number(row1['LON'], row2['LON'])
    sim_sog = similar_number(row1['SOG'], row2['SOG'])
    sim_cog = similar_number(row1['COG'], row2['COG'])
    sim_heading = similar_number(row1['Heading'], row2['Heading'])
    sim_vessel_name = similar_string(row1['VesselName'], row2['VesselName'])
    sim_imo = similar_string(row1['IMO'], row2['IMO'])
    sim_call_sign = similar_string(row1['CallSign'], row2['CallSign'])
    sim_vessel_type = similar_number(row1['VesselType'], row2['VesselType'])
    sim_status = similar_number(row1['Status'], row2['Status'])
    sim_length = similar_number(row1['Length'], row2['Length'])
    sim_width = similar_number(row1['Width'], row2['Width'])
    sim_draft = similar_number(row1['Draft'], row2['Draft'])
    sim_cargo = similar_number(row1['Cargo'], row2['Cargo'])
    sim_transceiver_class = similar_bool(row1['TransceiverClass'], row2['TransceiverClass'])
    sim = np.array(
        [
            sim_time,
            sim_lat, sim_lon, sim_sog, sim_cog, sim_heading, sim_vessel_name, sim_imo, sim_call_sign,
         sim_vessel_type, sim_status, sim_length, sim_width, sim_draft, sim_cargo, sim_transceiver_class])
    # 带权重的相似度
    wei_sim = np.sum(sim * wei)
    is_sim = wei_sim > 0.95
    # print(wei_sim)
    # print(is_sim)
    return is_sim
    # print(
    #     f"sim_MMSI\n{sim_mmsi}\n"
    #     f"sim_time\n{sim_time}\n"
    #     f"sim_lat\n{sim_lat}\n"
    #     f"sim_lon\n{sim_lon}\n"
    #     f"sim_sog\n{sim_sog}\n"
    #     f"sim_cog\n{sim_cog}\n"
    #     f"sim_heading\n{sim_heading}\n"
    #     f"sim_vessel_name\n{sim_vessel_name}\n"
    #     f"sim_imo\n{sim_imo}\n"
    #     f"sim_call_sign\n{sim_call_sign}\n"
    #     f"sim_vessel_type\n{sim_vessel_type}\n"
    #     f"sim_status\n{sim_status}\n"
    #     f"sim_length\n{sim_length}\n"
    #     f"sim_width\n{sim_width}\n"
    #     f"sim_draft\n{sim_draft}\n"
    #     f"sim_cargo\n{sim_cargo}\n"
    #     f"sim_transceiver_class\n{sim_transceiver_class}\n"
    # )


# 改进的动态滑动窗口策略
def dynamic_window(data, wei, initial_window_size, threshold):
    global mmsi_count_has_sim, mmsi_count
    mmsi_count_has_sim += 1
    print(f"{mmsi_count_has_sim}/{mmsi_count}")
    window_size = initial_window_size
    counter = 0
    repetition_number = []
    for begin in range(0, len(data)):
        # print(f"repetition\n{repetition_number}\n")
        # print(f"begin\n{begin}\n")
        if begin in repetition_number:
            continue
        compared = 0
        window_begin = begin
        now = window_begin + 1
        while compared < window_size:
            if now >= len(data):
                break
            # 窗口尺寸扩大
            # 将窗口内数据分别与窗口第一个数据进行相似度计算，
            # 当检测到数据𝑊(𝑘)与数据𝑊(𝑖)相似时，其中𝑖 < 𝑘 ≤ 𝑗，对窗口进行扩大
            if similar(wei, data.iloc[window_begin], data.iloc[now]):
                window_change = window_size - 1
                window_size = now - window_begin + 1 + window_change
                repetition_number.append(now)
                data.loc[data.index[now], 'ISSIMILAR'] = 1
                # print(data)

                # 窗口尺寸缩小
                # 当检测到一个不重复数据，则𝑐𝑜𝑢𝑛𝑡𝑒𝑟的值加一，当𝑐𝑜𝑢𝑛𝑡𝑒𝑟超过阈值，提前结束窗口，并缩小窗口
            else:
                counter += 1
                if counter > threshold:
                    window_change = window_size - counter - 1
                    window_size = now - window_begin + 1 - window_change
                    counter = 0
                    break
            compared += 1
            now += 1
    return data

def save_file(df, output_path, new_filename):
    # 保存处理后的数据集
    output_path = os.path.join(output_path, new_filename)
    df.to_csv(output_path, index=False)

# 遍历文件夹中的所有CSV文件
# for filename in os.listdir(input_folder):
#     if filename.endswith('.csv'):

def data_clean(df):

    global mmsi_count_has_sim, mmsi_count
    # 相似判定中的分组mmsi数量
    mmsi_count_has_sim = 0

    # Load the dataset

    # print(df.head())

    # 1. 将 AIS 数据按 MMSI 和时间升幂排序
    df.sort_values(by=['MMSI', 'BaseDateTime'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("1 将 AIS 数据按 MMSI 和时间升幂排序 end")
    print("len(df):{}\n".format(len(df)))
    # print(df)

    # 2. 删除 MMSI 不为 9 位的数据
    df = df[df['MMSI'].apply(lambda x: len(str(x)) == 9)]
    print("2 删除 MMSI 不为 9 位的数据 end")
    print("len(df):{}\n".format(len(df)))

    # 3. 删除MMSI相同、IMO不同的情况，一般为套牌船
    # print(df)
    df = df.groupby('MMSI').filter(lambda x: x['IMO'].nunique() <= 1)
    # print(df)
    print("3 删除MMSI相同、IMO不同的情况，一般为套牌船 end")
    print("len(df):{}\n".format(len(df)))

    # 4. 删除一天内的AIS数据不足50条的轨迹
    # print(df)
    df = df.groupby('MMSI').filter(lambda x: x['MMSI'].count() > 50)
    # print(df)
    print("4 删除一天内的AIS数据不足1050条的轨迹 end")
    print("len(df):{}\n".format(len(df)))

    # 5. 删除状态为1的轨迹点
    df = df[df['Status'] != 1]
    print("5 删除状态为1的轨迹点 end")
    print("len(df):{}\n".format(len(df)))

    # 6. 删除船长小于 3 和船宽小于 2 的船舶数据
    df = df[(df['Length'] >= 3) & (df['Width'] >= 2)]
    print("6 删除船长小于 3 和船宽小于 2 的船舶数据 end")
    print("len(df):{}\n".format(len(df)))

    # 7. 删除超出有效范围的经度、维度、对地航速、对地航向数据
    df = df[(df['LON'] >= -180.0) & (df['LON'] <= 180.0)]
    df = df[(df['LAT'] >= -90.0) & (df['LAT'] <= 90.0)]
    df = df[(df['SOG'] > 0) & (df['SOG'] <= 24)]
    df = df[(df['COG'] >= 0) & (df['COG'] <= 409.6)]
    print("7 删除超出有效范围的经度、维度、对地航速、对地航向数据 end")
    print("len(df):{}\n".format(len(df)))

    # 8. 删除经纬度明显漂移的数据
    df = df.groupby('MMSI').apply(calculate_distance, include_groups=False).reset_index(level=0)
    # print(df)

    # 排除与前后两个点之间实际距离均大于理论距离的点
    distance_shift = df['Distance'].shift(-1).fillna(0)
    max_distance_shift = df['MaxDistance'].shift(-1).fillna(0)
    df = df[(df['Distance'] <= df['MaxDistance']) | (distance_shift <= max_distance_shift)]

    # df.drop(columns=['Distance', 'MaxDistance'], inplace=True)
    # print(df)
    print("8 删除经纬度明显漂移的数据 end")
    print("before df len:{}\n".format(len(df)))

    # # 9. 删除相似重复的数据
    # # 求去掉mmsi的各列权重
    # # df_without_mmsi = df.drop(columns=['MMSI', 'BaseDateTime', 'Distance', 'MaxDistance'], axis=1)
    # df_without_mmsi = df.drop(columns=['MMSI', 'Distance', 'MaxDistance'], axis=1)
    #
    # # print(df_without_mmsi)
    # num_unique = df_without_mmsi.nunique()
    # total_unique = num_unique.sum()
    # weight = num_unique / total_unique
    # print(f"种类数量:\n {num_unique}\n种类总数:\n{total_unique}\n权重:\n{weight}")
    # mmsi_count = df['MMSI'].nunique()
    # print('mmsi_count: {}'.format(df['MMSI'].nunique()))
    # # print(df)
    # # 删除相似数据
    # df['ISSIMILAR'] = 0
    # df = df.groupby('MMSI').apply(dynamic_window, wei=weight, initial_window_size=5, threshold=5,
    #                               include_groups=False).reset_index(level=0)
    # # print(df.head())
    #
    # df = df[df['ISSIMILAR'] == 0]
    # df = df.drop('ISSIMILAR', axis=1)
    # # print(df)
    # print("9 删除重复的数据 end")
    # print("after df len:{}\n".format(len(df)))

    df.drop(columns=['Distance', 'MaxDistance'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    # print(df)

    return df


def process_file(input_folder, lon_min, lon_max, lat_min, lat_max):
    df = pd.DataFrame()
    file_count = 0

    for filename in os.listdir(input_folder):
        # if filename != "AIS_2023_12_30.csv":
        #     continue
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)

            print("{}: {} begin".format(file_count, filename))
            # 读取csv文件的内容
            temp_df = pd.read_csv(file_path)

            temp_df = temp_df[((temp_df['LON'] >= lon_min) & (temp_df['LON'] < lon_max) &
                            (temp_df['LAT'] >= lat_min) & (temp_df['LAT'] < lat_max))]

            temp_df = data_clean(temp_df)

            # 只保留部分
            temp_df = temp_df[['MMSI', 'BaseDateTime', 'LAT', 'LON', 'COG', 'SOG']]

            # save_file(temp_df, '../data/AIS/AIS_2023_101112', filename)

            df = pd.concat([df, temp_df])
            # print("df len:{}\n".format(len(df)))

            print(f"{filename} end")


            # for lon_min, lon_max, lat_min, lat_max in zip(LON_min, LON_max, LAT_min, LAT_max):
            #
            #     temp_df1 = temp_df[((temp_df['LON'] >= lon_min) & (temp_df['LON'] < lon_max) &
            #                        (temp_df['LAT'] >= lat_min) & (temp_df['LAT'] < lat_max))]
            #     df = pd.concat([df, temp_df1])
            #     print("len(temp_df1):{} len(df):{}".format(len(temp_df1), len(df)))


            # 将读取的内容添加到df中
            # df = pd.concat([df, temp_df])

            print("{}: {} end".format(file_count, filename))

            file_count = file_count + 1

    return df


def save_file(df, output_path, new_filename):
    # 保存处理后的数据集
    output_path = os.path.join(output_path, new_filename)
    df.to_csv(output_path, index=False)


def gather():

    input_folders = ['../data/AIS_2023_09', '../data/AIS_2023_10', '../data/AIS_2023_11', '../data/AIS_2023_12']

    df = pd.DataFrame()
    # LON_min = [-95.5, -94.8,  -91.18, -88.7, -85.6, -83.5, -82.6, -81.4]
    # LON_max = [-83.5, -91.18, -89.5,  -85.6, -83.5, -82.6, -81.4, -79.0]
    # LAT_min = [ 23.5,  28.8,   28.8,   28.8,  28.8,  23.3,  23.3,  23.3]
    # LAT_max = [ 28.8,  29.2,   29.0,   30.0,  29.5,  27.0,  25.6,  24.8]

    # lon_min = -97.6
    # lon_max = -79
    # lat_min = 24.5
    # lat_max = 30.5

    lon_min = -95.6
    lon_max = -80.0
    lat_min = 28.5
    lat_max = 30.5

    file_count = 0

    for folder in input_folders:
        temp_df = process_file(folder, lon_min, lon_max, lat_min, lat_max)
        df = pd.concat([df, temp_df])
        file_count += 1



    save_file(df, '../data/AIS/AIS_2023_4month', 'AIS_2023_4month.csv')
    print("finish")


gather()


