import pandas as pd
import numpy as np
import os
import ast

def trips_point(df):
    # trips = df['trips']

    # 将每个部分转换为列表
    trips_index_list = []
    trips_lon_label = []
    trips_lat_label = []
    trips_lon_pred = []
    trips_lat_pred = []
    trips_pred_list = []
    trips_single_list = []
    trips_RMSE_list = []

    print(f"trip length:{len(df)}")
    # i = 0

    for i, row in df.iterrows():
        trip = row['trips']
        single = row['single_RMSE']
        RMSE = row['RMSE']
        if i % 10000 == 0:
            print("{}/{}".format(i, len(df)))

        vector_trip = ast.literal_eval(trip)

        # print(vector_trip[0])

        # 使用 zip 解压数据
        grids, coords_label, coords_pred, pred = zip(*vector_trip)

        # 提取经度和纬度
        lons_label, lats_label = zip(*coords_label)
        lons_pred, lats_pred = zip(*coords_pred)

        # 扩展列表
        # 为每个点添加行程索引
        trips_index_list.extend([i] * len(grids))
        trips_lon_label.extend(lons_label)
        trips_lat_label.extend(lats_label)
        trips_lon_pred.extend(lons_pred)
        trips_lat_pred.extend(lats_pred)
        trips_pred_list.extend(pred)
        trips_single_list.extend([single] * len(grids))
        trips_RMSE_list.extend([RMSE] * len(grids))

    # 添加到结果DataFrame
    trips = pd.DataFrame({'MMSI':trips_index_list,
                          'LAT_Label': trips_lat_label, 'LON_Label': trips_lon_label,
                          'LAT_Pred': trips_lat_pred, 'LON_Pred': trips_lon_pred,
                          'Pred': trips_pred_list,
                          'single_RMSE': trips_single_list, 'RMSE': trips_RMSE_list})

    return trips


def save_file(df, output_path, new_filename):
    # 保存处理后的数据集
    output_path = os.path.join(output_path, new_filename)
    df.to_csv(output_path, index=False)

def RMSE_point(data_path, data_name):
    df = pd.read_csv(os.path.join(data_path, data_name))
    # df = pd.read_csv(os.path.join(folder,  args.data_name + args.data_format))
    print('RMSE point read finish')

    df_point = trips_point(df)
    save_file(df_point, data_path, 'RMSE_point.csv')

    print('RMSE count finish')
    print('finish')


# data_path = os.path.join('../data/AIS', "AIS_2023_1112")
# RMSE_point(data_path, 'RMSE.csv')