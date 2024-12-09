import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


def save_file(df, output_path, new_filename):
    # 保存处理后的数据集
    output_path = os.path.join(output_path, new_filename)
    df.to_csv(output_path, index=False)


def trips_split(data_format, data_name):
    data_path = os.path.join('../data', 'AIS', data_name)
    if data_format == 'csv':
        df = pd.read_csv(os.path.join(data_path, 'trips_drop_cleaned_'+ data_name +'.csv'))
        print('trips_split read finish')

        # 分割比例
        train_size = 0.7  # 训练集大小为70%
        val_size = 0.1  # 验证集大小为10%

        # 使用 train_test_split 进行第一次分割，得到训练集和剩余数据
        df_train, df_remaining = train_test_split(df, test_size=(1 - train_size), random_state=42)
        print("len(df_train):{}, len(df_remaining){}".format(len(df_train),len(df_remaining)))

        # 使用 train_test_split 进行第二次分割，从剩余数据中获取验证集
        df_val, df_test = train_test_split(df_remaining, test_size=(1 - val_size / (1 - train_size)), random_state=42)

        # 保存到文件
        save_file(df_train, data_path, 'traj_train.csv')
        save_file(df_val, data_path, 'traj_val.csv')
        save_file(df_test, data_path, 'traj_test111.csv')

        print('finish')


# trips_split('csv', 'AIS_z')
