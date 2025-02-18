import numpy as np
import pandas as pd
import os


def create_graph(trips):
    # 创建新的DataFrame用于存储结果
    trips_graph = pd.DataFrame(columns=['src', 'dst', 'weight'])

    # 创建一个字典来统计每一对 (src, dst) 的出现次数
    counts = {}

    i = 0
    print(len(trips))

    for trip in trips:
        if i % 5000 == 0:
            print("{}/{}".format(i, len(trips)))
        i += 1
        src = -1
        dst = -1
        # 按照逗号分割每一部分，并转换为相应的数据类型
        trip_point = trip.split(';')
        for loc in trip_point:
            idx, lon, lat, cog, sog, time = loc.split(',')

            if src == -1:
                src = int(idx)
                continue

            dst = int(idx)

            # 如果 (src, dst) 组合已经存在，则增加计数
            if (src, dst) in counts:
                counts[(src, dst)] += 1
            else:
                counts[(src, dst)] = 1

            src = dst

    # 将统计结果转换为 DataFrame
    trips_graph = pd.DataFrame(
        [(src, dst, weight) for (src, dst), weight in counts.items()],
        columns=['src', 'dst', 'weight']
    )

    # for (src, dst), weight in counts.items():
    #     trips_graph = pd.concat([trips_graph, pd.DataFrame({'src': [src], 'dst': [dst], 'weight': [weight]})],ignore_index=True)

    return trips_graph


def save_file(df, output_path, new_filename):
    # 保存处理后的数据集
    output_path = os.path.join(output_path, new_filename)
    df.to_csv(output_path, index=False)


def trips_graph(data_format, data_name):
    data_path = os.path.join('../data', 'AIS', data_name)
    if data_format == 'csv':
        df = pd.read_csv(os.path.join(data_path, 'traj_train.csv'))
        print('trips_graph read finish')

        trips_graph = create_graph(df['trips_new'])

        print('create graph finish')

        trips_graph.sort_values(by=['src', 'dst'], inplace=True)

        save_file(trips_graph, data_path, 'graph_A.csv')

        print('save file finish')

        print('finish')

# trips_graph('csv', 'AIS_z')