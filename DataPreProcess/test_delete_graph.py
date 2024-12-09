import pandas as pd
import os
from constants import *
from scipy.optimize import bracket


def delete_test_graph(test, graph):
    # 获取 src 列的唯一值
    # unique_src = set(graph['src'])
    # print(unique_src)

    # src_counts = graph['src'].value_counts()
    src_counts = graph.groupby('src')['weight'].sum()

    trips_sparse = test['trips_sparse'].values.tolist()
    num_labels = test['num_labels'].values.tolist()

    index = 0

    for trip, num_label in zip(trips_sparse, num_labels):
        first_column = [row[0] for row in trip]

        count = 0

        for loc in first_column:
            # print('loc:{}'.format(loc))

            # if num_label[count] == 0 or (int(loc) in src_counts and src_counts[int(loc)] > test_delte):
            if int(loc) in src_counts and src_counts[int(loc)] > test_delte:
                count = count + 1
            else:
                test.drop(index, inplace=True)
                print('index:{} loc:{} count:{} num_labels[count]:{}\n'.format(index, loc, count, num_label[count]))
                break

            # count += 1

        index += 1

    test.reset_index(drop=True, inplace=True)
    # print(test)



def save_file(df, output_path, new_filename):
    # 保存处理后的数据集
    output_path = os.path.join(output_path, new_filename)
    df.to_csv(output_path, index=False)


def test_delete_graph(data_format, data_name):
    data_path = os.path.join('../data', 'AIS', data_name)
    if data_format == 'csv':
        df_graph = pd.read_csv(os.path.join(data_path, 'graph_A.csv'))
        df_test = pd.read_csv(os.path.join(data_path, 'traj_test111.csv'), converters={'trips_sparse': eval, 'num_labels': eval})
        delete_test_graph(df_test, df_graph)

        save_file(df_test, data_path, 'traj_test.csv')
        print('finish')




