import math
import itertools
import numpy as np
import pandas as pd
import pickle
import argparse
import networkx as nx
import torch
import os
import sys
import pyproj
from pyproj import Transformer
import torch.nn as nn
from itertools import combinations
from RMSE_point import RMSE_point

sys.path.append('../')

from model import Transformer_insertion
from detection_stage.model import Transformer_tagging
from utils import get_masks_and_count_tokens_src, get_masks_and_count_tokens_trg, calculate_laplacian_matrix
from dataloader import pad_arrays
from constants import *
from collections import defaultdict
from geopy.distance import great_circle
from fastdtw import fastdtw


def load_test_dataset(args, data_path, adj_path):
    test_path = os.path.join(data_path, 'traj_test.csv')
    lbs_test = pd.read_csv(test_path, converters={'trips_sparse': eval,
                                                  'num_labels': eval})
    # id2loc = pickle.load(open(os.path.join(data_path,"grid2center_Beijing.pickle"), 'rb'))
    id2loc = pickle.load(open(os.path.join(data_path, "grid2center_" + args.data_name + ".pickle"), 'rb'))
    print("test data size {}, location num {}".format(len(lbs_test), len(id2loc)))

    to3414 = Transformer.from_crs("epsg:4326", "epsg:4575", always_xy=True)

    def dataset_collate(trips):
        trips_collate = []
        for trip in trips:
            trip_collate = []
            trip = trip.split(';')
            for loc in trip:
                idx, lon, lat, cog, sog, time = loc.split(',')
                trip_collate.append([int(idx), float(lon), float(lat), float(cog), float(sog), int(time)])
            trips_collate.append(trip_collate)
        return trips_collate


    def data_to_input(trips):
        trips_input = []
        for trip in trips:
            res = []
            time_min = trip[0][-1]
            for (loc, lon, lat, cog, sog, time) in trip:
                if loc == 'BLK':
                    res.append((BLK_TOKEN, 5000, 20447840.4, 4419792.3 ,0 ,0))
                else:
                    coords = id2loc[loc]
                    res.append((int(loc)+TOTAL_SPE_TOKEN, time-time_min, coords[0], coords[1], cog, sog))
            trips_input.append(np.array(res))
        return trips_input


    def get_insertion_input_seq2seq(trips_drop, labels):
        res = []
        for trip_drop, label in zip(trips_drop, labels):
            assert len(trip_drop) == len(label)
            temp = []
            for loc, t in zip(trip_drop, label):
                if t == 0:
                    temp.append(loc)
                else:
                    temp.append(loc)
                    temp.append(['BLK', 'BLK', 'BLK', 'BLK'])

            assert len(temp) == len(trip_drop) + np.sum(np.array(label)!=0)
            res.append(temp)
        return res


    loc_size = len(id2loc)
    loc2id = {loc: id for id, loc in id2loc.items()}

    adj_pd = pd.read_csv(adj_path)
    adj_pd = adj_pd.add({'src': TOTAL_SPE_TOKEN, 'dst': TOTAL_SPE_TOKEN, 'weight': 0})
    G = nx.DiGraph()
    G.add_nodes_from(list(range(loc_size + TOTAL_SPE_TOKEN)))
    src, dst, weights = adj_pd['src'].values.tolist(), adj_pd['dst'].values.tolist(), adj_pd['weight'].values.tolist()
    G.add_weighted_edges_from(zip(src, dst, weights))
    adj_graph = nx.to_numpy_array(G)

    test_traj = lbs_test['trips_sparse'].values.tolist()
    test_tgt = dataset_collate(lbs_test['trips_new'].values.tolist())
    drop_ratios = lbs_test['drop_ratio'].values.tolist()
    num_labels = lbs_test['num_labels'].values.tolist()

    max_len = 60

    print("test num {}, target {}, " \
          .format(len(test_traj), len(test_tgt)))

    # + special tokens: PAD, BOS, EOS, NUL, BLK
    test_input = data_to_input(test_traj)
    # test_target = data_to_input(test_tgt)
    test_target = test_tgt

    distance = trips_distance(id2loc)

    # new_filename = 'distance_matrix.csv'
    # output_path = os.path.join(data_path, new_filename)
    # distance.to_csv(output_path, index=False)

    return test_input, test_target, loc_size, id2loc, max_len, adj_graph, drop_ratios, num_labels, distance


def trips_distance(grid2center):
    # 将字典中的坐标转换为 NumPy 数组
    keys = list(grid2center.keys())
    coordinates = np.array([grid2center[key] for key in keys])

    # 计算所有点之间的距离
    def calculate_distances(coordinates):
        # to4326 = Transformer.from_crs("epsg:4087", "epsg:4326", always_xy=True)
        # coordinates_trans = to4326.transform(coordinates[:, 1], coordinates[:, 0])

        # coordinates_trans = utm_to_wgs84(coordinates[:, 1], coordinates[:, 0])

        # Convert coordinates to radians
        # coords_rad = np.radians(coordinates_trans)
        #
        # # Extract latitudes and longitudes
        # lat1 = coords_rad[:, 0][:, np.newaxis]
        # lon1 = coords_rad[:, 1][:, np.newaxis]
        # lat2 = coords_rad[:, 0]
        # lon2 = coords_rad[:, 1]
        #
        # # Compute differences
        # dlat = lat2 - lat1
        # dlon = lon2 - lon1
        #
        # # Apply Haversine formula
        # a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        # c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        #
        # R = 6371.0
        #
        # # Distance matrix in kilometers
        # distance_matrix_km = R * c
        #
        # # Convert distance matrix to meters
        # distance_matrix_m = distance_matrix_km * 1000

        
        # num_points = len(coordinates)
        distance_matrix_m = np.linalg.norm(coordinates[:, np.newaxis] - coordinates, axis=2)
        return distance_matrix_m

    # 调用函数计算距离
    distance_matrix = calculate_distances(coordinates)

    # 将距离矩阵转换为 DataFrame
    distance = pd.DataFrame(distance_matrix, index=keys, columns=keys)

    return distance


def collate_multi_class_label(label):
    res = []
    if label == 0:
        res = []
    elif label == 1:
        res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT, PAD_COG, PAD_SOG] for _ in range(4)]
    elif label == 2:
        res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT, PAD_COG, PAD_SOG] for _ in range(9)]
    elif label == 3:
        res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT, PAD_COG, PAD_SOG] for _ in range(15)]
    elif label == 4:
        res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT, PAD_COG, PAD_SOG] for _ in range(25)]
    return res

def get_insertion_input_data(traj, num_label):
    res, masked_pos = [], []
    assert len(traj) == len(num_label)
    for (record, label) in zip(traj, num_label):
        if label == 0:
            res.append(record)
        else:
            res.append(record)
            blk_tokens = collate_multi_class_label(label)
            masked_pos.extend(list(range(len(res), len(res)+len(blk_tokens))))
            res.extend(blk_tokens)

    return np.array(res), np.array(masked_pos, dtype=np.int_)


# 找到上一个点位的网格号
def find_last_loc(traj_locs, masked_pos_batch, batch_pred_inputs):

    max_len = batch_pred_inputs.size(1)

    index = masked_pos_batch[:, max_len] - 1

    if max_len == 0:

        # 使用 np.arange 和高级索引提取指定元素
        selected_data = traj_locs[np.arange(index.shape[0]), index]

        # 将结果转换为二维数组
        last_loc = selected_data[:, None]

        # last_loc = traj_locs[:, index]
        return last_loc

    masked_pos_pres_batch = masked_pos_batch[:, :max_len]

    new_traj_locs = traj_locs.clone()

    # 更新数组 traj_locs
    new_traj_locs[np.arange(masked_pos_batch.shape[0])[:, None], masked_pos_pres_batch] = batch_pred_inputs

    # 将 index 和 traj_locs 转换为 NumPy 数组
    index_np = index.cpu().numpy()
    new_traj_locs_np = new_traj_locs.cpu().numpy()

    # 生成所有可能的索引
    all_indices = np.arange(traj_locs.shape[1])

    # 生成每个子数组的有效索引
    valid_indices = all_indices[None, :] <= index_np[:, None]

    # 生成每个子数组中大于等于5的值的掩码
    valid_values_mask = (new_traj_locs_np >= 5) & valid_indices

    max_indices = valid_values_mask.shape[1] - 1 - np.argmax(valid_values_mask[:, ::-1], axis=1)
    # 处理没有满足条件的索引
    no_valid_mask = ~valid_values_mask.any(axis=1)
    max_indices[no_valid_mask] = 0  # 设置为0，因为np.argmax返回0

    # 提取结果值
    last_loc = new_traj_locs[np.arange(len(index)), max_indices]

    # last_loc = last_loc.flatten()

    last_loc = last_loc.reshape(-1)

    return last_loc


def test_twostage(args):
    data_path = os.path.join(args.data_path, args.data_name)
    adj_path = os.path.join(data_path, 'graph_A.csv')
    detection_model_path = os.path.join(args.model_path, 'model_detection')
    recovery_model_path = os.path.join(args.model_path, 'model_recovery')


    test_input, test_target, loc_size, id2loc, max_len, adj_graph, drop_ratios, num_labels, distance = load_test_dataset(args, data_path, adj_path)

    # 调整 NumPy 显示选项
    # np.set_printoptions(threshold=np.inf)  # 显示所有元素

    # 打印数组
    # print(adj_graph)
    # 将数组导出到文件
    # np.savetxt(data_path + 'adj_graph.txt', adj_graph, delimiter=',')
    # with open(data_path + '/adj_graph.txt', 'w') as log_file:
    #     log_file.write(
    #         "len(adj_graph):{}\nadj_graph:{}\n" \
    #         .format(len(adj_graph), adj_graph))

    tagging_model = Transformer_tagging(
        model_dimension=args.hidden_size,
        fourier_dimension=args.hidden_size,
        time_dimension=args.hidden_size,
        vocab_size=loc_size + TOTAL_SPE_TOKEN,
        number_of_heads=args.num_heads,
        number_of_layers=args.num_layers,
        number_cls=args.num_cls,
        dropout_probability=args.dropout,
        device=args.device
    ).to(args.device)

    tagging_model.load_state_dict(torch.load(detection_model_path, map_location=args.device))

    insertion_model = Transformer_insertion(
        model_dimension=args.hidden_size,
        fourier_dimension=args.hidden_size,
        time_dimension=args.hidden_size,
        src_vocab_size=loc_size+TOTAL_SPE_TOKEN,
        trg_vocab_size=loc_size+TOTAL_SPE_TOKEN,
        number_of_heads=args.num_heads,
        number_of_layers=4,
        dropout_probability=args.dropout,
        max_len=max_len,
        device = args.device
    ).to(args.device)

    insertion_model.load_state_dict(torch.load(recovery_model_path, map_location=args.device))


    A = calculate_laplacian_matrix(adj_graph, mat_type='hat_rw_normd_lap_mat')
    A = torch.from_numpy(A).float().to_sparse().to(device=args.device)

    ### Stage 1: tagging for BLK token
    test_size, eval_batch = len(test_input), args.batch_size
    num_iter = int(np.ceil(len(test_input) / eval_batch))
    print("tagging stage: {}".format(num_iter))

    # 创建示例输入数据
    batch_size = 32
    seq_len = 32
    coor_dim = 30  # 假设坐标维度为30
    example_input_loc = torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long, device=args.device)
    example_input_time = torch.rand(batch_size, seq_len, 1, device=args.device)
    example_input_coors = torch.rand(batch_size, seq_len, coor_dim, device=args.device)  # 假设coors有30维
    example_src_mask = torch.ones(batch_size, 1, seq_len, dtype=torch.bool, device=args.device)
    example_input = (example_input_loc, example_input_time, example_input_coors, example_src_mask)

    # 使用summary函数
    # summary(tagging_model, input_size=[(512, 55), (512, 55, 1), (512, 55, 2), (512, 1, 1, 55), (17464, 17464)])
    # summary(TaggingModelWrapper(tagging_model, A, args.device),
    #         input_size=(32, 32))  # 输入形状 (batch_size, seq_len)


    tagging_model.eval()
    tagging_preds = []


    for i in range(num_iter):
        with torch.no_grad():
            traj_inp = test_input[i * eval_batch: min((i + 1) * eval_batch, test_size)]
            traj_inp = pad_arrays(traj_inp)
            # traj_inp = test_input[i]
            traj_inp_loc, traj_inp_time, traj_inp_coors, traj_inp_cogs, traj_inp_sogs = traj_inp[:, :, 0], traj_inp[:, :, 1:2], traj_inp[:, :, 2:4], traj_inp[:, :, 4:5], traj_inp[:, :, 5:6]
            traj_inp_loc = torch.tensor(traj_inp_loc, dtype=torch.long, device=args.device)
            traj_inp_time = torch.tensor(traj_inp_time, dtype=torch.float, device=args.device)
            traj_inp_coors = torch.tensor(traj_inp_coors, dtype=torch.float, device=args.device)
            traj_inp_cogs = torch.tensor(traj_inp_cogs, dtype=torch.float, device=args.device)
            traj_inp_sogs = torch.tensor(traj_inp_sogs, dtype=torch.float, device=args.device)

            src_mask, _ = get_masks_and_count_tokens_src(traj_inp_loc, PAD_TOKEN)

            outputs = tagging_model(traj_inp_loc, traj_inp_time, traj_inp_coors, traj_inp_cogs, traj_inp_sogs, src_mask, A, 'tagging')
            pred = torch.argmax(outputs, dim=-1).cpu().numpy()
            tagging_preds.extend(pred)

    assert len(tagging_preds) == len(test_input)

    # print(tagging_preds)

    cnt = 0
    print("processing tagging result to the input of stage 2 model")
    insertion_inputs, masked_pos = [], []
    for trip_drop, tag in zip(test_input, tagging_preds):
        tag = tag[:len(trip_drop)]
        insertion_input, masked = get_insertion_input_data(trip_drop, tag)

        insertion_inputs.append(insertion_input)
        masked_pos.append(masked)

    with open(data_path + '/masked_p.txt', 'w') as log_file:
        # 清空文件
        pass

    # with open(data_path + '/last_words_batch.txt', 'w') as log_file:
    #     # 清空文件
    #     pass

    ### Stage 2: insertion for BLK tokens

    insertion_model.eval()
    inputs = []
    final_preds = []
    for i in range(num_iter):
        with torch.no_grad():
            traj_inp = insertion_inputs[i * eval_batch: min((i + 1) * eval_batch, test_size)]
            masked_pos_np = masked_pos[i * eval_batch: min((i + 1) * eval_batch, test_size)]
            lengths = np.array(list(map(len, traj_inp)))
            masked_pos_lengths = np.array(list(map(len, masked_pos_np)))

            traj_inp = pad_arrays(traj_inp)
            traj_locs, traj_tms, traj_coors, traj_cogs, traj_sogs = traj_inp[:, :, 0], traj_inp[:, :, 1:2], traj_inp[:, :, 2:4], traj_inp[:, :, 4:5], traj_inp[:, :, 5:6]

            traj_locs = torch.tensor(traj_locs, dtype=torch.long, device=args.device)
            traj_tms = torch.tensor(traj_tms, dtype=torch.float, device=args.device)
            traj_coors = torch.tensor(traj_coors, dtype=torch.float, device=args.device)
            traj_cogs = torch.tensor(traj_cogs, dtype=torch.float, device=args.device)
            traj_sogs = torch.tensor(traj_sogs, dtype=torch.float, device=args.device)

            inputs.extend(traj_locs.cpu().numpy())
            masked_pos_batch = pad_arrays(masked_pos_np)

            masked_pos_batch = torch.tensor(masked_pos_batch, dtype=torch.long, device=args.device)
            batch_pred_inputs = torch.tensor([BLK_TOKEN] * traj_locs.size(0), dtype=torch.long,
                                             device=args.device).unsqueeze(1)

            for idx in range(masked_pos_batch.shape[1]):
                attn_mask, _ = get_masks_and_count_tokens_trg(torch.cat([traj_locs, batch_pred_inputs], dim=1),
                                                              PAD_TOKEN)
                batch_masked_pos_cur = masked_pos_batch[:, :idx + 1]

                trg_probs = insertion_model(traj_locs, traj_tms, traj_coors, traj_cogs, traj_sogs, attn_mask, A, 'recovery', batch_masked_pos_cur,
                                            batch_pred_inputs)

                last_words_batch = trg_probs[:, idx]  # B x vocab_size

                # 找到距离上一个轨迹点较近的候选点
                last_loc = find_last_loc(traj_locs, masked_pos_batch, batch_pred_inputs[:, 1:])

                # 将张量从 CUDA 设备转移到 CPU
                last_loc_cpu = last_loc.cpu()

                # 将张量转换为 NumPy 数组
                last_loc_np = last_loc_cpu.numpy()

                # 确保 last_loc - 5 的结果保持为正数
                last_loc_np = np.maximum(last_loc_np - 5, 0)

                # 将 last_loc_np 转换为 PyTorch 张量，并移动到与 distance 相同的设备
                last_loc_tensor = torch.tensor(last_loc_np, dtype=torch.int64, device=args.device).flatten()


                distance_tensor = torch.tensor(distance.values, device=args.device)

                # 从 distance 矩阵中选择特定的列
                distances_to_last_loc = torch.index_select(distance_tensor, dim=0, index=last_loc_tensor)

                # 生成布尔掩码，表示距离是否小于candidate_loc_distance
                mask = distances_to_last_loc <= args.candidate_loc_distance

                # 生成 5 行全为 True 的张量
                true_rows = torch.ones(last_words_batch.size(0), TOTAL_SPE_TOKEN, dtype=torch.bool, device=args.device)

                # 使用 torch.cat 拼接张量
                new_mask = torch.cat([true_rows, mask], dim=1)

                # 使用 torch.where 函数
                last_words_batch = torch.where(new_mask, last_words_batch, -float('inf'))

                # with open(data_path + '/last_words_batch.txt', 'a') as log_file:
                #     for i in range(last_words_batch.shape[1]):
                #         log_file.write("i:{} last_words_batch:{}\n".format(i, last_words_batch[:, i]))

                pred_locs = torch.argmax(last_words_batch, dim=-1)

                batch_pred_inputs = torch.cat([batch_pred_inputs, pred_locs.unsqueeze(1)], dim=1)

            output_pred_locs = batch_pred_inputs[:, 1:].cpu().numpy()  # remove the first blk token
            output_locs = traj_locs.cpu().numpy()
            batch_preds_post = []

            # print(masked_pos_lengths)

            for idx, (pred, masked_p, length, masked_pos_length) in enumerate(
                    zip(output_pred_locs, masked_pos_np, lengths, masked_pos_lengths)):
                masked_p = masked_p[:masked_pos_length]
                output_locs[idx, masked_p] = pred[:masked_pos_length]
                batch_preds_post.append(output_locs[idx, :length])

                with open(data_path + '/masked_p.txt', 'a') as log_file:
                    log_file.write("idx:{}\nlen(pred):{}\npred:\n{}\nlen(masked_p):{}\nmasked_p:\n{}\nlength:{}\nmasked_pos_length:{}\nlen(output_locs):{}\noutput_locs:{}\n" \
                                   .format(idx, len(pred), pred, len(masked_p), masked_p, length, masked_pos_length, len(output_locs[idx, :length]), output_locs[idx, :length]))

        final_preds.extend(batch_preds_post)

    assert len(final_preds) == len(test_input)
    print("length of preds: {}, evaluating".format(len(final_preds)))

    prec, rec, recovery, m_prec = evaluate(args, insertion_inputs, final_preds, test_target, num_labels, id2loc, max_len, data_path)


def evaluate(args, test_input, preds, test_target, num_labels, id2loc, maxlen, data_path):
    recall_total, precision_total, recovery_total, micro_precision_total = [], [], [], []

    with open(data_path + '/val_preds.txt', 'w') as log_file:
        # 清空文件
        pass

    RMSE = 0
    RMSE_count = 0
    false_count = 0

    pred_trips = []
    pred_trips_id = 0

    to4326 = Transformer.from_crs(f"epsg:{epsg}", "epsg:4326", always_xy=True)

    for idx, (drop, pred, label_origin, tag) in enumerate(zip(test_input, preds, test_target, num_labels)):
        label = [l[0] for l in label_origin]
        pred = [p - TOTAL_SPE_TOKEN for p in pred if p >= TOTAL_SPE_TOKEN]

        pred_seq = []

        # print('pred:{}\nlabel:{}'.format(pred, label))

        right = set(pred).intersection(set(label))
        with open(data_path + '/val_preds.txt', 'a') as log_file:
            log_file.write("idx:{}\npred length:{}\n{}\nlabel length:{}\n{}\nright\n{}\ntag length:{}\n{}\n" \
                           .format(idx, len(pred), pred, len(label), label, right, len(tag), tag))

        if len(pred) == len(tag):
            pred_seq = [
                [item[0], to4326.transform(item[1], item[2]), to4326.transform(item[1], item[2])] + [0]
                for item in label_origin
            ]

            pred_trips.append({'id': pred_trips_id, 'trips': pred_seq, 'single_RMSE': -1})
            pred_trips_id += 1
            # 未预测到
            false_count += 1

        elif len(pred) > len(tag):
            label_index = 0
            last_label_index = label_index
            pred_index = 0
            last_pred_index = 0
            min_square_distance = 0
            single_RMSE = 0
            single_RMSE_count = 0
            for i in range(len(tag)):
                # print("tag[i]:{}".format(tag[i]))
                # 找出被预测的地方
                if tag[i] == 0:
                    pred_seq.append([label_origin[label_index][0],
                                     to4326.transform(label_origin[label_index][1],label_origin[label_index][2]),
                                     to4326.transform(label_origin[label_index][1],label_origin[label_index][2])]
                                    + [0])  # 使用原来的
                    label_index += 1
                    continue

                pred_seq.append([label_origin[label_index][0],
                                 to4326.transform(label_origin[label_index][1],label_origin[label_index][2]),
                                 to4326.transform(label_origin[label_index][1],label_origin[label_index][2])]
                                + [0])  # 使用原来的
                label_index += 1
                # print("label_index:{} last_label_index:{} ".format(label_index, last_label_index))
                label_no_pred = label[last_label_index:label_index]
                label_need_pred = label[label_index: label_index + tag[i]]

                j = 0
                while j < label_index - last_label_index:
                    # print("pred:\n{}\nlabel:\n{}\ntag:\n{}\n".format(pred, label, tag))
                    if label_no_pred[j] != pred[pred_index]:
                        j = j - 1
                        # print("j:{}".format(j))
                        # print("new___label_no_pred[j]:{}".format(label_no_pred[j]))
                    pred_index += 1
                    j += 1

                # print("label_index:{}  pred_index:{}".format(label_index, pred_index))
                # print("tag[i]:{}\nlabel[label_index + tag[i]]:{}\npred[pred_index]:{}".
                #       format(tag[i], label[label_index + tag[i]], pred[pred_index]))
                if label[label_index + tag[i]] == pred[pred_index]:
                    label_index = label_index + tag[i]
                    last_label_index = label_index
                else:
                    last_pred_index = pred_index
                    # print(label[label_index + tag[i]])
                    while label[label_index + tag[i]] != pred[pred_index]:
                        pred_index += 1
                    pred_need_pred = pred[last_pred_index: pred_index]
                    if len(pred_need_pred) > len(label_need_pred):

                        converted_label_subset = label_origin[label_index: label_index + tag[i]]
                        converted_label = [(row[1], row[2]) for row in converted_label_subset]

                        # converted_label = [id2loc[id] for id in label_need_pred]
                        converted_pred = [id2loc[id] for id in pred_need_pred]

                        min_distance, best_subseq =  find_best_subsequence(converted_pred, converted_label)
                        min_square_distance += min_distance

                        print("min_square_distance:{}".format(min_square_distance))

                        single_RMSE += min_square_distance
                        single_RMSE_count += len(label_need_pred)
                        label_index = label_index + tag[i]
                        last_label_index = label_index

                        for coord_label, coord_pred in zip(converted_label, best_subseq):
                            x1, y1 = coord_label
                            lat_lon_label = to4326.transform(x1, y1)
                            x2, y2 = coord_pred
                            lat_lon_pred = to4326.transform(x2, y2)
                            pred_seq.append([-1, lat_lon_label, lat_lon_pred, 1])

                    else:
                        converted_label_subset = label_origin[label_index: label_index + tag[i]]
                        converted_label = [(row[1], row[2]) for row in converted_label_subset]

                        # converted_label = [id2loc[id] for id in label_need_pred]
                        converted_pred = [id2loc[id] for id in pred_need_pred]

                        # pred_loc = id2loc[pred[pred_index]]
                        # label_loc = id2loc[label[pred_index]]

                        min_distance, best_subseq = find_best_subsequence(converted_label, converted_pred)
                        min_square_distance += min_distance

                        print("min_square_distance:{}".format(min_square_distance))

                        single_RMSE += min_square_distance
                        single_RMSE_count += len(pred_need_pred)
                        label_index = label_index + tag[i]
                        last_label_index = label_index

                        for coord_label, coord_pred in zip(best_subseq, converted_pred):
                            x1, y1 = coord_label
                            lat_lon_label = to4326.transform(x1, y1)
                            x2, y2 = coord_pred
                            lat_lon_pred = to4326.transform(x2, y2)
                            pred_seq.append([-1, lat_lon_label, lat_lon_pred, 1])

            # print("single_RMSE_count:{}".format(single_RMSE_count))

            single_pred_RMSE = -1
            if single_RMSE_count != 0:

                tmp_single_RMSE = math.sqrt(single_RMSE / single_RMSE_count)

                if tmp_single_RMSE >= 0:
                    RMSE += single_RMSE
                    RMSE_count += single_RMSE_count

                    single_RMSE = tmp_single_RMSE
                    single_pred_RMSE = single_RMSE

                    print("single_RMSE:{} single_RMSE_count:{}".format(single_RMSE, single_RMSE_count))
                    with open(data_path + '/val_preds.txt', 'a') as log_file:
                        log_file.write("single_RMSE:{} single_RMSE_count:{}\nRMSE_count:{} RMSE:{}\n". \
                                       format(single_RMSE, single_RMSE_count, RMSE_count, math.sqrt(RMSE / RMSE_count)))

            pred_trips.append({'id': pred_trips_id, 'trips': pred_seq, 'single_RMSE': single_pred_RMSE})
            pred_trips_id += 1
        recall = len(set(pred).intersection(set(label))) / len(label)
        precision = len(set(pred).intersection(set(label))) / len(pred)

        drop = [p[0] - TOTAL_SPE_TOKEN for p in drop if p[0] >= TOTAL_SPE_TOKEN]

        expected = set(label) - set(drop)
        if len(expected) > 0:
            recovery = len(set(pred).intersection(expected)) / len(expected)
        else:
            recovery = 1

        pred_missing = [loc for loc in pred if loc not in drop]
        if len(pred_missing) > 0:
            micro_prec = len(set(pred_missing).intersection(expected)) / len(pred_missing)
        else:
            micro_prec = 0

        recall_total.append(recall)
        recovery_total.append(recovery)
        precision_total.append(precision)
        micro_precision_total.append(micro_prec)
        # hauss_dist_total.append(hauss)

    pred_RMSE = -1

    if RMSE_count != 0:
        RMSE = math.sqrt(RMSE / RMSE_count)
        pred_RMSE = RMSE
        print('RMSE:{}       RMSE_count:{}\nfalse_count:{}'.format(RMSE, RMSE_count, false_count))
        with open(data_path + '/val_preds.txt', 'a') as log_file:
            log_file.write("RMSE_count:{} total RMSE:{}\nfalse_count:{}\n". \
                           format(RMSE_count, RMSE, false_count))

    df = pd.DataFrame(pred_trips)
    df['RMSE'] = pred_RMSE
    save_file(df, data_path, 'RMSE.csv')
    RMSE_point(data_path, 'RMSE.csv')

    print("average recall {}, average precision {}, average micro-recall {}, average micro-precision {}". \
          format(np.mean(recall_total), np.mean(precision_total), np.mean(recovery_total),
                 np.mean(micro_precision_total)))
    prec, recall, recovery, m_prec = np.mean(precision_total), np.mean(recall_total), np.mean(
        recovery_total), np.mean(micro_precision_total)

    return prec, recall, recovery, m_prec


def euclidean_square_distance(p1, p2):
    to4326 = Transformer.from_crs(f"epsg:{epsg}", "epsg:4326")
    # to4326 = Transformer.from_crs("epsg:4575", "epsg:4326")
    # to4326 = Transformer.from_crs("epsg:4087", "epsg:4326")

    point1 = to4326.transform(p1[1], p1[0])
    point2 = to4326.transform(p2[1], p2[0])

    # print(point1)

    return (great_circle(point1, point2).meters) ** 2
    # 计算两点之间的欧氏距离的平方
    # return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def find_best_subsequence(long, short):
    len_long = len(long)
    len_short = len(short)
    min_distance = float('inf')
    best_subseq = None

    # for start in range(len_long - len_short + 1):
    #     subseq = long[start:start + len_short]
    #     distance, _ = fastdtw(np.array(short), np.array(subseq), dist=euclidean_square_distance)
    #     if distance < min_distance:
    #         min_distance = distance
    #         best_subseq = subseq

    # 生成所有长度为 len_short 的组合
    for indices in combinations(range(len_long), len_short):
        subseq = [long[i] for i in indices]
        distance, _ = fastdtw(np.array(short), np.array(subseq), dist=euclidean_square_distance)
        if distance < min_distance:
            min_distance = distance
            best_subseq = subseq

    return min_distance, best_subseq


def save_file(df, output_path, new_filename):
    # 保存处理后的数据集
    output_path = os.path.join(output_path, new_filename)
    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TrajectoryEnrichment')
    parser.add_argument("--dropout", type=float, default=0.1,
            help="dropout probability")
    parser.add_argument("--hidden_size", type=int, default=128,
            help="number of hidden dimension")
    parser.add_argument("--num_heads", type=int, default=4,
            help="number of heads")
    parser.add_argument("--out_size", type=int, default=128,
            help="number of output dim")
    parser.add_argument("--num_layers", type=int, default=4,
            help="number of encoder/decoder layers")
    parser.add_argument("--num_epochs", type=int, default=50,
            help="number of minimum training epochs")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="number of batch size")
    parser.add_argument("--num_cls", type=int, default=5,
                        help="number of classes")
    parser.add_argument("--num", type=int, default=1,
                        help="number of dropped segments for each trip")
    parser.add_argument("--rate", type=float, default=0.4,
                        help="dropping rate for each segment")
    parser.add_argument("--gpu", type=int, default= 1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=0.001,
            help="learning rate")
    parser.add_argument("--num_warmup_steps", type=int, default=3000,
                        help="number of warm-up steps")
    parser.add_argument('--model_path', type=str, default='../model/model_AIS_2023_101112',
                        help='Model path')
    parser.add_argument('--data_path', type=str, default='../data/AIS',
                        help='Dataset path')
    parser.add_argument("--data_name", type=str, default="AIS_2023_101112",
                        help="data name")
    parser.add_argument("--candidate_loc_distance", type=int, default=10000,
                        help="candidate loc distance")


    args = parser.parse_args()
    cuda_condition = torch.cuda.is_available() and args.gpu
    args.device = torch.device("cuda" if cuda_condition else "cpu")
    # args.data_path = os.path.join(args.dataset, args.data_path)
    args.sample = False

    print(args)
    test_twostage(args)
