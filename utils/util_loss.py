import numpy as np
import scipy.sparse as sp
import math


def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    user_emb = torch.nan_to_num(user_emb, 0)
    pos_item_emb = torch.nan_to_num(pos_item_emb, 0)
    neg_item_emb = torch.nan_to_num(neg_item_emb, 0)
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
    return torch.mean(loss)


def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb = torch.nan_to_num(emb, 0)
        emb_loss += torch.norm(emb, p=2) / emb.shape[0]
    return emb_loss * reg


def get_user_item_matrix(graph):
    edge_index = torch.stack(graph.edges())

    max_user = max(set(edge_index[0].tolist()))
    max_item = max(set(edge_index[1].tolist()))

    num_user = len(set(edge_index[0].tolist()))
    num_item = len(set(edge_index[1].tolist()))
    print("the number of users: {0}".format(num_user))
    print("the number of items: {0}".format(num_item))

    row, col, entries = [], [], []
    for edge in edge_index.t().tolist():
        row.append(edge[0])
        col.append(edge[1] - max_user - 1)
        entries.append(1.0)
    num_max = find_min_number(col)

    interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(num_user, num_item),
                                    dtype=np.float32)

    # save_sparse_matrix_to_pickle(interaction_mat, 'interaction_matrix.pkl')
    # print("最大用户编号{0}".format(max_user))
    # print("最大物品编号{0}".format(find_max_number(col)))

    return interaction_mat, num_user, num_item


def find_max_number(nums):
    max_num = nums[0]

    for num in nums:
        if num > max_num:
            max_num = num

    return max_num


def find_min_number(nums):
    min_num = nums[0]
    for num in nums:
        if num < min_num:
            min_num = num

    return min_num


def compute_beta(epoch, total_epochs):
    return 1 * 1 * epoch / total_epochs + 1 * (1 - epoch / total_epochs)


import pickle


def save_sparse_matrix_to_pickle(sparse_matrix, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(sparse_matrix, f)


def fast_evaluation(epoch, measure, bestPerformance):
    print('Evaluating the model...')

    if len(bestPerformance) > 0:
        count = 0
        performance = {}
        for m in measure[1:]:
            k, v = m.strip().split(':')
            performance[k] = float(v)
        for k in bestPerformance[1]:
            if bestPerformance[1][k] > performance[k]:
                count += 1
            else:
                count -= 1
        if count < 0:
            bestPerformance[1] = performance
            bestPerformance[0] = epoch + 1
            # bestPerformance[2] = rec_list
    else:
        bestPerformance.append(epoch + 1)
        performance = {}
        for m in measure[1:]:
            k, v = m.strip().split(':')
            performance[k] = float(v)
        bestPerformance.append(performance)

    print('-' * 120)
    print('Real-Time Ranking Performance ' + ' (Top-' + str(20) + ' Item Recommendation)')
    measure = [m.strip() for m in measure[1:]]
    print('*Current Performance*')
    print('Epoch:', str(epoch + 1) + ',', '  |  '.join(measure))
    bp = ''
    # for k in self.bestPerformance[1]:
    #     bp+=k+':'+str(self.bestPerformance[1][k])+' | '
    bp += 'Hit Ratio' + ':' + str(bestPerformance[1]['Hit Ratio']) + '  |  '
    bp += 'Precision' + ':' + str(bestPerformance[1]['Precision']) + '  |  '
    bp += 'Recall' + ':' + str(bestPerformance[1]['Recall']) + '  |  '
    # bp += 'F1' + ':' + str(self.bestPerformance[1]['F1']) + ' | '
    bp += 'NDCG' + ':' + str(bestPerformance[1]['NDCG'])
    print('*Best Performance* ')
    print('Epoch:', str(bestPerformance[0]) + ',', bp)
    print('-' * 120)
    return bestPerformance[0]


class Metric(object):
    def __init__(self):
        pass

    @staticmethod
    def hits(origin, res):
        hit_count = {}
        for user in origin:
            items = list(origin[user].keys())
            predicted = [item[0] for item in res[user]]
            hit_count[user] = len(set(items).intersection(set(predicted)))
        return hit_count

    @staticmethod
    def hit_ratio(origin, hits):
        """
        Note: This type of hit ratio calculates the fraction:
         (# retrieved interactions in the test set / #all the interactions in the test set)
        """
        total_num = 0
        for user in origin:
            items = list(origin[user].keys())
            total_num += len(items)
        hit_num = 0
        for user in hits:
            hit_num += hits[user]
        return round(hit_num / total_num, 5)

    # # @staticmethod
    # def hit_ratio(origin, hits):
    #     """
    #     Note: This type of hit ratio calculates the fraction:
    #      (# users who are recommended items in the test set / #all the users in the test set)
    #     """
    #     hit_num = 0
    #     for user in hits:
    #         if hits[user] > 0:
    #             hit_num += 1
    #     return hit_num / len(origin)

    @staticmethod
    def precision(hits, N):
        prec = sum([hits[user] for user in hits])
        return round(prec / (len(hits) * N), 5)

    @staticmethod
    def recall(hits, origin):
        recall_list = [hits[user] / len(origin[user]) for user in hits]
        recall = round(sum(recall_list) / len(recall_list), 5)
        return recall

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return round(2 * prec * recall / (prec + recall), 5)
        else:
            return 0

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error += abs(entry[2] - entry[3])
            count += 1
        if count == 0:
            return error
        return round(error / count, 5)

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += (entry[2] - entry[3]) ** 2
            count += 1
        if count == 0:
            return error
        return round(math.sqrt(error / count), 5)

    @staticmethod
    def NDCG(origin, res, N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            # 1 = related, 0 = unrelated
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG += 1.0 / math.log(n + 2, 2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG += 1.0 / math.log(n + 2, 2)
            sum_NDCG += DCG / IDCG
        return round(sum_NDCG / len(res), 5)


def ranking_evaluation(origin, res, N):
    measure = []
    for n in N:
        predicted = {}
        for user in res:
            predicted[user] = res[user][:n]
        indicators = []
        if len(origin) != len(predicted):
            print('The Lengths of test set and predicted set do not match!')
            exit(-1)
        hits = Metric.hits(origin, predicted)
        hr = Metric.hit_ratio(origin, hits)
        indicators.append('Hit Ratio:' + str(hr) + '\n')
        prec = Metric.precision(hits, n)
        indicators.append('Precision:' + str(prec) + '\n')
        recall = Metric.recall(hits, origin)
        indicators.append('Recall:' + str(recall) + '\n')
        # F1 = Metric.F1(prec, recall)
        # indicators.append('F1:' + str(F1) + '\n')
        # MAP = Measure.MAP(origin, predicted, n)
        # indicators.append('MAP:' + str(MAP) + '\n')
        NDCG = Metric.NDCG(origin, predicted, n)
        indicators.append('NDCG:' + str(NDCG) + '\n')
        # AUC = Measure.AUC(origin,res,rawRes)
        # measure.append('AUC:' + str(AUC) + '\n')
        measure.append('Top ' + str(n) + '\n')
        measure += indicators
    return measure


import torch


def get_origin_user_interaction_list(graph):
    user_interactions = {}

    edges = graph.edges()
    src_nodes, dst_nodes = edges

    user_set = set(src_nodes.numpy())

    for user_id in user_set:
        interaction_indices = (src_nodes == user_id).nonzero(as_tuple=True)[0]

        interacted_items = dst_nodes[interaction_indices].numpy()

        user_interactions[user_id] = {str(item_id): 1.0 for item_id in interacted_items}

    return user_interactions, user_set


def get_rec_list(user_set, scores, max_user_id):
    rec_list = {}
    user_list = list(user_set)

    user_indices_tensor = torch.tensor(user_list, dtype=torch.long)

    sorted_indices = torch.argsort(scores[user_indices_tensor], descending=True, dim=1)

    sorted_item_indices = sorted_indices + max_user_id
    # print(sorted_item_indices)

    for idx, user_index in enumerate(user_indices_tensor):
        sorted_user_scores = scores[user_index][sorted_indices[idx]]
        sorted_items = sorted_item_indices[idx]

        formatted_scores = list(zip(map(str, sorted_items.tolist()), sorted_user_scores.tolist()))
        rec_list[int(user_index)] = formatted_scores

    return rec_list


def normalize_graph_mat(adj_mat):
    shape = adj_mat.shape
    rowsum = np.array(adj_mat.sum(1))
    if shape[0] == shape[1]:
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
    else:
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_mat = d_mat_inv.dot(adj_mat)
    return norm_adj_mat


def generate_interaction_matrix_from_dgl(graph, user_num, item_num):
    # src_nodes, dst_nodes = graph.edges()
    # src_nodes = src_nodes.cpu().numpy()
    # dst_nodes = dst_nodes.cpu().numpy()
    src, dst = graph.edges()
    src_nodes = src.cpu().numpy()
    dst = dst.cpu().numpy()

    ratings = np.ones_like(src_nodes, dtype=np.float32)


    interaction_matrix = sp.csr_matrix(
        (ratings, (src_nodes, dst)),
        shape=(user_num + item_num, item_num + user_num),
        dtype=np.float32
    )

    return interaction_matrix
