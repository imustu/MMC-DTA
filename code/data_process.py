import pickle
import json
import torch
import os
import numpy as np
import scipy.sparse as sp
import networkx as nx
import pandas as pd
from torch_geometric import data as DATA
from collections import OrderedDict
from rdkit import Chem
from utils import DTADataset, sparse_mx_to_torch_sparse_tensor, minMaxNormalize, denseAffinityRefine
import random

def dic_normalize(dic):
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0

    return dic


pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


def load_data(dataset):
    affinity = pickle.load(open('data/' + dataset + '/affinities', 'rb'), encoding='latin1')
    if dataset == 'davis':
        affinity = -np.log10(affinity / 1e9)

    return affinity


def process_data(affinity_mat, dataset, num_pos, pos_threshold):
    dataset_path = 'data/' + dataset + '/'

    train_file = json.load(open(dataset_path + 'S1_train_set.txt'))
    train_index = []
    #len(train_file)=4.也就是说train_file中[[是有5份数据的，每份数据5009个。他文章写的是30056个数据，一共分成6份，其中5份是训练集，1份是测试集
    for i in range(len(train_file)):
        train_index += train_file[i]
    test_index = json.load(open(dataset_path + 'S1_test_set.txt'))

    rows, cols = np.where(np.isnan(affinity_mat) == False)
    train_rows, train_cols = rows[train_index], cols[train_index]
    train_Y = affinity_mat[train_rows, train_cols]
    #训练集包含药物ID、目标ID和亲和性分数
    train_dataset = DTADataset(drug_ids=train_rows, target_ids=train_cols, y=train_Y)
    test_rows, test_cols = rows[test_index], cols[test_index]
    test_Y = affinity_mat[test_rows, test_cols]
    test_dataset = DTADataset(drug_ids=test_rows, target_ids=test_cols, y=test_Y)
    #全0矩阵
    train_affinity_mat = np.zeros_like(affinity_mat)
    #把训练集填充进去，训练集没涵盖的地方是0
    train_affinity_mat[train_rows, train_cols] = train_Y
# 改
    #affinity_graph, drug_pos, target_pos = get_affinity_graph(dataset, train_affinity_mat, num_pos, pos_threshold)
    affinity_graph = get_affinity_graph(dataset,train_affinity_mat,num_pos,pos_threshold)
#改
    #return train_dataset, test_dataset, affinity_graph, drug_pos, target_pos
    return train_dataset,test_dataset,affinity_graph


#新改

def select_positive_negative_pairs(train_dataset):
    threshold = 0.1
    positive_pairs = []
    negative_pairs = []

    # 获取所有药物ID和对应的亲和力值
    drug_affinity = {}
    for data in train_dataset:
        drug_id = data.drug_id.item()
        protein_id = data.target_id.item()
        affinity = data.y.item()
        drug_affinity[(drug_id,protein_id)] = affinity

    # 遍历每个药物ID
    for drug_id in np.unique([data.drug_id.item() for data in train_dataset]):
        # 获取当前药物的所有蛋白质对的亲和力值
        protein_affinities = [protein_id for (d_id,protein_id),affinity in drug_affinity.items() if d_id == drug_id]
        if len(protein_affinities) < 2:
            continue

        # 随机选择一个蛋白质对作为锚点
        anchor_idx = np.random.choice(len(protein_affinities))
        anchor_protein_id = protein_affinities[anchor_idx]
        anchor_affinity = drug_affinity[(drug_id,anchor_protein_id)]

        # 寻找亲和力接近的蛋白质对作为正样本
        similar_protein_ids = [protein_id for protein_id in protein_affinities if
                               protein_id != anchor_protein_id and abs(
                                   drug_affinity[(drug_id,protein_id)] - anchor_affinity) < threshold]
        if similar_protein_ids:
            positive_protein_id = np.random.choice(similar_protein_ids)
            positive_pairs.append((drug_id,anchor_protein_id,positive_protein_id))

        # 寻找亲和力差距大的蛋白质对作为负样本
        dissimilar_protein_ids = [protein_id for protein_id in protein_affinities if
                                  protein_id != anchor_protein_id and abs(
                                      drug_affinity[(drug_id,protein_id)] - anchor_affinity) > threshold]
        if dissimilar_protein_ids:
            negative_protein_id = np.random.choice(dissimilar_protein_ids)
            negative_pairs.append((drug_id,anchor_protein_id,negative_protein_id))

    return positive_pairs,negative_pairs


#新改
#正负样本对的数量不同
def select_positive_negative_pairs_v2(train_dataset,threshold=0.05,max_diff=5):
    # 去掉亲和力为5的蛋白质-药物对，并按照亲和力从大到小排序
    filtered_data = [data for data in train_dataset if data.y.item() != 5]
    sorted_data = sorted(filtered_data,key=lambda x: x.y.item(),reverse=True)

    # 初始化正负样本集合
    positive_pairs = []
    negative_pairs = []
    pos_affs= []
    neg_affs= []
    # 生成正样本对
    for i in range(len(sorted_data) - 1):
        # 相邻的蛋白质-药物对，如果亲和力差距小于阈值 threshold 则加入正样本集合
        drug_id1,protein_id1,affinity1 = sorted_data[i].drug_id.item(),sorted_data[i].target_id.item(),sorted_data[
            i].y.item()
        drug_id2,protein_id2,affinity2 = sorted_data[i + 1].drug_id.item(),sorted_data[i + 1].target_id.item(), \
        sorted_data[i + 1].y.item()

        if abs(affinity1 - affinity2) < threshold:
            positive_pairs.append((drug_id1,protein_id1,protein_id2))
            pos_affs.append((affinity1,affinity2))

    # 生成负样本对
    n = len(sorted_data)
    for i in range(n):
        anchor_drug_id,anchor_protein_id,anchor_affinity = sorted_data[i].drug_id.item(),sorted_data[
            i].target_id.item(),sorted_data[i].y.item()

        # 从倒数第一个开始往前找符合条件的负样本
        for j in range(n - 1,i,-1):
            compare_protein_id = sorted_data[j].target_id.item()
            compare_affinity = sorted_data[j].y.item()

            # 检查亲和力差距
            if abs(anchor_affinity - compare_affinity) > max_diff:
                negative_pairs.append((anchor_drug_id,anchor_protein_id,compare_protein_id))
                neg_affs.append((anchor_affinity,compare_affinity))
            else:
                # 一旦亲和力差距 <= max_diff，停止本层循环，开始选择下一个 anchor
                break

    return positive_pairs,negative_pairs
#新改
#正负样本对的数量相同，去除多余的
def select_positive_negative_pairs_v3(train_dataset,threshold=0.05,max_diff=5):
    # 去掉亲和力为5的蛋白质-药物对，并按照亲和力从大到小排序
    filtered_data = [data for data in train_dataset if data.y.item() != 5]
    sorted_data = sorted(filtered_data,key=lambda x: x.y.item(),reverse=True)

    # 初始化正负样本集合
    positive_pairs = []
    negative_pairs = []
    pos_affs= []
    neg_affs= []
    # 生成正样本对
    for i in range(len(sorted_data) - 1):
        # 相邻的蛋白质-药物对，如果亲和力差距小于阈值 threshold 则加入正样本集合
        drug_id1,protein_id1,affinity1 = sorted_data[i].drug_id.item(),sorted_data[i].target_id.item(),sorted_data[
            i].y.item()
        drug_id2,protein_id2,affinity2 = sorted_data[i + 1].drug_id.item(),sorted_data[i + 1].target_id.item(), \
        sorted_data[i + 1].y.item()

        if abs(affinity1 - affinity2) < threshold:
            positive_pairs.append((drug_id1,protein_id1,protein_id2))
            pos_affs.append((affinity1,affinity2))

    # 生成负样本对
    n = len(sorted_data)
    for i in range(n):
        anchor_drug_id,anchor_protein_id,anchor_affinity = sorted_data[i].drug_id.item(),sorted_data[
            i].target_id.item(),sorted_data[i].y.item()

        # 从倒数第一个开始往前找符合条件的负样本
        for j in range(n - 1,i,-1):
            compare_protein_id = sorted_data[j].target_id.item()
            compare_affinity = sorted_data[j].y.item()

            # 检查亲和力差距
            if abs(anchor_affinity - compare_affinity) > max_diff:
                negative_pairs.append((anchor_drug_id,anchor_protein_id,compare_protein_id))
                neg_affs.append((anchor_affinity,compare_affinity))
            else:
                # 一旦亲和力差距 <= max_diff，停止本层循环，开始选择下一个 anchor
                break
            # 调整正负样本及其亲和力对的数量一致
    min_len = min(len(positive_pairs),len(negative_pairs))
    if len(positive_pairs) > min_len:
        indices = random.sample(range(len(positive_pairs)),min_len)
        positive_pairs = [positive_pairs[i] for i in indices]
        pos_affs = [pos_affs[i] for i in indices]
    elif len(negative_pairs) > min_len:
        indices = random.sample(range(len(negative_pairs)),min_len)
        negative_pairs = [negative_pairs[i] for i in indices]
        neg_affs = [neg_affs[i] for i in indices]

    return positive_pairs,negative_pairs

def get_affinity_graph(dataset, adj, num_pos, pos_threshold):
    dataset_path = 'data/' + dataset + '/'
    num_drug, num_target = adj.shape[0], adj.shape[1]
#改
    """
    dt_ = adj.copy()
    dt_ = np.where(dt_ >= pos_threshold, newLoss+mseLoss+ContracsLoss_ADAM_512.0, 0.0)
    dtd = np.matmul(dt_, dt_.T)
    dtd = dtd / dtd.sum(axis=-newLoss+mseLoss+ContracsLoss_ADAM_512).reshape(-newLoss+mseLoss+ContracsLoss_ADAM_512, newLoss+mseLoss+ContracsLoss_ADAM_512)
    dtd = np.nan_to_num(dtd)
    dtd += np.eye(num_drug, num_drug)
    dtd = dtd.astype("float32")
    d_d = np.loadtxt(dataset_path + 'drug-drug-sim.txt', delimiter=',')
    dAll = dtd + d_d
    
    drug_pos = np.zeros((num_drug, num_drug))
    for i in range(len(dAll)):
        one = dAll[i].nonzero()[0]
        if len(one) > num_pos:
            oo = np.argsort(-dAll[i, one])
            sele = one[oo[:num_pos]]
            drug_pos[i, sele] = newLoss+mseLoss+ContracsLoss_ADAM_512
        else:
            drug_pos[i, one] = newLoss+mseLoss+ContracsLoss_ADAM_512
    drug_pos = sp.coo_matrix(drug_pos)
    drug_pos = sparse_mx_to_torch_sparse_tensor(drug_pos)

    td_ = adj.T.copy()
    td_ = np.where(td_ >= pos_threshold, newLoss+mseLoss+ContracsLoss_ADAM_512.0, 0.0)
    tdt = np.matmul(td_, td_.T)
    tdt = tdt / tdt.sum(axis=-newLoss+mseLoss+ContracsLoss_ADAM_512).reshape(-newLoss+mseLoss+ContracsLoss_ADAM_512, newLoss+mseLoss+ContracsLoss_ADAM_512)
    tdt = np.nan_to_num(tdt)
    tdt += np.eye(num_target, num_target)
    tdt = tdt.astype("float32")
    t_t = np.loadtxt(dataset_path + 'target-target-sim.txt', delimiter=',')
    tAll = tdt + t_t
    target_pos = np.zeros((num_target, num_target))
    for i in range(len(tAll)):
        one = tAll[i].nonzero()[0]
        if len(one) > num_pos:
            oo = np.argsort(-tAll[i, one])
            sele = one[oo[:num_pos]]
            target_pos[i, sele] = newLoss+mseLoss+ContracsLoss_ADAM_512
        else:
            target_pos[i, one] = newLoss+mseLoss+ContracsLoss_ADAM_512
    target_pos = sp.coo_matrix(target_pos)
    target_pos = sparse_mx_to_torch_sparse_tensor(target_pos)
    """
#处理davis数据集，将为5的地方-5，也就是置0了
#但这个处理是在蛋白质-药物异构图中，将边去掉用的
    if dataset == "davis":
        adj[adj != 0] -= 5
        adj_norm = minMaxNormalize(adj, 0)
    elif dataset == "kiba":
        adj_refine = denseAffinityRefine(adj.T, 150)
        adj_refine = denseAffinityRefine(adj_refine.T, 40)
        adj_norm = minMaxNormalize(adj_refine, 0)
    adj_1 = adj_norm
    adj_2 = adj_norm.T
    adj = np.concatenate((
        np.concatenate((np.zeros([num_drug, num_drug]), adj_1), 1),
        np.concatenate((adj_2, np.zeros([num_target, num_target])), 1)
    ), 0)
    train_row_ids, train_col_ids = np.where(adj != 0)
    edge_indexs = np.concatenate((
        np.expand_dims(train_row_ids, 0),
        np.expand_dims(train_col_ids, 0)
    ), 0)
    edge_weights = adj[train_row_ids, train_col_ids]
    node_type_features = np.concatenate((
        np.tile(np.array([1, 0]), (num_drug, 1)),
        np.tile(np.array([0, 1]), (num_target, 1))
    ), 0)
    adj_features = np.zeros_like(adj)
    adj_features[adj != 0] = 1
    features = np.concatenate((node_type_features, adj_features), 1)
    affinity_graph = DATA.Data(x=torch.Tensor(features), adj=torch.Tensor(adj),
                               edge_index=torch.LongTensor(edge_indexs))
    affinity_graph.__setitem__("edge_weight", torch.Tensor(edge_weights))
    affinity_graph.__setitem__("num_drug", num_drug)
    affinity_graph.__setitem__("num_target", num_target)
#改
    #return affinity_graph, drug_pos, target_pos
    return affinity_graph


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))

    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    """
    print("原子类型："+str(atom.GetSymbol()))
    print("原子度："+str(atom.GetDegree()))
    print("连接的氢原子数："+str(atom.GetTotalNumHs()))
    print("原子隐含价："+str(atom.GetImplicitValence()))
    print("原子芳香性:"+str(atom.GetIsAromatic()))
    """

    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def get_drug_molecule_graph(ligands):
    smile_graph = OrderedDict()

    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        smile_graph[d] = smile_to_graph(lg)

    return smile_graph


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])

    return c_size, features, edge_index

"""
newLoss+mseLoss+ContracsLoss_ADAM_512.从 contact_dir 加载接触图文件：
    contact_dir 下的 .npy 文件包含的是每个蛋白质的氨基酸残基之间的接触图，也就是表示氨基酸残基之间是否存在相互作用或接触。这个接触图实际上是蛋白质的边（edges）信息，表示哪些氨基酸残基之间有边（关系）。
加载这个 .npy 文件后，代码通过 np.where(contact_map >= 0.5) 来筛选出相互作用关系的边，并将这些氨基酸残基之间的关系作为图的边（target_edge_index）。
2.从 aln_dir 提取氨基酸特征：
    aln_dir 目录下的比对文件提供了蛋白质的氨基酸特征。这些特征可以通过比对文件提取出每个氨基酸残基的保守性、物理化学特性、或者其它相关的序列信息。代码通过 target_to_feature 函数从这些比对文件中提取每个氨基酸的特征，最终得到每个蛋白质的节点特征（target_feature）。
这些特征代表了每个氨基酸残基的属性，可能包括序列保守性、演化信息等，这些信息将作为图中每个节点（氨基酸残基）的特征。
3.整合：
    图的节点和边的组合：
        节点：每个氨基酸残基作为图中的节点，节点的特征来自 aln 文件中的比对信息。
        边：节点之间的边来自接触图（contact_map），边表示氨基酸残基之间的物理接触或结构上的相互作用关系。
最终，蛋白质的图结构由两个部分组成：
    节点特征（target_feature），来自 aln 文件的比对特征。
    节点之间的边（target_edge_index），来自 contact_map 中的接触图
"""
def get_target_molecule_graph(proteins, dataset):
    msa_path = 'data/' + dataset + '/aln'
    contac_path = 'data/' + dataset + '/pconsc4'    

    target_graph = OrderedDict()
    for t in proteins.keys():
        g = target_to_graph(t, proteins[t], contac_path, msa_path)
        target_graph[t] = g

    return target_graph

#靶点_>蛋白质分子图

def target_to_graph(target_key, target_sequence, contact_dir, aln_dir):
    target_size = len(target_sequence)
    contact_file = os.path.join(contact_dir, target_key + '.npy')

    target_feature = target_to_feature(target_key, target_sequence, aln_dir)

    contact_map = np.load(contact_file)
    contact_map += np.matrix(np.eye(target_size))
    index_row, index_col = np.where(contact_map >= 0.5)
    target_edge_index = []
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
    target_edge_index = np.array(target_edge_index)

    return target_size, target_feature, target_edge_index


def target_feature(aln_file, pro_seq):
    pssm = PSSM_calculation(aln_file, pro_seq)
    # print(pssm.shape)
    other_feature = seq_feature(pro_seq)
    # print(other_feature.shape)

    return np.concatenate((np.transpose(pssm, (1, 0)), other_feature), axis=1)


def target_to_feature(target_key, target_sequence, aln_dir):
    aln_file = os.path.join(aln_dir, target_key + '.aln')
    feature = target_feature(aln_file, target_sequence)

    return feature


def PSSM_calculation(aln_file, pro_seq):
    pfm_mat = np.zeros((len(pro_res_table), len(pro_seq)))
    with open(aln_file, 'r') as f:
        line_count = len(f.readlines())
        for line in f.readlines():
            if len(line) != len(pro_seq):
                print('error', len(line), len(pro_seq))
                continue
            count = 0
            for res in line:
                if res not in pro_res_table:
                    count += 1
                    continue
                pfm_mat[pro_res_table.index(res), count] += 1
                count += 1
    pseudocount = 0.8
    ppm_mat = (pfm_mat + pseudocount / 4) / (float(line_count) + pseudocount)
    pssm_mat = ppm_mat

    return pssm_mat


def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]

    return np.array(res_property1 + res_property2)


def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))

    for i in range(len(pro_seq)):
        pro_hot[i, ] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i, ] = residue_features(pro_seq[i])

    return np.concatenate((pro_hot, pro_property), axis=1)




