import torch
import torch.nn as nn

from torch_geometric.nn import DenseGCNConv, GCNConv, global_mean_pool as gep
from torch_geometric.utils import dropout_adj


class GCNBlock(nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(GCNBlock, self).__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            conv_layer = GCNConv(gcn_layers_dim[i], gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, edge_index, edge_weight, batch):
        output = x
        embeddings = []
        for conv_layer_index in range(len(self.conv_layers)):
            output = self.conv_layers[conv_layer_index](output, edge_index, edge_weight)
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(gep(output, batch))

        return embeddings


class GCNModel(nn.Module):
    def __init__(self, layers_dim):
        super(GCNModel, self).__init__()

        self.num_layers = len(layers_dim) - 1
        self.graph_conv = GCNBlock(layers_dim, relu_layers_index=list(range(self.num_layers)))

    def forward(self, graph_batchs):
        embedding_batchs = list(
                map(lambda graph: self.graph_conv(graph.x, graph.edge_index, None, graph.batch), graph_batchs))
        embeddings = []
        for i in range(self.num_layers):
            embeddings.append(torch.cat(list(map(lambda embedding_batch: embedding_batch[i], embedding_batchs)), 0))

        return embeddings


class DenseGCNBlock(nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(DenseGCNBlock, self).__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            conv_layer = DenseGCNConv(gcn_layers_dim[i], gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, adj):
        output = x
        embeddings = []
        for conv_layer_index in range(len(self.conv_layers)):
            output = self.conv_layers[conv_layer_index](output, adj, add_loop=False)
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(torch.squeeze(output, dim=0))

        return embeddings


class DenseGCNModel(nn.Module):
    def __init__(self, layers_dim, edge_dropout_rate=0.):
        super(DenseGCNModel, self).__init__()

        self.edge_dropout_rate = edge_dropout_rate
        self.num_layers = len(layers_dim) - 1
        self.graph_conv = DenseGCNBlock(layers_dim, 0.1, relu_layers_index=list(range(self.num_layers)),
                                        dropout_layers_index=list(range(self.num_layers)))

    def forward(self, graph):
        xs, adj, num_d, num_t = graph.x, graph.adj, graph.num_drug, graph.num_target
        indexs = torch.where(adj != 0)
        edge_indexs = torch.cat((torch.unsqueeze(indexs[0], 0), torch.unsqueeze(indexs[1], 0)), 0)
        edge_indexs_dropout, edge_weights_dropout = dropout_adj(edge_index=edge_indexs, edge_attr=adj[indexs],
                                                                p=self.edge_dropout_rate, force_undirected=True,
                                                                num_nodes=num_d + num_t, training=self.training)
        adj_dropout = torch.zeros_like(adj)
        adj_dropout[edge_indexs_dropout[0], edge_indexs_dropout[1]] = edge_weights_dropout

        embeddings = self.graph_conv(xs, adj_dropout)

        return embeddings


class LinearBlock(nn.Module):
    def __init__(self, linear_layers_dim, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(LinearBlock, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(len(linear_layers_dim) - 1):
            layer = nn.Linear(linear_layers_dim[i], linear_layers_dim[i + 1])
            self.layers.append(layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x):
        output = x
        embeddings = []
        for layer_index in range(len(self.layers)):
            output = self.layers[layer_index](output)
            if layer_index in self.relu_layers_index:
                output = self.relu(output)
            if layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(output)

        return embeddings
class CrossAttention(nn.Module):
    def __init__(self, hidden_dim,device):
        super(CrossAttention, self).__init__()
        self.query_linear = nn.Linear(hidden_dim, hidden_dim)
        self.key_linear = nn.Linear(hidden_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, hidden_dim)
        self.query_linear = self.query_linear.to(device)
        self.key_linear = self.key_linear.to(device)
        self.value_linear = self.value_linear.to(device)
        self.scale = torch.sqrt(torch.tensor(hidden_dim/2, dtype=torch.float))

    def forward(self, query, key, value):
        Q = self.query_linear(query)
        K = self.key_linear(key)
        V = self.value_linear(value)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute weighted sum
        output = torch.matmul(attention_weights, V)
        return output
#对比学习的代码
class Contrast(nn.Module):
    def __init__(self, hidden_dim, output_dim, tau, lam):
        super(Contrast, self).__init__()

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim))
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)
    """
    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-newLoss+mseLoss+ContracsLoss_ADAM_512, keepdim=True)
        z2_norm = torch.norm(z2, dim=-newLoss+mseLoss+ContracsLoss_ADAM_512, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)

        return sim_matrix
"""
    #交叉注意力的求法
    def forward(self, za, zb,device):
        za_proj = self.proj(za)
        zb_proj = self.proj(zb)
        cross_attn = CrossAttention(za_proj.shape[1],device).to(device)  # Assuming za_proj and zb_proj have the same shape
        za_cross_attn = cross_attn(za_proj, zb_proj, zb_proj)
        zb_cross_attn = cross_attn(zb_proj, za_proj, za_proj)
        
        return torch.cat((za_cross_attn, zb_cross_attn), 1)


class CSCoDTA(nn.Module):
    def __init__(self, tau, lam, ns_dims, d_ms_dims, t_ms_dims, embedding_dim=128, dropout_rate=0.2):
        super(CSCoDTA, self).__init__()

        self.output_dim = embedding_dim * 2

        self.affinity_graph_conv = DenseGCNModel(ns_dims, dropout_rate)
        self.drug_graph_conv = GCNModel(d_ms_dims)
        self.target_graph_conv = GCNModel(t_ms_dims)
#改
        self.drug_contrast = Contrast(ns_dims[-1], embedding_dim, tau, lam)
        self.target_contrast = Contrast(ns_dims[-1], embedding_dim, tau, lam)


# 改
    #def forward(self,affinity_graph,drug_graph_batchs,target_graph_batchs,drug_pos,target_pos):
    def forward(self,affinity_graph,drug_graph_batchs,target_graph_batchs,device ):
        num_d = affinity_graph.num_drug

        affinity_graph_embedding = self.affinity_graph_conv(affinity_graph)[-1]
        drug_graph_embedding = self.drug_graph_conv(drug_graph_batchs)[-1]
        target_graph_embedding = self.target_graph_conv(target_graph_batchs)[-1]
        affinity_graph_embedding = affinity_graph_embedding.to(device)
        drug_graph_embedding = drug_graph_embedding.to(device)
        target_graph_embedding = target_graph_embedding.to(device)
#改
        #dru_loss, drug_embedding = self.drug_contrast(affinity_graph_embedding[:num_d], drug_graph_embedding, drug_pos)
        #tar_loss, target_embedding = self.target_contrast(affinity_graph_embedding[num_d:], target_graph_embedding,target_pos)
        drug_embedding = self.drug_contrast(affinity_graph_embedding[:num_d], drug_graph_embedding,device )
        target_embedding = self.target_contrast(affinity_graph_embedding[num_d:], target_graph_embedding,device )
        # return dru_loss + tar_loss, drug_embedding, target_embedding
        return drug_embedding, target_embedding


class PredictModule(nn.Module):
    def __init__(self, embedding_dim=128, output_dim=1):
        super(PredictModule, self).__init__()

        self.prediction_func, prediction_dim_func = (lambda x, y: torch.cat((x, y), -1), lambda dim: 4 * dim)
        mlp_layers_dim = [prediction_dim_func(embedding_dim), 1024, 512, output_dim]

        self.mlp = LinearBlock(mlp_layers_dim, 0.1, relu_layers_index=[0, 1], dropout_layers_index=[0, 1])

    def forward(self, data, drug_embedding, target_embedding):
        drug_id, target_id, y = data.drug_id, data.target_id, data.y

        drug_feature = drug_embedding[drug_id.int().cpu().numpy()]
        target_feature = target_embedding[target_id.int().cpu().numpy()]

        concat_feature = self.prediction_func(drug_feature, target_feature)
        mlp_embeddings = self.mlp(concat_feature)
        link_embeddings = mlp_embeddings[-2]
        out = mlp_embeddings[-1]

        return out, link_embeddings



import torch
import torch.nn as nn
import torch.nn.functional as F

class Contrast2(nn.Module):
#目前这个计算亲和力的部分的模型与CSCoDTA中调用的那个Contrast里对应的计算亲和力的地方的模型不同
    def __init__(self, embedding_dim,device):
        super(Contrast2, self).__init__()
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, 128).to(device),
            nn.ReLU(),
            nn.Linear(128, 64).to(device),
            nn.ReLU(),
            nn.Linear(64, 1).to(device)  # 输出一个亲和力值
        )

    def prediction_func(self, drug_feature, target_feature):
        # 拼接药物和目标特征
        return torch.cat((drug_feature, target_feature), dim=-1)

    def forward(self, positive_pairs, negative_pairs, drug_embedding, target_embedding,device):
        drug_embedding = drug_embedding.to(device)
        target_embedding = target_embedding.to(device)
        pos_losses = []
        neg_losses = []

        for drug_id1, protein_id1, protein_id2 in positive_pairs:
            if device== 'cuda:0':
                drug_feature = drug_embedding[drug_id1.int().cpu().numpy()]
                target_feature1 = target_embedding[protein_id1.int().cpu().numpy()]
                target_feature2 = target_embedding[protein_id2.int().cpu().numpy()]
            else:
                # cpu
                drug_feature = drug_embedding[drug_id1]
                target_feature1 = target_embedding[protein_id1]
                target_feature2 = target_embedding[protein_id2]

            concat_feature = self.prediction_func(drug_feature, target_feature1)
            affinity1 = self.mlp(concat_feature)

            concat_feature = self.prediction_func(drug_feature, target_feature2)
            affinity2 = self.mlp(concat_feature)

            pos_loss = F.mse_loss(affinity1.view(-1,1).float().to(device), affinity2.view(-1,1).float().to(device))
            pos_losses.append(pos_loss)

        for drug_id, protein_id1, protein_id2 in negative_pairs:
            if device == 'cuda:0':
                drug_feature = drug_embedding[drug_id.int().cpu().numpy()]
                target_feature1 = target_embedding[protein_id1.int().cpu().numpy()]
                target_feature2 = target_embedding[protein_id2.int().cpu().numpy()]
            else:
                #cpu
                drug_feature = drug_embedding[drug_id]
                target_feature1 = target_embedding[protein_id1]
                target_feature2 = target_embedding[protein_id2]

            concat_feature = self.prediction_func(drug_feature, target_feature1)
            affinity1 = self.mlp(concat_feature)

            concat_feature = self.prediction_func(drug_feature, target_feature2)
            affinity2 = self.mlp(concat_feature)

            neg_loss = F.mse_loss(affinity1.view(-1,1).float().to(device), affinity2.view(-1,1).float().to(device))
            neg_losses.append(neg_loss)

        # 平均损失
        total_loss = torch.mean(torch.stack(pos_losses)) - torch.mean(torch.stack(neg_losses))
        return total_loss