import numpy as np
import torch
from torch import nn
from torch.nn import ReLU, Linear
from torch_geometric.nn import GCNConv, GATConv, GatedGraphConv, GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool


class GraphGNN(nn.Module):
    def __init__(self, in_features_dim, out_features_dim, num_classes, word2vec, gnn_type):
        super(GraphGNN, self).__init__()
        self.in_features_dim = in_features_dim

        self.word_embed_size = word2vec.shape[1]
        word_size = word2vec.shape[0]
        unknown_word = np.zeros((1, self.word_embed_size))
        self.word2vec = torch.from_numpy(np.concatenate([unknown_word, word2vec], axis=0).astype(np.float32))
        self.lookup = torch.nn.Embedding(num_embeddings=word_size + 1,
                                         embedding_dim=self.word_embed_size).from_pretrained(self.word2vec, freeze=True)

        self.word_conv = torch.nn.Conv1d(in_channels=50, out_channels=50, kernel_size=2)

        if gnn_type == 'gcn':
            self.conv1 = GCNConv(in_features_dim, out_features_dim)
        elif gnn_type == 'gat':
            self.conv1 = GATConv(in_features_dim, out_features_dim)
        elif gnn_type == 'ggnn':
            self.conv1 = GatedGraphConv(out_features_dim, 2)
        elif gnn_type == 'gin':
            self.gin_mlp = torch.nn.Linear(in_features_dim, out_features_dim)
            self.conv1 = GINConv(self.gin_mlp)
        else:
            raise Exception("wrong gnn_type", gnn_type)
        self.relu1 = ReLU()
        self.lin = Linear(out_features_dim * 2, num_classes)

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)
        embed = self.embedding(x, edge_index)
        out1 = global_max_pool(embed, batch)
        out2 = global_mean_pool(embed, batch)
        input_in = torch.cat([out1, out2], dim=-1)
        out = self.lin(input_in)
        return out

    def embedding(self, x, edge_index):
        new_x = []
        for n in x:
            if n[-1] == -1:
                new_x.append(torch.zeros(50 + 8))
            else:
                node_tensor = self.lookup(n[:-1]).unsqueeze(dim=0)  # 1, 38, 50
                node_tensor = self.word_conv(node_tensor.permute(0, 2, 1))  # 1, 50, 37
                node_tensor = torch.mean(node_tensor, dim=-1)  # 1, 50
                node_type_info = torch.nn.functional.one_hot(torch.tensor([n[-1]], dtype=torch.long),
                                                             num_classes=8).float()
                node_tensor = torch.cat((node_tensor, node_type_info), dim=1)
                new_x.append(node_tensor.squeeze())
        x = torch.stack(new_x, dim=0)

        out1 = self.conv1(x, edge_index)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = self.relu1(out1)
        return out1
