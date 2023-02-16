import numpy as np
import os
import torch
from torch_geometric.data import Data, DataLoader


def create_data_list(graphs, features, labels):
    data_list = []
    for i in range(len(graphs)):
        x = torch.tensor(features[i])
        edge_index = torch.tensor(graphs[i])
        y = torch.tensor(labels[i].argmax())
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    return data_list

