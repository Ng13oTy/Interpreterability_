from math import sqrt
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks


class GNNExplainer:
    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
        'EPS': 1e-15,
    }

    def __init__(self, epochs: int = 100, lr: float = 0.01, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

        self.edge_mask = None

    def explain(self,
                model: torch.nn.Module,
                x: Tensor,
                edge_index: Tensor,
                target: Tensor,
                ):
        self._train(model, x, edge_index, target=target)
        mask = self.edge_mask.detach()
        mask = torch.sigmoid(mask)
        self._clean_model(model)
        return mask

    def _train(
            self,
            model: torch.nn.Module,
            x: Tensor,
            edge_index: Tensor,
            target: Tensor,
    ):
        self._initialize_masks(x, edge_index)
        set_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)
        parameters = [self.edge_mask]
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for _ in range(self.epochs):
            optimizer.zero_grad()
            y_hat, y = model(x, edge_index), target
            loss = self._loss(y_hat, y)

            loss.backward()
            optimizer.step()

    def _loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        loss = torch.nn.functional.cross_entropy(y_hat, y)

        m = self.edge_mask.sigmoid()

        edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
        loss = loss + self.coeffs['edge_size'] * edge_reduce(m)

        ent = -m * torch.log(m + self.coeffs['EPS']) - (
                1 - m) * torch.log(1 - m + self.coeffs['EPS'])
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        return loss

    def _initialize_masks(self, x: Tensor, edge_index: Tensor):
        device = x.device
        N, E = x.size(0), edge_index.size(1)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = Parameter(torch.randn(E, device=device) * std)

    def _clean_model(self, model):
        clear_masks(model)
        self.edge_mask = None
