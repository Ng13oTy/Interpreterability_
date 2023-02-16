import logging
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import ReLU, Sequential

from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import (
    ExplanationType,
    ModelMode,
    ModelTaskLevel,
)
from torch_geometric.nn import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.utils import get_message_passing_embeddings


class PGExplainer:
    coeffs = {
        'edge_size': 0.05,
        'edge_ent': 1.0,
        'temp': [5.0, 2.0],
        'bias': 0.0,
    }

    def __init__(self, epochs: int, lr: float = 0.003, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

        self.mlp = Sequential(
            Linear(-1, 64),
            ReLU(),
            Linear(64, 1),
        )
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        self._curr_epoch = -1

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.mlp)

    def train(
            self,
            epoch: int,
            model: torch.nn.Module,
            x: Tensor,
            edge_index: Tensor,
            target: Tensor,
    ):
        z = get_message_passing_embeddings(model, x, edge_index)[-1]
        self.optimizer.zero_grad()
        temperature = self._get_temperature(epoch)
        inputs = self._get_inputs(z, edge_index)
        logits = self.mlp(inputs).view(-1)
        edge_mask = self._concrete_sample(logits, temperature)
        set_masks(model, edge_mask, edge_index, apply_sigmoid=True)

        y_hat, y = model(x, edge_index), target

        loss = self._loss(y_hat, y, edge_mask)
        loss.backward()
        self.optimizer.step()

        clear_masks(model)
        self._curr_epoch = epoch

        return float(loss)

    def explain(
            self,
            model: torch.nn.Module,
            x: Tensor,
            edge_index: Tensor,
            target: Tensor,
    ):
        z = get_message_passing_embeddings(model, x, edge_index,)[-1]

        inputs = self._get_inputs(z, edge_index)
        logits = self.mlp(inputs).view(-1)
        mask = logits.detach()
        mask = torch.sigmoid(mask)
        return mask

    def _get_temperature(self, epoch: int) -> float:
        temp = self.coeffs['temp']
        return temp[0] * pow(temp[1] / temp[0], epoch / self.epochs)

    def _get_inputs(self, embedding: Tensor, edge_index: Tensor) -> Tensor:
        zs = [embedding[edge_index[0]], embedding[edge_index[1]]]
        return torch.cat(zs, dim=-1)

    def _concrete_sample(self, logits: Tensor,
                         temperature: float = 1.0) -> Tensor:
        bias = self.coeffs['bias']
        eps = (1 - 2 * bias) * torch.rand_like(logits) + bias
        return (eps.log() - (1 - eps).log() + logits) / temperature

    def _loss(self, y_hat: Tensor, y: Tensor, edge_mask: Tensor) -> Tensor:
        loss = torch.nn.functional.cross_entropy(y_hat, y)

        # Regularization loss:
        mask = edge_mask.sigmoid()
        size_loss = mask.sum() * self.coeffs['edge_size']
        mask = 0.99 * mask + 0.005
        mask_ent = -mask * mask.log() - (1 - mask) * (1 - mask).log()
        mask_ent_loss = mask_ent.mean() * self.coeffs['edge_ent']

        return loss + size_loss + mask_ent_loss
