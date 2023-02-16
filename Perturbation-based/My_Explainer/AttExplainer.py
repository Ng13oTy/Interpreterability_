import logging
from typing import List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.config import ExplanationType, ModelTaskLevel
from torch_geometric.nn.conv.message_passing import MessagePassing


class AttExplainer:
    def __init__(self, reduce: str = 'mean'):
        super().__init__()
        self.reduce = reduce

    def explain(
            self,
            model: torch.nn.Module,
            x: Tensor,
            edge_index: Tensor,
            target: Tensor,
    ):
        alphas: List[Tensor] = []

        def hook(module, msg_kwargs, out):
            if 'alpha' in msg_kwargs[0]:
                alphas.append(msg_kwargs[0]['alpha'].detach())
            elif getattr(module, '_alpha', None) is not None:
                alphas.append(module._alpha.detach())

        hook_handles = []
        for module in model.modules():  # Register message forward hooks:
            if isinstance(module, MessagePassing):
                hook_handles.append(module.register_message_forward_hook(hook))

        model(x, edge_index)

        for handle in hook_handles:  # Remove hooks:
            handle.remove()

        if len(alphas) == 0:
            raise ValueError("Could not collect any attention coefficients. "
                             "Please ensure that your model is using "
                             "attention-based GNN layers.")

        for i, alpha in enumerate(alphas):
            alpha = alpha[:edge_index.size(1)]  # Respect potential self-loops.
            if alpha.dim() == 2:
                alpha = getattr(torch, self.reduce)(alpha, dim=-1)
                if isinstance(alpha, tuple):  # Respect `torch.max`:
                    alpha = alpha[0]
            elif alpha.dim() > 2:
                raise ValueError(f"Can not reduce attention coefficients of "
                                 f"shape {list(alpha.size())}")
            alphas[i] = alpha

        if len(alphas) > 1:
            alpha = torch.stack(alphas, dim=-1)
            alpha = getattr(torch, self.reduce)(alpha, dim=-1)
            if isinstance(alpha, tuple):  # Respect `torch.max`:
                alpha = alpha[0]
        else:
            alpha = alphas[0]

        alpha = alpha.detach()
        return alpha
