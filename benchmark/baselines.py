from __future__ import annotations

from dataclasses import dataclass


try:
    import torch
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyTorch is required for baseline models. Install torch in your environment.") from e


@dataclass(frozen=True)
class BaselineHParams:
    hidden_dim: int = 64
    dropout: float = 0.5
    lr: float = 1e-3
    weight_decay: float = 5e-4
    max_epochs: int = 100
    patience: int = 10


class MLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 2, dropout: float = 0.5):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 2, dropout: float = 0.5, aggregator: str = "mean"):
        try:
            from dgl.nn import SAGEConv
        except Exception as e:  # pragma: no cover
            raise RuntimeError("DGL is required for the GraphSAGE baseline. Install dgl (and torch).") from e

        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim, aggregator_type=aggregator)
        self.conv2 = SAGEConv(hidden_dim, out_dim, aggregator_type=aggregator)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, g, x):
        import torch.nn.functional as F

        h = self.conv1(g, x)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(g, h)
        return h


def build_baseline(model_id: str, *, in_dim: int, hparams: BaselineHParams):
    model_id = str(model_id)
    if model_id == "mlp":
        return MLP(in_dim=in_dim, hidden_dim=int(hparams.hidden_dim), dropout=float(hparams.dropout))
    if model_id == "sage":
        return GraphSAGE(in_dim=in_dim, hidden_dim=int(hparams.hidden_dim), dropout=float(hparams.dropout))
    raise ValueError(f"Unknown baseline model_id: {model_id}")
