import torch.nn as nn


class RegressionHead(nn.Module):
    """
    Simple regression head: aggregate tokens, dropout, then linear projection.

    Output shape:
        - `[B]` when ``num_outputs == 1`` (the head squeezes the trailing dim
          so downstream code doesn't need to ``view(-1)``).
        - `[B, num_outputs]` otherwise (multi-output regression).
    """

    def __init__(
        self,
        embed_dim,
        num_outputs=1,
        dropout=0.1,
        patch_aggregation_method="avg",
        cls_token_available=True,
    ):
        super().__init__()
        self.num_outputs = num_outputs
        self.patch_aggregation_method = patch_aggregation_method
        self.cls_token_available = cls_token_available
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_outputs)

    def forward(self, x):
        if self.patch_aggregation_method == "cls_token":
            assert self.cls_token_available
            x = x[:, 0]
        elif self.patch_aggregation_method == "avg":
            x = x[:, 1:].mean(dim=1) if self.cls_token_available else x.mean(dim=1)
        elif self.patch_aggregation_method == "sum":
            x = x[:, 1:].sum(dim=1) if self.cls_token_available else x.sum(dim=1)

        x = self.dropout(x)
        x = self.fc(x)
        if self.num_outputs == 1:
            x = x.squeeze(-1)
        return x


class RegressionHead_MLP(nn.Module):
    """MLP variant of :class:`RegressionHead`, mirroring OrdinalRegressionHead_MLP."""

    def __init__(
        self,
        embed_dim,
        num_outputs=1,
        dropout=0.1,
        patch_aggregation_method="avg",
        cls_token_available=True,
    ):
        super().__init__()
        self.num_outputs = num_outputs
        self.patch_aggregation_method = patch_aggregation_method
        self.cls_token_available = cls_token_available

        hidden_dim1 = 256
        hidden_dim2 = 128

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, num_outputs),
        )

    def forward(self, x):
        if self.patch_aggregation_method == "cls_token":
            assert self.cls_token_available
            x = x[:, 0]
        elif self.patch_aggregation_method == "avg":
            x = x[:, 1:].mean(dim=1) if self.cls_token_available else x.mean(dim=1)
        elif self.patch_aggregation_method == "sum":
            x = x[:, 1:].sum(dim=1) if self.cls_token_available else x.sum(dim=1)

        x = self.mlp(x)
        if self.num_outputs == 1:
            x = x.squeeze(-1)
        return x
