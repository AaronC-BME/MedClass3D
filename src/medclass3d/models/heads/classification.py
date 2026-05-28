import torch.nn as nn
from timm.layers import ClassifierHead


class ClassificationHead(nn.Module):
    """Linear classification head.

    For ResEncoder, which pre-pools its encoder output to ``[B, C]`` via a mean
    over spatial dims, pass ``patch_aggregation_method=None``. For ViT-style
    backbones emitting ``[B, N, C]`` token tensors, use ``cls_token`` / ``avg``
    / ``sum``.

    Output shape: ``[B, num_classes]`` (logits).
    """

    def __init__(
        self,
        embed_dim,
        num_classes,
        dropout=0.1,
        patch_aggregation_method="avg",
        cls_token_available=True,
    ):
        super().__init__()
        self.fc = ClassifierHead(embed_dim, num_classes, "", dropout)
        self.patch_aggregation_method = patch_aggregation_method
        self.cls_token_available = cls_token_available

    def forward(self, x):
        if self.patch_aggregation_method is not None:
            if self.patch_aggregation_method == "cls_token":
                assert self.cls_token_available
                x = x[:, 0]
            elif self.patch_aggregation_method == "avg":
                x = x[:, 1:].mean(dim=1) if self.cls_token_available else x.mean(dim=1)
            elif self.patch_aggregation_method == "sum":
                x = x[:, 1:].sum(dim=1) if self.cls_token_available else x.sum(dim=1)

        return self.fc(x)


class ClassificationHead_MLP(nn.Module):
    """MLP variant of :class:`ClassificationHead`.

    Two-layer MLP (256 → 128 → num_classes) with dropout between each layer.
    """

    def __init__(
        self,
        embed_dim,
        num_classes,
        dropout=0.1,
        patch_aggregation_method="avg",
        cls_token_available=True,
    ):
        super().__init__()
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
            nn.Linear(hidden_dim2, num_classes),
        )

    def forward(self, x):
        if self.patch_aggregation_method is not None:
            if self.patch_aggregation_method == "cls_token":
                assert self.cls_token_available
                x = x[:, 0]
            elif self.patch_aggregation_method == "avg":
                x = x[:, 1:].mean(dim=1) if self.cls_token_available else x.mean(dim=1)
            elif self.patch_aggregation_method == "sum":
                x = x[:, 1:].sum(dim=1) if self.cls_token_available else x.sum(dim=1)

        return self.mlp(x)
