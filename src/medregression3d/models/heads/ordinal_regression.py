import torch
import torch.nn as nn

class OrdinalRegressionHead(nn.Module):
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
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(embed_dim, num_classes - 1)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))  # CORAL-style bias

    def forward(self, x):
        if self.patch_aggregation_method == "cls_token":
            assert self.cls_token_available
            x = x[:, 0]
        elif self.patch_aggregation_method == "avg":
            x = x[:, 1:].mean(dim=1) if self.cls_token_available else x.mean(dim=1)
        elif self.patch_aggregation_method == "sum":
            x = x[:, 1:].sum(dim=1) if self.cls_token_available else x.sum(dim=1)

        x = self.dropout(x)
        logits = self.linear(x) + self.bias
        probas = torch.sigmoid(logits)
        return logits, probas
    
class OrdinalRegressionHead_MLP(nn.Module):
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
            nn.Linear(hidden_dim2, num_classes - 1),  # CORAL-style output (K - 1)
        )

        self.bias = nn.Parameter(torch.zeros(num_classes - 1))  # CORAL-style bias

    def forward(self, x):
        if self.patch_aggregation_method == "cls_token":
            assert self.cls_token_available
            x = x[:, 0]
        elif self.patch_aggregation_method == "avg":
            x = x[:, 1:].mean(dim=1) if self.cls_token_available else x.mean(dim=1)
        elif self.patch_aggregation_method == "sum":
            x = x[:, 1:].sum(dim=1) if self.cls_token_available else x.sum(dim=1)

        logits = self.mlp(x) + self.bias
        probas = torch.sigmoid(logits)
        return logits, probas