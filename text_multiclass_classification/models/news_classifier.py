from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class NewsClassifier(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        num_embeddings: int,
        num_channels: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        if pretrained_embeddings is None:
            self.emb = nn.Embedding(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_size,
                padding_idx=padding_idx,
            )
        else:
            self.emb = nn.Embedding(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_size,
                padding_idx=padding_idx,
                _weight=pretrained_embeddings,
            )

        self.conv_net = nn.Sequential(
            nn.Conv1d(
                in_channels=embedding_size, out_channels=num_channels, kernel_size=3
            ),
            nn.ELU(),
            nn.Conv1d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                stride=2,
            ),
            nn.ELU(),
            nn.Conv1d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                stride=2,
            ),
            nn.ELU(),
            nn.Conv1d(
                in_channels=num_channels, out_channels=num_channels, kernel_size=3
            ),
            nn.ELU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=num_channels, out_features=hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward pass of the classifier.

        Args:
            x (torch.Tensor): An input data tensor. x.shape
                should be (batch_size, num_classes).

        Returns:
             torch.Tensor: The resulting tensor. The output
                shape of the tensor should be (batch_size, num_classes).
        """
        # After the embedding layer the shape is [batch_size, seq_len, emb_dim]
        # but the conv1d layers require input shape [batch_size, input_channels, seq_len]
        # that's why we permute the shape in order to become (in our case)
        # [batch_size, emb_dim, seq_len]
        x_embedded = self.emb(x).permute(0, 2, 1)

        features = self.conv_net(x_embedded)

        # average and remove the extra dimension
        remaining_size = features.size(dim=2)
        features = self.dropout(F.avg_pool1d(features, remaining_size).squeeze(dim=2))

        # mlp classifier
        return self.classifier(features)
