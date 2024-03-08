from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from text_multiclass_classification.models.elman_rnn import ElmanRNN


class NewsClassifierWithCNN(nn.Module):
    """
    Args:
        num_embeddings (int): Number of embedding vectors.
            Usually the length of vocabulary.
        embedding_size (int): Size of the embedding vectors.
        num_channels (int): Number of convolutional kernels
            per layer.
        hidden_size (int): The size of hidden dimension for
            the linear layers.
        num_classes (int): The number of classification classes.
        dropout (float, optional): The dropout value (default=0.1).
        padding_idx (int, optional): An index representing a null
            position (default=0).
        pretrained_embeddings (torch.Tensor, optional). Pretrained
            word embeddings, if provided. Defaults to `None`
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_size: int,
        num_channels: int,
        hidden_size: int,
        num_classes: int,
        dropout: float = 0.1,
        padding_idx: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
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
            nn.Linear(in_features=num_channels, out_features=hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=num_classes),
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
        # After the embedding layer the shape is [batch, seq_len, emb_dim]
        # but the conv1d layers require input shape [batch, input_channels, seq_len]
        # that's why we permute the shape in order to become (in our case)
        # [batch_size, emb_dim, seq_len]
        x_embedded = self.emb(x).permute(0, 2, 1)

        features = self.conv_net(x_embedded)

        # average and remove the extra dimension
        remaining_size = features.size(dim=2)
        features = self.dropout(F.avg_pool1d(features, remaining_size).squeeze(dim=2))

        # mlp classifier
        return self.classifier(features)


class NewsClassifierWithRNN(nn.Module):
    """
    Args:
        num_embeddings (int): Number of embedding vectors.
            Usually the length of the vocabulary.
        embedding_size (int): Size of the embedding vectors.
        num_classes (int): The number of classification classes.
        hidden_size (int): The size of hidden dimension for
            the both linear and rnn layers.
        batch_first (bool, optional): Whether the input tensors
            will have batch or the sequence size on 0th dimension.
            Defaults to `True`.
        dropout (float, optional): The dropout value (default=0.1).
        padding_idx (int, optional): An index representing a null
            position (default=0).
        pretrained_embeddings (torch.Tensor, optional). Pretrained
            word embeddings, if provided. Defaults to `None`
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_size: int,
        num_classes: int,
        hidden_size: int,
        batch_first: bool = True,
        dropout: float = 0.1,
        padding_idx: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

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

        self.rnn = ElmanRNN(
            input_size=embedding_size, hidden_size=hidden_size, batch_first=batch_first
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_size, out_features=num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """The forward pass of the news classifier.

        Args:
            x (torch.Tensor): An input data tensor. The input shape
                should be (batch_size, input_dim).

        Returns:
            torch.Tensor: The resulting tensor. Output shape should
                be (batch_size, output_dim).
        """
        x_embedded = self.emb(x)
        y_out = self.rnn(x_embedded)
        y_out = y_out[:, -1, :]

        y_out = self.classifier(y_out)
        return y_out
