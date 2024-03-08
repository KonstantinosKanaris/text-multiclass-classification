from typing import List, Optional

import torch
from torch import nn


class ElmanRNN(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, batch_first: bool = False
    ) -> None:
        super().__init__()

        self.batch_first: bool = batch_first
        self.hidden_size: int = hidden_size

        self.rnn_cell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

    def _initialize_hidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(size=(batch_size, self.hidden_size))

    def forward(
        self, x: torch.Tensor, initial_hidden: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """The forward pass of the `ElmanRNN` model.

        Args:
            x (torch.Tensor): An input data tensor.
                If `self.batch_first` is `True` then
                x.shape = (batch, seq_size, feat_size),
                else x.shape = (seq_size, batch, feat_size).
            initial_hidden (torch.Tensor, optional): The initial hidden state
                for the RNN, if any. Defaults to `None`.

        Returns:
            torch.Tensor: The outputs of the RNN at each time step.
                If `self.batch_first` is `True` then
                output.shape = (batch, seq_size, hidden_size) else
                output.shape = (seq_size, batch, hidden_size.
        """
        if self.batch_first:
            batch_size, seq_size, feat_size = x.size()
            x = x.permute(1, 0, 2)
        else:
            seq_size, batch_size, feat_size = x.size()

        hiddens: List[torch.Tensor] = []

        if initial_hidden is None:
            initial_hidden = self._initialize_hidden(batch_size=batch_size).to(x.device)

        hidden_t = initial_hidden

        for t in range(seq_size):
            hidden_t = self.rnn_cell(x[t], hidden_t)
            hiddens.append(hidden_t)

        stacked_hiddens = torch.stack(hiddens)

        if self.batch_first:
            stacked_hiddens = stacked_hiddens.permute(1, 0, 2)

        return stacked_hiddens
