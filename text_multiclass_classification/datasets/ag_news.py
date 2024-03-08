from __future__ import annotations

from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from text_multiclass_classification.datasets.utils import (
    basic_text_normalization,
    download_data,
)
from text_multiclass_classification.utils.vectorizers import SequenceVectorizer

URL = {
    "train": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",  # noqa: E501
    "test": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",  # noqa: E501
}


class NewsDataset(Dataset):
    def __init__(
        self,
        news_df: pd.DataFrame,
        vectorizer: SequenceVectorizer,
    ) -> None:
        self.news_df = news_df
        self._vectorizer = vectorizer

        # +1 if using only begin_seq, +2 if using both begin and
        # end seq tokens
        self._max_seq_length = self.news_df.title.str.len().max() + 2

    def get_vectorizer(self) -> SequenceVectorizer:
        """Returns the vectorizer."""
        return self._vectorizer

    def get_num_batches(self, batch_size) -> int:
        """Given a batch size, returns the number of batches
        in the dataset.

        Args:
            batch_size (int): The batch size.
        Returns:
            int: The number of batches in the dataset.
        """
        return len(self) // batch_size

    @classmethod
    def load_dataset_from_csv(cls, news_csv: str) -> NewsDataset:
        """Entry point for instantiating the `NewsDataset` class
        from a csv file containing the data.

        Args:
            news_csv (str): The path to the dataset's csv file.

        Returns:
            NewsDataset: An instance of the `NewsDataset` class.
        """
        news_df = pd.read_csv(filepath_or_buffer=news_csv)
        news_df.title = news_df.title.map(basic_text_normalization)
        return cls(
            news_df=news_df,
            vectorizer=SequenceVectorizer.from_dataframe(
                df=news_df, category_col="category", text_col="title"
            ),
        )

    @classmethod
    def load_dataset_from_url(
        cls,
        news_csv_url: str,
        save_path: str,
    ) -> NewsDataset:
        """Entry point for instantiating the `NewsDataset` class
        from an url where the data is located at, as a csv file.

        Args:
            news_csv_url (str): The url to the dataset's csv file.
            save_path (Str): A local filepath to save the downloaded data.

        Returns:
            NewsDataset: An instance of the `NewsDataset` class.
        """
        news_csv = download_data(source=news_csv_url, destination=save_path)
        return cls.load_dataset_from_csv(news_csv)

    def __len__(self) -> int:
        return len(self.news_df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """The primary entry point method for PyTorch dataset.

        Args:
            index (int): The index to the data point.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple holding the
                data point's:
                .. code-block:: text

                    features (X)
                    label (y)
        """
        row = self.news_df.iloc[index]

        title_vector, vec_length = self._vectorizer.vectorize(
            row.title, self._max_seq_length
        )
        category_index = self._vectorizer.category_vocab.lookup_token(row.category)

        return title_vector, torch.tensor(data=category_index, dtype=torch.int64)
