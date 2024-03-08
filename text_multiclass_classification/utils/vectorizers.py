from __future__ import annotations

import string
from collections import Counter
from typing import Tuple

import pandas as pd
import torch

from text_multiclass_classification.utils.vocabulary import (
    SequenceVocabulary,
    Vocabulary,
)


class SequenceVectorizer:
    """Responsible for vectorizing a provided text sequence.

    Args:
        text_vocab (SequenceVocabulary): Vocabulary constructed
            from dataset's collection of texts for classification.
        category_vocab (Vocabulary): Vocabulary constructed
            from dataset's classification categories.
    """

    def __init__(
        self, text_vocab: SequenceVocabulary, category_vocab: Vocabulary
    ) -> None:
        self.text_vocab: SequenceVocabulary = text_vocab
        self.category_vocab: Vocabulary = category_vocab

    def vectorize(self, text: str, vector_length=-1) -> Tuple[torch.Tensor, int]:
        """Creates a 1D fixed-length tensor (vector representation)
        for the provided text.

        Each word in the text sequence is converted to its corresponding
        index in the vocabulary. The text sequence is then wrapped with
        boundary markers and is padded with 0s to keep all vectors the
        same size.

        Args:
            text (str): Sequence of word strings separated with spaces.
            vector_length (int, optional): Length of the output
                vector. -1 means the length of the input tokenized
                text (default=-1).

        Returns:
            Tuple[torch.Tensor, int]: The vectorized text and an integer
                indicating the vector length.
        """
        indices = [self.text_vocab.begin_seq_index]
        indices.extend(self.text_vocab.lookup_token(token) for token in text.split(" "))
        indices.append(self.text_vocab.end_seq_index)

        if vector_length < 0:
            vector_length = len(indices)

        out_vector = torch.zeros(size=(vector_length,), dtype=torch.int64)
        out_vector[: len(indices)] = torch.tensor(indices, dtype=torch.int64)
        out_vector[len(indices) :] = self.text_vocab.mask_index  # noqa: E203

        return out_vector, len(indices)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        category_col: str,
        text_col: str,
        cutoff: int = 25,
    ):
        """Indicates the entry point for instantiating the
        `TextVectorizer` class from a dataframe.

        Constructs the category and text vocabularies, from the
        provided dataframes' category and text columns, filtering out
        infrequent tokens based on a cutoff frequency threshold.

        Args:
            df (pd.DataFrame): The target dataframe.
            category_col (str): The name of the column that
                contains the dataset's classification categories.
            text_col (str): The name of the column that contains
                the dataset's text to classify.
            cutoff (int, optional): Frequency threshold (default=25).

        Returns:
            Vectorizer: An instance of the `SequenceVectorizer` class.
        """
        category_vocab = Vocabulary()
        for category in sorted(set(df[category_col])):
            category_vocab.add_token(token=category)

        word_counts: Counter = Counter()
        for text in df[text_col].values:
            for token in text.split(" "):
                if token not in string.punctuation:
                    word_counts[token] += 1

        text_vocab = SequenceVocabulary()
        for word, word_count in word_counts.items():
            if word_count >= cutoff:
                text_vocab.add_token(token=word)

        return cls(text_vocab, category_vocab)
