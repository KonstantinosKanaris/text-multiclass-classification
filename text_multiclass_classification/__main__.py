"""The python-template orchestrator.

Executes the core module of the project.
"""

import argparse

import torch
import torchmetrics
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler

from text_multiclass_classification import __title__, logger
from text_multiclass_classification.datasets.ag_news import NewsDataset
from text_multiclass_classification.engine.trainer import TrainingExperiment
from text_multiclass_classification.models.news_classifier import NewsClassifier
from text_multiclass_classification.utils.embeddings import PreTrainedEmbeddings


def parse_arguments() -> argparse.Namespace:
    """
    Constructs parsers and subparsers.

    Returns:
        argparse.Namespace:
            The parser/subparser with its arguments.
    """
    parser = argparse.ArgumentParser(
        description=f"Command Line Interface for {__title__}"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="The path to the configuration file.",
    )

    subparsers = parser.add_subparsers(
        description="Project functionalities", dest="mode"
    )

    train = subparsers.add_parser(
        name="train", help="This is the subparser for training."
    )

    train.add_argument(
        "--config",
        type=str,
        required=True,
        help="The config file (yaml) required for training."
        "Contains data paths, hyperparameters, etc.",
    )

    train.add_argument(
        "--resume_from_checkpoint",
        type=str,
        choices={"yes", "no"},
        required=False,
        default="no",
        help="If `yes` the training will resume from the last " "saved checkpoint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # arguments = parse_arguments()

    glove_embeddings_path = "./.vector_cache/glove.6B.100d.txt"
    news_csv_path = "./data/ag_news/processed/ag_news.csv"

    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    EMBEDDING_SIZE = 100
    NUM_CHANNELS = 100
    HIDDEN_DIM = 100

    embeddings_wrapper = PreTrainedEmbeddings.from_embeddings_file(
        embedding_file=glove_embeddings_path
    )

    dataset = NewsDataset.load_dataset_from_csv(news_csv=news_csv_path)
    vectorizer = dataset.get_vectorizer()

    input_embeddings = embeddings_wrapper.make_embedding_matrix(
        words=list(vectorizer.text_vocab.token_to_idx.keys())
    )

    model = NewsClassifier(
        embedding_size=EMBEDDING_SIZE,
        num_embeddings=len(vectorizer.text_vocab),
        num_classes=len(vectorizer.category_vocab.token_to_idx),
        num_channels=NUM_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        dropout=0.1,
        # pretrained_embeddings=input_embeddings,
        padding_idx=0,
    )

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    accuracy_fn: torchmetrics.Metric = torchmetrics.Accuracy(
        task="multiclass", num_classes=len(vectorizer.category_vocab.token_to_idx)
    )

    ssf = StratifiedShuffleSplit(n_splits=2, test_size=0.2)

    trainer = TrainingExperiment(
        model=model,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        optimizer=optimizer,
        checkpoint_path="checkpoints/model_3.pth",
        epochs=5,
        patience=3,
    )

    for fold, (train_ids, val_ids) in enumerate(
        ssf.split(X=dataset, y=dataset.news_df.category)
    ):
        train_dataloader = DataLoader(
            dataset=dataset,
            batch_size=BATCH_SIZE,
            drop_last=True,
            # num_workers=os.cpu_count(),
            pin_memory=True,
            sampler=SubsetRandomSampler(train_ids),
        )
        val_dataloader = DataLoader(
            dataset=dataset,
            batch_size=BATCH_SIZE,
            drop_last=True,
            # num_workers=os.cpu_count(),
            pin_memory=True,
            sampler=SubsetRandomSampler(val_ids),
        )

        trainer.train(train_dataloader=train_dataloader, val_dataloader=val_dataloader)

    # if arguments.resume_from_checkpoint == "yes":
    #     resume = True
    # else:
    #     resume = False
    #
    # config = load_yaml_file(filepath=arguments.config)

    logger.debug("End of program.")
