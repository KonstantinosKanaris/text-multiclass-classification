import os
from typing import Any, Dict

import torch
import torchmetrics
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler

from text_multiclass_classification import logger
from text_multiclass_classification.datasets.ag_news import NewsDataset
from text_multiclass_classification.engine.trainer import TrainingExperiment
from text_multiclass_classification.factories.client import Client
from text_multiclass_classification.utils.aux import Timer, create_writer
from text_multiclass_classification.utils.embeddings import PreTrainedEmbeddings


class ExperimentManager:
    """A class to manage and run experiments with PyTorch models
    on custom data.

    This class provides methods to run multiple experiments specified
    in a configuration file.Each experiment consists of parameters such
    as the experiment name, directories for tracking and saving models,
    data paths for training and testing, and hyperparameters for training.

    Args:
        config (Dict[str, Any]):
            A dictionary containing experiment configurations.
        resume_from_checkpoint (bool):
            If 'True' the selected model will resume training
            from the last selected checkpoint. Defaults to `False.

    Attributes:
        config (Dict[str, Any]):
            The experiment configurations loaded from the
            configuration file.
        resume_from_checkpoint (bool):
            If 'True' the selected model will resume training
            from the last selected checkpoint. Defaults to `False.
        client: (Client):
            A factory client to select model, and optimizer.

    Example:

        To use this class, instantiate an `ExperimentManager` object with
        the experiment configurations, and then call the `run_experiments()`
        method to execute the experiments.

        >>> experiment_config = {
        ...     'tracking_dir': './runs',
        ...     'experiments': [
        ...         {
        ...             'name': 'experiment_1',
        ...             'data': {
        ...                 'train_csv': 'path_to_csv_dir/csv_file.csv'
        ...             },
        ...             'hyperparameters': {
        ...                 'general': {
        ...                     'num_epochs': 10,
        ...                     'batch_size': 32
        ...                 },
        ...                 'optimizer': {
        ...                     'optimizer_name': 'adam',
        ...                     'learning_rate': 0.001,
        ...                     'weight_decay': 0.3,
        ...                 },
        ...                 'early_stopping': {
        ...                     'patience': 3,
        ...                     'delta': 0.05
        ...                 },
        ...                 'model': {
        ...                     'model_name': 'news_classifier_rnn',
        ...                     'embeddings_path': 'glove_embeddings/glove.6B.100d.txt'
        ...                 }
        ...             }
        ...         },
        ...         ...
        ...     ]
        ... }
        >>> experiment_manager = ExperimentManager(config=experiment_config)
        >>> experiment_manager.run_experiments()
    """

    def __init__(
        self, config: Dict[str, Any], resume_from_checkpoint: bool = False
    ) -> None:
        self.config: Dict[str, Any] = config
        self.resume_from_checkpoint: bool = resume_from_checkpoint

        self.client: Client = Client()

    def run_experiments(self) -> None:
        """Runs multiple experiments with PyTorch models on custom data.

        This method iterates over each experiment specified in the
        configuration file and runs it using the `run_experiment()`
        method.
        """
        for i, experiment in enumerate(self.config["experiments"]):
            logger.info(f"Experiment {i+1}")
            self.run_experiment(experiment=experiment)

    def run_experiment(self, experiment: Dict[str, Any]) -> None:
        """Runs a training experiment with a PyTorch model on custom data.

        This method performs an end-to-end training experiment using the
        provided experiment parameters.It includes the following steps:

        .. code-block:: text

            1. Data preparation: Creates the train dataset and vectorizer.
            2. Model setup: Initializes the model architecture.
            3. Loss/Accuracy and optimizer setup: Defines the loss and
            accuracy functions and the optimizer.
            4. Experiment tracking: Sets up experiment tracking using
               `TensorBoard`.
            5. Model training: Trains the model for the specified number
               of epochs using `StratifiedShuffleSplit` for cross-validation.
            6. Logging: Logs the training info.

        Args:
            experiment (Dict[str, Any]): Contains training hyperparameters
                for each experiment.

        Example:
            To run an experiment, provide the experiment details in
            the following format:

            >>> experiment_example = {
            ...     'name': 'experiment_1',
            ...     'data': {
            ...         'train_csv': 'path_to_csv_dir/csv_file.csv'
            ...     },
            ...     'hyperparameters': {
            ...         'general': {
            ...             'num_epochs': 10,
            ...             'batch_size': 32
            ...         },
            ...         'optimizer': {
            ...             'optimizer_name': 'adam',
            ...             'learning_rate': 0.001,
            ...             'weight_decay': 0.3,
            ...         },
            ...         'early_stopping': {
            ...             'patience': 3,
            ...              'delta': 0.05
            ...         },
            ...         'model': {
            ...             'model_name': 'news_classifier'
            ...             'embeddings_path': './glove_embeddings/glove.6B.100d.txt'
            ...         }
            ...     }
            ... }
        """
        dataset = NewsDataset.load_dataset_from_csv(
            news_csv=experiment["data"]["train_csv"]
        )
        vectorizer = dataset.get_vectorizer()

        embeddings_path = experiment["hyperparameters"]["model"]["embeddings_path"]
        if embeddings_path and os.path.exists(embeddings_path):
            embeddings_wrapper = PreTrainedEmbeddings.from_embeddings_file(
                embedding_file=experiment["hyperparameters"]["model"]["embeddings_path"]
            )
            input_embeddings = embeddings_wrapper.make_embedding_matrix(
                words=list(vectorizer.text_vocab.token_to_idx.keys())
            )
        else:
            input_embeddings = None

        model = self.client.models_client(
            model_name=experiment["hyperparameters"]["model"]["model_name"],
            num_classes=len(vectorizer.category_vocab.token_to_idx),
            num_embeddings=len(vectorizer.text_vocab),
            pretrained_embeddings=input_embeddings,
        )

        optimizer = self.client.optimizers_client(
            model_params=model.parameters(),
            learning_rate=experiment["hyperparameters"]["optimizer"]["learning_rate"],
            weight_decay=experiment["hyperparameters"]["optimizer"]["weight_decay"],
        )

        loss_fn = nn.CrossEntropyLoss()
        accuracy_fn = torchmetrics.Accuracy(
            task="multiclass", num_classes=len(vectorizer.category_vocab.token_to_idx)
        )

        writer = create_writer(
            start_dir=self.config["tracking_dir"],
            experiment_name=experiment["name"],
            model_name=experiment["hyperparameters"]["model"]["model_name"],
        )

        model_name = (
            f"{experiment['name']}_"
            f"{experiment['hyperparameters']['model']['model_name']}.pth"
        )
        checkpoint_path = os.path.join(self.config["checkpoints_dir"], model_name)

        trainer = TrainingExperiment(
            model=model,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            optimizer=optimizer,
            checkpoint_path=checkpoint_path,
            epochs=experiment["hyperparameters"]["general"]["num_epochs"],
            patience=experiment["hyperparameters"]["early_stopping"]["patience"],
            delta=experiment["hyperparameters"]["early_stopping"]["delta"],
            resume=self.resume_from_checkpoint,
            writer=writer,
        )

        ssf = StratifiedShuffleSplit(n_splits=2, test_size=0.2)

        logger.info("-------------------------- Training --------------------------")
        logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}\n")
        with Timer() as t:
            for fold, (train_ids, val_ids) in enumerate(
                ssf.split(X=dataset, y=dataset.news_df.category)
            ):
                logger.info(f"Fold: {fold+1}")
                train_dataloader = DataLoader(
                    dataset=dataset,
                    batch_size=experiment["hyperparameters"]["general"]["batch_size"],
                    drop_last=True,
                    num_workers=os.cpu_count() if os.cpu_count() else 0,
                    pin_memory=True,
                    sampler=SubsetRandomSampler(train_ids),
                )
                val_dataloader = DataLoader(
                    dataset=dataset,
                    batch_size=experiment["hyperparameters"]["general"]["batch_size"],
                    drop_last=True,
                    num_workers=os.cpu_count() if os.cpu_count() else 0,
                    pin_memory=True,
                    sampler=SubsetRandomSampler(val_ids),
                )

                logger.info(
                    f"Training on {len(train_dataloader)} batches of "
                    f"{train_dataloader.batch_size} samples."
                )
                logger.info(
                    f"Evaluating on {len(val_dataloader)} batches of "
                    f"{val_dataloader.batch_size} samples."
                )

                trainer.train(
                    train_dataloader=train_dataloader, val_dataloader=val_dataloader
                )
        logger.info(f"Training took {t.elapsed} seconds.")
