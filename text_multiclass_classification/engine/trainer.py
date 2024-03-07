"""
Contains a class for training and testing a PyTorch model.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.utils.data
import torch.utils.tensorboard
import torchmetrics
from tqdm.auto import tqdm

from text_multiclass_classification import logger
from text_multiclass_classification.utils.aux import (
    EarlyStopping,
    load_general_checkpoint,
)


class TrainingExperiment:
    """Class for conducting a training experiment for a PyTorch model
    on custom data.

    Args:
        checkpoint_path (str):
            The file path to save or load the model checkpoint.
        model (torch.nn.Module): The PyTorch model to be trained.
        loss_fn (torch.nn.Module):
            The loss function used for optimization.
        accuracy_fn (torchmetrics.Metric): Accuracy metric.
        optimizer (torch.optim.Optimizer):
            The optimizer used for updating model parameters.
        epochs (int, optional):
            The number of training epochs. (default=5).
        patience (int, optional):
            Number of epochs to wait before early stopping.
            (default=5)
        delta (float, optional):
            Early stopping specific. Minimum change in monitored
            quantity to qualify as an improvement. (default=0).
        writer (torch.utils.tensorboard.writer.SummaryWriter, optional):
            Optional 'SummaryWriter' instance for logging training metrics
            to TensorBoard. Defaults to `None`.
        resume (bool): If True, resumes training from the specified checkpoint.
            Defaults to `False`.

    Example of results dictionary for 2 epochs::

        {
            "train_loss": [2.0616, 1.0537],
            "train_acc": [0.3945, 0.4257],
            "val_loss": [1.2641, 1.5706],
            "val_acc": [0.3400, 0.3573]
        }
    """

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        accuracy_fn: torchmetrics.Metric,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: str,
        epochs: int = 5,
        patience: int = 5,
        delta: float = 0,
        resume: bool = False,
        writer: Optional[torch.utils.tensorboard.writer.SummaryWriter] = None,
    ) -> None:
        self.model: torch.nn.Module = model
        self.loss_fn: torch.nn.Module = loss_fn
        self.accuracy_fn: torchmetrics.Metric = accuracy_fn
        self.optimizer: torch.optim.Optimizer = optimizer
        self.checkpoint_path: str = checkpoint_path
        self.epochs: int = epochs
        self.patience: int = patience
        self.delta: float = delta
        self.resume: bool = resume
        self.writer: Optional[torch.utils.tensorboard.writer.SummaryWriter] = writer
        self.early_stopping: EarlyStopping = EarlyStopping(
            patience=self.patience,
            delta=self.delta,
            path=self.checkpoint_path,
            verbose=True,
        )

    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
    ) -> Dict[str, List[float]]:
        """Trains a PyTorch model on custom data.

        Performs the training of a PyTorch model using the provided data
        loaders, loss function, and optimizer. It also evaluates the model
        on the validation data at the end of each epoch. Checkpointing is
        supported, optionally allowing for the resumption of training from
        a saved checkpoint.

        The training process includes early stopping to prevent over-fitting,
        where training is stopped if the validation loss does not improve for a
        certain number of epochs.

        Calculates, prints and stores evaluation metrics throughout.

        Stores metrics to specified writer `log_dir` if present. Refer
        to `text_multiclass_classification.utils.aux.create_writer`
        function for more.

        Args:
            train_dataloader (torch.utils.data.DataLoader):
                A `DataLoader` instance for providing batches of
                training data.
            val_dataloader (torch.utils.data.DataLoader):
                A `DataLoader` instance for providing batches of
                testing data.

        Returns:
            Dict[str, List[float]]:
                A dictionary containing the training and testing metrics including
                'train_loss', 'train_acc', 'val_loss', and 'val_acc'.
        """
        logger.info("-------------------------- Training --------------------------")
        logger.info(
            f"Training on {len(train_dataloader)} batches of "
            f"{train_dataloader.batch_size} samples."
        )
        logger.info(
            f"Evaluating on {len(val_dataloader)} batches of "
            f"{val_dataloader.batch_size} samples."
        )
        logger.info(f"Training model: {self.model.__class__.__name__}")
        logger.info(f"Optimizer: {self.optimizer.__class__.__name__}")
        logger.info(f"Loss function: {self.loss_fn.__class__.__name__}")
        logger.info("Early Stopping: Yes")
        logger.info(f"Target device: {self.__class__.DEVICE}")
        logger.info(f"Epochs: {self.epochs}\n")

        self.model.to(self.__class__.DEVICE)

        results: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        start_epoch = 0
        if self.resume:
            checkpoint = load_general_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                filepath=self.checkpoint_path,
            )
            self.model = checkpoint["model"].to(self.__class__.DEVICE)
            self.optimizer = checkpoint["optimizer"]
            loss_value = checkpoint["val_loss"]
            start_epoch = checkpoint["epoch"] + 1

            logger.info(
                f"Resume training from general checkpoint: {self.checkpoint_path}."
            )
            logger.info(f"Last training loss value: {loss_value:.4f}")
            logger.info(f"Resuming from {start_epoch + 1} epoch...")

        for epoch in tqdm(range(start_epoch, self.epochs), position=0, leave=True):
            train_loss, train_acc = self._train_step(dataloader=train_dataloader)
            val_loss, val_acc = self._val_step(dataloader=val_dataloader)

            # Print out what's happening
            logger.info(
                f"===>>> epoch: {epoch + 1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_acc: {val_acc:.4f}"
            )

            # Update results dictionary
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["val_loss"].append(val_loss)
            results["val_acc"].append(val_acc)

            # Track experiments with SummaryWriter
            if self.writer:
                self.writer.add_scalars(
                    main_tag="Loss",
                    tag_scalar_dict={
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                    global_step=epoch,
                )
                self.writer.add_scalars(
                    main_tag="Accuracy",
                    tag_scalar_dict={"train_acc": train_acc, "val_acc": val_acc},
                    global_step=epoch,
                )
                self.writer.close()

            self.early_stopping(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                val_loss=val_loss,
            )
            if self.early_stopping.early_stop:
                logger.info("Training stopped due to early stopping.")
                break
            else:
                continue

        return results

    def _train_step(
        self, dataloader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """Trains a PyTorch model for a single epoch.

        Turns a target PyTorch model to `train` mode and then
        runs through all the required training steps (forward
        pass, loss calculation, optimizer step).

        Args:
            dataloader (torch.utils.data.DataLoader):
                A `DataLoader` instance for providing batches of
                training data.

        Returns:
          Tuple[float, float]:
            The training loss and training accuracy metrics in the form
            (train_loss, train_accuracy). For example: (0.1112, 0.8743).
        """
        # Put the model in train mode
        self.model.train()

        # Setup train loss and train accuracy values
        train_loss, train_acc = 0.0, 0.0

        # Use prefetch_generator for iterating through data
        # pbar = enumerate(BackgroundGenerator(dataloader))

        # Loop through data loader data batches
        for batch, (X, y) in enumerate(dataloader):
            # Send the data to the target device
            X, y = X.to(self.__class__.DEVICE), y.to(self.__class__.DEVICE)

            # 1. Forward pass (returns logits)
            y_pred = self.model(X)

            # 2. Calculate and accumulate loss
            loss = self.loss_fn(y_pred, y)
            train_loss += loss.item()

            # 3. Optimize zero grad
            self.optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            self.optimizer.step()

            # Calculate and accumulate accuracy metric across all batches
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += self.accuracy_fn(y_pred_class, y).item()

        # Divide total train loss and accuracy by length of dataloader
        train_loss /= len(dataloader)
        train_acc /= len(dataloader)

        return train_loss, train_acc

    def _val_step(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Tests a PyTorch model for a single epoch.

        Turns a target PyTorch model to `eval` model and then performs
        a forward pass on a validation dataset.

        Args:
            dataloader (torch.utils.data.DataLoader):
                A `DataLoader` instance for providing batches of
                testing data.

        Returns:
          Tuple[float, float]:
            The validation loss and validation accuracy metrics in the form
            (val_loss, val_accuracy). For example: (0.0223, 0.8985).
        """
        # Put model in eval mode
        self.model.eval()

        # Use prefetch_generator and tqdm for iterating through data
        # pbar = enumerate(BackgroundGenerator(dataloader))

        # Setup test loss and test accuracy values
        val_loss, val_acc = 0.0, 0.0

        # Turn on inference context manager
        with torch.inference_mode():
            # Loop through data loader data batches
            for batch, (X, y) in enumerate(dataloader):
                # Send data to the target device
                X, y = X.to(self.__class__.DEVICE), y.to(self.__class__.DEVICE)

                # 1. Forward pass
                test_y_pred = self.model(X)

                # 2. Calculate and accumulate loss
                loss = self.loss_fn(test_y_pred, y)
                val_loss += loss.item()

                # Calculate and accumulate accuracy
                val_pred_labels = test_y_pred.argmax(dim=1)
                val_acc += self.accuracy_fn(val_pred_labels, y).item()

        # Divide total test loss and accuracy by length of dataloader
        val_loss /= len(dataloader)
        val_acc /= len(dataloader)

        return val_loss, val_acc
