from typing import Iterator, Optional

import torch

from text_multiclass_classification import logger
from text_multiclass_classification.factories.factories import (
    ModelsFactory,
    OptimizersFactory,
)
from text_multiclass_classification.models.news_classifier import (
    NewsClassifierWithCNN,
    NewsClassifierWithRNN,
)
from text_multiclass_classification.utils import (
    constants,
    custom_exceptions,
    error_messages,
)


class Client:
    """
    A client for obtaining neural network models, optimizers, and data
    transformations.

    This class provides methods to obtain neural network models, optimizers,
    and data transformations based on specified names. It internally uses
    factories to manage the creation and retrieval of these components.

    Attributes:
        models_factory (ModelsFactory):
            A factory for registering and obtaining neural network models.
        optimizers_factory (OptimizersFactory):
            A factory for registering and obtaining optimizers.
    """

    def __init__(self) -> None:
        self.models_factory: ModelsFactory = ModelsFactory()
        self.optimizers_factory: OptimizersFactory = OptimizersFactory()

    def models_client(
        self,
        model_name: str,
        num_classes: int,
        num_embeddings: int,
        pretrained_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.nn.Module:
        """
        Returns a neural network model based on the provided model name.

        This function serves as a client to obtain different pre-defined
        neural network models with specific numbers of output classes.
        The model_name argument determines which specific model will be
        instantiated.

        Supported model names:
            - 'text_classifier_cnn': A text classifier with cnn layers
                for text feature extraction.
            - 'text_classifier_rnn': A text classifier with rnn layer
                for text sequence representation.

        Args:
            model_name (str): The name of the model to be instantiated.
            num_classes (int): The number of output classes for the model.
            num_embeddings (int): Number of embedding vectors.
                Usually the length of the vocabulary.
            pretrained_embeddings (torch.Tensor, optional). Pretrained
                word embeddings, if provided. Defaults to `None`

        Returns:
            torch.nn.Module: An instance of the specified neural network model.

        Raises:
            UnsupportedModelNameError:
                If the specified model name is not supported.

        Examples:
            >>> from text_multiclass_classification.factories import client
            >>> client = client.Client()
            >>> model_instance = client.models_client(
            ...     name='news_classifier_RNN',
            ...     num_classes=4,
            ...     num_embeddings=100
            ... )
            >>> print(model_instance)
            NewsClassifierWithRNN(
              (emb): Embedding(4582, 100, padding_idx=0)
              (rnn): ElmanRNN(
                (rnn_cell): RNNCell(100, 100)
              )
              (classifier): Sequential(
                (0): Dropout(p=0.1, inplace=False)
                (1): Linear(in_features=100, out_features=100, bias=True)
                (2): ReLU()
                (3): Dropout(p=0.1, inplace=False)
                (4): Linear(in_features=100, out_features=4, bias=True)
              )
            )
        """
        match model_name.lower():
            case constants.NEWS_CLASSIFIER_WITH_CNN_MODEL_NAME:
                self.models_factory.register_model(
                    name=model_name.lower(), model=NewsClassifierWithCNN
                )
            case constants.NEWS_CLASSIFIER_WITH_RNN_MODEL_NAME:
                self.models_factory.register_model(
                    name=model_name.lower(),
                    model=NewsClassifierWithRNN,
                )
            case _:
                logger.error(f"{error_messages.unsupported_model_name} `{model_name}`.")
                raise custom_exceptions.UnsupportedModelNameError(
                    f"{error_messages.unsupported_model_name} `{model_name}`."
                )

        model = self.models_factory.get_model(
            name=model_name.lower(),
            num_classes=num_classes,
            num_embeddings=num_embeddings,
            pretrained_embeddings=pretrained_embeddings,
        )
        return model

    def optimizers_client(
        self,
        model_params: Iterator[torch.nn.parameter.Parameter],
        optimizer_name: str = "sgd",
        learning_rate: float = 0.001,
        weight_decay: float = 0,
    ) -> torch.optim.Optimizer:
        """
        Returns an optimizer based on the provided optimizer name.

        This function serves as a client to obtain different pre-defined
        optimizers for training neural network models. The optimizer_name
        argument determines which specific optimizer will be instantiated.

        Supported optimizer names:
            - 'sgd': Stochastic Gradient Descent (SGD) optimizer.
            - 'adam': Adam optimizer.

        Args:
            optimizer_name (str, optional):
                The name of the optimizer to be instantiated.
                Defaults to ``sgd``.
            model_params (Iterator[torch.nn.parameter.Parameter]):
                An iterator of model parameters.
            learning_rate (float, optional): The learning rate (default=0.001).
            weight_decay (float, optional): L2 penalty (default=0).

        Returns:
            torch.optim.Optimizer: An instance of the specified optimizer.

        Raises:
            UnsupportedOptimizerNameError:
                If the specified optimizer name is not supported.

        Example:
            >>> # Initialize a simple PyTorch model
            >>> import torch
            >>> from torch import nn
            >>> class SimpleModel(nn.Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.fc = nn.Linear(10, 2)
            ...
            ...     def forward(self, x):
            ...         return self.fc(x)
            >>> simple_model = SimpleModel()
            >>>
            >>> # Get optimizer
            >>> from text_multiclass_classification.factories import client
            >>> client = client.Client()
            >>> hyperparams = {
            ...     "learning_rate": learning_rate,
            ...     "weight_decay": weight_decay,
            ... }
            >>> optimizer_instance = client.optimizers_client(
            ...     name='sgd',
            ...     model_params=simple_model.parameters(),
            ...     hyperparameters=hyperparams,
            ... )
            >>> print(optimizer_instance)
            SGD (
            Parameter Group 0
                dampening: 0
                differentiable: False
                foreach: None
                lr: 0.001
                maximize: False
                momentum: 0
                nesterov: False
                weight_decay: 0.3
            )
        """
        hyperparameters = {
            "lr": learning_rate,
            "weight_decay": weight_decay,
        }
        match optimizer_name.lower():
            case constants.SGD_OPTIMIZER_NAME:
                self.optimizers_factory.register_optimizer(
                    name=optimizer_name.lower(),
                    optimizer=torch.optim.SGD,
                )
            case constants.ADAM_OPTIMIZER_NAME:
                self.optimizers_factory.register_optimizer(
                    name=optimizer_name.lower(), optimizer=torch.optim.Adam
                )
            case _:
                logger.error(
                    f"{error_messages.unsupported_optimizer_name}"
                    f"`{optimizer_name}`."
                )
                raise custom_exceptions.UnsupportedOptimizerNameError(
                    f"{error_messages.unsupported_optimizer_name}"
                    f"`{optimizer_name}`."
                )

        optimizer = self.optimizers_factory.get_optimizer(
            name=optimizer_name.lower(),
            model_params=model_params,
            hyperparameters=hyperparameters,
        )
        return optimizer
