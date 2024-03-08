from typing import Any, Dict, Iterator, Optional

import torch


class ModelsFactory:
    """
    A factory class for registering and instantiating PyTorch models.

    Attributes:
        _models (Dict[str, Any]):
            A dictionary to store registered models.
    """

    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}

    def register_model(self, name: str, model: Any) -> None:
        """
        Registers a PyTorch model with the given name.

        Args:
            name (str): The name of the model to register.
            model (Any):
                The PyTorch model class or function.
        """
        self._models[name] = model

    def get_model(
            self,
            name: str,
            num_classes: int,
            num_embeddings: int,
            pretrained_embeddings: Optional[torch.Tensor] = None
    ) -> Any:
        """
        Instantiates and returns a PyTorch model by name.

        Args:
            name (str): The name of the model to instantiate.
            num_classes (int): The number of output classes for the model.
            num_embeddings (int): Number of embedding vectors.
                Usually the length of the vocabulary.
            pretrained_embeddings (torch.Tensor, optional). Pretrained
                word embeddings, if provided. Defaults to `None`

        Returns:
            torch.nn.Module: An instance of the specified PyTorch model.
        """
        model = self._models[name]
        return model(
            num_classes=num_classes,
            num_embeddings=num_embeddings,
            pretrained_embeddings=pretrained_embeddings
        )


class OptimizersFactory:
    """
    A factory class for registering and instantiating PyTorch optimizers.

    Attributes:
        _optimizers (Dict[str, Any]):
            A dictionary to store registered optimizers.
    """

    def __init__(self) -> None:
        self._optimizers: Dict[str, Any] = {}

    def register_optimizer(self, name: str, optimizer: Any) -> None:
        """
        Registers a PyTorch optimizer with the given name.

        Args:
            name (str): The name of the optimizer to register.
            optimizer (Any):
                The PyTorch optimizer class.
        """
        self._optimizers[name] = optimizer

    def get_optimizer(
        self,
        name: str,
        model_params: Iterator[torch.nn.parameter.Parameter],
        hyperparameters: Dict[str, Any],
    ) -> Any:
        """Instantiates and returns a PyTorch optimizer by name.

        Args:
            name (str): The name of the optimizer to instantiate.
            model_params (Iterator[torch.nn.parameter.Parameter]):
                Iterator of the model parameters.
            hyperparameters (Dict[str, Any]):
                Hyperparameters to be passed to the optimizer.

        Returns:
            torch.optim.Optimizer:
                An instance of the specified PyTorch optimizer.
        """
        optimizer = self._optimizers[name]
        return optimizer(model_params, **hyperparameters)
