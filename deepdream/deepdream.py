from .utils import create_simple_logger, create_wandb_logger, ImagePlotter
from .env import env
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import numpy as np
import PIL.Image
from tqdm.auto import tqdm

from typing import List, Any, Dict, Tuple, Union
from collections import OrderedDict
import logging

logger = create_simple_logger("deepdream", env.LOG_LEVEL)

END = "\033[0m"
BOLD = "\033[1m"
BROWN = "\033[0;33m"
ITALIC = "\033[3m"


def print_model_tree(
    model_tree: Dict[str, Union[nn.Module, Dict[str, nn.Module]]],
    indent: int = 0,
    add_modules: bool = False,
):
    for name, module in model_tree.items():
        ended = False
        print(" " * indent + f"{BOLD}{name}{END}:", end="")

        if isinstance(module, dict):
            if not ended:
                print()
            print_model_tree(module, indent + 2, add_modules=add_modules)
        else:
            if add_modules:
                print(f"{' ' * (indent+2)}{ITALIC}{module}{END}", end="")
        if not ended:
            print()


def create_all_possible_submodule_keys(
    tree: Dict[str, Union[nn.Module, Dict[str, nn.Module]]], prefix: str = ""
) -> List[str]:
    keys = []
    for name, module in tree.items():
        keys.append(f"{prefix}{name}")
        if isinstance(module, dict):
            keys.extend(
                create_all_possible_submodule_keys(module, prefix=f"{prefix}{name}.")
            )
    return keys


def get_model_tree(
    model: nn.Module,
) -> Dict[str, Union[nn.Module, Dict[str, nn.Module]]]:
    model_tree = OrderedDict()
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            model_tree[name] = get_model_tree(module)
        else:
            # if the module has no children, it is a leaf node, add it to the tree
            model_tree[name] = module

    return model_tree


def deprocess(image: torch.Tensor) -> PIL.Image:
    image = image.squeeze(0).detach().cpu().numpy()
    image = np.moveaxis(image, 0, -1)
    image = (image - image.min()) / (image.max() - image.min())
    image = np.uint8(image * 255)
    return PIL.Image.fromarray(image)


def get_transforms(h: int, w: int):
    return transforms.Compose(
        [
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            # change image to -1 to 1
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


class DeepDreamModel(nn.Module):
    """A model that can be used for DeepDream. It takes a base model and a list of block names to select. Using the block names, it creates a new model that only contains the selected blocks. The forward method of this model forwards the input through the selected blocks and returns the activations of the selected blocks. The activations are stored in the `activations` attribute of the model.

    Parameters
    ----------
    base_model: nn.Module
        The base model to use for DeepDream.
    logger: logging.Logger, optional
        A logger to use for logging, by default None
    correct_activation_order: bool, optional
        If True, the activations are corrected to be in the same order as the `block_names_to_select`, by default True. If the `block_names_to_select` lists the layers in a different order than they are in the model, the order in which the activations are stored will be different. This parameter corrects that order.
    """

    def __init__(
        self,
        base_model: nn.Module,
        logger: logging.Logger = None,
        correct_activation_order: bool = True,
    ):
        """Initializes the DeepDreamModel."""
        super(DeepDreamModel, self).__init__()
        self.logger = logger or create_simple_logger(
            self.__class__.__name__, env.LOG_LEVEL
        )
        self.correct_activation_order = correct_activation_order
        self.base_model = base_model
        self.base_model_tree = get_model_tree(base_model)
        self.all_possible_submodule_keys = create_all_possible_submodule_keys(
            self.base_model_tree
        )
        self.activations = []

    def __str__(self) -> str:
        string = str(self.blocks)
        # add a tab to each line
        final_string = f"""{self.__class__.__name__}(\n{string}\n)"""
        return final_string

    def __repr__(self) -> str:
        return self.__str__()

    def find_index_of_layer(
        self,
        all_possible_submodule_keys: List[str],
        layer_to_search: str,
        return_max_only: bool = True,
    ) -> List[int]:
        """Finds the index of the layer in the `all_possible_submodule_keys` list. Returns the index of the layer in the list. If `return_max_only` is True, it returns only the maximum index of the layer. If `return_max_only` is False, it returns all the indices of the layer."""
        matches = []
        for i, layer in enumerate(all_possible_submodule_keys):
            if layer_to_search in layer:
                matches.append(i)

        # if no matches are found, raise an error
        if not matches:
            m = f"Layer {layer_to_search} not found in the model. Use the method `print_model_tree` to see the model tree. {all_possible_submodule_keys}"
            self.logger.error(m)
            raise ValueError(m)
        if return_max_only:
            return [max(matches)]
        else:
            return matches

    def get_blocks(
        self,
        base_model: nn.Module = None,
        block_names_to_select: List[str] = None,
    ) -> Tuple[List[nn.Module], List[str]]:
        """Gets the blocks from the model that are needed for DeepDream. The blocks are selected based on the `block_names_to_select` list. A hook is added to the activations of the selected blocks to store the activations in the `activations` attribute of the model.

        Parameters
        ----------
        base_model: nn.Module, optional
            The base model to use for DeepDream. If None, the base model passed to the constructor is used, by default None
        block_names_to_select: List[str]
            A list of block names to select from the model. The activations of these blocks are stored in the `activations` attribute of the model.

        Returns
        -------
        Tuple[List[nn.Module], List[str]]
            A tuple containing the blocks and the names of the blocks selected from the model.
        """
        if base_model is None:
            base_model = self.base_model

        if block_names_to_select is None:
            m = "block_names_to_select is None. Please provide a list of block names to select."
            self.logger.error(m)
            raise ValueError(m)

        # to be used to sort the activations and get the index of the last layer
        indices = []
        for l in block_names_to_select:
            indices.extend(
                self.find_index_of_layer(
                    self.all_possible_submodule_keys,
                    l,
                    return_max_only=True,  # we only need the max index
                )
            )

        self.logger.info(f"Indices of layers: {indices}")
        self.logger.info(f"Layers used for activations hooks: {block_names_to_select}")
        # get the order of the layers
        self.activation_order = np.argsort(indices)
        # add a forward hook to all the activations
        [
            base_model.get_submodule(layer).register_forward_hook(
                lambda module, input, output: self.activations.append(output)
            )
            for layer in block_names_to_select
        ]

        # The index of the last layer to take
        max_index = max(indices)
        self.logger.debug(
            f"Max index: {max_index}, Indices: {indices}, Layers: {block_names_to_select}"
        )

        # take only those layers that are needed
        layers_for_model_ = [
            s.split(".")[0] for s in self.all_possible_submodule_keys[: max_index + 1]
        ]
        layers_for_model = []
        for t in layers_for_model_:
            if t not in layers_for_model:
                layers_for_model.append(t)

        self.blocks = [base_model.get_submodule(k) for k in layers_for_model]
        self.names = [k for k in layers_for_model]

        return self.blocks, self.names

    def print_model_tree(self, add_modules: bool = False) -> None:
        """Prints the model tree of the base model."""
        print_model_tree(self.base_model_tree, add_modules=add_modules)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """The forward method of the model. It forwards the input through the selected blocks and returns the activations of the selected blocks."""
        self.activations = []
        # forwards the input through the blocks
        for name, block in zip(self.names, self.blocks):
            # print the exception if there is an error in the block for debugging
            try:
                x = block(x)
            except Exception as e:
                m = f"Error in block: {name}, {e}"
                self.logger.error(m)
                raise ValueError(m)

        if self.correct_activation_order:
            # correct the order of the activations, so that they are in the same order as the `block_names_to_select`
            temp = list(zip(list(self.activation_order), self.activations))
            temp.sort(key=lambda x: x[0])
            self.activations_correct_order = [x[1] for x in temp]

        else:
            self.activations_correct_order = self.activations
        return self.activations_correct_order


class DeepDream:
    """The DeepDream class that can be used to perform DeepDream on an image using a base model. The base model is passed to the constructor along with a list of block names to select. The block names are used to select the blocks from the model that are needed for DeepDream. The class uses `DeepDreamModel` to create a model that only contains the selected blocks. The `dream` method can be used to perform DeepDream on an image.

    Parameters
    ----------
    base_model: nn.Module
        The base model to use for DeepDream.
    block_names_to_select: List[str]
        A list of block names to select from the model. The activations of these blocks are stored in the `activations` attribute of the model.
    logger: logging.Logger, optional
        A logger to use for logging, by default None
    wandb_logger_config: dict[str, Any], optional
        A dictionary containing the configuration for the wandb logger. If provided, it must contain the `name`, and `project` keys. See the `create_wandb_logger` function for more details. If None, no wandb logger is created, by default None
    """

    def __init__(
        self,
        base_model: nn.Module,
        block_names_to_select: List[str],
        logger: logging.Logger = None,
        wandb_logger_config: dict[str, Any] = None,
    ):
        self.logger = logger or create_simple_logger(
            self.__class__.__name__, env.LOG_LEVEL
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using Device: {self.device}")
        self._create_deepdream_model(
            base_model,
            block_names_to_select,
        )

        if wandb_logger_config:
            # create wandb logger and log the configuration
            config = wandb_logger_config.copy()
            config["base_model"] = base_model.__class__.__name__
            config["block_names_to_select"] = block_names_to_select
            wandb_logger_config["config"] = config
            self.wandb_logger = create_wandb_logger(**wandb_logger_config)
        else:
            self.wandb_logger = None

    def _create_deepdream_model(
        self,
        base_model: nn.Module,
        block_names_to_select: List[str],
    ):
        """Creates the DeepDreamModel using the base model and the block names to select."""
        base_model = base_model.to(self.device)
        base_model.eval()
        self.dream_model = DeepDreamModel(base_model)
        self.dream_model_blocks, self.dream_model_block_names = (
            self.dream_model.get_blocks(
                base_model,
                block_names_to_select,
            )
        )

    def normalize_grad(self, image: torch.Tensor) -> torch.Tensor:
        """Normalizes the gradient of the image before performing gradient ascent."""
        # normalize the gradient
        gradient = image.grad.data
        gradient /= gradient.std() + 1e-8
        return gradient

    def activation_loss_norm(self, activations):
        loss = 0
        for activation in activations:
            activation_norm = activation.norm()
            loss -= activation_norm  # gradient ascent
        return loss

    def activation_loss_mean(self, activations):
        loss = 0
        for activation in activations:
            loss -= activation.mean()  # gradient ascent
        return loss

    def _deep_dream(
        self,
        model: nn.Module,
        image: torch.Tensor,
        iterations: int,
        lr: float,
        loss_type: str = "mean",
        plot_image_interval: Union[str, int] = None,
        image_suffix: str = "",
    ) -> torch.Tensor:
        """The logic for performing DeepDream on an image. It performs gradient ascent on the image using the activations of the model."""
        plot_image = True if plot_image_interval is not None else False

        model.eval()
        image = image.unsqueeze(0).clone()  # add batch dimension
        image = image.to(self.device)
        image.requires_grad = True
        optimizer = optim.Adam([image], lr=lr)
        if plot_image:
            image_plotter = ImagePlotter()
        if loss_type == "mean":
            self.logger.debug("Using mean of activations for loss")
            loss_fn = self.activation_loss_mean
        elif loss_type == "norm":
            self.logger.debug("Using norm of activations for loss")
            loss_fn = self.activation_loss_norm
        else:
            m = f"`loss_type` must either be mean or norm, found: {loss_type}"
            self.logger.error(m)
            raise ValueError(m)

        for i in tqdm(range(iterations)):
            optimizer.zero_grad()
            activations = model(image)
            loss = loss_fn(activations)
            loss.backward()
            self.normalize_grad(image)
            self.logger.info(f"Iteration {i+1}/{iterations}, Loss: {loss.item()}")
            optimizer.step()
            # clip the image
            image.data = image.data.clamp_(-1, 1)
            # log the image to wandb each 5 iterations
            if self.wandb_logger and i % 5 == 0:
                wandb_image = self.wandb_logger.Image(deprocess(image))
                self.wandb_logger.log({"image": wandb_image})

            if plot_image:
                if isinstance(plot_image_interval, int) and (
                    (i + 1) % plot_image_interval == 0
                ):
                    image_plotter.update_image(
                        deprocess(image.detach().clone()),
                        title=f"It {i+1}/{iterations}, Loss: {loss.item():.4f}{image_suffix}",
                    )
        if plot_image_interval == "final_image":
            image_plotter.update_image(
                deprocess(image.detach().clone()),
                title=f"Loss: {loss.item():.4f} at Epoch End",
            )

        return image.detach()

    def resize_tf_image(
        self,
        image: torch.Tensor,
        size: tuple,
        preprocess: transforms.Compose,
    ) -> torch.Tensor:
        """Resizes the image using the given size and preprocesses it."""
        pil_image = deprocess(image)
        # PIL has (w, h) format
        size = tuple(reversed(size))
        pil_image = pil_image.resize(size)
        pil_image = preprocess(pil_image)
        return pil_image

    def dream(
        self,
        image: PIL.Image,
        iterations: int = 20,
        lr: float = 0.05,
        octave_scale: float = 1.3,
        octaves: List[int] = [0],
        loss_type: str = "mean",
        plot_image_interval: Union[str, int] = None,
    ) -> PIL.Image:
        """The main method to perform DeepDream on an image. It takes an image and performs DeepDream on it using the selected blocks from the model.

        Parameters
        ----------
        image: PIL.Image
            The image on which to perform DeepDream.
        iterations: int, optional
            The number of iterations to perform, by default 20
        lr: float, optional
            The learning rate to use for gradient ascent, by default 0.05
        octave_scale: float, optional
            The scale to use for resizing the image in each octave, by default 1.3
        octaves: List[int], optional
            A list of octaves to use for DeepDream, by default [0], which means only one octave is used
        loss_type: str, optional
            Loss function to use. Must be either 'mean' or 'norm'. By Default 'mean'
        plot_image_interval: Union[str, int], optional
            The interval with which to plot the dreamed image. If None, no image is plotted. If string, must be equal to 'final_image' which plots the final image.

        Returns
        -------
        PIL.Image
            The final DeepDream image.
        """
        original_shape = image.size
        # PIL has (w, h) format
        original_shape = tuple(reversed(original_shape))
        transform = get_transforms(*original_shape)
        new_image = transform(image).to(self.device)

        for i in octaves:
            self.logger.info(f"Processing octave {i}")
            new_size = tuple([int(dim * (octave_scale**i)) for dim in original_shape])
            self.logger.info(f"New size: {new_size}")
            new_image = self.resize_tf_image(
                new_image.detach(), new_size, get_transforms(*new_size)
            )
            new_image = self._deep_dream(
                self.dream_model,
                new_image.detach(),
                iterations,
                lr,
                loss_type=loss_type,
                plot_image_interval=plot_image_interval,
                image_suffix=f" Octave: {i}",
            )

        final_image = self.resize_tf_image(
            new_image.detach(), original_shape, get_transforms(*original_shape)
        )
        return deprocess(final_image.detach())
