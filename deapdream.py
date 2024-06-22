from utils import create_simple_logger, create_wandb_logger, ImagePlotter
from env import env
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import inception_v3, Inception_V3_Weights

import numpy as np
import IPython.display as display
import PIL.Image
from tqdm.auto import tqdm

from typing import List, Union, Any
import logging

logger = create_simple_logger("deapdream", env.LOG_LEVEL)


class DeapDreamModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        block_names_to_select: List[str],
        logger: logging.Logger = None,
    ):
        super(DeapDreamModel, self).__init__()
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.blocks, self.block_indices_to_maximize = self._get_blocks(
            base_model, block_names_to_select
        )

    def _get_blocks(
        self, base_model: nn.Module, block_names_to_select: List[str]
    ) -> List[nn.Module]:

        block_names, blocks = zip(*base_model.named_children())
        self.logger.info(f"Number of blocks: {len(blocks)}")
        max_block_number = 0
        block_indices_to_maximize = []

        for i, name in enumerate(block_names):
            for b in block_names_to_select:
                if b in name:
                    block_indices_to_maximize.append(i)
                    max_block_number = i
        self.logger.info(f"Block indices to maximize: {block_indices_to_maximize}")

        layers_to_take_for_model = []
        for i in range(max_block_number + 1):
            layers_to_take_for_model.append(blocks[i])
        return layers_to_take_for_model, block_indices_to_maximize

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        activations = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.block_indices_to_maximize:
                activations.append(x)
        return activations

    def __str__(self) -> str:
        string = str(self.blocks)
        # add a tab to each line
        final_string = f"""{self.__class__.__name__}(\n{string}\n)"""
        return final_string

    def __repr__(self) -> str:
        return self.__str__()


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


class DeepDream:
    def __init__(
        self,
        base_model: nn.Module,
        block_names_to_select: List[str],
        logger: logging.Logger = None,
        wandb_logger_config: dict[str, Any] = None,
    ):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using Device: {self.device}")

        self.base_model = base_model.to(self.device)
        self.base_model.eval()
        self.model = DeapDreamModel(self.base_model, block_names_to_select).to(
            self.device
        )

        if wandb_logger_config:
            # create wandb logger and log the configuration
            self.wandb_logger = create_wandb_logger(**wandb_logger_config)
            config = wandb_logger_config.get("config", {})
            config["model"] = self.model.__class__.__name__
            config["block_names_to_select"] = block_names_to_select
            self.wandb_logger.config.update(config)

    def normalize_grad(self, image: torch.Tensor) -> torch.Tensor:
        # normalize the gradient
        gradient = image.grad.data
        gradient /= gradient.std() + 1e-8
        return gradient

    def _deep_dream(
        self, model: nn.Module, image: torch.Tensor, iterations: int, lr: float
    ) -> torch.Tensor:
        model.eval()
        image = image.unsqueeze(0).clone()  # add batch dimension
        image.requires_grad = True
        optimizer = optim.Adam([image], lr=lr)
        for i in tqdm(range(iterations)):
            optimizer.zero_grad()
            activations = model(image)
            loss = 0
            for activation in activations:
                loss -= activation.mean()  # perform gradient ascent
            loss.backward()
            self.normalize_grad(image)
            print(f"Iteration {i+1}/{iterations}, Loss: {loss.item()}")
            optimizer.step()
            # clip the image
            image.data.clamp_(-1, 1)
            # log the image to wandb each 5 iterations
            if self.wandb_logger and i % 5 == 0:
                wandb_image = self.wandb_logger.Image(deprocess(image))
                self.wandb_logger.log({"image": wandb_image})

        return image

    def resize_tf_image(
        self,
        image: torch.Tensor,
        size: tuple,
        preprocess: transforms.Compose,
    ) -> torch.Tensor:
        pil_image = deprocess(image)
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
        plot_image: bool = False,
    ) -> PIL.Image:
        original_shape = image.size
        transform = get_transforms(*original_shape)
        new_image = transform(image).to(self.device)
        if plot_image:
            image_plotter = ImagePlotter()
        for i in octaves:
            print(f"Processing octave {i}")
            new_size = tuple([int(dim * (octave_scale**i)) for dim in original_shape])
            print(f"New size: {new_size}")
            new_image = self.resize_tf_image(
                new_image, new_size, get_transforms(*new_size)
            )
            new_image = self._deep_dream(self.model, new_image, iterations, lr)
            if plot_image:
                image_plotter.update_image(deprocess(new_image), title=f"Octave {i}")
        final_image = self.resize_tf_image(
            new_image, original_shape, get_transforms(*original_shape)
        )
        return deprocess(final_image)
