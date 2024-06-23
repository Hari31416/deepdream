from env import env

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Any
import logging
from PIL import Image
import wandb
from IPython.display import display


def create_simple_logger(
    logger_name: str, level: str = env.LOG_LEVEL
) -> logging.Logger:
    """Creates a simple logger with the given name and level. The logger has a single handler that logs to the console.

    Parameters
    ----------
    logger_name : str
        Name of the logger.
    level : str or int
        Level of the logger. Can be a string or an integer. If a string, it should be one of the following: "debug", "info", "warning", "error", "critical".

    Returns
    -------
    logging.Logger
        The logger object.
    """
    level_to_int_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    if isinstance(level, str):
        level = level_to_int_map[level.lower()]
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = create_simple_logger("utils", env.LOG_LEVEL)


def is_jupyter_notebook() -> bool:
    """Checks if the code is being run in a Jupyter notebook.

    Returns
    -------
    bool
        True if the code is being run in a Jupyter notebook, False otherwise.
    """
    is_jupyter = False
    try:
        # noinspection PyUnresolvedReferences
        from IPython import get_ipython

        # noinspection PyUnresolvedReferences
        if get_ipython() is None or "IPKernelApp" not in get_ipython().config:
            pass
        else:
            is_jupyter = True
    except (ImportError, NameError):
        pass
    if is_jupyter:
        logger.debug("Running in Jupyter notebook.")
    else:
        logger.debug("Not running in a Jupyter notebook.")
    return is_jupyter


class ImagePlotter:
    """A class to display images. It can be used to display images in a loop."""

    def __init__(
        self,
        cmap: str = "viridis",
        **kwargs: dict[str, any],
    ):
        """Initializes the ImagePlotter object.

        Parameters
        ----------
        title : str, optional
            The title of the figure. Default is "".
        cmap : str, optional
            The colormap to be used. Default is "viridis".
        kwargs
            Additional keyword arguments to be passed to the `plt.subplots` method.
        """

        self.fig, self.ax = plt.subplots(**kwargs)
        self.im = None
        self.cmap = cmap

    def update_image(
        self,
        image: Union[np.ndarray, Image.Image],
        title: str = "",
        path_to_save: Union[str, None] = None,
    ) -> None:
        # convert pil image to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)

        channels = image.shape[-1]
        if channels == 1 and self.cmap not in ["gray", "Greys"]:
            cmap = "gray"
        else:
            cmap = self.cmap

        if self.im is None:
            self.im = self.ax.imshow(image, cmap=cmap)
        else:
            self.im.set_data(image)
        self.ax.set_title(title)
        self.ax.title.set_fontsize(15)
        self.ax.axis("off")
        plt.draw()
        plt.pause(0.01)
        # display the figure if running in a Jupyter notebook
        if is_jupyter_notebook():
            display(self.fig, clear=True)
        if path_to_save is not None:
            self.fig.savefig(path_to_save)


def create_wandb_logger(
    name: Union[str, None] = None,
    project: Union[str, None] = None,
    config: Union[dict[str, any], None] = None,
    tags: Union[list[str], None] = None,
    notes: str = "",
    group: Union[str, None] = None,
    job_type: str = "",
    logger: Union[logging.Logger, None] = None,
) -> wandb.sdk.wandb_run.Run:
    """Creates a new run on Weights & Biases and returns the run object.

    Parameters
    ----------
    project : str | None, optional
        The name of the project. If None, it must be provided in the config. Default is None.
    name : str | None, optional
        The name of the run. If None, it must be provided in the config. Default is None.
    config : dict[str, any] | None, optional
        The configuration to be logged. Default is None. If `project` and `name` are not provided, they must be present in the config.
    tags : list[str] | None, optional
        The tags to be added to the run. Default is None.
    notes : str, optional
        The notes to be added to the run. Default is "".
    group : str | None, optional
        The name of the group to which the run belongs. Default is None.
    job_type : str, optional
        The type of job. Default is "train".
    logger : logging.Logger | None, optional
        The logger to be used by the object. If None, a simple logger is created using `create_simple_logger`. Default is None.

    Returns
    -------
    wandb.Run
        The run object.
    """
    logger = logger or create_simple_logger("create_wandb_logger")
    if config is None:
        logger.debug("No config provided. Using an empty config.")
        config = {}

    if name is None and "name" not in config.keys():
        m = "Run name must be provided either as an argument or in the config."
        logger.error(m)
        raise ValueError(m)

    if project is None and "project" not in config.keys():
        m = "Project name must be provided either as an argument or in the config."
        logger.error(m)
        raise ValueError(m)

    # If the arguments are provided, they take precedence over the config
    name = name or config.get("name")
    project = project or config.get("project")
    notes = notes or config.get("notes")
    tags = tags or config.get("tags")
    group = group or config.get("group")
    job_type = job_type or config.get("job_type")

    logger.info(
        f"Initializing Weights & Biases for project {project} with run name {name}."
    )
    wandb.init(
        project=project,
        name=name,
        config=config,
        tags=tags,
        notes=notes,
        group=group,
        job_type=job_type,
    )
    return wandb
