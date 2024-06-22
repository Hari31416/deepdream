import os
from dotenv import load_dotenv

load_dotenv()


class env:
    """A class that contains the environment variables used by the project."""

    LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING")
    WANDB_API_KEY = f"{os.getenv('WANDB_API_KEY')}"
