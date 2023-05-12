import torch
import requests
import numpy as np
from typing import List, Optional, Union

import utils
from logger import logger
from modeling.inference_model import (
    GenerationResult,
    GenerationSettings,
    InferenceModel,
)

from modeling.inference_models.parents.openai_gooseai import model_loader as openai_gooseai_model_loader



class OpenAIAPIError(Exception):
    def __init__(self, error_type: str, error_message) -> None:
        super().__init__(f"{error_type}: {error_message}")


class model_loader(openai_gooseai_model_loader):
    """InferenceModel for interfacing with OpenAI's generation API."""
    
    def __init__(self):
        super().__init__()
        self.url = "https://api.goose.ai/v1/engines"
    
    def is_valid(self, model_name, model_path, menu_path):
        return  model_name == "GooseAI"