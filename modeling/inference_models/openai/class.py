import torch
import requests
import numpy as np
from typing import List, Optional, Union
import os

import utils
from logger import logger
from modeling.inference_model import (
    GenerationResult,
    GenerationSettings,
    InferenceModel,
)

from modeling.inference_models.openai_gooseai import model_backend as openai_gooseai_model_backend

model_backend_name = "OpenAI"
model_backend_type = "OpenAI" #This should be a generic name in case multiple model backends are compatible (think Hugging Face Custom and Basic Hugging Face)

class OpenAIAPIError(Exception):
    def __init__(self, error_type: str, error_message) -> None:
        super().__init__(f"{error_type}: {error_message}")


class model_backend(openai_gooseai_model_backend):
    """InferenceModel for interfacing with OpenAI's generation API."""
    
    def __init__(self):
        super().__init__()
        self.url = "https://api.openai.com/v1/engines"
        self.source = "OpenAI"
    
    def is_valid(self, model_name, model_path, menu_path):
        return model_name == "OAI"