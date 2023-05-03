import torch

# We have to go through aiserver to initalize koboldai_vars :(
from aiserver import koboldai_vars

from modeling.inference_model import InferenceModel
from modeling.inference_models.api import APIInferenceModel
from modeling.inference_models.generic_hf_torch import GenericHFTorchInferenceModel
from modeling.inference_models.horde import HordeInferenceModel

model: InferenceModel

# Preferably teeny tiny
TEST_MODEL_HF_ID = "EleutherAI/pythia-70m"
TEST_PROMPT = "Once upon a time I found myself"
TEST_GEN_TOKEN_COUNT = 20
TEST_SEED = 1337

# HF Torch


def test_generic_hf_torch_load() -> None:
    global model
    model = GenericHFTorchInferenceModel(
        TEST_MODEL_HF_ID, lazy_load=False, low_mem=False
    )
    model.load()


def test_generic_hf_torch_lazy_load() -> None:
    GenericHFTorchInferenceModel(TEST_MODEL_HF_ID, lazy_load=True, low_mem=False).load()


def test_generic_hf_torch_low_mem_load() -> None:
    GenericHFTorchInferenceModel(TEST_MODEL_HF_ID, lazy_load=False, low_mem=True).load()


def test_torch_inference() -> None:
    x = model.raw_generate(TEST_PROMPT, max_new=TEST_GEN_TOKEN_COUNT, seed=TEST_SEED)
    print(x.decoded)
    assert len(x.encoded) == 1, "Bad output shape (too many batches!)"
    assert (
        len(x.encoded[0]) == TEST_GEN_TOKEN_COUNT
    ), f"Wrong token amount (requested {TEST_GEN_TOKEN_COUNT})"

    y = model.raw_generate(TEST_PROMPT, max_new=TEST_GEN_TOKEN_COUNT, seed=TEST_SEED)

    assert torch.equal(
        x.encoded[0], y.encoded[0]
    ), f"Faulty full determinism! {x.decoded} vs {y.decoded}"

    print(x)


# Horde
def test_horde_load() -> None:
    global model
    # TODO: Make this a property and sync it with kaivars
    koboldai_vars.cluster_requested_models = []
    model = HordeInferenceModel()
    model.load()


def test_horde_inference() -> None:
    x = model.raw_generate(TEST_PROMPT, max_new=TEST_GEN_TOKEN_COUNT, seed=TEST_SEED)
    assert (
        len(x.encoded[0]) == TEST_GEN_TOKEN_COUNT
    ), f"Wrong token amount (requested {TEST_GEN_TOKEN_COUNT})"
    print(x)


# API
def test_api_load() -> None:
    global model
    model = APIInferenceModel()
    model.load()


def test_api_inference() -> None:
    x = model.raw_generate(TEST_PROMPT, max_new=TEST_GEN_TOKEN_COUNT, seed=TEST_SEED)
    # NOTE: Below test is flakey due to Horde worker-defined constraints
    # assert len(x.encoded[0]) == TEST_GEN_TOKEN_COUNT, f"Wrong token amount (requested {TEST_GEN_TOKEN_COUNT})"
    print(x)
