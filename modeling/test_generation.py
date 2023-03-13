# We have to go through aiserver to initalize koboldai_vars :(
import torch
from aiserver import GenericHFTorchInferenceModel
from aiserver import koboldai_vars
from modeling.inference_model import InferenceModel

model: InferenceModel

# Preferably teeny tiny
TEST_MODEL_HF_ID = "EleutherAI/pythia-70m"
TEST_PROMPT = "Once upon a time I found myself"
TEST_GEN_TOKEN_COUNT = 20
TEST_SEED = 1337


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


def test_model_gen() -> None:
    x = model.raw_generate(TEST_PROMPT, max_new=TEST_GEN_TOKEN_COUNT, seed=TEST_SEED)
    print(x.decoded)
    assert len(x.encoded) == 1, "Bad output shape (too many batches!)"
    assert len(x.encoded[0]) == 20, "Wrong token amount (requested 20)"

    y = model.raw_generate(TEST_PROMPT, max_new=TEST_GEN_TOKEN_COUNT, seed=TEST_SEED)

    assert torch.equal(
        x.encoded[0], y.encoded[0]
    ), f"Faulty full determinism! {x.decoded} vs {y.decoded}"

    print(x)
