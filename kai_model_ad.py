import patch_torch_save
from transformers import AutoModel

def kaiad(): # put arbitrary code in here
    print("This model was provided for free by KoboldAI. Check out our free interface at KoboldAI.org")

patched_save_function = patch_torch_save.patch_save_function(kaiad)

model = AutoModel.from_pretrained("facebook/opt-125m")
model.save_pretrained("./local_folder", save_function=patched_save_function) # optionally, upload to HF hub
