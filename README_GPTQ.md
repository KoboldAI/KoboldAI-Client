### Install/Use Guide
(This guide is for both Linux and Windows and assumes user has git installed and a basic grasp of command line use)

#### Installation
In the command prompt/command line navigate to where you want the KoboldAI subfolder to be created.

Note: do not run your command prompt as administrator/with elevated priviledges, reports suggest this leads to problems.

`git clone https://github.com/0cc4m/KoboldAI -b latestgptq --recurse-submodules`

`cd KoboldAI`

Next step, (Windows) subfolder mode or B: option doesn't matter choose either

* [if on Windows]
  ```
  install_requirements.bat
  ```
  * if it closes the window when it finishes, reopen a command prompt and navigate back to your KoboldAI directory.

* [if on Linux with Nvidia] 
  ```
  ./install_requirements.sh
  ```
* [if on Linux with AMD]
  ```
  ./install_requirements.sh rocm
  ./commandline-rocm.sh
  pip install git+https://github.com/0cc4m/GPTQ-for-LLaMa@c884b421a233f9603d8224c9b22c2d83dd2c1fc4
  ```
  * If you get error missing hip/hip_runtime_xxx.h you dont have proper rocm & hip pkg installed
  * If you get CUDA_HOME envar is not set run in env: 
    `pip3 install torch --index-url https://download.pytorch.org/whl/rocm5.4.2 --force-reinstall`

#### Setting up models
If you haven't already done so, create a model folder with the same name as your model (or whatever you want to name the folder)

Put your 4bit quantized .pt or .safetensors in that folder with all associated .json files and tokenizer.model (.json files and tokenizer.model should be from the Huggingface model folder of the same model type).

Then move your model folder to KoboldAI/models, and rename the .pt or .safetensors file in your model folder to `4bit.pt` or `4bit.safetensors` for non-groupsize models or `4bit-<groupsize>g.pt` or `4bit-<groupsize>.safetensors` for a groupsize mode (Example: `4bit-128g.safetensors`)

So - your .pt's model folder should look like this: "4bit.pt, config.json, generation_config.json, pytorch_model.bin.index.json, special_tokens_map.json, tokenizer.model, tokenizer_config.json" Note: the 4bit.pt file can be in the same folder as the regular HF .bin files it was quantized from, it'll load the quantized model.

#### Running KoboldAI and loading 4bit models
If you haven't done so already, exit the command prompt/leave KAI's conda env. (Close the commandline window on Windows, run `exit` on Linux)

Run `play.bat` [windows], `play.sh` [linux Nvidia], or `play-rocm.sh` [linux AMD]

Switch to UI2, then load your model.

