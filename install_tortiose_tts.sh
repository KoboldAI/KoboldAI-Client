#!/bin/bash
bin/micromamba run -r runtime -n koboldai pip install git+https://github.com/neonbjb/tortoise-tts OmegaConf deepspeed
bin/micromamba run -r runtime -n koboldai pip install torchaudio --index-url https://download.pytorch.org/whl/cu118
bin/micromamba run -r runtime -n koboldai pip install -r requirements.txt --no-dependencies
