#!/bin/bash
git submodule update --init --recursive
if [[ $1 = "cuda" ]]; then
wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
bin/micromamba create -f environments/huggingface.yml -r runtime -n koboldai -y
# Weird micromamba bug causes it to fail the first time, running it twice just to be safe, the second time is much faster
bin/micromamba create -f environments/huggingface.yml -r runtime -n koboldai -y

# Install quant_cuda module for 4-bit
bin/micromamba run -r runtime -n koboldai pip install https://github.com/0cc4m/GPTQ-for-LLaMa/releases/download/2023-04-10/quant_cuda-0.0.0-cp38-cp38-linux_x86_64.whl
exit
fi
if [[ $1 = "rocm" ]]; then
wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
bin/micromamba create -f environments/rocm.yml -r runtime -n koboldai-rocm -y
# Weird micromamba bug causes it to fail the first time, running it twice just to be safe, the second time is much faster
bin/micromamba create -f environments/rocm.yml -r runtime -n koboldai-rocm -y
exit
fi
echo Please specify either CUDA or ROCM
