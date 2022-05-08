#!/bin/bash
if [[ $1 = "cuda" ]]; then
wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
bin/micromamba create -f environments/huggingface.yml -r runtime -n koboldai -y
# Weird micromamba bug causes it to fail the first time, running it twice just to be safe, the second time is much faster
bin/micromamba create -f environments/huggingface.yml -r runtime -n koboldai -y
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
