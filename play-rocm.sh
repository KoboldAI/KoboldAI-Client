#!/bin/bash
export PYTHONNOUSERSITE=1
if [ ! -f "runtime/envs/koboldai-rocm/bin/python" ]; then
./install_requirements.sh rocm
fi
bin/micromamba run -r runtime -n koboldai-rocm python aiserver.py $*
