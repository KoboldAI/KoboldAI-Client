#!/bin/bash
export PYTHONNOUSERSITE=1
if [ ! -f "runtime/envs/koboldai-ipex/bin/python" ]; then
./install_requirements.sh ipex
fi

export LD_LIBRARY_PATH=$(realpath "runtime/envs/koboldai-ipex")/lib/:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/libstdc++.so
export NEOReadDebugKeys=1
export ClDeviceGlobalMemSizeAvailablePercent=100

bin/micromamba run -r runtime -n koboldai-ipex python aiserver.py $*