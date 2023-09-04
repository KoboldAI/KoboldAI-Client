#!/bin/bash
export PYTHONNOUSERSITE=1
if [ ! -f "runtime/envs/koboldai-ipex/bin/python" ]; then
./install_requirements.sh ipex
fi

#Set OneAPI environmet if it's not set by the user
if [ ! -x "$(command -v sycl-ls)" ]
then
    echo "Setting OneAPI environment"
    if [[ -z "$ONEAPI_ROOT" ]]
    then
        ONEAPI_ROOT=/opt/intel/oneapi
    fi
    source $ONEAPI_ROOT/setvars.sh
fi

export LD_PRELOAD=/usr/lib/libstdc++.so
export NEOReadDebugKeys=1
export ClDeviceGlobalMemSizeAvailablePercent=100

bin/micromamba run -r runtime -n koboldai-ipex python aiserver.py $*