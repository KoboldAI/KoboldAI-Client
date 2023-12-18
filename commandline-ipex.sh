if [ ! -x "$(command -v sycl-ls)" ]
then
    echo "Setting OneAPI environment"
    if [[ -z "$ONEAPI_ROOT" ]]
    then
        ONEAPI_ROOT=/opt/intel/oneapi
    fi
    source $ONEAPI_ROOT/setvars.sh
fi
export CONDA_AUTO_ACTIVATE_BASE=false
export PYTHONNOUSERSITE=1
export NEOReadDebugKeys=1
export ClDeviceGlobalMemSizeAvailablePercent=100
bin/micromamba run -r runtime -n koboldai-ipex bash
