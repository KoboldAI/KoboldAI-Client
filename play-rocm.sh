if [ ! -f "runtime/envs/koboldai-rocm/bin/python" ]; then
source ./install_requirements.sh rocm
fi
bin/micromamba run -r runtime -n koboldai-rocm python aiserver.py $*
