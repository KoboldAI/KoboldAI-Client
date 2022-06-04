#!/bin/bash
if [ ! -f "runtime/envs/koboldai/bin/python" ]; then
./install_requirements.sh cuda
fi
bin/micromamba run -r runtime -n koboldai python aiserver.py $*
