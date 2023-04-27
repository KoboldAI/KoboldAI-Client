export CONDA_AUTO_ACTIVATE_BASE=false
export PYTHONNOUSERSITE=1
bin/micromamba run -r runtime -n koboldai-rocm bash
