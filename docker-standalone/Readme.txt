These are the source files for the official versions of the standalone docker and are provided for completeness.
Using these files you will not use any of the local modifications you make, instead it will use the latest github version of KoboldAI as the basis.

If you wish to run KoboldAI containerised with access to the local directory you can do so using docker-cuda.sh or docker-rocm.sh instead.

We do not support ROCm in the standalone docker as it is intended for cloud deployment on CUDA systems.
If you wish to build a ROCm version instead, you can do so by modifying the Dockerfile and changing the install_requirements.sh from cuda to rocm.

Similarly you need to modify the Dockerfile to specify which branch of KoboldAI the docker is being built for.

Usage:
This docker will automatically assume the persistent volume is mounted to /content and will by default not store models there.
The following environment variables exist to adjust the behavior if desired.

KOBOLDAI_DATADIR=/content , this can be used to specify a different default location for your stories, settings, userscripts, etc in case your provider does not let you change the mounted folder path.
KOBOLDAI_MODELDIR= , This variable can be used to make model storage persistent, it can be the same location as your datadir but this is not required.
KOBOLDAI_ARGS= , This variable is built in KoboldAI and can be used to override the default launch options. Right now the docker by default will launch in remote mode, with output hidden from the logs and file management enabled.