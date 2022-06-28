These are the source files for the official versions of the standalone docker and are provided for completeness.
Using these files you will not use any of the local modifications you make, instead it will use the latest github version of KoboldAI as the basis.

If you wish to run KoboldAI containerised with access to the local directory you can do so using docker-cuda.sh or docker-rocm.sh instead.

We do not support ROCm in the standalone docker as it is intended for cloud deployment on CUDA systems.
If you wish to build a ROCm version instead, you can do so by modifying the Dockerfile and changing the install_requirements.sh from cuda to rocm.

Similarly you need to modify the Dockerfile to specify which branch of KoboldAI the docker is being built for.