# This dockerfile is meant to serve as a rocm base image.  It registers the debian rocm package repository, and
# installs the rocm-dev package.

FROM rocm/pytorch
RUN conda install --all -y Flask-SocketIO
RUN pip3 install git+https://github.com/finetuneanon/transformers@gpt-neo-localattention3
