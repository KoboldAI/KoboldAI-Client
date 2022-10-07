FROM debian
RUN apt update && apt install wget aria2 git bzip2 -y
RUN git clone https://github.com/koboldai/koboldai-client /opt/koboldai
WORKDIR /opt/koboldai
RUN ./install_requirements.sh cuda
COPY docker-helper.sh /opt/koboldai/docker-helper.sh
EXPOSE 5000/tcp
CMD /opt/koboldai/docker-helper.sh
