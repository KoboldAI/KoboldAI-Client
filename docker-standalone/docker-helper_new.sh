#!/bin/bash
cd /opt/koboldai
if [[ -n update ]];then
        git pull --recurse-submodules
fi

#The goal here is to allow any directory in /content to be mapped to the appropriate dir in the koboldai dir
if [[ ! -d "/content" ]];then
        mkdir /content
fi

for FILE in /content/*
do
    FILENAME="$(basename $FILE)"
	rm -rf /opt/koboldai/$FILENAME
	ln -s $FILE /opt/koboldai/
done


#Previous parameters are now env vars in the docker container so they can be overwritten as desired
PYTHONUNBUFFERED=1 ./play.sh
