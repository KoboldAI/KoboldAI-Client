#!/bin/bash
cd /opt/koboldai
if [[ -n update ]];then
        git pull
fi

#The goal here is to allow any directory in /content to be mapped to the appropriate dir in the koboldai dir
if [[ ! -d "/content" ]];then
        mkdir /content
fi

for FILE in /content/*;do
        rm -rf /opt/koboldai/$FILE
        ln -s /content/$FILE /opt/koboldai/
        #mount --bind /content/$FILE /opt/koboldai/$FILE
done


#Previous parameters are now env vars in the docker container so they can be overwritten as desired
PYTHONUNBUFFERED=1 ./play.sh
