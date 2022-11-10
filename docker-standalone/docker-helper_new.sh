#!/bin/bash
cd /opt/koboldai
if [[ -n update ]];then
	git pull
	cd KoboldAI-Horde/
	git pull
	cd ..
fi

#The goal here is to allow any directory in /content to be mapped to the appropriate dir in the koboldai dir
if [[ ! -d "/content" ]];then
	mkdir /content
fi

for FILE in *;do 
	if [[ -d "/opt/koboldai/$FILE" ]];then
		rm -rf /opt/koboldai/$FILE
	fi
	ln -s /content/$FILE /opt/koboldai/$FILE
done


#Previous parameters are now env vars in the docker container so they can be overwritten as desired
PYTHONUNBUFFERED=1 ./play.sh
