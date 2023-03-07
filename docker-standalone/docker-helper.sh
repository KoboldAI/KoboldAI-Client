#!/bin/bash
cd /opt/koboldai

if [[ -n update ]];then
        git pull --recurse-submodules && ./install_requirements.sh cuda
	git submodule update --init --recursive
fi

if [[ ! -v KOBOLDAI_DATADIR ]];then
	mkdir /content
	KOBOLDAI_DATADIR=/content
fi

mkdir $KOBOLDAI_DATADIR/stories
mkdir $KOBOLDAI_DATADIR/settings
mkdir $KOBOLDAI_DATADIR/softprompts
mkdir $KOBOLDAI_DATADIR/userscripts
mkdir $KOBOLDAI_DATADIR/presets
mkdir $KOBOLDAI_DATADIR/themes

cp -rn stories/* $KOBOLDAI_DATADIR/stories/
cp -rn userscripts/* $KOBOLDAI_DATADIR/userscripts/
cp -rn softprompts/* $KOBOLDAI_DATADIR/softprompts/
cp -rn presets/* $KOBOLDAI_DATADIR/presets/
cp -rn themes/* $KOBOLDAI_DATADIR/themes/

if [[ -v KOBOLDAI_MODELDIR ]];then
	mkdir $KOBOLDAI_MODELDIR/models
	mkdir $KOBOLDAI_MODELDIR/functional_models
	rm models
	rm -rf models/
	ln -s $KOBOLDAI_MODELDIR/models/ models
	ln -s $KOBOLDAI_MODELDIR/functional_models/ functional_models
fi

for FILE in $KOBOLDAI_DATADIR*
do
    FILENAME="$(basename $FILE)"
	rm /opt/koboldai/$FILENAME
	rm -rf /opt/koboldai/$FILENAME
	ln -s $FILE /opt/koboldai/
done

PYTHONUNBUFFERED=1 ./play.sh --remote --quiet --override_delete --override_rename
