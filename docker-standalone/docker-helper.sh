#!/bin/bash
cd /opt/koboldai
git pull
#./install_requirements.sh cuda

if [[ ! -v KOBOLDAI_DATADIR ]];then
	mkdir /content
	KOBOLDAI_DATADIR=/content
fi

mkdir $KOBOLDAI_DATADIR/stories
if [[ -v KOBOLDAI_MODELDIR ]];then
	mkdir $KOBOLDAI_MODELDIR/models
fi
mkdir $KOBOLDAI_DATADIR/settings
mkdir $KOBOLDAI_DATADIR/softprompts
mkdir $KOBOLDAI_DATADIR/userscripts
#mkdir $KOBOLDAI_MODELDIR/cache

cp -rn stories/* $KOBOLDAI_DATADIR/stories/
cp -rn userscripts/* $KOBOLDAI_DATADIR/userscripts/
cp -rn softprompts/* $KOBOLDAI_DATADIR/softprompts/

rm stories
rm -rf stories/
rm userscripts
rm -rf userscripts/
rm softprompts
rm -rf softprompts/

if [[ -v KOBOLDAI_MODELDIR ]];then
	rm models
	rm -rf models/
	#rm cache
	#rm -rf cache/
fi

ln -s $KOBOLDAI_DATADIR/stories/ stories
ln -s $KOBOLDAI_DATADIR/settings/ settings
ln -s $KOBOLDAI_DATADIR/softprompts/ softprompts
ln -s $KOBOLDAI_DATADIR/userscripts/ userscripts
if [[ -v KOBOLDAI_MODELDIR ]];then
	ln -s $KOBOLDAI_MODELDIR/models/ models
	#ln -s $KOBOLDAI_MODELDIR/cache/ cache
fi

PYTHONUNBUFFERED=1 ./play.sh --remote --quiet --override_delete --override_rename
