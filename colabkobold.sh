#!/bin/bash
# KoboldAI Easy Deployment Script by Henk717

# read the options
TEMP=`getopt -o m:i:p:c:d:a:l:z:g:t:n: --long model:,init:,path:,configname:,download:,aria2:,dloc:7z:git:tar:ngrok: -- "$@"`
eval set -- "$TEMP"

# extract options and their arguments into variables.
while true ; do
    case "$1" in
        -m|--model)
            model=" --model $2" ; shift 2 ;;
        -i|--init)
            init=$2 ; shift 2 ;;
        -p|--path)
            path=" --path /content/$2" ; shift 2 ;;
        -c|--configname)
            configname=" --configname $2" ; shift 2 ;;
        -n|--ngrok)
            configname=" --ngrok" ; shift 2 ;;
        -d|--download)
            download="$2" ; shift 2 ;;
        -a|--aria2)
            aria2="$2" ; shift 2 ;;
        -l|--dloc)
            dloc="$2" ; shift 2 ;;
        -z|--7z)
            z7="$2" ; shift 2 ;;
        -t|--tar)
            tar="$2" ; shift 2 ;;
        -g|--git)
            git="$2" ; shift 2 ;;
        --) shift ; break ;;
        *) echo "Internal error!" ; exit 1 ;;
    esac
done

# Create the Launch function so we can run KoboldAI at different places in the script
function launch
{
    #End the script if "--init only" was specified.
    if [ "$init" == "only" ]; then
        echo Initialization complete...
        exit 0
    else
    cd /content/KoboldAI-Client
    python3 aiserver.py$model$path$configname$ngrok --remote --override_delete --override_rename
    exit
    fi
}

# Don't allow people to mess up their system
if [[ ! -d "/content" ]]; then
    echo You can only use this script on Google Colab
    echo Use aiserver.py to play KoboldAI locally.
    echo Check our Readme for Colab links if you wish to play on Colab.
    exit
fi

# Redefine the download location
if [ "$dloc" == "colab" ]; then
    dloc="/content"
else
    dloc="/content/drive/MyDrive/KoboldAI/models"
fi

# Create Folder Structure and Install KoboldAI
if [ "$init" != "skip" ]; then
    if [ -f "/content/installed" ]; then
    echo KoboldAI already installed... Skipping installation....
    cd /content
    else
    cd /content
    if [ ! -z ${git+x} ]; then
        if [ "$git" == "united" ]; then
            git clone https://github.com/henk717/KoboldAI-Client
        fi
        git clone $git
    else
        git clone https://github.com/koboldai/KoboldAI-Client
    fi

    mkdir /content/drive/MyDrive/KoboldAI/
    mkdir /content/drive/MyDrive/KoboldAI/stories/
    mkdir /content/drive/MyDrive/KoboldAI/models/
    mkdir /content/drive/MyDrive/KoboldAI/settings/
    mkdir /content/drive/MyDrive/KoboldAI/softprompts/

    cd /content/KoboldAI-Client
    rm stories
    rm -rf stories/
    ln -s /content/drive/MyDrive/KoboldAI/stories/ stories
    ln -s /content/drive/MyDrive/KoboldAI/settings/ settings
    ln -s /content/drive/MyDrive/KoboldAI/softprompts/ softprompts

    if [ "$model" == " --model TPUMeshTransformerGPTJ" ]; then
        pip install -r requirements_mtj.txt
    else
        pip install -r requirements.txt
    fi
    touch /content/installed
    fi
fi

# Models extracted? Then we skip anything beyond this point for faster loading.
if [ -f "/content/extracted" ]; then
    launch
fi

#Download routine for regular Downloads
if [ ! -z ${download+x} ]; then
    wget -c $download -P $dloc
fi

#Download routine for Aria2c scripts
if [ ! -z ${aria2+x} ]; then
    apt install aria2 -y
    curl -L $aria2 | aria2c -c -i- -d$dloc --user-agent=KoboldAI
fi

#Extract the model with 7z
if [ ! -z ${z7+x} ]; then
    7z x -o/content/ -aos $dloc/$z7
    touch /content/extracted
fi

#Extract the model in a ZSTD Tar file
if [ ! -z ${tar+x} ]; then
    git clone https://github.com/VE-FORBRYDERNE/pv
    cd pv
    ./configure
    make
    make install
    cd ..
    apt install zstd -y
    pv $tar | tar -I zstd -x
    touch /content/extracted
fi

launch

