#!/bin/bash
# KoboldAI Easy Colab Deployment Script by Henk717

# read the options
TEMP=`getopt -o m:i:p:c:d:x:a:l:z:g:t:n:b:s:r: --long model:,init:,path:,configname:,download:,aria2:,dloc:,xloc:,7z:,git:,tar:,ngrok:,branch:,savemodel:,localtunnel:,lt:,revision:,backend: -- "$@"`
eval set -- "$TEMP"

# extract options and their arguments into variables.
while true ; do
    case "$1" in
        -m|--model)
            model=" --model $2" ; shift 2 ;;
        -r|--revision)
            revision=" --revision $2" ; shift 2 ;;
        -i|--init)
            init=$2 ; shift 2 ;;
        -p|--path)
            mpath="$2" ; shift 2 ;;
        -c|--configname)
            configname=" --configname $2" ; shift 2 ;;
        -n|--ngrok)
            ngrok=" --ngrok" ; shift 2 ;;
        --lt|--localtunnel)
            localtunnel=" --localtunnel" ; shift 2 ;;
        -d|--download)
            download="$2" ; shift 2 ;;
        -a|--aria2)
            aria2="$2" ; shift 2 ;;
        -l|--dloc)
            dloc="$2" ; shift 2 ;;
        -x|--xloc)
            xloc="$2" ; shift 2 ;;
        -z|--7z)
            z7="$2" ; shift 2 ;;
        -t|--tar)
            tar="$2" ; shift 2 ;;
        -g|--git)
            git="$2" ; shift 2 ;;
        -b|--branch)
            branch="$2" ; shift 2 ;;
        -s|--savemodel)
            savemodel=" --savemodel" ; shift 2 ;;
        --backend)
            backend=" --model_backend $2" ; shift 2 ;;
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
    echo "Launching KoboldAI with the following options : python3 aiserver.py$model$kmpath$configname$ngrok$localtunnel$savemodel$revision$backend --colab"
    python3 aiserver.py$model$kmpath$configname$ngrok$localtunnel$savemodel$revision$backend --colab
    exit
    fi
}

git_default_branch() {
  (git remote show $git | grep 'HEAD branch' | cut -d' ' -f5) 2>/dev/null
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

# Redefine the extraction location
if [ "$xloc" == "drive" ]; then
    xloc="/content/drive/MyDrive/KoboldAI/models/"
    dloc="/content"
else
    xloc="/content/"
fi

# Redefine the Path to be in the relevant location
if [[ -v mpath ]];then
mpath="$xloc$mpath"
kmpath=" --path $mpath"
fi

# Create folders on Google Drive
mkdir /content/drive/MyDrive/KoboldAI/
mkdir /content/drive/MyDrive/KoboldAI/stories/
mkdir /content/drive/MyDrive/KoboldAI/models/
mkdir /content/drive/MyDrive/KoboldAI/settings/
mkdir /content/drive/MyDrive/KoboldAI/softprompts/
mkdir /content/drive/MyDrive/KoboldAI/userscripts/
mkdir /content/drive/MyDrive/KoboldAI/presets/
mkdir /content/drive/MyDrive/KoboldAI/themes/

if [ "$init" == "drive" ]; then
	echo Google Drive folders created.
	exit 0
fi
    
# Install and/or Update KoboldAI
if [ "$init" != "skip" ]; then
    cd /content
    if [ ! -z ${git+x} ]; then
        if [ "$git" == "Official" ]; then
            git=https://github.com/koboldai/KoboldAI-Client
        fi
        if [ "$git" == "United" ]; then
            git=https://github.com/henk717/KoboldAI-Client
            #branch=b603690e4cc3dff7491cbc2fd0b445a18b7b88d4
        fi
        if [ "$git" == "united" ]; then
            git=https://github.com/henk717/KoboldAI-Client
            #branch=b603690e4cc3dff7491cbc2fd0b445a18b7b88d4
        fi
    else
        git=https://github.com/koboldai/KoboldAI-Client
    fi

    mkdir /content/KoboldAI-Client
    cd /content/KoboldAI-Client

    git init
    git remote remove origin
    git remote add origin $git
    git fetch --all

    if [ ! -z ${branch+x} ]; then
        git checkout $branch -f
        git reset --hard origin/$branch
    else
        git checkout $(git_default_branch) -f
        git reset --hard origin/$(git_default_branch)
    fi

    git submodule update --init --recursive
    
    cd /content/KoboldAI-Client

    cp -rn stories/* /content/drive/MyDrive/KoboldAI/stories/
    cp -rn userscripts/* /content/drive/MyDrive/KoboldAI/userscripts/
    cp -rn softprompts/* /content/drive/MyDrive/KoboldAI/softprompts/
    cp -rn presets/* /content/drive/MyDrive/KoboldAI/presets/
    cp -rn themes/* /content/drive/MyDrive/KoboldAI/themes/
    rm -rf AI-Horde-Worker/
    rm -rf KoboldAI-Horde-Bridge/
    rm stories
    rm -rf stories/
    rm userscripts
    rm -rf userscripts/
    rm softprompts
    rm -rf softprompts/
    rm models
    rm -rf models/
    rm presets
    rm -rf presets/
    rm themes
    rm -rf themes/
    ln -s /content/drive/MyDrive/KoboldAI/stories/ stories
    ln -s /content/drive/MyDrive/KoboldAI/settings/ settings
    ln -s /content/drive/MyDrive/KoboldAI/softprompts/ softprompts
    ln -s /content/drive/MyDrive/KoboldAI/userscripts/ userscripts
    ln -s /content/drive/MyDrive/KoboldAI/models/ models
    ln -s /content/drive/MyDrive/KoboldAI/presets/ presets
    ln -s /content/drive/MyDrive/KoboldAI/themes/ themes

    # Remove conflicting dependencies
    sudo apt remove python3-blinker -y

    if [ -n "${COLAB_TPU_ADDR+set}" ]; then
        pip install -r requirements_mtj.txt
    else
        pip install -r requirements.txt
    fi
    
    # Make sure Colab has the system dependencies
    sudo apt update
    sudo apt install netbase aria2 -y
    npm install -g localtunnel
fi

cd /content

# Models extracted? Then we skip anything beyond this point for faster loading.
if [ -f "/content/extracted" ]; then
    launch
fi

# Is the model extracted on Google Drive? Skip the download and extraction
# Only on Google Drive since it has a big impact there if we don't, and locally we have better checks in place
if [ "$xloc" == "/content/drive/MyDrive/KoboldAI/models/"  ] && [[ -d $mpath ]];then
    launch
fi

#Download routine for regular Downloads
if [ ! -z ${download+x} ]; then
    wget -c $download -P $dloc
fi

#Download routine for Aria2c scripts
if [ ! -z ${aria2+x} ]; then
    curl -L $aria2 | aria2c -x 10 -s 10 -j 10 -c -i- -d$dloc --user-agent=KoboldAI --file-allocation=none
fi

#Extract the model with 7z
if [ ! -z ${z7+x} ]; then
    7z x -o$xloc $dloc/$z7 -aos
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
    pv $dloc/$tar | tar -I zstd -C $xloc -x
    touch /content/extracted
fi

launch
