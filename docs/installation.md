## Install KoboldAI on your own computer

KoboldAI has a large number of dependencies you will need to install on your computer, unfortunately Python does not make it easy for us to provide instructions that work for everyone. The instructions below will work on most computers, but if you have multiple versions of Python installed conflicts can occur.

### Downloading the latest version of KoboldAI

KoboldAI is a rolling release on our github, the code you see is also the game. You can the software by clicking on the green Code button at the top of the page and clicking Download ZIP.

The easiest way for Windows users is to use the [offline installer](https://sourceforge.net/projects/koboldai/files/latest/download) below.

### Installing KoboldAI offline bundle on Windows 7 or higher using the KoboldAI Offline Installer (Easiest)

1.  [Download the latest offline installer from here](https://sourceforge.net/projects/koboldai/files/latest/download)
2.  Run the installer to place KoboldAI on a location of choice, KoboldAI is portable software and is not bound to a specific harddrive. (Because of long paths inside our dependencies you may not be able to extract it many folders deep).
3.  Update KoboldAI to the latest version with update-koboldai.bat if desired.
4.  Use KoboldAI offline using play.bat or remotely with remote-play.bat

### Installing KoboldAI Github release on Windows 10 or higher using the KoboldAI Runtime Installer

1.  Extract the .zip to a location you wish to install KoboldAI, you will need roughly 20GB of free space for the installation (this does not include the models).
2.  Open install\_requirements.bat as **administrator**.
3.  Choose the regular version of Transformers (Option 1), finetuneanon is depreciated and no longer recommended.
4.  You will now be asked to choose the installation mode, we **strongly** recommend the Temporary B: drive option. This option eliminates most installation issues and also makes KoboldAI portable. The B: drive will be gone after a reboot and will automatically be recreated each time you play KoboldAI.
5.  The installation will now automatically install its requirements, some stages may appear to freeze do not close the installer until it asks you to press a key. Before pressing a key to exit the installer please check if errors occurred. Most problems with the game crashing are related to installation/download errors. Disabling your antivirus can help if you get errors.
6.  Use play.bat to start KoboldAI.

### Installing KoboldAI on Linux using the KoboldAI Runtime (Easiest)

1.  Clone the URL of this Github repository (For example git clone [https://github.com/koboldai/koboldai-client](https://github.com/koboldai/koboldai-client) )
2.  AMD user? Make sure ROCm is installed if you want GPU support. Is yours not compatible with ROCm? Follow the usual instructions.
3.  Run play.sh or if your AMD GPU supports ROCm use play-rocm.sh

KoboldAI will now automatically configure its dependencies and start up, everything is contained in its own conda runtime so we will not clutter your system. The files will be located in the runtime subfolder. If at any point you wish to force a reinstallation of the runtime you can do so with the install\_requirements.sh file. While you can run this manually it is not neccesary.

### Manual installation / Mac

We can not provide a step by step guide for manual installation due to the vast differences between the existing software configuration and the systems of our users.

If you would like to manually install KoboldAI you will need some python/conda package management knowledge to manually do one of the following steps :

1.  Use our bundled environments files to install your own conda environment, this should also automatically install CUDA (Recommended, you can get Miniconda from https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links). The recommended configuration is huggingface.yml for CUDA users and rocm.yml for ROCm users.
2.  If conda is proving difficult you could also look inside requirements.txt for the required dependencies and try to install them yourself. This will likely be a mixture of pip and your native package manager, just installing our requirements.txt is not recommended since we assume local users will run conda to get all dependencies. For local installations definitely prioritize conda as that is a better way for us to enforce that you have the compatible versions.
3.  Clone our Github or download the zip file.
4.  Now start KoboldAI with aiserver.py and not with our play.bat or play.sh files.

### AMD GPU's (Linux only)

AMD GPU's have terrible compute support, this will currently not work on Windows and will only work for a select few Linux GPU's. [You can find a list of the compatible GPU's here](https://github.com/RadeonOpenCompute/ROCm#Hardware-and-Software-Support). Any GPU that is not listed is guaranteed not to work with KoboldAI and we will not be able to provide proper support on GPU's that are not compatible with the versions of ROCm we require. Make sure to first install ROCm on your Linux system using a guide for your distribution, after that you can follow the usual linux instructions above.

### Troubleshooting

There are multiple things that can go wrong with the way Python handles its dependencies, unfortunately we do not have direct step by step solutions for every scenario but there are a few common solutions you can try.

#### ModuleNotFoundError

This is ALWAYS either a download/installation failure or a conflict with other versions of Python. This is very common if users chose the subfolder option during the installation while putting KoboldAI in a location that has spaces in the path. When an antivirus sandboxes the installation or otherwise interferes with the downloads, systems with low disk space or when your operating system was not configured for Long FIle Paths (The installer will do this on Windows 10 and higher if you run it as administrator, anything other than Windows 10 is not supported by our installers).

Another reason the installation may have failed is if you have conflicting installations of Python on your machine, if you press the Windows Key + R and enter %appdata% in the Run Dialog it will open the folder Python installs dependencies on some systems. If you have a Python folder in this location rename this folder and try to run the installer again. It should now no longer get stuck on existing dependencies. Try the game and see if it works well. If it does you can try renaming the folder back to see if it remains functional.

The third reason the installation may have failed is if you have conda/mamba on your system for other reasons, in that case we recommend either removing your existing installations of python/conda if you do not need them and testing our installer again. Or using conda itself with our bundled environment files to let it create its runtime manually. **Keep in mind that if you go the manual route you should NEVER use play.bat but should instead run aiserver.py directly**.

In general, the less versions of Python you have on your system the higher your chances of it installing correctly. We are consistently trying to mitigate these installation conflicts in our installers but for some users we can not yet avoid all conflicts.

#### GPU not found errors

GPU not found errors can be caused by one of two things, either you do not have a suitable Nvidia GPU (It needs Compute Capability 5.0 or higher to be able to play KoboldAI). Your Nvidia GPU is supported by KoboldAI but is not supported by the latest version of CUDA. Your Nvidia GPU is not yet supported by the latest version of CUDA or you have a dependency conflict like the ones mentioned above.

Like with Python version conflicts we recommend uninstalling CUDA from your system if you have manually installed it and do not need it for anything else and trying again. If your GPU needs CUDA10 to function open environments\\finetuneanon.yml and add a line that says - cudatoolkit=10.2 underneath dependencies: . After this you can run the installer again (Pick the option to delete the existing files) and it will download a CUDA10 compatible version.

If you do not have a suitable Nvidia GPU that can run on CUDA10 or Higher and that supports Compute Capabilities 5.0 or higher we can not help you get the game detected on the GPU. Unless you are following our ROCm guide with a compatible AMD GPU.

#### vocab.json / config.json is not found error

If you get these errors you either did not select the correct folder for your custom model or the model you have downloaded is not (yet) compatible with KoboldAI. There exist a few models out there that are compatible and provide a pytorch\_model.bin file but do not ship all the required files. In this case try downloading a compatible model of the same kind (For example another GPT-Neo if you downloaded a GPT-Neo model) and replace the pytorch\_model.bin file with the one you are trying to run. Chances are this will work fine.
