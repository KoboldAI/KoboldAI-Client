Thanks for checking out the KoboldAI Client! Keep up with news and updates on the subreddit:
https://www.reddit.com/r/KoboldAI/

[ABOUT]

This is a browser front-end for playing with multiple local & remote AI models.
The purpose is to provide a smoother, web-based UI experience than the various command-line AI apps.
I'm pushing this out now that the major quality-of-life fearures have been roughed in (generate,
undo, edit-by-line, memory, save/load, etc), which means there will probably be bugs.

This application uses Transformers (https://huggingface.co/transformers/) to interact with the AI models
via Tensorflow. Tensorflow has CUDA/GPU support for shorter generation times, but I do not have anything 
in this test release to set up CUDA/GPU support on your system. If you have a high-end GPU with
sufficient VRAM to run your model of choice, see (https://www.tensorflow.org/install/gpu) for
instructions on enabling GPU support.

Transformers/Tensorflow can still be used on CPU if you do not have high-end hardware, but generation
times will be much longer. Alternatively, KoboldAI also supports InferKit (https://inferkit.com/).
This will allow you to send requests to a remotely hosted Megatron-11b model for fast generation times
on any hardware. This is a paid service, but signing up for a free account will let you generate up
to 40,000 characters, and the free account will work with KoboldAI.

[SETUP]

1. Install a 64-bit version of Python.
	(Development was done on 3.7, I have not tested newer versions)
	Windows download link: https://www.python.org/ftp/python/3.7.9/python-3.7.9-amd64.exe
2. When installing Python make sure "Add Python to PATH" is selected.
	(If pip isn't working, run the installer again and choose Modify to choose Optional features.)
3. Run install_requirements.bat.
	(This will install the necessary python packages via pip)
4. Run play.bat
5. Select a model from the list. Flask will start and give you a message that it's ready to connect.
6. Open a web browser and enter http://127.0.0.1:5000/

[FOR INFERKIT INTEGRATION]

If you would like to use InferKit's Megatron-11b model, sign up for a free account on their website.
https://inferkit.com/
After verifying your email address, sign in and click on your profile picture in the top right.
In the drop down menu, click "API Key".
On the API Key page, click "Reveal API Key" and copy it. When starting KoboldAI and selecting the
InferKit API model, you will be asked to paste your API key into the terminal. After entering,
the API key will be stored in the client.settings file for future use.
You can see your remaining budget for generated characters on their website under "Billing & Usage".

[ENABLE COLORS IN WINDOWS 10 COMMAND LINE]

If you see strange numeric tags in the console output, then your console of choice does not have
color support enabled. On Windows 10, you can enable color support by lanching the registry editor
and adding the REG_DWORD key VirtualTerminalLevel to Computer\HKEY_CURRENT_USER\Console and setting
its value to 1.

[ENABLE GPU FOR SUPPORTED VIDEO CARDS]

1. Install NVidia CUDA toolkit from https://developer.nvidia.com/cuda-10.2-download-archive
2. Visit PyTorch's website(https://pytorch.org/get-started/locally/) and select Pip under "Package" 
and your version of CUDA under "Compute Platform" (I linked 10.2) to get the pip3 command.
3. Copy and paste pip3 command into command prompt to install torch with GPU support

Be aware that when using GPU mode, inference will be MUCH faster but if your GPU doesn't have enough 
VRAM to load the model it will crash the application.

[IMPORT AI DUNGEON GAMES]

To import your games from AI Dungeon, first grab CuriousNekomimi's AI Dungeon Content Archive Toolkit:
https://github.com/CuriousNekomimi/AIDCAT
Follow the video instructions for getting your access_token, and run aidcat.py in command prompt.
Choose option [1] Download your saved content.
Choose option [2] Download your adventures.
Save the JSON file to your computer using the prompt.
Run KoboldAI, and after connecting to the web GUI, press the Import button at the top.
Navigate to the JSON file exported from AIDCAT and select it. A prompt will appear in the GUI 
presenting you with all Adventures scraped from your AI Dungeon account.
Select an Adventure and click the Accept button.

[HOST GPT-NEO ON GOOGLE COLAB]

If your computer does not have an 8GB GPU to run GPT-Neo locally, you can now run a Google Colab
notebook hosting a GPT-Neo-2.7B model remotely and connect to it using the KoboldAI client.
See the instructions on the Colab at the link below:
https://colab.research.google.com/drive/1uGe9f4ruIQog3RLxfUsoThakvLpHjIkX?usp=sharing