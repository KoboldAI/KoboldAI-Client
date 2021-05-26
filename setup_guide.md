
# How to Setup Kobold AI

This is a semi-short guide on how to get Kobold AI running, and it includes a few other optional things that may interest you like importing data or setting up remote use (Google Colab, OpenAI, etc)

## Prerequisites
For making everything work:
 * [64-bit Python 3](.github/DOWNLOADING.md)
(When installing, make sure "Add Python to PATH" is selected.)

For GPU:
 * [NVidia CUDA toolkit](https://developer.nvidia.com/cuda-10.2-download-archive)
 * [PyTorch](https://pytorch.org/get-started/locally/)
(select Pip under "Package" and your version of CUDA under "Compute Platform" (I linked 10.2) to get the pip3 command. Then, paste it into command prompt to install torch with GPU support)

## Enabling GPU for Supported Video Cards

A few things to note:

**You can't enable GPU if you have an AMD GPU!** Enabling GPU requires NVidia CUDA toolkit, which is for NVidia cards only.

**You need to install the CUDA toolkit and PyTorch if you want to use GPU.** In case you missed that.

Be aware that when using GPU mode, inference will be MUCH faster but if your GPU doesn't have enough 
VRAM to load the model it will crash the application.

## Instructions

1. Run install_requirements.bat.
	(This will install the necessary python packages via pip)
2. Run play.bat
3. Select a model from the list. Flask will start and give you a message that it's ready to connect.
4. Open a web browser and enter http://127.0.0.1:5000/

## Enable Colors in Windows 10 Command Line

If you see strange numeric tags in the console output, then your console of choice does not have
color support enabled. On Windows 10, you can enable color support by lanching the registry editor
and adding the REG_DWORD key VirtualTerminalLevel to Computer\HKEY_CURRENT_USER\Console and setting
its value to 1.

## Additional Setup

At this point, KoboldAI may be ready depending on how you want to use it. Here's a couple other nifty things you can do if it's worth setting up for you.

### Importing AI Dungeon Games

To import your games from AI Dungeon, [first grab CuriousNekomimi's AI Dungeon Content Archive Toolkit](https://github.com/CuriousNekomimi/AIDCAT)
Follow the video instructions for getting your access_token, and run aidcat.py in command prompt.
Choose option [1] Download your saved content.
Choose option [2] Download your adventures.
Save the JSON file to your computer using the prompt.
Run KoboldAI, and after connecting to the web GUI, press the Import button at the top.
Navigate to the JSON file exported from AIDCAT and select it. A prompt will appear in the GUI 
presenting you with all Adventures scraped from your AI Dungeon account.
Select an Adventure and click the Accept button.

### Hosting GPT-NEO on Google Colab

If your computer does not have an 8GB GPU to run GPT-Neo locally, you can now run a Google Colab
notebook hosting a GPT-Neo-2.7B model remotely and connect to it using the KoboldAI client.
[Instructions on running the Colab, from google itself](https://colab.research.google.com/drive/1uGe9f4ruIQog3RLxfUsoThakvLpHjIkX?usp=sharing)


### For Inferkit Integration

If you would like to use InferKit's Megatron-11b model, [sign up for a free account on their website.](https://inferkit.com/)

After verifying your email address, sign in and click on your profile picture in the top right.
In the drop down menu, click "API Key".
On the API Key page, click "Reveal API Key" and copy it. When starting KoboldAI and selecting the
InferKit API model, you will be asked to paste your API key into the terminal. After entering,
the API key will be stored in the client.settings file for future use.
You can see your remaining budget for generated characters on their website under "Billing & Usage".

### OpenAI Support

KoboldAI now supports OpenAI's API. Their models are incredibly strong, but there is a little bit of a complication here, that being moderated API usage. **If you're going to use the API in a way that keeps God in heaven, it's most likely not worth going through the trouble of getting an API key.**

If that doesn't dissuade you, [you'll need to get an API key from them.](https://beta.openai.com/?app=creative-gen)
