# KoboldAI - Your gateway to GPT writing

This is a browser-based front-end for AI-assisted writing with multiple local & remote AI models. It offers the standard array of tools, including Memory, Author's Note, World Info, Save & Load, adjustable AI settings, formatting options, and the ability to import existing AI Dungeon adventures. You can also turn on Adventure mode and play the game like AI Dungeon Unleashed.

## Multiple ways to play

Stories can be played like a Novel, a text adventure game or used as a chatbot with an easy toggles to change between the multiple gameplay styles. This makes KoboldAI both a writing assistant, a game and a platform for so much more. The way you play and how good the AI will be depends on the model or service you decide to use. No matter if you want to use the free, fast power of Google Colab, your own high end graphics card, an online service you have an API key for (Like OpenAI or Inferkit) or if you rather just run it slower on your CPU you will be able to find a way to use KoboldAI that works for you.

### Adventure mode

By default KoboldAI will run in a generic mode optimized for writing, but with the right model you can play this like AI Dungeon without any issues. You can enable this in the settings and bring your own prompt, try generating a random prompt or download one of the prompts available at [prompts.aidg.club](https://prompts.aidg.club) .

The gameplay will be slightly different than the gameplay in AI Dungeon because we adopted the Type of the Unleashed fork, giving you full control over all the characters because we do not automatically adapt your sentences behind the scenes. This means you can more reliably control characters that are not you.

As a result of this what you need to type is slightly different, in AI Dungeon you would type ***take the sword*** while in KoboldAI you would type it like a sentence such as ***You take the sword*** and this is best done with the word You instead of I.

To speak simply type : *You say "We should probably gather some supplies first"*
Just typing the quote might work, but the AI is at its best when you specify who does what in your commands.

If you want to do this with your friends we advise using the main character as You and using the other characters by their name if you are playing on a model trained for Adventures. These models assume there is a You in the story. This mode does usually not perform well on Novel models because they do not know how to handle the input those are best used with regular story writing where you take turns with the AI.

### Writing assistant

If you want to use KoboldAI as a writing assistant this is best done in the regular mode with a model optimized for Novels. These models do not make the assumption that there is a You character and focus on Novel like writing. For writing these will often give you better results than Adventure or Generic models. That said, if you give it a good introduction to the story large generic models like 6B can be used if a more specific model is not available for what you wish to write. You can also try to use models that are not specific to what you wish to do, for example a NSFW Novel model for a SFW story if a SFW model is unavailable. This will mean you will have to correct the model more often because of its bias, but can still produce good enough results if it is familiar enough with your topic.

### Chatbot Mode

In chatbot mode you can use a suitable model as a chatbot, this mode automatically adds your name to the beginning of the sentences and prevents the AI from talking as you. To use it properly you must write your story opening as both characters in the following format (You can use your own text) :

``` ChatBot Opening Example
Bot : Hey!
You : Hey Boyname, how have you been?
Bot : Been good! How about you?
You : Been great to, excited to try out KoboldAI
Bot : KoboldAI is really fun!
You : For sure! What is your favorite game?
```

Its recommended to have your own input be the last input, especially in the beginning its possible that the AI mixes up the names. In that case either retry or manually correct the name. This behavior improves as the chat progresses. Some models may swap names if they are more familiar with a different name that is similar to the name you defined for the bot. In that case you can either do the occasional manual correction or choose a name for your chatbot that the AI likes better.

This mode works the best on either a Generic model or a chatbot model specifically designed for it, some models like the AvrilAI model are instead designed to be used in Adventure mode and do not conform to the format above. These models typically ship with adventure mode enabled by default and should not be switched over to chatbot mode.

Novel or Adventure models are not recommended for this feature but might still work but can derail away from the conversation format quickly.



## Play KoboldAI online for free on Google Colab (The easiest way to play)

If you would like to play KoboldAI online for free on a powerful computer you can use Google Colaboraty. We provide two editions, a TPU and a GPU edition with a variety of models available. These run entirely on Google's Servers and will automatically upload saves to your Google Drive if you choose to save a story (Alternatively, you can choose to download your save instead so that it never gets stored on Google Drive). Detailed instructions on how to use them are at the bottom of the Colab's.

Each edition features different models and requires different hardware to run, this means that if you are unable to obtain a TPU or a GPU you might still be able to use the other version. The models you can use are listed underneath the edition. To open a Colab click the big link featuring the editions name.

### [Click here for the TPU Edition Colab](https://colab.research.google.com/github/KoboldAI/KoboldAI-Client/blob/main/colab/TPU.ipynb)

| Model                          | Size   | Type     | Drive Space | Description                                                  |
| ------------------------------ | ------ | --------- | ----------- | ------------------------------------------------------------ |
| Skein 6B by VE_FORBDRYDERNE    | 6B TPU | Hybrid    | 0 GB         | Skein is our flagship 6B model, it is a hybrid between a Adventure model and a Novel model. Best used with either Adventure mode or the You Bias userscript enabled. Skein has been trained on high quality Novels along with CYOA adventure stories and is not as wackey as the Adventure model. It also has tagging support. |
| Adventure 6B by VE_FORBRYDERNE | 6B TPU | Adventure | 0 GB         | Adventure is a 6B model designed to mimick the behavior of AI Dungeon. It is exclusively for Adventure Mode and can take you on the epic and wackey adventures that AI Dungeon players love. It also features the many tropes of AI Dungeon as it has been trained on very similar data. It must be used in second person (You). |
| Lit 6B by Haru                 | 6B TPU | NSFW      | 8 GB /  12 GB | Lit is a great NSFW model trained by Haru on both a large set of Literotica stories and high quality novels along with tagging support. Creating a high quality model for your NSFW stories. This model is exclusively a novel model and is best used in third person. |
| Generic 6B by EleutherAI       | 6B TPU | Generic   | 10 GB / 12 GB | GPT-J-6B is what all other models are based on, if you need something that has no specific bias towards any particular subject this is the model for you. Best used when the other models are not suitable for what you wish to do. Such as homework assistance, blog writing, coding and more. It needs more hand holding than other models and is more prone to undesirable formatting changes. |
| C1 6B by Haru                  | 6B TPU | Chatbot   | 8 GB /  12 GB | C1 has been trained on various internet chatrooms, it makes the basis for an interesting chatbot model and has been optimized to be used in the Chatmode. |

### [Click here for the GPU Edition Colab](https://colab.research.google.com/github/KoboldAI/KoboldAI-Client/blob/main/colab/GPU.ipynb)

| Model                                                        | Size     | Type      | Description                                                  |
| ------------------------------------------------------------ | -------- | ---------- | ------------------------------------------------------------ |
| [GPT-Neo-2.7B-Picard](https://huggingface.co/KoboldAI/GPT-Neo-2.7B-Picard) by Mr Seeker | 2.7B GPU | Novel      | Picard is a model trained for SFW Novels based on GPT-Neo-2.7B. It is focused on Novel Type writing without the NSFW bias. While the name suggests a sci-fi model this model is designed for Novels of a variety of genre's. It is meant to be used in KoboldAI's regular mode. |
| [GPT-Neo-2.7B-AID](https://huggingface.co/KoboldAI/GPT-Neo-2.7B-AID) by melastacho | 2.7B GPU | Adventure | Also know as Adventure 2.7B this is a clone of the AI Dungeon Classic model and is best known for the epic wackey adventures that AI Dungeon Classic players love. |
| [GPT-Neo-2.7B-Horni-LN](https://huggingface.co/KoboldAI/GPT-Neo-2.7B-Horni-LN) by finetune | 2.7B GPU | Novel | This model is based on GPT-Neo-2.7B-Horni and retains its NSFW knowledge, but was then further biased towards SFW novel stories. If you seek a balance between a SFW Novel model and a NSFW model this model should be a good choice. |
| [GPT-Neo-2.7B-Horni](https://huggingface.co/KoboldAI/GPT-Neo-2.7B-Horni) by finetune | 2.7B GPU | NSFW       | This model is tuned on Literotica to produce a Novel Type model biased towards NSFW content. Can still be used for SFW stories but will have a bias towards NSFW content. It is meant to be used in KoboldAI's regular mode. |
| [GPT-Neo-2.7B-Shinen](https://huggingface.co/KoboldAI/GPT-Neo-2.7B-Shinen) by Mr Seeker | 2.7B GPU | NSFW       | Shinen is an alternative to the Horni model designed to be more explicit. If Horni is to tame for you shinen might produce better results. While it is a Novel model it is unsuitable for SFW stories due to its heavy NSFW bias. Shinen will not hold back. It is meant to be used in KoboldAI's regular mode. |
| [GPT-Neo-2.7B](https://huggingface.co/EleutherAI/gpt-neo-2.7B) by EleutherAI | 2.7B GPU    | Generic    | This is the base model for all the other 2.7B models, it is best used when you have a use case that we have no other models available for, such as writing blog articles or programming. It can also be a good basis for the experience of some of the softprompts if your softprompt is not about a subject the other models cover. |

### Model Types
| Type     | Description                                                  |
| --------- | ------------------------------------------------------------ |
| Novel     | For regular story writing, not compatible with Adventure mode or other specialty modes. |
| NSFW      | Indicates that the model is strongly biased towards NSFW content and is not suitable for children, work environments or livestreaming. Most NSFW models are also Novel models in nature. |
| Adventure | These models are excellent for people willing to play KoboldAI like a Text Adventure game and are meant to be used with Adventure mode enabled. Even if you wish to use it as a Novel Type model you should always have Adventure mode on and set it to story. These models typically have a strong bias towards the use of the word You and without Adventure mode enabled break the story flow and write actions on your behalf. |
| Chatbot   | These models are specifically trained for chatting and are best used with the Chatmode enabled. Typically trained on either public chatrooms or private chats. |
| Hybrid    | Hybrid models are a blend between different Types, for example they are trained on both Novel stories and Adventure stories. These models are great variety models that you can use for multiple different playTypes and modes, but depending on your usage you may need to enable Adventure Mode or the You bias (in userscripts). |
| Generic   | Generic models are not trained towards anything specific, typically used as a basis for other tasks and models. They can do everything the other models can do, but require much more handholding to work properly. Generic models are an ideal basis for tasks that we have no specific model for, or for experiencing a softprompt in its raw form. |


## Install KoboldAI on your own computer

KoboldAI has a large number of dependencies you will need to install on your computer, unfortunately Python does not make it easy for us to provide instructions that work for everyone. The instructions below will work on most computers, but if you have multiple versions of Python installed conflicts can occur.

### Downloading the latest version of KoboldAI

KoboldAI is a rolling release on our github, the code you see is also the game. You can the software by clicking on the green Code button at the top of the page and clicking Download ZIP.

The easiest way for Windows users is to use the [offline installer](https://sourceforge.net/projects/koboldai/files/latest/download) below.

### Installing KoboldAI offline bundle on Windows 7 or higher using the KoboldAI Offline Installer (Easiest)

1. [Download the latest offline installer from here](https://sourceforge.net/projects/koboldai/files/latest/download)
2. Run the installer to place KoboldAI on a location of choice, KoboldAI is portable software and is not bound to a specific harddrive. (Because of long paths inside our dependencies you may not be able to extract it many folders deep).
3. Update KoboldAI to the latest version with update-koboldai.bat if desired.
4. Use KoboldAI offline using play.bat or remotely with remote-play.bat

### Installing KoboldAI Github release on Windows 10 or higher using the KoboldAI Runtime Installer

1. Extract the .zip to a location you wish to install KoboldAI, you will need roughly 20GB of free space for the installation (this does not include the models).
2. Open install_requirements.bat as **administrator**.
3. Choose the regular version of Transformers (Option 1), finetuneanon is depreciated and no longer recommended.
4. You will now be asked to choose the installation mode, we **strongly** recommend the Temporary B: drive option. This option eliminates most installation issues and also makes KoboldAI portable. The B: drive will be gone after a reboot and will automatically be recreated each time you play KoboldAI.
5. The installation will now automatically install its requirements, some stages may appear to freeze do not close the installer until it asks you to press a key. Before pressing a key to exit the installer please check if errors occurred. Most problems with the game crashing are related to installation/download errors. Disabling your antivirus can help if you get errors.
6. Use play.bat to start KoboldAI.

### Manual installation / Linux / Mac

We can not provide a step by step guide for manual installation due to the vast differences between the existing software configuration and the systems of our users.

If you would like to manually install KoboldAI you will need some python/conda package management knowledge to manually do one of the following steps :

1. Use our bundled environments files to install your own conda environment, this should also automatically install CUDA (Recommended, you can get Miniconda from https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links). The recommended configuration is huggingface.yml for CUDA users and rocm.yml for ROCm users.
2. If you have a working copy of Docker for either CUDA or ROCm try play-cuda.sh or play-rocm.sh to launch the docker versions. In this case the installation is mostly automatic.
3. If conda is proving difficult you could also look inside requirements.txt for the required dependencies and try to install them yourself. This will likely be a mixture of pip and your native package manager, just installing our requirements.txt is not recommended since to speed things up we do not force any version changes. For local installations definitely prioritize conda as that is a better way for us to enforce you have the latest compatible versions.

### AMD GPU's

AMD GPU's have terrible compute support, this will currently not work on Windows and will only work for a select few Linux GPU's. [You can find a list of the compatible GPU's here](https://github.com/RadeonOpenCompute/ROCm#Hardware-and-Software-Support). Any GPU that is not listed is guaranteed not to work with KoboldAI and we will not be able to provide proper support on GPU's that are not compatible with the versions of ROCm we require. 

### Troubleshooting

There are multiple things that can go wrong with the way Python handles its dependencies, unfortunately we do not have direct step by step solutions for every scenario but there are a few common solutions you can try.

#### ModuleNotFoundError

This is ALWAYS either a download/installation failure or a conflict with other versions of Python. This is very common if users chose the subfolder option during the installation while putting KoboldAI in a location that has spaces in the path. When an antivirus sandboxes the installation or otherwise interferes with the downloads, systems with low disk space or when your operating system was not configured for Long FIle Paths (The installer will do this on Windows 10 and higher if you run it as administrator, anything other than Windows 10 is not supported by our installers).

Another reason the installation may have failed is if you have conflicting installations of Python on your machine, if you press the Windows Key + R and enter %appdata% in the Run Dialog it will open the folder Python installs dependencies on some systems. If you have a Python folder in this location rename this folder and try to run the installer again. It should now no longer get stuck on existing dependencies. Try the game and see if it works well. If it does you can try renaming the folder back to see if it remains functional.

The third reason the installation may have failed is if you have conda/mamba on your system for other reasons, in that case we recommend either removing your existing installations of python/conda if you do not need them and testing our installer again. Or using conda itself with our bundled environment files to let it create its runtime manually. **Keep in mind that if you go the manual route you should NEVER use play.bat but should instead run aiserver.py directly**.

In general, the less versions of Python you have on your system the higher your chances of it installing correctly. We are consistently trying to mitigate these installation conflicts in our installers but for some users we can not yet avoid all conflicts.

#### GPU not found errors

GPU not found errors can be caused by one of two things, either you do not have a suitable Nvidia GPU (It needs Compute Capability 5.0 or higher to be able to play KoboldAI). Your Nvidia GPU is supported by KoboldAI but is not supported by the latest version of CUDA. Your Nvidia GPU is not yet supported by the latest version of CUDA or you have a dependency conflict like the ones mentioned above.

Like with Python version conflicts we recommend uninstalling CUDA from your system if you have manually installed it and do not need it for anything else and trying again. If your GPU needs CUDA10 to function open environments\finetuneanon.yml and add a line that says - cudatoolkit=10.2 underneath dependencies: . After this you can run the installer again (Pick the option to delete the existing files) and it will download a CUDA10 compatible version.

If you do not have a suitable Nvidia GPU that can run on CUDA10 or Higher and that supports Compute Capabilities 5.0 or higher we can not help you get the game detected on the GPU. Unless you are following our ROCm guide with a compatible AMD GPU.

#### vocab.json / config.json is not found error

If you get these errors you either did not select the correct folder for your custom model or the model you have downloaded is not (yet) compatible with KoboldAI. There exist a few models out there that are compatible and provide a pytorch_model.bin file but do not ship all the required files. In this case try downloading a compatible model of the same kind (For example another GPT-Neo if you downloaded a GPT-Neo model) and replace the pytorch_model.bin file with the one you are trying to run. Chances are this will work fine.

## KoboldAI Compatible Models

Most of the high quality models have been integrated in the menu, these models have their download link removed since the easiest way to obtain them is to run them directly from the menu. KoboldAI will automatically download and convert the models to a offline format for later use.

If you have old 6B versions which end in -hf they will no longer be compatible with the newer versions of transformers and will no longer behave correctly. It is highly recommended that you install the official version of transformers (offline installers for KoboldAI contain this version by default) and redownload these models from the menu to get compatible versions. If you have very limited internet we will for a limited time also offer finetuneanon's fork in the install_requirements.bat file, when using that option you will not be able to use the 6B models in our main menu so definitely upgrade when your internet allows.

The VRAM requirements amounts are the recommended amounts for fast smooth play, playing with lower VRAM is possible but then you may need to either lower the amount of tokens in the settings, or you may need to put less layers on your GPU causing a significant performance loss. 

**For CPU players and during the loading regular RAM usage is double of what we list here.**



| **Model**                                                    | Type                              | **(V)RAM** | Repetition Penalty | Description                                                  |
| ------------------------------------------------------------ | --------------------------------- | ---------- | ------------------ | ------------------------------------------------------------ |
| Skein 6B by VE_FORBDRYERNE | Adventure Novel / 6B / Neo Custom | 16GB       | 1.1                | Skein is our flagship 6B model, it is a hybrid between a Adventure model and a Novel model. Best used with either Adventure mode or the You Bias userscript enabled. Skein has been trained on high quality Novels along with CYOA adventure stories and is not as wackey as the Adventure model. It also has tagging support. |
| Adventure 6B by VE_FORBRYDERNE | Adventure / 6B / Neo Custom       | 16GB       | 1.2                | Adventure is a 6B model designed to mimick the behavior of AI Dungeon. It is exclusively for Adventure Mode and can take you on the epic and wackey adventures that AI Dungeon players love. It also features the many tropes of AI Dungeon as it has been trained on very similar data. It must be used in second person (You). |
| Adventure 2.7B by melastashco | Adventure / 2.7B / Neo Custom     | 8GB        | 2.0                | This is one of the closest replications of the original AI Dungeon Classic model. Tuned on the same data that got uploaded alongside AI Dungeon. In KoboldAI we noticed this model performs better than the conversions of the original AI Dungeon model. It has all the traits you expect of AI Dungeon Classic while not having as many artifacts as this model was trained specifically for KoboldAI. Must be played with Adventure mode enabled to prevent it from doing actions on your behalf. |
| Horni 2.7B by finetuneanon | Novel / 2.7B / Neo Custom         | 8GB        | 2.0                | One of the best novel models available for 2.7B focused on NSFW content. This model trains the AI to write in a story like fashion using a very large collection of Literotica stories. It is one of the original finetuned models for 2.7B. |
| Horni-LN 2.7B by finetuneanon | Novel / 2.7B / Neo Custom         | 8GB        | 2.0                | This model is much like the one above, but has been additionally trained on regular light novels. More likely to go SFW and is more focused towards themes found in these light novels over general cultural references. This is a good model for Novel writing especially if you want to add erotica to the mix. |
| Picard 2.7B by Mr Seeker | Novel / 2.7B / Neo Custom         | 8GB        | 2.0                | Picard is another Novel model, this time exclusively focused on SFW content of various genres. Unlike the name suggests this goes far beyond Star Trek stories and is not exclusively sci-fi. |
| Shinen 2.7B by Mr Seeker | Novel / 2.7B / Neo Custom         | 8GB        | 2.0                | The most NSFW of them all, Shinen WILL make things sexual. This model will assume that whatever you are doing is meant to be a sex story and will sexualize constantly. It is designed for people who find Horni to tame. It was trained on SexStories instead of Literotica and was trained on tags making it easier to guide the AI to the right context. |
| [AID-16Bit](https://storage.henk.tech/KoboldAI/aid-16bit.zip) | Adventure / 1.5B / GPT-2 Custom   | 4GB        | 2.0                | The original AI Dungeon Classic model converted to Pytorch and then converted to a 16-bit Model making it half the size. |
| [model_v5_pytorch](https://storage.henk.tech/KoboldAI/model_v5_pytorch.zip) (AI Dungeon's Original Model) | Adventure / 1.5B / GPT-2 Custom   | 8GB       | 2.0                | This is the original AI Dungeon Classic model converted to the Pytorch format compatible with AI Dungeon Clover and KoboldAI. We consider this model inferior to the GPT-Neo version because it has more artifacting due to its conversion. This is however the most authentic you can get to AI Dungeon Classic. |
| [Novel 774M](https://storage.henk.tech/KoboldAI/Novel%20model%20774M.rar) | Novel / 774M / GPT-2 Custom       | 4GB        | 2.0                | Novel 774M is made by the AI Dungeon Clover community, because of its small size and novel bias it is more suitable for CPU players that want to play with speed over substance or players who want to test a GPU with a low amount of VRAM. These performance savings are at the cost of story quality and you should not expect the kind of in depth story capabilities that the larger models offer. It was trained for SFW stories. |
| [Smut 774M](https://storage.henk.tech/KoboldAI/Smut%20model%20774M%2030K.rar) | Novel / 774M / GPT-2 Custom       | 4GB        | 2.0                | The NSFW version of the above, its a smaller GPT-2 based model made by the AI Dungeon Clover community. Gives decent speed on a CPU at the cost of story quality like the other 774M models. |
| [Mia (GPT-Neo-125M-AID)](https://huggingface.co/KoboldAI/GPT-Neo-125M-AID) by Henk717 | Adventure / 125M / Neo Custom     | 1GB        | 2.0                | Mia is the smallest Adventure model, it runs at very fast speeds on the CPU which makes it a good testing model for developers who do not have GPU access. Because of its small size it will constantly attempt to do actions on behalf of the player and it will not produce high quality stories. If you just need a small model for a quick test, or if you want to take the challenge of trying to run KoboldAI entirely on your phone this would be an easy model to use due to its small RAM requirements and fast (loading) speeds. |

## Softprompts

Softprompts (also known as Modules in other products) are addons that can change the output of existing models. For example you may load a softprompt that biases the AI towards a certain subject and style like transcripts from your favorite TV show. 

Since these softprompts are often based on existing franchises we currently do not bundle any of them with KoboldAI due to copyright concerns (We do not want to put the entire project at risk). Instead look at community resources like #softprompts on the [KoboldAI Discord](https://discord.gg/XuQWadgU9k) or the [community hosted mirror](https://storage.henk.tech/KoboldAI/softprompts/) .

That way we are better protected from any DMCA claims as things can be taken down easier than directly on Github. If you have a copyright free softprompt that you made from scratch and is not based on existing IP that you would like to see officially bundled with KoboldAI issue a pull request with your softprompt.

Training softprompts can be done for free with the [mtj-softtuner colab](https://colab.research.google.com/github/VE-FORBRYDERNE/mtj-softtuner/blob/main/mtj-softtuner.ipynb) , in that case you can leave most of the settings default. Your source data needs to be a folder with text files that are UTF-8 formatted and contain Unix line endings.

## Userscripts

Userscripts are scripts that can automate tasks in KoboldAI, or modify the AI behavior / input / output.
Scripting is done in LUA5.4 (Lua does not need to be separately installed as long as you got all the python requirements) and has sandboxing to help protect you from malicious behavior. Even with these measures in place we strongly advise you only run userscripts from places you trust and/or understand, otherwise consult the community for advice on how safe the script might be.

Inside the userscripts folder you will find our kaipreset scripts, these are default scripts that we think will be useful for our users. These scripts are automatically overwritten when you update KoboldAI, if you wish to modify these scripts make sure to first rename them to something else that does not contain kaipreset so your changes are not lost. These scripts range from a You Bias filter that prevents the AI from addressing characters as you. Ways to be able to prevent the AI from using words, word replacements and more. 

Along with our preset scripts we also ship examples in the examples folder that merely serve as a demonstration and do not enhance your usage of KoboldAI. To use these scripts make sure to move them out of the examples folder before either using or modifying the script.

Lastly the all the features of our userscript API are documented inside the API Documentation files inside the userscripts folder.

For our TPU versions keep in mind that scripts modifying AI behavior relies on a different way of processing that is slower than if you leave these userscripts disabled even if your script only sporadically uses this modifier. If you want to partially use a script at its full speed than you can enable "No Gen Modifiers" to ensure that the parts that would make the TPU slow are not active.

## Contributors

This project contains work from the following contributors :

- The Gantian - Creator of KoboldAI, has created most features such as the interface, the different AI model / API integrations and in general the largest part of the project.
- VE FORBRYDERNE - Contributed many features such as the Editing overhaul, Adventure Mode, expansions to the world info section, breakmodel integration, scripting support, softpromtps and much more. As well as vastly improving the TPU compatibility and integrating external code into KoboldAI so we could use official versions of Transformers with virtually no downsides.
- Henk717 - Contributed the installation scripts, this readme, random story generator, the docker scripts, the foundation for the commandline interface and other smaller changes as well as integrating multiple parts of the code of different forks to unite it all. He also optimized the model loading so that downloaded models get converted to efficient offline models and that in future models are more likely to work out of the box.  Not all code Github attributes to Henk717 is by Henk717 as some of it has been integrations of other people's work. We try to clarify this in the contributors list as much as we can.
- Ebolam - Automatic Saving
- Frogging101 - top_k / tfs support (Part of this support was later redone by VE to integrate what was originally inside of finetuneanon's transformers)
- UWUplus (Ralf) - Contributed storage systems for community colabs, as well as cleaning up and integrating the website dependencies/code better. He is also the maintainer of flask-cloudflared which we use to generate the cloudflare links.
- Javalar - Initial Performance increases on the story_refresh
- LexSong - Initial environment file adaptation for conda that served as a basis for the install_requirements.bat overhaul.
- Arrmansa - Breakmodel support for other projects that served as a basis for VE FORBRYDERNE's integration.
- Jojorne - Small improvements to the response selection for gens per action.

As well as various Model creators who will be listed near their models, and all the testers who helped make this possible!

Did we miss your contribution? Feel free to issue a commit adding your name to this list.

## License

KoboldAI is licensed with a AGPL license, in short this means that it can be used by anyone for any purpose. However, if you decide to make a publicly available instance your users are entitled to a copy of the source code including all modifications that you have made (which needs to be available trough an interface such as a button on your website), you may also not distribute this project in a form that does not contain the source code (Such as compiling / encrypting the code and distributing this version without also distributing the source code that includes the changes that you made. You are allowed to distribute this in a closed form if you also provide a separate archive with the source code.).

umamba.exe is bundled for convenience because we observed that many of our users had trouble with command line download methods, it is not part of our project and does not fall under the AGPL license. It is licensed under the BSD-3-Clause license.
