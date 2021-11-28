# KoboldAI - Your gateway to GPT writing

This is a browser-based front-end for AI-assisted writing with multiple local & remote AI models. It offers the standard array of tools, including Memory, Author's Note, World Info, Save & Load, adjustable AI settings, formatting options, and the ability to import existing AI Dungeon adventures. You can also turn on Adventure mode and play the game like AI Dungeon Unleashed.

## Multiple ways to play

Stories can be played like a Novel, or played like a text adventure game with an easy toggle to change between the two gameplay styles. This makes KoboldAI both a writing assistant and a game. The way you play and how good the AI will be depends on the model or service you decide to use. No matter if you want to use the free, fast power of Google Colab, your own high end graphics card, an online service you have an API key for (Like OpenAI or Inferkit) or if you rather just run it slower on your CPU you will be able to find a way to use KoboldAI that works for you.

### Adventure mode

By default KoboldAI will run in a generic mode optimized for writing, but with the right model you can play this like AI Dungeon without any issues. You can enable this in the settings and bring your own prompt, try generating a random prompt or download one of the prompts available at [prompts.aidg.club](https://prompts.aidg.club) .

The gameplay will be slightly different than the gameplay in AI Dungeon because we adopted the style of the Unleashed fork, giving you full control over all the characters because we do not automatically adapt your sentences behind the scenes. This means you can more reliably control characters that are not you.

As a result of this what you need to type is slightly different, in AI Dungeon you would type ***take the sword*** while in KoboldAI you would type it like a sentence such as ***You take the sword*** and this is best done with the word You instead of I.

To speak simply type : *You say "We should probably gather some supplies first"*
Just typing the quote might work, but the AI is at its best when you specify who does what in your commands.

If you want to do this with your friends we advice using the main character as You and using the other characters by their name if you are playing on a model trained for Adventures. These models assume there is a You in the story. This mode does usually not perform well on Novel models because they do not know how to handle the input those are best used with regular story writing where you take turns with the AI.

### Writing assistant

If you want to use KoboldAI as a writing assistant this is best done in the regular mode with a model optimized for Novels. These models do not make the assumption that there is a You character and focus on Novel like writing. For writing these will often give you better results than Adventure or Generic models. That said, if you give it a good introduction to the story large generic models like 6B can be used if a more specific model is not available for what you wish to write. You can also try to use models that are not specific to what you wish to do, for example a NSFW Novel model for a SFW story if a SFW model is unavailable. This will mean you will have to correct the model more often because of its bias, but can still produce good enough results if it is familiar enough with your topic.

## Play KoboldAI online for free on Google Colab (The easiest way to play)

We provide multiple ready made versions to get you going, click on the name for a link to the specific version. These run entirely on Google's Servers and will automatically upload saves to your Google Drive if you choose to manually save a story. Each version has slightly different instructions on how to use them (Many need some space on your google drive to run, others may need some manual steps) that are listed on the page.

TPU editions work on any configuration of TPU Google gives out at the time of writing. GPU editions are subject to a GPU lottery and may crash on launch if you are unlucky (Especially if a lot of users are using up the good GPU's or you have been using Colab often).

[Click here to open the Recommended version](https://henk.tech/colabkobold)

| Version                                                      | Model                                                        | Size     | Style           | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ | -------- | --------------- | ------------------------------------------------------------ |
| [Adventure 6B](https://colab.research.google.com/drive/1vdAsD0xCc_YsAXqBUxb_QAwPOXkFJtxm?usp=sharing#sandboxMode=true) | [gpt-j-6b-adventure-jax](https://wandb.ai/ve-forbryderne/adventure/runs/carol-data/files/models) by ve_forbryderne (Download the -hf version if you plan to run this locally) | 6B TPU   | Adventure       | This is the Recommended version for AI Dungeon players, this is effectively a Free Griffin but with more control. This Colab edition provides better memory than Griffin would have given you, allowing for a more coherent experience. And while it will still generate characters like The Great Litch Lord that AI Dungeon players are familiar with it was trained on stories beyond AI Dungeon and is more balanced in its approaches. This is a TPU edition so it can fit a lot in memory |
| [Skein](https://colab.research.google.com/drive/1ZAKgkSyyfiZN87npKYaRM8vL4OF2Btfg?usp=sharing#sandboxMode=true) | gpt-j-6b-skein-jax by ve_forbryderne (Download the -hf version if you plan to run this locally) | 6B TPU   | Novel/Adventure | Skein is a hybrid between a Novel model and the Adventure model. Because of this it needs a bit more context about the writing style (Needing a few retries in the random story generator if you use this). It was trained on both Light Novels and choose your own adventure stories along side extra information to help it understand story themes better. It is recommended to play this with Adventure mode enabled to prevent it from doing "Actions" even if you wish to use it for Novel writing. If you wish to use it for Novel writing you can do this by toggling the input to Story. |
| [Generic 6B TPU](https://colab.research.google.com/drive/1pG9Gz9PrqklNBESPNaXvfctMVnvwf_Q8#forceEdit=true&sandboxMode=true&scrollTo=jcxnaOk5Th4x) | [Original GPT-6-JAX Slim](https://the-eye.eu/public/AI/GPT-J-6B/step_383500_slim.tar.gz) (Requires a TPU and does not work local) | 6B TPU   | Novel           | The recommended model if you want a generic experience. This model is not optimized for anything in particular and works best when you give it a longer introduction. Make sure to include examples for the AI to learn from and write the first part of the story entirely yourself. Then it should be able to learn from your style and continue from there. Very sensitive to a high temp because it knows webpages and code, so when configured incorrectly it will easily end a story with 'Rate my blogpost, follow me on twitter' and the likes. |
| [Horni](https://colab.research.google.com/drive/1QwjkK_JeK9aYEkyM_6nrJXQARFMnBDmG?usp=sharing#sandboxMode=true) (Formerly Novel/NSFW) | [GPT-Neo-2.7B-Horni](https://storage.henk.tech/KoboldAI/gpt-neo-2.7B-horni.7z) by finetune | 2.7B GPU | Novel           | One of the oldest models in our collection, tuned on Literotica to produce a Novel style model optimized towards NSFW content. Can still be used for SFW stories but will have a bias towards NSFW content. Because this is an older 2.7B model it is only compatible as a GPU instance. Most GPU's in Colab are powerful enough to run this well but it will crash if you get something weak like a Nvidia P7. |
| [Picard](https://colab.research.google.com/drive/1VNVKtbPaTcmkQzy8bEQkd9SUiUJBdbEL?usp=sharing#sandboxMode=true) | [Picard](https://storage.henk.tech/KoboldAI/gpt-neo-2.7B-picard.7z) by Mr Seeker | 2.7B GPU | Novel           | Picard is a model trained for SFW Novels based on GPT-Neo-2.7B. It is focused on Novel style writing without the NSFW bias. While the name suggests a sci-fi model this model is designed for Novels of a variety of genre's. Most GPU's in Colab are powerful enough to run this well but it will crash if you get something weak like a Nvidia P7. |
| [Shinen](https://colab.research.google.com/drive/1-7Lkj-np2DaSnmq1OdPYkel6W2rh4E-0?usp=sharing#sandboxMode=true) | [Shinen](https://storage.henk.tech/KoboldAI/gpt-neo-2.7B-shinen.7z) by Mr Seeker | 2.7B GPU | Novel           | Shinen is an alternative to the Horni model designed to be more explicit. If Horni is to tame for you shinen might produce better results. While it is a Novel model it is unsuitable for SFW stories due to its heavy NSFW bias. Shinen will not hold back. Most GPU's in Colab are powerful enough to run this well but it will crash if you get something weak like a Nvidia P7. |

## Install KoboldAI on your own computer

KoboldAI has a large number of dependencies you will need to install on your computer, unfortunately Python does not make it easy for us to provide instructions that work for everyone. The instructions below will work on most computers, but if you have multiple versions of Python installed conflicts can occur.

### Downloading the latest version of KoboldAI

KoboldAI is a rolling release on our github, the code you see is also the game. The easiest way to download the game is by clicking on the green Code button at the top of the page and clicking Download ZIP.

### Installing KoboldAI on Windows 10 or higher using the KoboldAI Runtime Installer

1. Extract the .zip to a location you wish to install KoboldAI, you will need roughly 20GB of free space for the installation (this does not include the models).
2. Open install_requirements.bat as administrator.
3. Choose either the Finetuneanon or the Regular version of transformers (Finetuneanon works better for GPU players but breaks CPU mode, only use this version if you have a modern Nvidia GPU with enough VRAM for the model you wish to run).
4. You will now be asked to choose the installation mode, we **strongly** recommend the Temporary K: drive option for anyone who does not already have a K: drive on their computer. This option eliminates most installation issues and also makes KoboldAI portable. The K: drive will be gone after a reboot and will automatically be recreated each time you play KoboldAI.
5. The installation will now automatically install its requirements, some stages may appear to freeze do not close the installer until it asks you to press a key. Before pressing a key to exit the installer please check if errors occurred. Most problems with the game crashing are related to installation/download errors. Disabling your antivirus can help if you get errors.
6. Use play.bat to play the game.

### Manual installation / Linux / Mac

We can not provide a step by step guide for manual installation due to the vast differences between the existing software configuration and the systems of our users.

If you would like to manually install KoboldAI you will need some python/conda package management knowledge to manually do one of the following steps :

1. Use our bundled environments files to install your own conda environment, this should also automatically install CUDA.
2. If you do not want to use conda install the requirements listed in requirements.txt and make sure that CUDA is properly installed.
3. Adapt and use our bundled docker files to create your own KoboldAI docker instance.

### Using an AMD GPU on Linux

AMD GPU's have terrible compute support, this will currently not work on Windows and will only work for a select few Linux GPU's. [You can find a list of the compatible GPU's here](https://github.com/RadeonOpenCompute/ROCm#Hardware-and-Software-Support). Any GPU that is not listed is guaranteed not to work with KoboldAI and we will not be able to provide proper support on GPU's that are not compatible with the versions of ROCm we require. This guide requires that you already followed the appropriate steps to configure both [ROCm](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html) and [Docker]([Install Docker Engine | Docker Documentation](https://docs.docker.com/engine/install/)) and is for advanced users only.

1. Make sure you have installed both the latest version of [Docker](https://docs.docker.com/engine/install/), docker-compose and [ROCm](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html) on your system and have configured your user to have access to the Docker group (Sudo can interfere with the dialogues).
2. Assign our play-rocm.sh file execute permissions (chmod +x play-rocm.sh).
3. Run our play-rocm.sh file, it should now automatically install and create a suitable runtime for KoboldAI with AMD support and directly run the game afterwards. For X11 forwarding support you will need to run this as sudo at least once at the local machine. Otherwise use the command line options to load KoboldAI if you are playing this remotely.
4. Currently models automatically downloaded by the game are discarded on exit in the Docker version, it is strongly recommended that you manually download a model and load this using the custom model features to prevent unnecessary downloads.

If you hit strange errors with the ROCm version where it fails on the installation be sure you are running the latest version of Docker and Docker-compose. Some versions will fail on the root elevation or lack the appropriate formats.

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

#### "LayerNormKernelImpl" not implemented for 'Half'

This error only occurs when you are trying to run a model on the CPU mode while Finetuneanon's version of Transformers is installed. If you want/need to use the CPU mode use the install_requirements.bat file with the Official Transformers option and choose to delete all existing files.

#### vocab.json / config.json is not found error

If you get these errors you either did not select the correct folder for your custom model or the model you have downloaded is not (yet) compatible with KoboldAI. There exist a few models out there that are compatible and provide a pytorch_model.bin file but do not ship all the required files. In this case try downloading a compatible model of the same kind (For example another GPT-Neo if you downloaded a GPT-Neo model) and replace the pytorch_model.bin file with the one you are trying to run. Chances are this will work fine.

## KoboldAI Compatible Models

The models listed in the KoboldAI menu are generic models meant to easily get you going based on the Huggingface service. For higher quality models and fully offline use you will need to manually download a suitable model for your style. These are some of the models the community has available for you all tested to be compatible with KoboldAI and will be the brain of the AI.



| **Model**                                                    | Type                              | **(V)RAM** | Repetition Penalty | Description                                                  |
| ------------------------------------------------------------ | --------------------------------- | ---------- | ------------------ | ------------------------------------------------------------ |
| [gpt-j-6b-adventure-jax-hf](https://api.wandb.ai/files/ve-forbryderne/adventure/carol-data/models/gpt-j-6b-adventure-hf.7z) | Adventure / 6B / Neo Custom       | 16GB       | 1.2                | This model has been trained on the AI Dungeon set with additional stories thrown in. It is the most well rounded AI Dungeon like model and can be seen as an improved Griffin. If you wish to play KoboldAI like AI Dungeon this is the one to pick. It works great with the random story generator if your temp is 0.5 . |
| [gpt-j-6b-skein-jax-hf](https://api.wandb.ai/files/ve-forbryderne/skein/files/gpt-j-6b-skein-hf.7z) | Adventure Novel / 6B / Neo Custom | 16GB       | 1.1                | A hybrid of a few different datasets aimed to create a balanced story driven experience. If the adventure model is to focused on its own adventures and you want something a bit more generic this is the one for you. This model understands tags and adventure mode but can also be used as a writing assistant for your Novel. Its a good middle ground between a finetuned model and a generic model. It needs more guidance than some of the other models do making it less suitable for random story generation, but still focusses on writing rather than websites or code. If you want to use a model for existing story idea's this is a great choice. |
| [gpt-neo-2.7B-aid](https://storage.henk.tech/KoboldAI/gpt-neo-2.7B-aid.7z) | Adventure / 2.7B / Neo Custom     | 8GB        | 2.0                | This is one of the closest replications of the original AI Dungeon Classic model. Tuned on the same data that got uploaded alongside AI Dungeon. In KoboldAI we noticed this model performs better than the conversions of the original AI Dungeon model. It has all the traits you expect of AI Dungeon Classic while not having as many artifacts as this model was trained specifically for KoboldAI. Must be played with Adventure mode enabled to prevent it from doing actions on your behalf. |
| [gpt-neo-2.7B-horni](https://storage.henk.tech/KoboldAI/gpt-neo-2.7B-horni.7z) | Novel / 2.7B / Neo Custom         | 8GB        | 2.0                | One of the best novel models available for 2.7B focused on NSFW content. This model trains the AI to write in a story like fashion using a very large collection of Literotica stories. It is one of the original finetuned models for 2.7B. |
| [gpt-neo-2.7B-horni-ln](https://storage.henk.tech/KoboldAI/gpt-neo-2.7B-horni-ln.7z) | Novel / 2.7B / Neo Custom         | 8GB        | 2.0                | This model is much like the one above, but has been additionally trained on regular light novels. More likely to go SFW and is more focused towards themes found in these light novels over general cultural references. This is a good model for Novel writing especially if you want to add erotica to the mix. |
| [gpt-neo-2.7B-picard](https://storage.henk.tech/KoboldAI/gpt-neo-2.7B-picard.7z) | Novel / 2.7B / Neo Custom         | 8GB        | 2.0                | Picard is another Novel model, this time exclusively focused on SFW content of various genres. Unlike the name suggests this goes far beyond Star Trek stories and is not exclusively sci-fi. |
| [gpt-neo-2.7B-shinen](https://storage.henk.tech/KoboldAI/gpt-neo-2.7B-shinen.7z) | Novel / 2.7B / Neo Custom         | 8GB        | 2.0                | The most NSFW of them all, Shinen WILL make things sexual. This model will assume that whatever you are doing is meant to be a sex story and will sexualize constantly. It is designed for people who find Horni to tame. It was trained on SexStories instead of Literotica and was trained on tags making it easier to guide the AI to the right context. |
| [GPT-J-6B (Converted)](https://storage.henk.tech/KoboldAI/gpt-j-6b.7z) | Generic / 6B / Neo Custom         | 16GB       | 1.1                | This is the basis for all the other GPT-J-6B models, it has been trained on The Pile and is an open alternative for GPT Curie. Because it is a generic model it is not particularly good at anything and needs a long introduction to understand what you want to do. It is however the most flexible because it has no bias. If you want to do something that has no specific model available, such as writing a webpage article or coding this can be a good one to try. This specific version was converted by our community to be able to run as a GPT-Neo model on your GPU. |
| [AID-16Bit](https://storage.henk.tech/KoboldAI/aid-16bit.zip) | Adventure / 1.5B / GPT-2 Custom   | 4GB        | 2.0                | The original AI Dungeon Classic model converted to Pytorch and then converted to a 16-bit Model making it half the size. |
| [model_v5_pytorch](https://storage.henk.tech/KoboldAI/model_v5_pytorch.zip) (AI Dungeon's Original Model) | Adventure / 1.5B / GPT-2 Custom   | 8GB       | 2.0                | This is the original AI Dungeon Classic model converted to the Pytorch format compatible with AI Dungeon Clover and KoboldAI. We consider this model inferior to the GPT-Neo version because it has more artifacting due to its conversion. This is however the most authentic you can get to AI Dungeon Classic. |
| [Novel 774M](https://storage.henk.tech/KoboldAI/Novel%20model%20774M.rar) | Novel / 774M / GPT-2 Custom       | 4GB        | 2.0                | Novel 774M is made by the AI Dungeon Clover community, because of its small size and novel bias it is more suitable for CPU players that want to play with speed over substance or players who want to test a GPU with a low amount of VRAM. These performance savings are at the cost of story quality and you should not expect the kind of in depth story capabilities that the larger models offer. It was trained for SFW stories. |
| [Smut 774M](https://storage.henk.tech/KoboldAI/Smut%20model%20774M%2030K.rar) | Novel / 774M / GPT-2 Custom       | 4GB        | 2.0                | The NSFW version of the above, its a smaller GPT-2 based model made by the AI Dungeon Clover community. Gives decent speed on a CPU at the cost of story quality like the other 774M models. |
| [Mia](https://storage.henk.tech/KoboldAI/Mia.7z)             | Adventure / 125M / Neo Custom     | 1GB        | 2.0                | Mia is the smallest Adventure model, it runs at very fast speeds on the CPU which makes it a good testing model for developers who do not have GPU access. Because of its small size it will constantly attempt to do actions on behalf of the player and it will not produce high quality stories. If you just need a small model for a quick test, or if you want to take the challenge of trying to run KoboldAI entirely on your phone this would be an easy model to use due to its small RAM requirements and fast (loading) speeds. |



## Contributors

This project contains work from the following contributors :

- The Gantian - Creator of KoboldAI, has created most features such as the interface, the different AI model / API integrations and in general the largest part of the project.
- VE FORBRYDERNE - Contributed many features such as the Editing overhaul, Adventure Mode, expansions to the world info section, breakmodel integration and much more.
- Henk717 - Contributed the installation scripts, this readme, random story generator, the docker scripts, the foundation for the commandline interface and other smaller changes as well as integrating multiple parts of the code of different forks to unite it all. Not all code Github attributes to Henk717 is by Henk717 as some of it has been integrations of other people's work. We try to clarify this in the contributors list as much as we can.
- Frogging101 - top_k / tfs support
- UWUplus (Ralf) - Contributed storage systems for community colabs, as well as cleaning up and integrating the website dependencies/code better. He is also the maintainer of flask-cloudflared which we use to generate the cloudflare links.
- Javalar - Initial Performance increases on the story_refresh
- LexSong - Initial environment file adaptation for conda that served as a basis for the install_requirements.bat overhaul.
- Arrmansa - Breakmodel support for other projects that served as a basis for VE FORBRYDERNE's integration.

As well as various Model creators who will be listed near their models, and all the testers who helped make this possible!

Did we miss your contribution? Feel free to issue a commit adding your name to this list.

## License

KoboldAI is licensed with a AGPL license, in short this means that it can be used by anyone for any purpose. However, if you decide to make a publicly available instance your users are entitled to a copy of the source code including all modifications that you have made (which needs to be available trough an interface such as a button on your website), you may also not distribute this project in a form that does not contain the source code (Such as compiling / encrypting the code and distributing this version without also distributing the source code that includes the changes that you made. You are allowed to distribute this in a closed form if you also provide a separate archive with the source code.).

umamba.exe is bundled for convenience because we observed that many of our users had trouble with command line download methods, it is not part of our project and does not fall under the AGPL license. It is licensed under the BSD-3-Clause license.
