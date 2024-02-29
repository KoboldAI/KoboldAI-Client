## KoboldAI - Your gateway to GPT writing

This is a browser-based front-end for AI-assisted writing with multiple local & remote AI models. It offers the standard array of tools, including Memory, Author's Note, World Info, Save & Load, adjustable AI settings, formatting options, and the ability to import existing AI Dungeon adventures. You can also turn on Adventure mode and play the game like AI Dungeon Unleashed.

## Multiple ways to play

Stories can be played like a Novel, a text adventure game or used as a chatbot with an easy toggles to change between the multiple gameplay styles. This makes KoboldAI both a writing assistant, a game and a platform for so much more. The way you play and how good the AI will be depends on the model or service you decide to use. No matter if you want to use the free, fast power of Google Colab, your own high end graphics card, an online service you have an API key for (Like OpenAI or Inferkit) or if you rather just run it slower on your CPU you will be able to find a way to use KoboldAI that works for you.

### Adventure mode

By default KoboldAI will run in a generic mode optimized for writing, but with the right model you can play this like AI Dungeon without any issues. You can enable this in the settings and bring your own prompt, try generating a random prompt or download one of the prompts available at [/aids/ Prompts](https://aetherroom.club/).

The gameplay will be slightly different than the gameplay in AI Dungeon because we adopted the Type of the Unleashed fork, giving you full control over all the characters because we do not automatically adapt your sentences behind the scenes. This means you can more reliably control characters that are not you.

As a result of this what you need to type is slightly different, in AI Dungeon you would type _**take the sword**_ while in KoboldAI you would type it like a sentence such as _**You take the sword**_ and this is best done with the word You instead of I.

To speak simply type : _You say "We should probably gather some supplies first"_  
Just typing the quote might work, but the AI is at its best when you specify who does what in your commands.

If you want to do this with your friends we advise using the main character as You and using the other characters by their name if you are playing on a model trained for Adventures. These models assume there is a You in the story. This mode does usually not perform well on Novel models because they do not know how to handle the input those are best used with regular story writing where you take turns with the AI.

### Writing assistant

If you want to use KoboldAI as a writing assistant this is best done in the regular mode with a model optimized for Novels. These models do not make the assumption that there is a You character and focus on Novel like writing. For writing these will often give you better results than Adventure or Generic models. That said, if you give it a good introduction to the story large generic models like 13B can be used if a more specific model is not available for what you wish to write. You can also try to use models that are not specific to what you wish to do, for example a NSFW Novel model for a SFW story if a SFW model is unavailable. This will mean you will have to correct the model more often because of its bias, but can still produce good enough results if it is familiar enough with your topic.

### Chatbot Mode

In chatbot mode you can use a suitable model as a chatbot, this mode automatically adds your name to the beginning of the sentences and prevents the AI from talking as you. To use it properly you must write your story opening as both characters in the following format (You can use your own text) :

```plaintext
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

## [Models the TPU can run:](https://colab.research.google.com/github/KoboldAI/KoboldAI-Client/blob/main/colab/TPU.ipynb)

| Model | Style | Description |
| --- | --- | --- |
| [Nerys](https://huggingface.co/KoboldAI/fairseq-dense-13B-Nerys) by Mr Seeker | Novel/Adventure | Nerys is a hybrid model based on Pike (A newer Janeway), on top of the Pike dataset you also get some Light Novels, Adventure mode support and a little bit of Shinen thrown in the mix. The end result is a very diverse model that is heavily biased towards SFW novel writing, but one that can go beyond its novel training and make for an excellent adventure model to. Adventure mode is best played from a second person perspective, but can be played in first or third person as well. Novel writing can be done best from the first or third person. |
| [Erebus](https://huggingface.co/KoboldAI/OPT-13B-Erebus) by Mr Seeker | NSFW | Erebus is our community's flagship NSFW model, being a combination of multiple large datasets that include Literotica, Shinen and erotic novels from Nerys and featuring thourough tagging support it covers the vast majority of erotic writing styles. This model is capable of replacing both the Lit and Shinen models in terms of content and style and has been well received as (one of) the best NSFW models out there. If you wish to use this model for commercial or non research usage we recommend choosing the 20B version as that one is not subject to the restrictive OPT license. |
| [Janeway](https://huggingface.co/KoboldAI/fairseq-dense-13B-Janeway) by Mr Seeker | Novel | Janeway is a model created from Picard's dataset combined with a brand new collection of ebooks. This model is trained on 20% more content than Picard and has been trained on literature from various genres. Although the model is mainly focussed on SFW, romantic scenes might involve a degree of nudity. |
| [Shinen](https://huggingface.co/KoboldAI/fairseq-dense-13B-Shinen) by Mr Seeker | NSFW | Shinen is an NSFW model trained on a variety of stories from the website Sexstories it contains many different kinks. It has been merged into the larger (and better) Erebus model. |
| [Skein](https://huggingface.co/KoboldAI/GPT-J-6B-Skein) by VE\_FORBRYDERNE | Adventure | Skein is best used with Adventure mode enabled, it consists of a 4 times larger adventure dataset than the Adventure model making it excellent for text adventure gaming. On top of that it also consists of light novel training further expanding its knowledge and writing capabilities. It can be used with the You filter bias if you wish to write Novels with it, but dedicated Novel models can perform better for this task. |
| [Adventure](https://huggingface.co/KoboldAI/GPT-J-6B-Adventure) by VE\_FORBRYDERNE | Adventure | Adventure is a 6B model designed to mimick the behavior of AI Dungeon. It is exclusively for Adventure Mode and can take you on the epic and wackey adventures that AI Dungeon players love. It also features the many tropes of AI Dungeon as it has been trained on very similar data. It must be used in second person (You). |
| [Lit](https://huggingface.co/hakurei/lit-6B) ([V2](https://huggingface.co/hakurei/litv2-6B-rev3)) by Haru | NSFW | Lit is a great NSFW model trained by Haru on both a large set of Literotica stories and high quality novels along with tagging support. Creating a high quality model for your NSFW stories. This model is exclusively a novel model and is best used in third person. |
| [OPT](https://huggingface.co/facebook/opt-13b) by Metaseq | Generic | OPT is considered one of the best base models as far as content goes, its behavior has the strengths of both GPT-Neo and Fairseq Dense. Compared to Neo duplicate and unnecessary content has been left out, while additional literature was added in similar to the Fairseq Dense model. The Fairseq Dense model however lacks the broader data that OPT does have. The biggest downfall of OPT is its license, which prohibits any commercial usage, or usage beyond research purposes. |
| [Neo(X)](https://huggingface.co/EleutherAI/gpt-neox-20b) by EleutherAI | Generic | NeoX is the largest EleutherAI model currently available, being a generic model it is not particularly trained towards anything and can do a variety of writing, Q&A and coding tasks. 20B's performance is closely compared to the 13B models and it is worth trying both especially if you have a task that does not involve english writing. Its behavior will be similar to the GPT-J-6B model since they are trained on the same dataset but with more sensitivity towards repetition penalty and with more knowledge. |
| [Fairseq Dense](https://huggingface.co/KoboldAI/fairseq-dense-13B) | Generic | Trained by Facebook Researchers this model stems from the MOE research project within Fairseq. This particular version has been converted by us for use in KoboldAI. It is known to be on par with the larger 20B model from EleutherAI and considered as better for pop culture and language tasks. Because the model has never seen a new line (enter) it may perform worse on formatting and paragraphing. Compared to other models the dataset focuses primarily on literature and contains little else. |
| [GPT-J-6B](https://huggingface.co/EleutherAI/gpt-j-6B) by EleutherAI | Generic | This model serves as the basis for most other 6B models (Some being based on Fairseq Dense instead). Being trained on the Pile and not biased towards anything in particular it is suitable for a variety of tasks such as writing, Q&A and coding tasks. You will likely get better result with larger generic models or finetuned models. |

## [Models the Colab GPU can run:](https://colab.research.google.com/github/KoboldAI/KoboldAI-Client/blob/main/colab/GPU.ipynb)

| Model | Style | Description |
| --- | --- | --- |
| [Nerys](https://huggingface.co/KoboldAI/fairseq-dense-2.7B-Nerys) by Mr Seeker | Novel/Adventure | Nerys is a hybrid model based on Pike (A newer Janeway), on top of the Pike dataset you also get some Light Novels, Adventure mode support and a little bit of Shinen thrown in the mix. The end result is a very diverse model that is heavily biased towards SFW novel writing, but one that can go beyond its novel training and make for an excellent adventure model to. Adventure mode is best played from a second person perspective, but can be played in first or third person as well. Novel writing can be done best from the first or third person. |
| [Tiefighter 13B by KoboldAI](https://huggingface.co/KoboldAI/LLaMA2-13B-Tiefighter) | Hybrid | Tiefighter 13B is a very versitile fiction Hybrid, it can write, chat and play adventure games and can also answer regular instructions (Although we do not recommend this model for factual use due to its fictional nature). This is an excellent starting model, for the best results avoid using Second person writing in your chats unless you are wanting it to become a text adventure.|
| [Janeway](https://huggingface.co/KoboldAI/GPT-Neo-2.7B-Janeway) by Mr Seeker | Novel | Janeway is a model created from Picard's dataset combined with a brand new collection of ebooks. This model is trained on 20% more content than Picard and has been trained on literature from various genres. Although the model is mainly focussed on SFW, romantic scenes might involve a degree of nudity. |
| [Picard](https://huggingface.co/KoboldAI/GPT-Neo-2.7B-Picard) by Mr Seeker | Novel | Picard is a model trained for SFW Novels based on Neo 2.7B. It is focused on Novel style writing without the NSFW bias. While the name suggests a sci-fi model this model is designed for Novels of a variety of genre's. It is meant to be used in KoboldAI's regular mode. |
| [AID](https://huggingface.co/KoboldAI/GPT-Neo-2.7B-AID) by melastacho | Adventure | Also know as Adventure 2.7B this is a clone of the AI Dungeon Classic model and is best known for the epic wackey adventures that AI Dungeon Classic players love. |
| [OPT](https://huggingface.co/facebook/opt-2.7b) by Metaseq | Generic | OPT is considered one of the best base models as far as content goes, its behavior has the strengths of both GPT-Neo and Fairseq Dense. Compared to Neo duplicate and unnecessary content has been left out, while additional literature was added in similar to the Fairseq Dense model. The Fairseq Dense model however lacks the broader data that OPT does have. The biggest downfall of OPT is its license, which prohibits any commercial usage, or usage beyond research purposes. |
| [Fairseq Dense](https://huggingface.co/KoboldAI/fairseq-dense-2.7B) | Generic | Trained by Facebook Researchers this model stems from the MOE research project within Fairseq. This particular version has been converted by us for use in KoboldAI. It is known to be on par with the larger models from EleutherAI and considered as better for pop culture and language tasks. Because the model has never seen a new line (enter) it may perform worse on formatting and paragraphing. Compared to other models the dataset focuses primarily on literature and contains little else. |
| [MythoMax 13B](https://huggingface.co/TheBloke/MythoMax-L2-13B-GPTQ) by Gryphe | Roleplay | An improved, potentially even perfected variant of MythoMix, my MythoLogic-L2 and Huginn merge using a highly experimental tensor type merge technique¹. |
| [Holomax 13B by KoboldAI](https://huggingface.co/KoboldAI/LLaMA2-13B-Holomax) | Adventure | This is an expansion merge to the well-praised MythoMax model from Gryphe (60%) using MrSeeker's KoboldAI Holodeck model (40%). The goal of this model is to enhance story-writing capabilities while preserving the desirable traits of the MythoMax model as much as possible (It does limit chat reply length). |
| [Airoboros 13B](https://huggingface.co/jondurbin/airoboros-13b) by Jon Durbin | Generic | This is an instruction fine-tuned llama-2 model, using synthetic instructions generated by airoboros⁵. |
| [Emerhyst 13B](https://huggingface.co/Undi95/Emerhyst-13B) by Undi | Roleplay | An attempt using BlockMerge_Gradient to get better result. In addition, LimaRP v3 was used⁷. |
| [Chronos 13B](https://huggingface.co/elinas/chronos-13b) by Elinas | Generic | This model is primarily focused on chat, roleplay, and storywriting, but can accomplish other tasks such as simple reasoning and coding. Chronos generates very long outputs with coherent text, largely due to the human inputs it was trained on. |
| [Spring Dragon by Henk717](https://huggingface.co/Henk717/spring-dragon) | Adventure | This model is a recreation attempt of the AI Dungeon 2 Dragon model. To achieve this, the "text_adventures.txt" dataset was used, which was bundled with the original AI Dungeon 2 GitHub release prior to the online service. It is worth noting that the same dataset file was used to create the Dragon model, where Dragon is a GPT-3 175B Davinci model from 2020. |
| [Holodeck By KoboldAI](https://huggingface.co/KoboldAI/LLAMA2-13B-Holodeck-1) | Adventure |LLAMA2 13B-Holodeck is a finetune created using Meta's llama 2 model.The training data contains around 3000 ebooks in various genres. Most parts of the dataset have been prepended using the following text: [Genre: <genre1>, <genre2>|
| [Neo](https://huggingface.co/EleutherAI/gpt-neo-2.7B) by EleutherAI | Generic | This is the base model for all the other 2.7B models, it is best used when you have a use case that we have no other models available for, such as writing blog articles or programming. It can also be a good basis for the experience of some of the softprompts if your softprompt is not about a subject the other models cover. |
| [Various 2.7b models]() by various | Various smaller models are also possible to load in GPU colab. | |
### Styles

| Type | Description |
| --- | --- |
| Novel | For regular story writing, not compatible with Adventure mode or other specialty modes. |
| NSFW | Indicates that the model is strongly biased towards NSFW content and is not suitable for children, work environments or livestreaming. Most NSFW models are also Novel models in nature. |
| Adventure | These models are excellent for people willing to play KoboldAI like a Text Adventure game and are meant to be used with Adventure mode enabled. Even if you wish to use it as a Novel Type model you should always have Adventure mode on and set it to story. These models typically have a strong bias towards the use of the word You and without Adventure mode enabled break the story flow and write actions on your behalf. |
| Hybrid | Hybrid models are a blend between different Types, for example they are trained on both Novel stories and Adventure stories. These models are great variety models that you can use for multiple different playTypes and modes, but depending on your usage you may need to enable Adventure Mode or the You bias (in userscripts). |
| Generic | Generic models are not trained towards anything specific, typically used as a basis for other tasks and models. They can do everything the other models can do, but require much more handholding to work properly. Generic models are an ideal basis for tasks that we have no specific model for, or for experiencing a softprompt in its raw form. |

## Tips to get the most out of Google Colab

*   Google will occationally show a Captcha, typically after it has been open for 30 minutes but it can be more frequent if you often use Colab. Make sure to do these properly, or you risk getting your instance shut down and getting a lower priority towards the TPU's.
*   KoboldAI uses Google Drive to store your files and settings, if you wish to upload a softprompt or userscript this can be done directly on the Google Drive website. You can also use this to download backups of your KoboldAI related files or upload models of your own.
*   Don't want to save your stories on Google Drive for privacy reasons? Do not use KoboldAI's save function and instead click Download as .json, this will automatically download the story to your own computer without ever touching Google's harddrives. You can load this back trough the Load from file option.
*   Google shut your instance down unexpectedly? You can still make use of the Download as .json button to recover your story as long as you did not close the KoboldAI window. You can then load this back up in your next session.
*   Done with KoboldAI? Go to the Runtime menu, click on Manage Sessions and terminate your open sessions that you no longer need. This trick can help you maintain higher priority towards getting a TPU.
*   Models stored on Google Drive typically load faster than models we need to download from the internet.

## Install KoboldAI on your own computer

KoboldAI has a large number of dependencies you will need to install on your computer, unfortunately Python does not make it easy for us to provide instructions that work for everyone. The instructions below will work on most computers, but if you have multiple versions of Python installed conflicts can occur.

### Downloading the latest version of KoboldAI

KoboldAI is a rolling release on our github, the code you see is also the game. You can download the software by clicking on the green Code button at the top of the page and clicking Download ZIP, or use the `git clone` command instead. Then, on Windows you need to you run install_requirements.bat (using admin mode is recommanded to avoid errors), and once it's done, or if you're on Linux, either play.bat/sh or remote-play.bat/sh to run it.

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
    Intel ARC user? Make sure OneAPI is installed if you want GPU support.
3.  Run play.sh if you use an Nvidia GPU or you want to use CPU only    
    Run play-rocm.sh if you use an AMD GPU supported by ROCm    
    Run play-ipex.sh if you use an Intel ARC GPU

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

### Intel ARC GPU's (Linux or WSL)

Make sure to first install OneAPI on your Linux or WSL system using a guide for your distribution, after that you can follow the usual linux instructions above.

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

## Softprompts

Softprompts (also known as Modules in other products) are addons that can change the output of existing models. For example you may load a softprompt that biases the AI towards a certain subject and style like transcripts from your favorite TV show.

Since these softprompts are often based on existing franchises we currently do not bundle any of them with KoboldAI due to copyright concerns (We do not want to put the entire project at risk). Instead look at community resources like #softprompts on the [KoboldAI Discord](https://discord.gg/XuQWadgU9k) or the [community hosted mirror](https://storage.henk.tech/KoboldAI/softprompts/).

That way we are better protected from any DMCA claims as things can be taken down easier than directly on Github. If you have a copyright free softprompt that you made from scratch and is not based on existing IP that you would like to see officially bundled with KoboldAI issue a pull request with your softprompt.

Training softprompts can be done for free with the [Easy Softprompt Tuner](https://colab.research.google.com/gist/henk717/281fd57ebd2e88d852ef9dcc3f29bebf/easy-softprompt-tuner.ipynb#sandboxMode=true), in that case you can leave most of the settings default. Your source data needs to be a folder with text files that are UTF-8 formatted and contain Unix line endings.

## Userscripts

Userscripts are scripts that can automate tasks in KoboldAI, or modify the AI behavior / input / output.  
Scripting is done in LUA5.4 (Lua does not need to be separately installed as long as you got all the python requirements) and has sandboxing to help protect you from malicious behavior. Even with these measures in place we strongly advise you only run userscripts from places you trust and/or understand, otherwise consult the community for advice on how safe the script might be.

Inside the userscripts folder you will find our kaipreset scripts, these are default scripts that we think will be useful for our users. These scripts are automatically overwritten when you update KoboldAI, if you wish to modify these scripts make sure to first rename them to something else that does not contain kaipreset so your changes are not lost. These scripts range from a You Bias filter that prevents the AI from addressing characters as you. Ways to be able to prevent the AI from using words, word replacements and more.

Along with our preset scripts we also ship examples in the examples folder that merely serve as a demonstration and do not enhance your usage of KoboldAI. To use these scripts make sure to move them out of the examples folder before either using or modifying the script.

Lastly the all the features of our userscript API are documented inside the API Documentation files inside the userscripts folder.

For our TPU versions keep in mind that scripts modifying AI behavior relies on a different way of processing that is slower than if you leave these userscripts disabled even if your script only sporadically uses this modifier. If you want to partially use a script at its full speed than you can enable "No Gen Modifiers" to ensure that the parts that would make the TPU slow are not active.

## API

KoboldAI has a REST API that can be accessed by adding /api to the URL that Kobold provides you (For example http://127.0.0.1:5000/api).  
When accessing this link in a browser you will be taken to the interactive documentation.

## Contributors

This project contains work from the following contributors :

*   The Gantian - Creator of KoboldAI, has created most features such as the interface, the different AI model / API integrations and in general the largest part of the project.
*   VE FORBRYDERNE - Contributed many features such as the Editing overhaul, Adventure Mode, expansions to the world info section, breakmodel integration, scripting support, API, softpromtps and much more. As well as vastly improving the TPU compatibility and integrating external code into KoboldAI so we could use official versions of Transformers with virtually no downsides.
*   Henk717 - Contributed the installation scripts, this readme, random story generator, the docker scripts, the foundation for the commandline interface and other smaller changes as well as integrating multiple parts of the code of different forks to unite it all. He also optimized the model loading so that downloaded models get converted to efficient offline models and that in future models are more likely to work out of the box. Not all code Github attributes to Henk717 is by Henk717 as some of it has been integrations of other people's work. We try to clarify this in the contributors list as much as we can.
*   Ebolam - Automatic Saving, back/redo, pinning, web loading of models
*   one-some, Logits Viewer and Token Streaming
*   db0, KoboldAI Horde
*   Frogging101 - top\_k / tfs support (Part of this support was later redone by VE to integrate what was originally inside of finetuneanon's transformers)
*   UWUplus (Ralf) - Contributed storage systems for community colabs, as well as cleaning up and integrating the website dependencies/code better. He is also the maintainer of flask-cloudflared which we use to generate the cloudflare links.
*   Javalar - Initial Performance increases on the story\_refresh
*   LexSong - Initial environment file adaptation for conda that served as a basis for the install\_requirements.bat overhaul.
*   Arrmansa - Breakmodel support for other projects that served as a basis for VE FORBRYDERNE's integration.
*   Jojorne - Small improvements to the response selection for gens per action.
*   OccultSage (GooseAI) - Improved support for GooseAI/OpenAI

As well as various Model creators who will be listed near their models, and all the testers who helped make this possible!

Did we miss your contribution? Feel free to issue a commit adding your name to this list.

## License

KoboldAI is licensed with a AGPL license, in short this means that it can be used by anyone for any purpose. However, if you decide to make a publicly available instance your users are entitled to a copy of the source code including all modifications that you have made (which needs to be available trough an interface such as a button on your website), you may also not distribute this project in a form that does not contain the source code (Such as compiling / encrypting the code and distributing this version without also distributing the source code that includes the changes that you made. You are allowed to distribute this in a closed form if you also provide a separate archive with the source code.).

umamba.exe is bundled for convenience because we observed that many of our users had trouble with command line download methods, it is not part of our project and does not fall under the AGPL license. It is licensed under the BSD-3-Clause license. Other files with differing licenses will have a reference or embedded version of this license within the file. It has been sourced from https://anaconda.org/conda-forge/micromamba/files and its source code can be found here : https://github.com/mamba-org/mamba/tree/master/micromamba
