
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

## [TPU Edition Model Descriptions](https://colab.research.google.com/github/KoboldAI/KoboldAI-Client/blob/main/colab/TPU.ipynb)

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

## [GPU Edition Model Descriptions](https://colab.research.google.com/github/KoboldAI/KoboldAI-Client/blob/main/colab/GPU.ipynb)

| Model | Style | Description |
| --- | --- | --- |
| [Nerys](https://huggingface.co/KoboldAI/fairseq-dense-2.7B-Nerys) by Mr Seeker | Novel/Adventure | Nerys is a hybrid model based on Pike (A newer Janeway), on top of the Pike dataset you also get some Light Novels, Adventure mode support and a little bit of Shinen thrown in the mix. The end result is a very diverse model that is heavily biased towards SFW novel writing, but one that can go beyond its novel training and make for an excellent adventure model to. Adventure mode is best played from a second person perspective, but can be played in first or third person as well. Novel writing can be done best from the first or third person. |
| [Erebus](https://huggingface.co/KoboldAI/OPT-2.7B-Erebus) by Mr Seeker | NSFW | Erebus is our community's flagship NSFW model, being a combination of multiple large datasets that include Literotica, Shinen and erotic novels from Nerys and featuring thourough tagging support it covers the vast majority of erotic writing styles. This model is capable of replacing both the Lit and Shinen models in terms of content and style and has been well received as (one of) the best NSFW models out there. If you wish to use this model for commercial or non research usage we recommend choosing the 20B version as that one is not subject to the restrictive OPT license. |
| [Janeway](https://huggingface.co/KoboldAI/GPT-Neo-2.7B-Janeway) by Mr Seeker | Novel | Janeway is a model created from Picard's dataset combined with a brand new collection of ebooks. This model is trained on 20% more content than Picard and has been trained on literature from various genres. Although the model is mainly focussed on SFW, romantic scenes might involve a degree of nudity. |
| [Picard](https://huggingface.co/KoboldAI/GPT-Neo-2.7B-Picard) by Mr Seeker | Novel | Picard is a model trained for SFW Novels based on Neo 2.7B. It is focused on Novel style writing without the NSFW bias. While the name suggests a sci-fi model this model is designed for Novels of a variety of genre's. It is meant to be used in KoboldAI's regular mode. |
| [AID](https://huggingface.co/KoboldAI/GPT-Neo-2.7B-AID) by melastacho | Adventure | Also know as Adventure 2.7B this is a clone of the AI Dungeon Classic model and is best known for the epic wackey adventures that AI Dungeon Classic players love. |
| [Horni LN](https://huggingface.co/KoboldAI/GPT-Neo-2.7B-Horni-LN) by finetune | Novel | This model is based on Horni 2.7B and retains its NSFW knowledge, but was then further biased towards SFW novel stories. If you seek a balance between a SFW Novel model and a NSFW model this model should be a good choice. |
| [Horni](https://huggingface.co/KoboldAI/GPT-Neo-2.7B-Horni) by finetune | NSFW | This model is tuned on Literotica to produce a Novel style model biased towards NSFW content. Can still be used for SFW stories but will have a bias towards NSFW content. It is meant to be used in KoboldAI's regular mode. |
| [Shinen](https://huggingface.co/KoboldAI/GPT-Neo-2.7B-Shinen) by Mr Seeker | NSFW | Shinen is an alternative to the Horni model designed to be more explicit. If Horni is to tame for you Shinen might produce better results. While it is a Novel model it is unsuitable for SFW stories due to its heavy NSFW bias. Shinen will not hold back. It is meant to be used in KoboldAI's regular mode. |
| [OPT](https://huggingface.co/facebook/opt-2.7b) by Metaseq | Generic | OPT is considered one of the best base models as far as content goes, its behavior has the strengths of both GPT-Neo and Fairseq Dense. Compared to Neo duplicate and unnecessary content has been left out, while additional literature was added in similar to the Fairseq Dense model. The Fairseq Dense model however lacks the broader data that OPT does have. The biggest downfall of OPT is its license, which prohibits any commercial usage, or usage beyond research purposes. |
| [Fairseq Dense](https://huggingface.co/KoboldAI/fairseq-dense-2.7B) | Generic | Trained by Facebook Researchers this model stems from the MOE research project within Fairseq. This particular version has been converted by us for use in KoboldAI. It is known to be on par with the larger models from EleutherAI and considered as better for pop culture and language tasks. Because the model has never seen a new line (enter) it may perform worse on formatting and paragraphing. Compared to other models the dataset focuses primarily on literature and contains little else. |
| [Neo](https://huggingface.co/EleutherAI/gpt-neo-2.7B) by EleutherAI | Generic | This is the base model for all the other 2.7B models, it is best used when you have a use case that we have no other models available for, such as writing blog articles or programming. It can also be a good basis for the experience of some of the softprompts if your softprompt is not about a subject the other models cover. |

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
