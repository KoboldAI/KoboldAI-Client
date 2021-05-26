
# Introduction to Kobold AI

Thanks for checking out the KoboldAI Client!

This is a browser-based front-end for AI-assisted writing with multiple local & remote AI models. 
It offers the standard array of tools, including Memory, Author's Note, World Info, Save & Load, 
adjustable AI settings, formatting options, and the ability to import existing AI Dungeon adventures.

Current UI Snapshot:
![ui snapshot](https://i.imgur.com/mjk5Yre.jpeg) 

For local generation, KoboldAI uses Transformers to interact 
with the AI models. This can be done either on CPU, or GPU with sufficient hardware. If you have a 
high-end GPU with sufficient VRAM to run your model of choice, see one of the guides below for instructions on enabling GPU support.

Transformers/Tensorflow can still be used on CPU if you do not have high-end hardware, but generation
times will be much longer. Alternatively, KoboldAI also supports utilizing remotely-hosted models. 
The currently supported remote APIs are InferKit, Google Colab, and OpenAI. see the dedicated sections below for more info on these.

* **[Read more about Transformers, if you're interested](https://huggingface.co/transformers/)**

## Community

 * **[Get support and updates on the subreddit](https://www.reddit.com/r/KoboldAI/)**

## How to Setup

* **[Enabling GPU support from tensorflow.](https://www.tensorflow.org/install/gpu)**

* **[Our guide on how to set up KoboldAI.][setup_guide.md]**

* **[This is a very crude but well made guide about setting up KoboldAI with GPU support included.](https://rentry.org/itsnotthathard)**
