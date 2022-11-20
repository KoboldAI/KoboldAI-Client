# kobold

```lua
global kobold: KoboldLib
```

### Methods:
* `kobold.decode()`
* `kobold.encode()`
* `kobold.get_config_file()`
* `kobold.halt_generation()`
* `kobold.restart_generation()`

### Fields:
* `kobold.authorsnote`
* `kobold.authorsnotetemplate`
* `kobold.custmodpth`
* `kobold.feedback`
* `kobold.generated`
* `kobold.generated_cols`
* `kobold.generated_rows`
* `kobold.is_config_file_open`
* `kobold.logits`
* `kobold.logits_cols`
* `kobold.logits_rows`
* `kobold.memory`
* `kobold.modelbackend`
* `kobold.modeltype`
* `kobold.num_outputs`
* `kobold.outputs`
* `kobold.settings`
* `kobold.spfilename`
* `kobold.story`
* `kobold.submission`
* `kobold.worldinfo`

# kobold.decode()

```lua
function KoboldLib.decode(tok: integer|table<integer, integer>) -> string
```

***Callable from:*** anywhere

Decodes the given token or list of tokens using the current tokenizer. If `kobold.backend` is `'readonly'` or `'api'`, the tokenizer used is the GPT-2 tokenizer, otherwise the model's tokenizer is used. This function is the inverse of `kobold.encode()`.

```lua
print(kobold.decode({15496, 2159}))  -- 'Hello World'
```

### Parameters:
* tok (`integer|table<integer, integer>`): Array of token IDs to decode, or the token ID of a single token.

### Returns:
* `string`: Decoded tokens.

# kobold.encode()

```lua
function KoboldLib.encode(str: string) -> table<integer, integer>
```

***Callable from:*** anywhere

Encodes the given string using the current tokenizer into an array of tokens. If `kobold.backend` is `'readonly'` or `'api'`, the tokenizer used is the GPT-2 tokenizer, otherwise the model's tokenizer is used. This function is the inverse of `kobold.decode()`.

```lua
local tokens = kobold.encode("Hello World")
print(#tokens)  -- 2
print(tokens[1])  -- 15496
print(tokens[2])  -- 2159
```

### Parameters:
* tok (`integer|table<integer, integer>`): Array of token IDs to decode, or the token ID of a single token.

### Returns:
* `string`: Decoded tokens.

# kobold.get_config_file()

```lua
function KoboldLib.get_config_file(clear?: boolean) -> file*
```

***Callable from:*** anywhere

Returns a file handle representing your script's configuration file, which is usually the file in the userscripts folder with the same filename as your script but with ".conf" appended at the end. This function throws an error on failure.

If the configuration file does not exist when this function is called, the configuration file will first be created as a new empty file.

If KoboldAI does not possess an open file handle to the configuration file, this function opens the file in `w+b` mode if the `clear` parameter is a truthy value, otherwise the file is opened in `r+b` mode. These are mostly the same -- the file is opened in binary read-write mode and then seeked to the start of the file -- except the former mode deletes the contents of the file prior to opening it and the latter mode does not.

If KoboldAI does possess an open file handle to the configuration file, that open file handle is returned without seeking or deleting the contents of the file. You can check if KoboldAI possesses an open file handle to the configuration file by using `kobold.is_config_file_open`.

```lua
local example_config = "return 'Hello World'"
local cfg
do
    -- If config file is empty, write example config
    local f <close> = kobold.get_config_file()
    f:seek("set")
    if f:read(1) == nil then f:write(example_config) end
    f:seek("set")

    -- Read config
    local err
    cfg, err = load(f:read("a"))
    if err ~= nil then error(err) end
    cfg = cfg()
end
```

### Parameters:
* clear? (`bool`): If KoboldAI does not possess an open file handle to the configuration object, this determines whether the file will be opened in `w+b` or `r+b` mode. This parameter defaults to `false`.

### Returns:
* `file*`: File handle for the configuration file.

# kobold.halt_generation()

```lua
function KoboldLib.halt_generation() -> nil
```

***Callable from:*** anywhere

If called from an input modifier, prevents the user's input from being sent to the model and skips directly to the output modifier.

If called from a generation modifier, stops generation after the current token is generated and then skips to the output modifier. In other words, if, when you call this function, `kobold.generated` has n columns, it will have exactly n+1 columns when the output modifier is called.

If called from an output modifier, has no effect.

# kobold.restart_generation()

```lua
function KoboldLib.restart_generation(sequence?: integer) -> nil
```

***Callable from:*** output modifier

After the current output is sent to the GUI, starts another generation using the empty string as the submission.

Whatever ends up being the output selected by the user or by the `sequence` parameter will be saved in `kobold.feedback` when the new generation begins.

### Parameters:
* sequence? (`integer`): If you have multiple Gens Per Action, this can be used to choose which sequence to use as the output, where 1 is the first, 2 is the second and so on. If you set this to 0, the user will be prompted to choose the sequence instead. Defaults to 0.

# kobold.authorsnote

```lua
field KoboldLib.authorsnote: string
```

***Readable from:*** anywhere
***Writable from:*** anywhere (triggers regeneration when written to from generation modifier)

The author's note as set from the "Memory" button in the GUI.

Modifying this field from inside of a generation modifier triggers a regeneration, which means that the context is recomputed after modification and generation begins again with the new context and previously generated tokens. This incurs a small performance penalty and should not be performed in excess.

# kobold.authorsnotetemplate

```lua
field KoboldLib.authorsnotetemplate: string
```

***Readable from:*** anywhere
***Writable from:*** anywhere (triggers regeneration when written to from generation modifier)

The author's note template as set from the "Memory" button in the GUI.

Modifying this field from inside of a generation modifier triggers a regeneration, which means that the context is recomputed after modification and generation begins again with the new context and previously generated tokens. This incurs a small performance penalty and should not be performed in excess.

# kobold.custmodpth

```lua
field KoboldLib.custmodpth: string
```

***Readable from:*** anywhere
***Writable from:*** nowhere

Path to a directory that the user chose either via the file dialog that appears when KoboldAI asks you to choose the path to your custom model or via the `--path` command-line flag.

If the user loaded a built-in model from the menu, this is instead the model ID of the model on Hugging Face's model hub, such as "KoboldAI/GPT-Neo-2.7B-Picard" or "hakurei/lit-6B".

# kobold.feedback

```lua
field KoboldLib.feedback: string?
```

***Readable from:*** anywhere
***Writable from:*** nowhere

If this is a repeat generation caused by `kobold.restart_generation()`, this will be a string containing the previous output. If not, this will be `nil`.

# kobold.generated

```lua
field KoboldLib.generated: table<integer, table<integer, integer>>?
```

***Readable from:*** generation modifier and output modifier, but only if `kobold.modelbackend` is not `'api'` or `'readonly'`
***Writable from:*** generation modifier

Two-dimensional array of tokens generated thus far. Each row represents one sequence, each column one token.

# kobold.generated_cols

```lua
field KoboldLib.generated_cols: integer
```

***Readable from:*** anywhere
***Writable from:*** nowhere

Number of columns in `kobold.generated`. In other words, the number of tokens generated thus far, which is equal to the number of times that the generation modifier has been called, not including the current time if this is being read from a generation modifier.

If `kobold.modelbackend` is `'api'` or `'readonly'`, this returns 0 instead.

# kobold.generated_rows

```lua
field KoboldLib.generated_rows: integer
```

***Readable from:*** anywhere
***Writable from:*** nowhere

Number of rows in `kobold.generated`, equal to `kobold.settings.numseqs`.

# kobold.is_config_file_open

```lua
field KoboldLib.is_config_file_open: boolean
```

***Readable from:*** anywhere
***Writable from:*** nowhere

Whether or not KoboldAI possesses an open file handle to your script's configuration file. See `kobold.get_config_file()` for more details.


# kobold.logits

```lua
field KoboldLib.logits: table<integer, table<integer, number>>?
```

***Readable from:*** generation modifier, but only if `kobold.modelbackend` is not `'api'` or `'readonly'`
***Writable from:*** generation modifier

Two-dimensional array of [logits](https://datascience.stackexchange.com/questions/31041/what-does-logits-in-machine-learning-mean) prior to being filtered by top-p sampling, etc. Each row represents one sequence, each column one of the tokens in the model's vocabulary. The ith column represents the logit score of token i-1, so if you want to access the logit score of token 18435 (" Hello" with a leading space), you need to access column 18436. You may alter this two-dimensional array to encourage or deter certain tokens from appearing in the output in a stochastic manner.

Don't modify this table unnecessarily unless you know what you are doing! The bias example scripts show how to use this feature properly.

# kobold.logits_cols

```lua
field KoboldLib.logits_cols: integer
```

***Readable from:*** anywhere
***Writable from:*** nowhere

Number of columns in `kobold.logits`, equal to the vocabulary size of the current model. Most models based on GPT-2 (e.g. GPT-Neo and GPT-J) have a vocabulary size of 50257. GPT-J models in particular have a vocabulary size of 50400 instead, although GPT-J models aren't trained to use the rightmost 143 tokens of the logits array.

If `kobold.modelbackend` is `'api'` or `'readonly'`, this returns 0 instead.

# kobold.logits_rows

```lua
field KoboldLib.logits_rows: integer
```

***Readable from:*** anywhere
***Writable from:*** nowhere

Number of rows in `kobold.generated`, equal to `kobold.settings.numseqs`. a local KoboldAI install.

# kobold.memory

```lua
field KoboldLib.memory: string
```

***Readable from:*** anywhere
***Writable from:*** anywhere (triggers regeneration when written to from generation modifier)

The memory as set from the "Memory" button in the GUI.

Modifying this field from inside of a generation modifier triggers a regeneration, which means that the context is recomputed after modification and generation begins again with the new context and previously generated tokens. This incurs a small performance penalty and should not be performed in excess.

# kobold.modelbackend

```lua
field KoboldLib.modelbackend: string
```

***Readable from:*** anywhere
***Writable from:*** nowhere

One of the following values:

* `'api'`: InferKit, OpenAI or legacy Colab mode
* `'readonly'`: Read-only no AI mode
* `'transformers'`: Models running on your own computer, and Colab GPU backend (currently used for 2.7B models on Colab)
* `'mtj'`: Colab TPU backend (currently used for 6B models on Colab)

# kobold.modeltype

```lua
field KoboldLib.modeltype: string
```

***Readable from:*** anywhere
***Writable from:*** nowhere

One of the following values:

* `'api'`: InferKit, OpenAI or legacy Colab mode
* `'readonly'`: Read-only no AI mode
* `'unknown'`
* `'gpt2'`: GPT-2-Small
* `'gpt2-medium'`
* `'gpt2-large'`
* `'gpt2-xl'`
* `'gpt-neo-125M'`
* `'gpt-neo-1.3B'`
* `'gpt-neo-2.7B'`
* `'gpt-j-6B'`

# kobold.num_outputs

```lua
field KoboldLib.num_outputs: integer
```

***Readable from:*** anywhere
***Writable from:*** nowhere

Number of rows in `kobold.outputs`. This is equal to `kobold.settings.numseqs` unless you're using a non-Colab third-party API such as OpenAI or InferKit, in which case this is 1. If you decide to write to `kobold.settings.numseqs` from an output modifier, this value remains unchanged.

# kobold.outputs

```lua
field KoboldLib.outputs: table<integer, string>
```

***Readable from:*** output modifier
***Writable from:*** output modifier

Model output before applying output formatting. One row per "Gens Per Action", unless you're using OpenAI or InferKit, in which case this always has exactly one row.

# kobold.settings

```lua
field KoboldLib.settings: KoboldSettings
```

***Readable from:*** anywhere
***Writable from:*** anywhere (does not affect other scripts when written to since each script has its own copy of this object)

Contains most of the settings. They have the same names as in gensettings&#46;py, so the top-p value is `kobold.settings.settopp`.

All the settings can be read from anywhere and written from anywhere, except `kobold.settings.numseqs` which can only be written to from an input modifier or output modifier.

Modifying certain fields from inside of a generation modifier triggers a regeneration, which means that the context is recomputed after modification and generation begins again with the new context and previously generated tokens. This incurs a small performance penalty and should not be performed in excess. Currently, only the following fields and their aliases cause this to occur:
* `kobold.settings.settknmax` (Max Tokens)
* `kobold.settings.anotedepth` (Author's Note Depth)
* `kobold.settings.setwidepth` (World Info Depth)
* `kobold.settings.setuseprompt` (Always Use Prompt)

# kobold.spfilename

***Readable from:*** anywhere
***Writable from:*** anywhere

```lua
field kobold.spfilename: string?
```

The name of the soft prompt file to use (as a string), including the file extension. If not using a soft prompt, this is `nil` instead.

You can also set the soft prompt to use by setting this to a string or `nil`.

Modifying this field from inside of a generation modifier triggers a regeneration, which means that the context is recomputed after modification and generation begins again with the new context and previously generated tokens. This incurs a small performance penalty and should not be performed in excess.

# kobold.story

***Readable from:*** anywhere
***Writable from:*** nowhere

```lua
field KoboldLib.story: KoboldStory
```

Contains the chunks of the current story. Don't use `pairs` or `ipairs` to iterate over the story chunks, use `kobold.story:forward_iter()` or `kobold.story:reverse_iter()`, which guarantee amortized worst-case iteration time complexity linear to the number of chunks in the story regardless of what the highest chunk number is.

You can index this object to get a story chunk (as a `KoboldStoryChunk` object) by its number, which is an integer. The prompt chunk, if it exists, is guaranteed to be chunk 0. Aside from that, the chunk numbers are not guaranteed to be contiguous or ordered in any way.

```lua
local prompt_chunk = kobold.story[0]  -- KoboldStoryChunk object referring to the prompt chunk
```

### Methods:

* `kobold.story:forward_iter()`
* `kobold.story:reverse_iter()`

# kobold.story:forward_iter()

***Callable from***: anywhere

```lua
function KoboldStory:forward_iter() -> fun(): KoboldStoryChunk, table, nil
```

Returns a stateful iterator that efficiently iterates through story chunks from top to bottom.

```lua
for chunk in kobold.story:forward_iter() do
    print(chunk.num, chunk.content)
end
```

# kobold.story:reverse_iter()

***Callable from***: anywhere

```lua
function KoboldStory:reverse_iter() -> fun(): KoboldStoryChunk, table, nil
```

Returns a stateful iterator that efficiently iterates through story chunks from bottom to top.

```lua
for chunk in kobold.story:reverse_iter() do
    print(chunk.num, chunk.content)
end
```

# KoboldStoryChunk

Represents a story chunk.

### Fields:

* `KoboldStoryChunk.content`
* `KoboldStoryChunk.num`

# KoboldStoryChunk.content

***Readable from:*** anywhere
***Writable from:*** anywhere (triggers regeneration when written to from generation modifier)

```lua
field KoboldStoryChunk.content: string
```

The text inside of the story chunk.

Modifying this field from inside of a generation modifier triggers a regeneration, which means that the context is recomputed after modification and generation begins again with the new context and previously generated tokens. This incurs a small performance penalty and should not be performed in excess.

# KoboldStoryChunk.num

***Readable from:*** anywhere
***Writable from:*** nowhere

```lua
field KoboldStoryChunk.num: integer
```

The number of the story chunk. Chunk 0 is guaranteed to be the prompt chunk if it exists; no guarantees can be made about the numbers of other chunks.

# kobold.submission

***Readable from:*** anywhere
***Writable from:*** input modifier

```lua
field kobold.submission: string
```

The user-submitted text after being formatted by input formatting. If this is a repeated generation incurred by `kobold.restart_generation()`, then this is the empty string.

# kobold.worldinfo

```lua
field KoboldLib.worldinfo: KoboldWorldInfo
```

Represents the world info entries.

Indexing this object at index i returns the ith world info entry from the top in amortized constant worst-case time as a `KoboldWorldInfoEntry`. This includes world info entries that are inside folders.

```lua
local entry = kobold.worldinfo[5]  -- Retrieves fifth entry from top as a KoboldWorldInfoEntry
```

You can use `ipairs` or a numeric loop to iterate from top to bottom:

```lua
for index, entry in ipairs(kobold.worldinfo) do
    print(index, entry.content)
end
```

```lua
for index = 1, #kobold.worldinfo do
    print(index, kobold.worldinfo[index].content)
end
```

### Methods:
* `kobold.story:compute_context()`
* `kobold.story:finduid()`
* `kobold.story:is_valid()`

### Fields:
* `kobold.story.folders`

# kobold.worldinfo:compute_context()

***Callable from:*** anywhere

```lua
function KoboldWorldInfo:compute_context(submission: string, entries?: KoboldWorldInfoEntry|table<any, KoboldWorldInfoEntry>, kwargs?: table<string, any>) -> string
```

Computes the context that would be sent to the generator with the user's current settings if `submission` were the user's input after being formatted by input formatting. The context would include memory at the top, followed by active world info entries, followed by some story chunks with the author's note somewhere, followed by `submission`.

### Parameters

* submission (`string`): String to use as simulated user's input after being formatted by input formatting.
* entries? (`KoboldWorldInfoEntry|table<any, KoboldWorldInfoEntry>`): A `KoboldWorldInfoEntry` or table thereof that indicates an allowed subset of world info entries to include in the context. Defaults to all world info entries.
* kwargs? (`table<string, any>`): Table of optional keyword arguments from the following list. Defaults to `{}`.
    * scan_story? (`boolean`): Whether or not to scan the past few actions of the story for world info keys in addition to the submission like how world info normally behaves. If this is set to `false`, only the `submission` is scanned for world info keys. Defaults to `true`.
    * include_anote? (`boolean`): Whether to include the author's note in the story.  Defaults to `true`, pass `false` to suppress including the author's note.

### Returns

`string`: Computed context.

# kobold.worldinfo:finduid()

***Callable from:*** anywhere

```lua
function KoboldWorldInfo:finduid(u: integer) -> KoboldWorldInfoEntry?
```

Returns the world info entry with the given UID in amortized constant worst-case time, or `nil` if not found.

### Parameters

* u (`integer`): UID.

### Returns

* `KoboldWorldInfoEntry?`: The world info entry with requested UID, or `nil` if no such entry exists.

# kobold.worldinfo:is_valid()

***Callable from:*** anywhere

```lua
function KoboldWorldInfo:is_valid() -> boolean
```

This always returns true.

# kobold.worldinfo.folders

***Readable from:*** anywhere
***Writable from:*** nowhere

```lua
field KoboldWorldInfo.folders: KoboldWorldInfoFolderSelector
```

Can be indexed in amortized constant worst-case time and iterated over and has a `finduid` method just like `kobold.worldinfo`, but gets folders (as `KoboldWorldInfoFolder` objects) instead.

```lua
local folder = kobold.worldinfo.folders[5]  -- Retrieves fifth folder from top as a KoboldWorldInfoFolder
```

```lua
for index, folder in ipairs(kobold.worldinfo.folders) do
    print(index, folder.name)
end
```

```lua
for index = 1, #kobold.worldinfo.folders do
    print(index, kobold.worldinfo.folders[index].name)
end
```

### Methods
* `kobold.story.folders:finduid()`
* `kobold.story.folders:is_valid()`

# KoboldWorldInfoEntry

Represents a world info entry.

### Methods:
* `KoboldWorldInfoEntry:compute_context()`
* `KoboldWorldInfoEntry:is_valid()`

### Fields:
* `KoboldWorldInfoEntry.comment`
* `KoboldWorldInfoEntry.constant`
* `KoboldWorldInfoEntry.content`
* `KoboldWorldInfoEntry.folder`
* `KoboldWorldInfoEntry.key`
* `KoboldWorldInfoEntry.keysecondary`
* `KoboldWorldInfoEntry.num`
* `KoboldWorldInfoEntry.selective`
* `KoboldWorldInfoEntry.uid`

# KoboldWorldInfoEntry:compute_context()

***Callable from:*** anywhere

```lua
function KoboldWorldInfoEntry:compute_context(submission: string, kwargs?: table<string, any>) -> string
```

The same as calling `kobold.worldinfo:compute_context()` with this world info entry as the argument.

### Parameters

* submission (`string`): String to use as simulated user's input after being formatted by input formatting.
* kwargs? (`table<string, any>`): Table of optional keyword arguments from the following list. Defaults to `{}`.
    * scan_story? (`boolean`): Whether or not to scan the past few actions of the story for world info keys in addition to the submission like how world info normally behaves. If this is set to `false`, only the `submission` is scanned for world info keys. Defaults to `true`.
    * include_anote? (`boolean`): Whether to include the author's note in the story.  Defaults to `true`, pass `false` to suppress including the author's note.

### Returns

`string`: Computed context.

# KoboldWorldInfoEntry:is_valid()

***Callable from:*** anywhere

```lua
function KoboldWorldInfoEntry:is_valid() -> boolean
```

Returns true if this world info entry still exists (i.e. wasn't deleted), otherwise returns false.

# KoboldWorldInfoEntry.comment

***Readable from:*** anywhere
***Writable from:*** anywhere

```lua
field KoboldWorldInfoEntry.comment: string
```

The world info entry's comment that appears in its topmost text box.

# KoboldWorldInfoEntry.constant

***Readable from:*** anywhere
***Writable from:*** anywhere (triggers regeneration when written to from generation modifier)

```lua
field KoboldWorldInfoEntry.constant: boolean
```

Whether or not this world info entry is constant. Constant world info entries are always included in the context regardless of whether or not its keys match the story chunks in the context.

Modifying this field from inside of a generation modifier triggers a regeneration, which means that the context is recomputed after modification and generation begins again with the new context and previously generated tokens. This incurs a small performance penalty and should not be performed in excess.

# KoboldWorldInfoEntry.content

***Readable from:*** anywhere
***Writable from:*** anywhere (triggers regeneration when written to from generation modifier)

```lua
field KoboldWorldInfoEntry.content: string
```

The text in the "What To Remember" text box that gets included in the context when the world info entry is active.

Modifying this field from inside of a generation modifier triggers a regeneration, which means that the context is recomputed after modification and generation begins again with the new context and previously generated tokens. This incurs a small performance penalty and should not be performed in excess.

# KoboldWorldInfoEntry.folder

***Readable from:*** anywhere
***Writable from:*** nowhere

```lua
field KoboldWorldInfoEntryfolder: integer?
```

UID of the folder the world info entry is in, or `nil` if it's outside of a folder.

# KoboldWorldInfoEntry.key

***Readable from:*** anywhere
***Writable from:*** anywhere (triggers regeneration when written to from generation modifier)

```lua
field KoboldWorldInfoEntry.key: string
```

For non-selective world info entries, this is the world info entry's comma-separated list of keys. For selective world info entries, this is the comma-separated list of primary keys.

Modifying this field from inside of a generation modifier triggers a regeneration, which means that the context is recomputed after modification and generation begins again with the new context and previously generated tokens. This incurs a small performance penalty and should not be performed in excess.

# KoboldWorldInfoEntry.keysecondary

***Readable from:*** anywhere
***Writable from:*** anywhere (triggers regeneration when written to from generation modifier)

```lua
field KoboldWorldInfoEntry.keysecondary: string
```

For non-selective world info entries, the value of this field is undefined and writing to it has no effect. For selective world info entries, this is the comma-separated list of secondary keys.

Modifying this field from inside of a generation modifier triggers a regeneration, which means that the context is recomputed after modification and generation begins again with the new context and previously generated tokens. This incurs a small performance penalty and should not be performed in excess.

# KoboldWorldInfoEntry.num

***Readable from:*** anywhere
***Writable from:*** nowhere

```lua
field KoboldWorldInfoEntry.num: integer
```

This is 0 if the entry is the first one from the top, 1 if second from the top, and so on.

# KoboldWorldInfoEntry.selective

***Readable from:*** anywhere
***Writable from:*** anywhere (triggers regeneration when written to from generation modifier)
 
```lua
field KoboldWorldInfoEntry.selective: boolean
```

Whether or not the world info entry is selective. Selective entries have both primary and secondary keys.

Modifying this field from inside of a generation modifier triggers a regeneration, which means that the context is recomputed after modification and generation begins again with the new context and previously generated tokens. This incurs a small performance penalty and should not be performed in excess.

# KoboldWorldInfoEntry.uid

***Readable from:*** anywhere
***Writable from:*** nowhere

```lua
field KoboldWorldInfoEntry.uid: integer
```

UID of the world info entry.

# KoboldWorldInfoFolderSelector:finduid()

***Callable from:*** anywhere

```lua
function KoboldWorldInfoFolderSelector:finduid(u: integer) -> KoboldWorldInfoFolder?
```

Returns the world info folder with the given UID in amortized constant worst-case time, or `nil` if not found.

### Parameters

* u (`integer`): UID.

### Returns

* `KoboldWorldInfoFolder?`: The world info folder with requested UID, or `nil` if no such folder exists.

# KoboldWorldInfoFolderSelector:is_valid()

***Callable from:*** anywhere

```lua
function KoboldWorldInfoFolderSelector:is_valid() -> boolean
```

This always returns true.

# KoboldWorldInfoFolder

Represents a world info folder.

### Methods:
* `KoboldWorldInfoFolder:compute_context()`
* `KoboldWorldInfoFolder:finduid()`
* `KoboldWorldInfoFolder:is_valid()`

### Fields:
* `KoboldWorldInfoFolder.name`
* `KoboldWorldInfoFolder.uid`

# KoboldWorldInfoFolder:compute_context()

***Callable from:*** anywhere

```lua
function KoboldWorldInfoFolder:compute_context(submission: string, entries?: KoboldWorldInfoEntry|table<any, KoboldWorldInfoEntry>, kwargs?: table<string, any>) -> string
```

Computes the context that would be sent to the generator with the user's current settings if `submission` were the user's input after being formatted by input formatting. The context would include memory at the top, followed by active world info entries, followed by some story chunks with the author's note somewhere, followed by `submission`.

Unlike `kobold.worldinfo:compute_context()`, this function doesn't include world info keys outside of the folder.

### Parameters

* submission (`string`): String to use as simulated user's input after being formatted by input formatting.
* entries? (`KoboldWorldInfoEntry|table<any, KoboldWorldInfoEntry>`): A `KoboldWorldInfoEntry` or table thereof that indicates an allowed subset of world info entries to include in the context. Entries that are not inside of the folder are still not included. Defaults to all world info entries in the folder.
* kwargs? (`table<string, any>`): Table of optional keyword arguments from the following list. Defaults to `{}`.
    * scan_story? (`boolean`): Whether or not to scan the past few actions of the story for world info keys in addition to the submission like how world info normally behaves. If this is set to `false`, only the `submission` is scanned for world info keys. Defaults to `true`.
    * include_anote? (`boolean`): Whether to include the author's note in the story.  Defaults to `true`, pass `false` to suppress including the author's note.

### Returns

`string`: Computed context.

# KoboldWorldInfoFolder:finduid()

***Callable from:*** anywhere

```lua
function KoboldWorldInfoFolder:finduid(u: integer) -> KoboldWorldInfoEntry?
```

Returns the world info entry inside of the folder with the given UID in amortized constant worst-case time, or `nil` if not found.

### Parameters

* u (`integer`): UID.

### Returns

* `KoboldWorldInfoEntry?`: The world info entry with requested UID, or `nil` if no such entry exists or if it's outside of the folder.

# KoboldWorldInfoFolder:is_valid()

***Callable from:*** anywhere

```lua
function KoboldWorldInfoFolder:is_valid() -> boolean
```

Returns whether or not the folder still exists (i.e. wasn't deleted).

# KoboldWorldInfoFolder&#46;name

***Readable from:*** anywhere
***Writable from:*** anywhere

```lua
field KoboldWorldInfoFolder.name: string
```

Name of the world info folder as defined in its one text box.

# KoboldWorldInfoFolder.uid

***Readable from:*** anywhere
***Writable from:*** nowhere

```lua
field KoboldWorldInfoFolder.uid: integer
```

UID of the world info folder.
