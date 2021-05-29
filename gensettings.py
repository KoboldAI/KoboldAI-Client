gensettingstf = [{
	"uitype": "slider",
	"unit": "float",
	"label": "Temperature",
	"id": "settemp", 
	"min": 0.1,
	"max": 2.0,
	"step": 0.05,
	"default": 1.0,
    "tooltip": "Randomness of sampling. High values can increase creativity but may make text less sensible. Lower values will make text more predictable but can become repetitious."
	},
	{
	"uitype": "slider",
	"unit": "float",
	"label": "Top p Sampling",
	"id": "settopp", 
	"min": 0.1,
	"max": 1.0,
	"step": 0.05,
	"default": 1.0,
    "tooltip": "Used to discard unlikely text in the sampling process. Lower values will make text more predictable but can become repetitious."
	},
	{
	"uitype": "slider",
	"unit": "float",
	"label": "Repetition Penalty",
	"id": "setreppen", 
	"min": 1.0,
	"max": 2.0,
	"step": 0.05,
	"default": 1.0,
    "tooltip": "Used to penalize words that were already generated or belong to the context."
	},
	{
	"uitype": "slider",
	"unit": "int",
	"label": "Amount to Generate",
	"id": "setoutput", 
	"min": 16,
	"max": 512,
	"step": 2,
	"default": 60,
    "tooltip": "Number of tokens the AI should generate. Higher numbers will take longer to generate."
	},
    {
	"uitype": "slider",
	"unit": "int",
	"label": "Max Tokens",
	"id": "settknmax", 
	"min": 512,
	"max": 2048,
	"step": 8,
	"default": 512,
    "tooltip": "Max number of tokens of context to submit to the AI for sampling. Make sure this is higher than Amount to Generate. Higher values increase VRAM/RAM usage."
	},
    {
	"uitype": "slider",
	"unit": "int",
	"label": "Gens Per Action",
	"id": "setnumseq", 
	"min": 1,
	"max": 5,
	"step": 1,
	"default": 1,
    "tooltip": "Number of results to generate per submission. Increases VRAM/RAM usage."
	},
    {
	"uitype": "slider",
	"unit": "int",
	"label": "W Info Depth",
	"id": "setwidepth", 
	"min": 1,
	"max": 5,
	"step": 1,
	"default": 1,
    "tooltip": "Number of historic actions to scan for W Info keys."
	},
    {
	"uitype": "toggle",
	"unit": "bool",
	"label": "Always Add Prompt",
	"id": "setuseprompt", 
	"min": 0,
	"max": 1,
	"step": 1,
	"default": 1,
    "tooltip": "Whether the prompt should be sent in the context of every action."
	}]

gensettingsik =[{
	"uitype": "slider",
	"unit": "float",
	"label": "Temperature",
	"id": "settemp", 
	"min": 0.1,
	"max": 2.0,
	"step": 0.05,
	"default": 1.0,
    "tooltip": "Randomness of sampling. High values can increase creativity but may make text less sensible. Lower values will make text more predictable but can become repetitious."
	},
	{
	"uitype": "slider",
	"unit": "float",
	"label": "Top p Sampling",
	"id": "settopp", 
	"min": 0.1,
	"max": 1.0,
	"step": 0.05,
	"default": 1.0,
    "tooltip": "Used to discard unlikely text in the sampling process. Lower values will make text more predictable but can become repetitious."
	},
    {
	"uitype": "slider",
	"unit": "int",
	"label": "Amount to Generate",
	"id": "setikgen", 
	"min": 50,
	"max": 3000,
	"step": 2,
	"default": 200,
    "tooltip": "Number of characters the AI should generate."
	},
    {
	"uitype": "slider",
	"unit": "int",
	"label": "W Info Depth",
	"id": "setwidepth", 
	"min": 1,
	"max": 5,
	"step": 1,
	"default": 1,
    "tooltip": "Number of historic actions to scan for W Info keys."
	},
    {
	"uitype": "toggle",
	"unit": "bool",
	"label": "Always Add Prompt",
	"id": "setuseprompt", 
	"min": 0,
	"max": 1,
	"step": 1,
	"default": 1,
    "tooltip": "Whether the prompt should be sent in the context of every action."
	}]

formatcontrols = [{
    "label": "Trim incomplete sentences",
    "id": "frmttriminc",
    "tooltip": "Remove text after last sentence closure.  If no closure is found, all tokens will be returned."
    },
    {
    "label": "Remove blank lines",
    "id": "frmtrmblln",
    "tooltip": "Replace double newlines (\\n\\n) with single newlines to avoid blank lines."
    },
    {
    "label": "Remove special characters",
    "id": "frmtrmspch",
    "tooltip": "Remove special characters (@,#,%,^, etc)"
    },
    {
    "label": "Add sentence spacing",
    "id": "frmtadsnsp",
    "tooltip": "If the last action ended with punctuation, add a space to the beginning of the next action."
    }]