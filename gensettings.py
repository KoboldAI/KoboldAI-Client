gensettingstf = [
	{
	"uitype": "slider",
	"unit": "int",
	"label": "Amount to Generate",
	"id": "setoutput", 
	"min": 16,
	"max": 512,
	"step": 2,
	"default": 80,
    "tooltip": "Number of tokens the AI should generate. Higher numbers will take longer to generate."
	},
   {
	"uitype": "slider",
	"unit": "float",
	"label": "Temperature",
	"id": "settemp", 
	"min": 0.1,
	"max": 2.0,
	"step": 0.01,
	"default": 0.5,
    "tooltip": "Randomness of sampling. High values can increase creativity but may make text less sensible. Lower values will make text more predictable but can become repetitious."
	},
	{
	"uitype": "slider",
	"unit": "float",
	"label": "Top p Sampling",
	"id": "settopp", 
	"min": 0.0,
	"max": 1.0,
	"step": 0.01,
	"default": 0.9,
    "tooltip": "Used to discard unlikely text in the sampling process. Lower values will make text more predictable but can become repetitious. (Put this value on 1 to disable its effect)"
	},
	{
	"uitype": "slider",
	"unit": "int",
	"label": "Top k Sampling",
	"id": "settopk",
	"min": 0,
	"max": 100,
	"step": 1,
	"default": 0,
    "tooltip": "Alternative sampling method, can be combined with top_p. (Put this value on 0 to disable its effect)"
	},
	{
	"uitype": "slider",
	"unit": "float",
	"label": "Tail-free Sampling",
	"id": "settfs", 
	"min": 0.0,
	"max": 1.0,
	"step": 0.01,
	"default": 1.0,
    "tooltip": "Alternative sampling method; it is recommended to disable top_p and top_k (set top_p to 1 and top_k to 0) if using this. 0.95 is thought to be a good value. (Put this value on 1 to disable its effect)"
	},
	{
	"uitype": "slider",
	"unit": "float",
	"label": "Typical Sampling",
	"id": "settypical", 
	"min": 0.0,
	"max": 1.0,
	"step": 0.01,
	"default": 1.0,
    "tooltip": "Alternative sampling method described in the paper \"Typical Decoding for Natural Language Generation\" (10.48550/ARXIV.2202.00666). The paper suggests 0.2 as a good value for this setting. Set this setting to 1 to disable its effect."
	},
	{
	"uitype": "slider",
	"unit": "float",
	"label": "Top a Sampling",
	"id": "settopa", 
	"min": 0.0,
	"max": 1.0,
	"step": 0.01,
	"default": 0.0,
    "tooltip": "Alternative sampling method that reduces the randomness of the AI whenever the probability of one token is much higher than all the others. Higher values have a stronger effect. Set this setting to 0 to disable its effect."
	},
	{
	"uitype": "slider",
	"unit": "float",
	"label": "Repetition Penalty",
	"id": "setreppen", 
	"min": 1.0,
	"max": 3.0,
	"step": 0.01,
	"default": 1.1,
    "tooltip": "Used to penalize words that were already generated or belong to the context (Going over 1.2 breaks 6B models)."
	},
	{
	"uitype": "slider",
	"unit": "int",
	"label": "Rep Penalty Range",
	"id": "setreppenrange", 
	"min": 0,
	"max": 4096,
	"step": 4,
	"default": 0,
    "tooltip": "Repetition penalty range. If set higher than 0, only applies repetition penalty to the last few tokens of your story rather than applying it to the entire story. This slider controls the amount of tokens at the end of your story to apply it to."
	},
	{
	"uitype": "slider",
	"unit": "float",
	"label": "Rep Penalty Slope",
	"id": "setreppenslope", 
	"min": 0.0,
	"max": 10.0,
	"step": 0.1,
	"default": 0.0,
    "tooltip": "Repetition penalty slope. If BOTH this setting and Rep Penalty Range are set higher than 0, will use sigmoid interpolation to apply repetition penalty more strongly on tokens that are closer to the end of your story. This setting controls the tension of the sigmoid curve; higher settings will result in the repetition penalty difference between the start and end of your story being more apparent. Setting this to 1 uses linear interpolation; setting this to 0 disables interpolation."
	},
    {
	"uitype": "slider",
	"unit": "int",
	"label": "Max Tokens",
	"id": "settknmax", 
	"min": 512,
	"max": 2048,
	"step": 8,
	"default": 1024,
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
	"default": 3,
    "tooltip": "Number of historic actions to scan for W Info keys."
	},
    {
	"uitype": "toggle",
	"unit": "bool",
	"label": "Auto Save",
	"id": "autosave", 
	"min": 0,
	"max": 1,
	"step": 1,
	"default": 0,
    "tooltip": "Whether the game is saved after each action."
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
	},
	{
	"uitype": "toggle",
	"unit": "bool",
	"label": "Adventure Mode",
	"id": "setadventure", 
	"min": 0,
	"max": 1,
	"step": 1,
	"default": 0,
    "tooltip": "Turn this on if you are playing a Choose your Adventure model."
	},
	{
	"uitype": "toggle",
	"unit": "bool",
	"label": "Chat Mode",
	"id": "setchatmode", 
	"min": 0,
	"max": 1,
	"step": 1,
	"default": 0,
    "tooltip": "This mode optimizes KoboldAI for chatting."
	},
	{
	"uitype": "toggle",
	"unit": "bool",
	"label": "Dynamic WI Scan",
	"id": "setdynamicscan", 
	"min": 0,
	"max": 1,
	"step": 1,
	"default": 0,
    "tooltip": "Scan the AI's output for world info keys as it's generating the output."
	},
	{
	"uitype": "toggle",
	"unit": "bool",
	"label": "No Prompt Generation",
	"id": "setnopromptgen", 
	"min": 0,
	"max": 1,
	"step": 1,
	"default": 0,
    "tooltip": "When enabled the AI does not generate when you enter the prompt, instead you need to do an action first."
	},
	{
	"uitype": "toggle",
	"unit": "bool",
	"label": "Random Story Persist",
	"id": "setrngpersist",
	"min": 0,
	"max": 1,
	"step": 1,
	"default": 0,
    "tooltip": "When enabled, the Memory text box in the Random Story dialog will be prefilled by default with your current story's memory instead of being empty."
	},
	{
	"uitype": "toggle",
	"unit": "bool",
	"label": "No Genmod",
	"id": "setnogenmod",
	"min": 0,
	"max": 1,
	"step": 1,
	"default": 0,
  "tooltip": "Disables userscript generation modifiers."
	},
	{
	"uitype": "toggle",
	"unit": "bool",
	"label": "Full Determinism",
	"id": "setfulldeterminism",
	"min": 0,
	"max": 1,
	"step": 1,
	"default": 0,
  "tooltip": "Causes generation to be fully deterministic -- the model will always output the same thing as long as your story, settings and RNG seed are the same. If this is off, only the sequence of outputs that the model makes will be deterministic."
	},
    {
	"uitype": "toggle",
	"unit": "bool",
	"label": "Token Streaming",
	"id": "setoutputstreaming",
	"min": 0,
	"max": 1,
	"step": 1,
	"default": 0,
  "tooltip": "Shows outputs to you as they are made. Does not work with more than one gens per action."
	},
    {
	"uitype": "toggle",
	"unit": "bool",
	"label": "Probability Viewer",
	"id": "setshowprobs",
	"min": 0,
	"max": 1,
	"step": 1,
	"default": 0,
  "tooltip": "Shows token selection probabilities. Does not work with more than one gens per action."
	},
    {
	"uitype": "toggle",
	"unit": "bool",
	"label": "Show Field Budget",
	"id": "setshowbudget",
	"min": 0,
	"max": 1,
	"step": 1,
	"default": 0,
  "tooltip": "Shows token usage when typing in relevant text boxes. <b>May lag slower devices.</b>"
	},
    {
	"uitype": "toggle",
	"unit": "bool",
	"label": "Debug",
	"id": "debug",
	"min": 0,
	"max": 1,
	"step": 1,
	"default": 0,
  "tooltip": "Show debug info"
	},
]

gensettingsik =[{
	"uitype": "slider",
	"unit": "float",
	"label": "Temperature",
	"id": "settemp", 
	"min": 0.1,
	"max": 2.0,
	"step": 0.05,
	"default": 0.5,
    "tooltip": "Randomness of sampling. High values can increase creativity but may make text less sensible. Lower values will make text more predictable but can become repetitious."
	},
	{
	"uitype": "slider",
	"unit": "float",
	"label": "Top p Sampling",
	"id": "settopp", 
	"min": 0.0,
	"max": 1.0,
	"step": 0.05,
	"default": 1.1,
    "tooltip": "Used to discard unlikely text in the sampling process. Lower values will make text more predictable but can become repetitious."
	},
	{
	"uitype": "slider",
	"unit": "int",
	"label": "Top k Sampling",
	"id": "settopk",
	"min": 0,
	"max": 100,
	"step": 1,
	"default": 0,
    "tooltip": "Alternative sampling method, can be combined with top_p."
	},
	{
	"uitype": "slider",
	"unit": "float",
	"label": "Tail-free Sampling",
	"id": "settfs", 
	"min": 0.0,
	"max": 1.0,
	"step": 0.05,
	"default": 0.0,
    "tooltip": "Alternative sampling method; it is recommended to disable (set to 0) top_p and top_k if using this. 0.95 is thought to be a good value."
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
	"default": 3,
    "tooltip": "Number of historic actions to scan for W Info keys."
	},
    {
	"uitype": "toggle",
	"unit": "bool",
	"label": "Auto Save",
	"id": "autosave", 
	"min": 0,
	"max": 1,
	"step": 1,
	"default": 0,
    "tooltip": "Whether the game is saved after each action."
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
	},
	{
	"uitype": "toggle",
	"unit": "bool",
	"label": "Adventure Mode",
	"id": "setadventure", 
	"min": 0,
	"max": 1,
	"step": 1,
	"default": 0,
    "tooltip": "Turn this on if you are playing a Choose your Adventure model."
	},
	{
	"uitype": "toggle",
	"unit": "bool",
	"label": "Chat Mode",
	"id": "setchatmode", 
	"min": 0,
	"max": 1,
	"step": 1,
	"default": 0,
    "tooltip": "This mode optimizes KoboldAI for chatting."
	},
	{
	"uitype": "toggle",
	"unit": "bool",
	"label": "No Prompt Generation",
	"id": "setnopromptgen", 
	"min": 0,
	"max": 1,
	"step": 1,
	"default": 0,
    "tooltip": "When enabled the AI does not generate when you enter the prompt, instead you need to do an action first."
	},
	{
	"uitype": "toggle",
	"unit": "bool",
	"label": "Random Story Persist",
	"id": "setrngpersist",
	"min": 0,
	"max": 1,
	"step": 1,
	"default": 0,
    "tooltip": "When enabled, the Memory text box in the Random Story dialog will be prefilled by default with your current story's memory instead of being empty."
	},
    {
	"uitype": "toggle",
	"unit": "bool",
	"label": "Debug",
	"id": "debug",
	"min": 0,
	"max": 1,
	"step": 1,
	"default": 0,
  "tooltip": "Show debug info"
	}
]

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
    "label": "Automatic spacing",
    "id": "frmtadsnsp",
    "tooltip": "Add spaces automatically if needed"
    },
    {
    "label": "Single Line",
    "id": "singleline",
    "tooltip": "Only allows the AI to output anything before the enter"
    }]
