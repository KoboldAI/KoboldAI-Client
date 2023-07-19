var socket;
socket = io.connect(window.location.origin, {transports: ['polling', 'websocket'], closeOnBeforeunload: false, query:{"ui":  "2"}});


//Let's register our server communications
socket.on('connect', function(){connect();});
socket.on("disconnect", (reason, details) => {
  console.log("Lost connection from: "+reason); // "transport error"
  disconnect();
});
socket.on('reset_story', function(){reset_story();});
socket.on('var_changed', function(data){var_changed(data);});
socket.on('load_popup', function(data){load_popup(data);});
socket.on('popup_items', function(data){popup_items(data);});
socket.on('popup_breadcrumbs', function(data){popup_breadcrumbs(data);});
socket.on('popup_edit_file', function(data){popup_edit_file(data);});
//socket.on('show_model_menu', function(data){show_model_menu(data);});
socket.on('open_model_load_menu', function(data){show_model_menu(data);});
socket.on('selected_model_info', function(data){selected_model_info(data);});
socket.on('oai_engines', function(data){oai_engines(data);});
socket.on('buildload', function(data){buildload(data);});
socket.on('error_popup', function(data){error_popup(data);});
socket.on("world_info_entry", function(data){process_world_info_entry(data);});
socket.on("world_info_entry_used_in_game", function(data){world_info_entry_used_in_game(data);});
socket.on("world_info_folder", function(data){world_info_folder(data);});
socket.on("delete_new_world_info_entry", function(data){document.getElementById("world_info_-1").remove();});
socket.on("delete_world_info_entry", function(data){document.getElementById("world_info_"+data).remove();});
socket.on("delete_world_info_folder", function(data){document.getElementById("world_info_folder_"+data).remove();});
socket.on("error", function(data){show_error_message(data);});
socket.on("message", function(data){show_message(data);});
socket.on('load_cookies', function(data){load_cookies(data);});
socket.on('load_tweaks', function(data){load_tweaks(data);});
socket.on("wi_results", updateWISearchListings);
socket.on("request_prompt_config", configurePrompt);
socket.on("log_message", function(data){process_log_message(data);});
socket.on("debug_message", function(data){console.log(data);});
socket.on("scratchpad_response", recieveScratchpadResponse);
socket.on("show_error_notification", function(data) { reportError(data.title, data.text) });
socket.on("generated_wi", showGeneratedWIData);
//socket.onAny(function(event_name, data) {console.log({"event": event_name, "class": data.classname, "data": data});});

// Must be done before any elements are made; we track their changes.
initalizeTooltips();

//setup an observer on the game text
var chunk_delete_observer = new MutationObserver(function (records) {gametextwatcher(records)});


var vars_sync_time = {};
var presets = {};
var current_chunk_number = null;
var ai_busy_start = Date.now();
var popup_deleteable = false;
var popup_editable = false;
var popup_renameable = false;
var rename_return_emit_name = "popup_rename";
var popup_rows = [];
var popup_style = "";
var popup_sort = {};
var shift_down = false;
var world_info_data = {};
var world_info_folder_data = {};
var saved_settings = {};
var biases_data = {};
var finder_selection_index = -1;
var colab_cookies = null;
var wi_finder_data = [];
var wi_finder_offset = 0;
var selected_game_chunk = null;
var log = [];
var on_new_wi_item = null;
var finder_mode = "ui";
var finder_waiting_id = null;
var control_held = false;
var actions_data = {};
var setup_wi_toggles = [];
var scroll_trigger_element = undefined; //undefined means not currently set. If set to null, it's disabled.
var drag_id = null;
var story_commentary_characters = {};
var generating_summary = false;
const on_colab = $el("#on_colab").textContent == "true";
let story_id = -1;
var dirty_chunks = [];
var initial_socketio_connection_occured = false;
var selected_model_data;

// Each entry into this array should be an object that looks like:
// {class: "class", key: "key", func: callback}
let sync_hooks = [];

// name, desc, icon, func
var finder_actions = [
	{name: "Load Model", icon: "folder_open", type: "action", func: function() { socket.emit('load_model_button', {}); }},
	{name: "New Story", icon: "description", type: "action", func: function() { socket.emit('new_story', ''); }},
	{name: "Load Story", icon: "folder_open", type: "action", func: load_story_list},
	{name: "Save Story", icon: "save", type: "action", func: save_story},
	{name: "Download Story", icon: "file_download", type: "action", func: function() { document.getElementById('download_iframe').src = 'json'; }},
	{name: "Import Story", icon: "file_download", desc: "Import a prompt from aetherroom.club, formerly prompts.aidg.club", type: "action", func: openClubImport },

	// Imggen
	{name: "Download Generated Image", icon: "file_download", type: "action", func: imgGenDownload},
	{name: "View Generated Image", icon: "image", type: "action", func: imgGenView},
	{name: "Clear Generated Image", icon: "image_not_supported", type: "action", func: imgGenClear},

	// Locations
	{name: "Setting Presets", icon: "open_in_new", type: "location", func: function() { highlightEl(".var_sync_model_selected_preset") }},
	{name: "Memory", icon: "open_in_new", type: "location", func: function() { highlightEl("#memory") }},
	{name: "Author's Note", icon: "open_in_new", type: "location", func: function() { highlightEl("#authors_notes") }},
	{name: "Notes", icon: "open_in_new", type: "location", func: function() { highlightEl(".var_sync_story_notes") }},
	{name: "World Info", icon: "open_in_new", type: "location", func: function() { highlightEl("#WI_Area") }},
	
	// TODO: Direct theme selection
	// {name: "", icon: "palette", func: function() { highlightEl("#biasing") }},
];

const context_menu_actions = {
	gamescreen: [
		{label: "Speak", icon: "record_voice_over", enabledOn: "CARET", click: speak_audio, shouldShow: function () {return document.getElementById("story_gen_audio").checked;}},
		null,
		{label: "Cut", icon: "content_cut", enabledOn: "SELECTION", click: cut},
		{label: "Copy", icon: "content_copy", enabledOn: "SELECTION", click: copy},
		{label: "Paste", icon: "content_paste", enabledOn: "SELECTION", click: paste},
		// Null makes a seperation bar
		null,
		{label: "Add to Memory", icon: "assignment", enabledOn: "SELECTION", click: push_selection_to_memory},
		{label: "Add to World Info Entry", icon: "auto_stories", enabledOn: "SELECTION", click: push_selection_to_world_info},
		{label: "Add as Bias", icon: "insights", enabledOn: "SELECTION", click: push_selection_to_phrase_bias},
		{label: "Retry from here", icon: "refresh", enabledOn: "CARET", click: retry_from_here},
		null,
		{label: "Take Screenshot", icon: "screenshot_monitor", enabledOn: "SELECTION", click: screenshot_selection},
		// Not implemented! See view_selection_probabiltiies
		// null,
		// {label: "View Token Probabilities", icon: "assessment", enabledOn: "SELECTION", click: view_selection_probabilities},
		// {label: "View Token Probabilities", icon: "account_tree", enabledOn: "SELECTION", click: view_selection_probabilities},
	],
	"wi-img": [
		{label: "View", icon: "search", enabledOn: "ALWAYS", click: wiImageView},
		{label: "Replace", icon: "swap_horiz", enabledOn: "ALWAYS", click: wiImageReplace},
		{label: "Clear", icon: "clear", enabledOn: "ALWAYS", click: wiImageClear},
	],
	"generated-image": [
		{label: "View", icon: "search", enabledOn: "ALWAYS", click: imgGenView},
		{label: "Download", icon: "download", enabledOn: "ALWAYS", click: imgGenDownload},
		{label: "Retry", icon: "refresh", enabledOn: "ALWAYS", click: imgGenRetry},
		{label: "Clear", icon: "clear", enabledOn: "ALWAYS", click: imgGenClear},
	],
	"wi-img-upload-button": [
		{label: "Upload Image", icon: "file_upload", enabledOn: "ALWAYS", click: wiImageReplace},
		{label: "Use Generated Image", icon: "image", enabledOn: "GENERATED-IMAGE", click: wiImageUseGeneratedImage},
	]
};

let context_menu_cache = [];

const shortcuts = [
	{mod: "ctrl", key: "s", desc: "Save Story", func: save_story},
	{mod: "ctrl", key: "o", desc: "Open Story", func: load_story_list},
	{mod: "alt", key: "z", desc: "Undoes last story action", func: storyBack, criteria: canNavigateStoryHistory},
	{mod: "alt", key: "y", desc: "Redoes last story action", func: storyRedo, criteria: canNavigateStoryHistory},
	{mod: "alt", key: "r", desc: "Retries last story action", func: storyRetry, criteria: canNavigateStoryHistory},
	{mod: "ctrl", key: "m", desc: "Focuses Memory", func: () => focusEl("#memory")},
	{mod: "ctrl", key: "u", desc: "Focuses Author's Note", func: () => focusEl("#authors_notes")}, // CTRL-N is reserved :^(
	{mod: "ctrl", key: "g", desc: "Focuses game text", func: () => focusEl("#input_text")},
	{mod: "ctrl", key: "l", desc: '"Lock" screen (Not secure)', func: () => socket.emit("privacy_mode", {'enabled': true})},
	{mod: "ctrl", key: "k", desc: "Finder", func: open_finder},
	{mod: "ctrl", key: "/", desc: "Help screen", func: () => openPopup("shortcuts-popup")},
]

const chat = {
	STYLES: {LEGACY: 0, MESSAGES: 1, BUBBLES: 2},
	style: "legacy",
	lastEdit: null,

	get useV2() {
		return [
			this.STYLES.MESSAGES,
			this.STYLES.BUBBLES
		].includes(this.style) && story.mode === story.MODES.CHAT
	}
}

const story = {
	MODES: {STORY: 0, ADVENTURE: 1, CHAT: 2},
	mode: null,
}

function $el(selector) {
	// We do not preemptively fetch all elements upon execution (wall of consts)
	// due to the layer of mental overhead it adds to debugging and reading
	// code in general.
	return document.querySelector(selector);
}

const map1 = new Map()
map1.set('Top K Sampling', 0)
map1.set('Top A Sampling', 1)
map1.set('Top P Sampling', 2)
map1.set('Tail Free Sampling', 3)
map1.set('Typical Sampling', 4)
map1.set('Temperature', 5)
map1.set('Repetition Penalty', 6)
const map2 = new Map()
map2.set(0, 'Top K Sampling')
map2.set(1, 'Top A Sampling')
map2.set(2, 'Top P Sampling')
map2.set(3, 'Tail Free Sampling')
map2.set(4, 'Typical Sampling')
map2.set(5, 'Temperature')
map2.set(6, 'Repetition Penalty')
var calc_token_usage_timeout;
var game_text_scroll_timeout;
var auto_loader_timeout;
var world_info_scroll_timeout;
var font_size_cookie_timout;
var colab_cookie_timeout;
var setup_missing_wi_toggles_timeout;
var var_processing_time = 0;
var finder_last_input;
var current_action;
//-----------------------------------Server to UI  Functions-----------------------------------------------
function connect() {
	console.log("connected");
	//reset_story();
	if (initial_socketio_connection_occured) {
		location.reload();
	}
	initial_socketio_connection_occured = true;
	for (item of document.getElementsByTagName("body")) {
		item.classList.remove("NotConnected");
	}
	document.getElementById("disconnect_message").classList.add("hidden");
}

function disconnect() {
	console.log("disconnected");
	for (item of document.getElementsByTagName("body")) {
		item.classList.add("NotConnected");
	}
	document.getElementById("disconnect_message").classList.remove("hidden");
}

function storySubmit() {
	disruptStoryState();
	socket.emit('submit', {'data': document.getElementById('input_text').value, 'theme': document.getElementById('themetext').value});
	document.getElementById('input_text').value = '';
	document.getElementById('themetext').value = '';
}

function storyBack() {
	disruptStoryState();
	socket.emit('back', {});
}

function storyRedo() {
	disruptStoryState();
	socket.emit('redo', {});
}

function storyRetry() {
	disruptStoryState();
	socket.emit('retry', {});
}

function disruptStoryState() {
	// This function is responsible for wiping things away which are sensitive
	// to story state
	$el("#story-review").classList.add("hidden");
}

function reset_story() {
	console.log("Resetting story");
	location.reload();
	disruptStoryState();
	chunk_delete_observer.disconnect();
	clearTimeout(calc_token_usage_timeout);
	clearTimeout(game_text_scroll_timeout);
	clearTimeout(font_size_cookie_timout);
	clearTimeout(world_info_scroll_timeout);
	clearTimeout(auto_loader_timeout);
	finder_last_input = null;
	on_new_wi_item = null;
	current_chunk_number = null;
	scroll_trigger_element = undefined;
	
	//clear actions
	actions_data = {};
	var story_area = document.getElementById('Selected Text');
	let temp = []
	for (child of story_area.children) {
		if (child.id != 'story_prompt') {
			temp.push(child);
		}
	}
	for (const item of temp) { 
		item.remove();
	}
	
	//clear any options
	var option_area = document.getElementById("Select Options");
	while (option_area.firstChild) {
		option_area.removeChild(option_area.firstChild);
	}
	
	//clear world info
	world_info_data = {};
	world_info_folder({"root": []});
	var world_info_area = document.getElementById("WI_Area");
	while (world_info_area.firstChild) {
		world_info_area.removeChild(world_info_area.firstChild);
	}

	
	const storyPrompt = $el("#story_prompt");

	if (storyPrompt) {
		storyPrompt.setAttribute("world_info_uids", "");
	}
	document.getElementById('themerow').classList.remove("hidden");
	document.getElementById('input_text').placeholder = "Enter Prompt Here (shift+enter for new line)";
	text = "";
	for (i=0;i<70;i++) {
		text += "\xa0 ";
	}
	document.getElementById("welcome_text").innerText = text;
	document.getElementById("Selected Text").setAttribute("contenteditable", "false");
	if (document.getElementById("story_prompt").innerText == "") {
		document.getElementById("welcome_container").classList.remove("hidden");
		document.getElementById("Selected Text").setAttribute("contenteditable", "true");
		
	}
	document.getElementById('main-grid').setAttribute('option_length', 0);

	$(".chat-message").remove();
	addInitChatMessage();
	
	chunk_delete_observer.observe(document.getElementById('Selected Text'), { subtree: true, childList: true, characterData: true });
}

function fix_text(val) {
	if (typeof val === 'string' || val instanceof String) {
		if (val.includes("{")) {
			return JSON.stringify(val);
		} else {
			return val;
		}
	} else {
		return val;
	}
}

function create_options(action) {
	//Set all options before the next chunk to hidden
	if (action.id  != current_action+1) {
		return;
	}
	var option_chunk = document.getElementById("Select Options");
	
	//first, let's clear out our existing data
	while (option_chunk.firstChild) {
		option_chunk.removeChild(option_chunk.firstChild);
	}
	
	//Let's check if we only have a single redo option. In that case we din't show as the user can use the redo button
	seen_prev_selection = false;
	show_options = false;
	for (item of action.action.Options) {
		if (!(item['Previous Selection']) && !(item['Edited'])) {
			show_options = true;
			break;
		} else if (item['Previous Selection']) {
			if (seen_prev_selection) {
				show_options = true;
				break;
			} else {
				seen_prev_selection = true;
			}
		}
	}
	if (!(show_options)) {
		document.getElementById('main-grid').setAttribute('option_length', 0);
		return;
	}
	
	document.getElementById('main-grid').setAttribute('option_length', action.action.Options.length);
	
	var table = document.createElement("div");
	table.classList.add("sequences");
	//Add Redo options
	let added_options=0;
	i=0;
	for (item of action.action.Options) {
		if ((item['Previous Selection']) && (item.text != "")) {
			var row = document.createElement("div");
			row.classList.add("sequence_row");
			var textcell = document.createElement("span");
			textcell.textContent = item.text;
			textcell.classList.add("sequence");
			textcell.setAttribute("option_id", i);
			textcell.setAttribute("option_chunk", action.id);
			var iconcell = document.createElement("span");
			iconcell.setAttribute("option_id", i);
			iconcell.setAttribute("option_chunk", action.id);
			iconcell.classList.add("sequnce_icon");
			var icon = document.createElement("span");
			icon.id = "Pin_"+i;
			icon.classList.add("material-icons-outlined");
			icon.classList.add("option_icon");
			icon.classList.add("cursor");
			icon.textContent = "cached";
			iconcell.append(icon);
			delete_icon = $e("span", iconcell, {"classes": ["material-icons-outlined", "cursor", 'option_icon'], 
												"tooltip": "Delete Option", 'option_id': i,
												'option_chunk': action.id, 'textContent': 'delete'});
			delete_icon.onclick = function () {
									socket.emit("delete_option", {"chunk": this.getAttribute("option_chunk"), "option": this.getAttribute("option_id")});
							  };
			textcell.onclick = function () {
									socket.emit("Use Option Text", {"chunk": this.getAttribute("option_chunk"), "option": this.getAttribute("option_id")});
							  };
			row.append(textcell);
			row.append(iconcell);
			table.append(row);
			added_options+=1;
		}
		i+=1;
	}
	//Add general options
	i=0;
	for (item of action.action.Options) {
		if (!(item.Edited) && !(item['Previous Selection']) && (item.text != "")) {
			var row = document.createElement("div");
			row.classList.add("sequence_row");
			var textcell = document.createElement("span");
			textcell.textContent = item.text;
			textcell.classList.add("sequence");
			textcell.setAttribute("option_id", i);
			textcell.setAttribute("option_chunk", action.id);
			var iconcell = document.createElement("span");
			iconcell.setAttribute("option_id", i);
			iconcell.setAttribute("option_chunk", action.id);
			iconcell.classList.add("sequnce_icon");
			var icon = document.createElement("span");
			icon.id = "Pin_"+i;
			icon.classList.add("material-icons-outlined");
			icon.classList.add("option_icon");
			icon.classList.add("cursor");
			icon.classList.add("pin");
			icon.textContent = "push_pin";
			if (!(item.Pinned)) {
				icon.setAttribute('style', "filter: brightness(50%);");
			} else {
				icon.classList.add('rotate_45');
			}
			iconcell.append(icon);
			iconcell.onclick = function () {
									socket.emit("Pinning", {"chunk": this.getAttribute("option_chunk"), "option": this.getAttribute("option_id")});
							   };
			textcell.onclick = function () {
									socket.emit("Use Option Text", {"chunk": this.getAttribute("option_chunk"), "option": this.getAttribute("option_id")});
							  };
			row.append(textcell);
			row.append(iconcell);
			table.append(row);
			added_options+=1;
		}
		i+=1;
	}
	if (added_options > 0) {
		option_chunk.append(table);
	}
	
	
	//make sure our last updated chunk is in view
	//option_chunk.scrollIntoView();
}

function process_actions_data(data) {
	start_time = Date.now();
	if (Array.isArray(data.value)) {
		actions = data.value;
	} else {
		actions = [data.value];
	}
	if (actions.length == 0) {return;}
	let action_type = "????";
	let first_action = -100;
	//console.log('Chunk Exists: '+(document.getElementById("Selected Text Chunk " + actions[actions.length-1].id)));
	//console.log('Passed action has no selected text: '+(actions[actions.length-1].action['Selected Text'] == ""));
	//console.log('Old action has no selected text: '+(actions_data[Math.max.apply(null,Object.keys(actions_data).map(Number))]['Selected Text'] == ""));
	//console.log(actions_data[Math.max.apply(null,Object.keys(actions_data).map(Number))]['Selected Text']);
	//console.log('All action IDs already seen: '+((actions[0].id in actions_data) && (actions[actions.length-1].id in actions_data)));
	//we need to figure out if this is changes to existing text, added text at the end, or infinite scroll text at the begining
	if (!(document.getElementById("Selected Text Chunk " + actions[actions.length-1].id))) {
		//We don't have this item yet, either we're an append or an prepend
		if ((Object.keys(actions_data).length > 1) && (actions[actions.length-1].id < Math.min.apply(null,Object.keys(actions_data).map(Number).filter(function(x){return x>0})))) {
			//adding to the begining
			action_type = "prepend";
			first_action = Math.min.apply(null,Object.keys(actions_data).map(Number).filter(function(x){return x>0}));
		} else if (actions[actions.length-1].id > Math.max.apply(null,Object.keys(actions_data).map(Number))) {
			action_type = "append";
		}
	} else if (actions[actions.length-1].action['Selected Text'] == "") {
		action_type = "options text only or deleted chunk";
	} else if (actions_data[Math.max.apply(null,Object.keys(actions_data).map(Number))]['Selected Text'] == "") {
		action_type = "append";
	} else if ((actions[0].id in actions_data) && (actions[actions.length-1].id in actions_data)) {
		//update
		action_type = "update";
	}
	for (action of actions) {
		actions_data[parseInt(action.id)] = action.action;
		do_story_text_updates(action);
		create_options(action);
	}
	
	clearTimeout(game_text_scroll_timeout);
	game_text_scroll_timeout = setTimeout(run_infinite_scroll_update.bind(null, action_type, actions, first_action), 200);
	clearTimeout(auto_loader_timeout);
	
	
	hide_show_prompt();
	//console.log("Took "+((Date.now()-start_time)/1000)+"s to process");
	
}

function parseChatMessages(text) {
	let messages = [];

	for (const line of text.split("\n")) {
		let [author, text] = line.split(":", 2);
		if (!author && !text) continue;

		// If there is no ":" in the text, it's a system message.
		if (!text) {
			text = author;
			author = "System";
		}

		messages.push({author: author, text: text});
	}
	return messages;
}

function do_story_text_updates(action) {
	story_area = document.getElementById('Selected Text');
	current_chunk_number = action.id;
	let item = null;

	if (chat.useV2) {
		//console.log(`[story_text_update] ${action.id}`)
		if (action.id === chat.lastEdit) {
			// Swallow update if we just caused it
			chat.lastEdit = null;
			return;
		}

		deleteChatPromptIfEmpty();

		const messageEl = $el(`[action-id="${action.id}"]`);
		let previous = messageEl ? messageEl.previousElementSibling : null;

		// Remove old ones
		$(`[action-id="${action.id}"]`).remove();

		for (const message of parseChatMessages(action.action["Selected Text"])) {
			previous = addMessage(message.author, message.text, action.id, previous);
		}
	} else {
		if (document.getElementById('Selected Text Chunk '+action.id)) {
			item = document.getElementById('Selected Text Chunk '+action.id);
			//clear out the item first
			while (item.firstChild) { 
				item.removeChild(item.firstChild);
			}
		} else {
			item = document.createElement("span");
			item.id = 'Selected Text Chunk '+action.id;
			item.classList.add("rawtext");
			item.setAttribute("chunk", action.id);
			item.setAttribute("tabindex", parseInt(action.id)+1);
			//item.addEventListener("focus", (event) => {
			//	set_edit(event.target);
			//});
			
			//need to find the closest element
			closest_element = document.getElementById("story_prompt");
			eval_element = 0
			while (eval_element < action.id ) {
				if (document.getElementById("Selected Text Chunk " + eval_element)) {
					closest_element = document.getElementById("Selected Text Chunk " + eval_element);
				}
				eval_element += 1;
			}
			if (closest_element.nextElementSibling) {
				story_area.insertBefore(item, closest_element.nextElementSibling);
			} else {
				story_area.append(item);
			}
		}
		
		
		if (action.action['Selected Text'].charAt(0) == ">") {
			item.classList.add("action_mode_input");
		} else {
			item.classList.remove("action_mode_input");
		}

		if ('wi_highlighted_text' in action.action) {
			for (chunk of action.action['wi_highlighted_text']) {
				chunk_element = document.createElement("span");
				chunk_element.innerText = chunk['text'];
				if (chunk['WI matches'] != null) {
					chunk_element.classList.add("wi_match");
					chunk_element.setAttribute("tooltip", chunk['WI Text']);
					chunk_element.setAttribute("wi-uid", chunk['WI matches']);
				}
				item.append(chunk_element);
			}
		} else {
			chunk_element = document.createElement("span");
			chunk_element.innerText = action.action['Selected Text'];
			item.append(chunk_element);
		}
		item.original_text = action.action['Selected Text'];
		item.classList.remove("pulse")
		item.classList.remove("single_pulse");
		item.classList.add("single_pulse");
	}
}

function do_prompt(data) {
	if (!document.getElementById("story_prompt")) {
		//Someone deleted our prompt. Just refresh to clean things up
		location.reload();
	}
	let full_text = "";
	for (chunk of data.value) {
		full_text += chunk['text'];
	}

	if (chat.useV2) {
		// We run do_prompt multiple times; delete old prompt messages
		$(".chat-message").remove();

		for (const message of parseChatMessages(full_text)) {
			addMessage(message.author, message.text, -1, null, null);
		}
	} else {
		// Normal
		let elements_to_change = document.getElementsByClassName("var_sync_story_prompt");
		for (item of elements_to_change) {
			//clear out the item first
			while (item.firstChild) { 
				item.removeChild(item.firstChild);
			}
			for (chunk of data.value) {
				chunk_element = document.createElement("span");
				chunk_element.innerText = chunk['text'];
				if (chunk['WI matches'] != null) {
					chunk_element.classList.add("wi_match");
					chunk_element.setAttribute("tooltip", chunk['WI Text']);
					chunk_element.setAttribute("wi-uid", chunk['WI matches']);
				}
				item.append(chunk_element);
			}
			item.original_text = full_text;
			item.classList.remove("pulse");
			assign_world_info_to_action(-1, null);
		}
	}

	// Sometimes full text ends up not being built
	if (!full_text) {
	}
	actions_data[-1] = {'Selected Text': full_text};

	//if we have a prompt we need to disable the theme area, or enable it if we don't
	if (data.value[0].text != "") {
		document.getElementById('input_text').placeholder = "Enter text here (shift+enter for new line)";
		document.getElementById('themerow').classList.add("hidden");
		document.getElementById('themetext').value = "";
		document.getElementById("welcome_container").classList.add("hidden");
		//enable editing
		document.getElementById("Selected Text").setAttribute("contenteditable", "true");
	} else {
		document.getElementById('input_text').placeholder = "Enter Prompt Here (shift+enter for new line)";
		document.getElementById('input_text').disabled = false;
		document.getElementById('themerow').classList.remove("hidden");
		addInitChatMessage();
	}
	
}

function do_story_text_length_updates(action) {
	if (document.getElementById('Selected Text Chunk '+action.id)) {
		document.getElementById('Selected Text Chunk '+action.id).setAttribute("token_length", action.action["Selected Text Length"]);
	//} else {
		//console.log('Selected Text Chunk '+action.id);
		//console.log(action);
	}
	
}


function save_story() { socket.emit("save_story", null, response => save_as_story(response)); }
function load_story_list() { socket.emit("load_story_list", ""); }

function do_presets(data) {
	for (select of document.getElementsByClassName('presets')) {
		//clear out the preset list
		while (select.firstChild) {
			select.removeChild(select.firstChild);
		}
		//add our blank option
		var option = document.createElement("option");
		option.value="";
		option.text="Presets";
		select.append(option);
		presets = data.value;
		
		
		for (const [key, value] of Object.entries(data.value)) {
			var option_group = document.createElement("optgroup");
			option_group.label = key;
			option_group.classList.add("preset_group");
			for (const [group, group_value] of Object.entries(value)) {
				var option = document.createElement("option");
				option.text=group;
				option.disabled = true;
				option.classList.add("preset_group");
				option_group.append(option);
				for (const [preset, preset_value] of Object.entries(group_value)) {
					var option = document.createElement("option");
					option.value=preset_value['preset'];
					option.text=preset_value.preset;
					// Don't think we can use custom tooltip here (yet)
					option.title = preset_value.description;
					option_group.append(option);
				}
			}
			select.append(option_group);
		}
	}
}

function update_status_bar(data) {
	var percent_complete = data.value;
	var percent_bar = document.getElementsByClassName("statusbar_inner");
	document.getElementById('status_bar_percent').textContent = Math.round(percent_complete,1)+"%"
	for (item of percent_bar) {
		item.setAttribute("style", "width:"+percent_complete+"%");
		
		if ((percent_complete == 0) || (percent_complete == 100)) {
			item.parentElement.classList.add("hidden");
			document.getElementById("inputrow_container").classList.remove("status_bar");
		} else {
			item.parentElement.classList.remove("hidden");
			document.getElementById("inputrow_container").classList.add("status_bar");
		}
	}
	if ((percent_complete == 0) || (percent_complete == 100)) {
		updateTitle();
	} else {
		document.title = `(${percent_complete}%) KoboldAI Client`;
	}
}

function do_ai_busy(data) {
	if (data.value) {
		ai_busy_start = Date.now();
		favicon.start_swap()
		current_chunk = parseInt(document.getElementById("action_count").textContent)+1;
		if (document.getElementById("Select Options Chunk " + current_chunk)) {
			document.getElementById("Select Options Chunk " + current_chunk).classList.add("hidden")
		}
	} else {
		runtime = Date.now() - ai_busy_start;
		if (document.getElementById("Execution Time")) {
			document.getElementById("Execution Time").textContent = Math.round(runtime/1000).toString().toHHMMSS();
		}
		favicon.stop_swap()
		document.getElementById('btnsubmit').textContent = "Submit";
		for (item of document.getElementsByClassName('statusbar_outer')) {
			item.classList.add("hidden");
		}
		if (document.getElementById("user_beep_on_complete").checked) {
			beep();
		}
	}
}

function hide_show_prompt() {
	const promptEl = $el("#story_prompt");
	if (!promptEl) return;

	if (Math.min.apply(null,Object.keys(actions_data).map(Number).filter(function(x){return x>=0})) == Infinity) {
		promptEl.classList.remove("hidden");
	} else if (Math.min.apply(null,Object.keys(actions_data).map(Number).filter(function(x){return x>=0})) > 0) {
		//we have actions and our minimum action we have in the UI is above the start of the game
		//we need to keep the story prompt hidden
		promptEl.classList.add("hidden");
	} else {
		promptEl.classList.remove("hidden");
	}
}

function var_changed(data) {
	//if (data.name == "sp") {
	//	console.log({"name": data.name, "data": data});
	//}

	for (const entry of sync_hooks) {
		if (data.classname !== entry.class) continue;
		if (data.name !== entry.name) continue;
		entry.func(data.value);
	}
	
	if (data.name in vars_sync_time) {
		if (vars_sync_time[data.name] > Date.parse(data.transmit_time)) {
			return;
		}
	}
	vars_sync_time[data.name] = Date.parse(data.transmit_time);

	if ((data.classname == 'actions') && (data.name == 'Action Count')) {
		current_action = data.value;
		if (document.getElementsByClassName("action_image")[0]) {
			document.getElementsByClassName("action_image")[0].setAttribute("chunk", data.value);
		}
		if (current_action <= 0) {
			//console.log("setting action_count to "+current_action);
			const storyPrompt = $el("#story_prompt");
			if (storyPrompt) storyPrompt.classList.remove("hidden");
			scroll_trigger_element = undefined;
			document.getElementById("Selected Text").onscroll = undefined;
		}
		hide_show_prompt();
	}

	if (data.classname === "story" && data.name === "story_id") story_id = data.value;
	
	if ((data.classname == 'story') && (data.name == 'privacy_mode')) {
		privacy_mode(data.value);
	}

	if (data.classname === "story" && data.name === "storymode") {
		story.mode = data.value;
		updateChatStyle();
	} else if (data.classname == "story" && data.name == "chat_style") {
		chat.style = data.value;
		updateChatStyle();
	}
	
	if ((data.classname == "user") && (data.name == "ui_level")) {
		set_ui_level(data.value);
	}
	
	//Special Case for Actions
	if ((data.classname == "story") && (data.name == "actions")) {
		process_actions_data(data)
	//Special Case for Presets
	} else if ((data.classname == 'model') && (data.name == 'presets')) {
		do_presets(data);
	//Special Case for prompt
	} else if ((data.classname == 'story') && (data.name == 'prompt_wi_highlighted_text')) {
		do_prompt(data);
	//Special Case for phrase biasing
	} else if ((data.classname == 'story') && (data.name == 'biases')) {
		do_biases(data);
	//Special Case for substitutions
	} else if ((data.classname == 'story') && (data.name == 'substitutions')) {
		load_substitutions(data.value);
	//Special Case for sample_order
	} else if ((data.classname == 'model') && (data.name == 'sampler_order')) {
		for (const [index, item] of data.value.entries()) {
			Array.from(document.getElementsByClassName("sample_order"))[index].textContent = map2.get(item);
		}
	//Special Case for SP
	} else if ((data.classname == 'system') && (data.name == 'splist')) {
		item = document.getElementById("sp");
		while (item.firstChild) {
			item.removeChild(item.firstChild);
		}
		option = document.createElement("option");
		option.textContent = "Not in Use";
		option.value = "";
		item.append(option);
		for (sp of data.value) {
			option = document.createElement("option");
			option.textContent = sp[1][0];
			option.value = sp[0];
			option.setAttribute("title", sp[1][1]);
			item.append(option);
		}
	//Special case for context viewer
	} else if (data.classname == "story" && data.name == "context") {
		update_context(data.value);
	//special case for story_actionmode
	} else if (data.classname == "story" && data.name == "actionmode") {
		const button = document.getElementById('adventure_mode');
		if (data.value == 1) {
			button.childNodes[1].textContent = "Adventure";
		} else {
			button.childNodes[1].textContent = "Story";
		}
	//Special Case for story picture
	} else if (data.classname == "story" && data.name == "picture") {
		image_area = document.getElementById("action image");

		let maybeImage = image_area.getElementsByClassName("action_image")[0];
		if (maybeImage) maybeImage.remove();

		$el("#image-loading").classList.add("hidden");

		if (data.value != "") {
			var image = new Image();
			image.src = 'data:image/png;base64,'+data.value;
			image.classList.add("action_image");
			image.setAttribute("context-menu", "generated-image");
			image.addEventListener("click", imgGenView);
			image_area.appendChild(image);
		}
	}  else if (data.classname == "story" && data.name == "picture_prompt") {
		if (data.value) document.getElementById("action image").setAttribute("tooltip", data.value);
	//special case for welcome text since we want to allow HTML
	} else if (data.classname == 'model' && data.name == 'welcome') {
		document.getElementById('welcome_text').innerHTML = data.value;
	//Basic Data Syncing
	} else {
		var elements_to_change = document.getElementsByClassName("var_sync_"+data.classname.replace(" ", "_")+"_"+data.name.replace(" ", "_"));
		for (item of elements_to_change) {
			if (Array.isArray(data.value)) {
				if (item.tagName.toLowerCase() === 'select') {
					while (item.firstChild) {
						item.removeChild(item.firstChild);
					}
					for (option_data of data.value) {
						option = document.createElement("option");
						option.textContent = option_data;
						option.value = option_data;
						item.append(option);
					}
					if (item.id == 'selected_theme') {
						//Fix the select box
						theme = getCookie("theme", "monochrome");
						for (element of item.childNodes) {
							if (element.value == theme) {
								element.selected = true;
							}
						}
					}
				} else if (item.tagName.toLowerCase() === 'input') {
					item.value = fix_text(data.value);
				} else {
					item.textContent = fix_text(data.value);
				}
			} else {
				if ((item.tagName.toLowerCase() === 'input') || (item.tagName.toLowerCase() === 'select')) {
					if (item.getAttribute("type") == "checkbox") {
						if (item.checked != data.value) {
							//not sure why the bootstrap-toggle won't respect a standard item.checked = true/false, so....
							item.parentNode.click();
						}
					} else {
						item.value = fix_text(data.value);
					}
				} else {
					item.textContent = fix_text(data.value);
				}
			}

			// TODO: Add old value and new value to detail?
			item.dispatchEvent(new CustomEvent("sync", {
				detail: {},
				bubbles: false,
				cancelable: true,
				composed: false,
			}));
		}
		//alternative syncing method
		var elements_to_change = document.getElementsByClassName("var_sync_alt_"+data.classname.replace(" ", "_")+"_"+data.name.replace(" ", "_"));
		for (item of elements_to_change) {
			item.setAttribute(data.classname.replace(" ", "_")+"_"+data.name.replace(" ", "_"), fix_text(data.value));
		}
		
		
		
	}
	
	//if we changed the gen amount, make sure our option area is set/not set
	//if ((data.classname == 'model') && (data.name == 'numseqs')) {
	//	if (data.value == 1) {
	//		//allow our options to collapse to 0%, but no more than 30% (in case there is a redo or the like)
	//		var r = document.querySelector(':root');
	//		r.style.setProperty('--story_options_size', 'fit-content(30%)');
	//	} else {
	//		//static 30%
	//		var r = document.querySelector(':root');
	//		r.style.setProperty('--story_options_size', 'fit-content(30%)');
	//	}
	//}
	
	//if we're updating generated tokens, let's show that in our status bar
	if ((data.classname == 'model') && (data.name == 'tqdm_progress')) {
		update_status_bar(data);
	}
	
	//If we have ai_busy, start the favicon swapping
	if ((data.classname == 'system') && (data.name == 'aibusy')) {
		do_ai_busy(data);
	}
	
	//set the selected theme to the cookie value
	//if ((data.classname == "system") && (data.name == "theme_list")) {
	//	Change_Theme(getCookie("theme", "Monochrome"));
	//}
	
	//Set all options before the next chunk to hidden
	if ((data.classname == "actions") && (data.name == "Action Count")) {
		var option_container = document.getElementById("Select Options");
		var current_chunk = parseInt(document.getElementById("action_count").textContent)+1;
		
		var children = option_container.children;
		for (var i = 0; i < children.length; i++) {
			var chunk = children[i];
			if (chunk.id == "Select Options Chunk " + current_chunk) {
				chunk.classList.remove("hidden");
			} else {
				chunk.classList.add("hidden");
			}
		}
	}
	
	
	update_token_lengths();
}

function load_popup(data) {
	popup_deleteable = data.deleteable;
	popup_editable = data.editable;
	popup_renameable = data.renameable;
	rename_return_emit_name = data.rename_return_emit_name;
	var popup = document.getElementById("file-browser");
	var popup_title = document.getElementById("popup_title");
	popup_title.textContent = data.popup_title;
	if (data.popup_title == "Select Story to Load") {
		document.getElementById("import_story_button").classList.remove("hidden");
	} else {
		document.getElementById("import_story_button").classList.add("hidden");
	}
	var popup_list = document.getElementById("popup_list");
	//first, let's clear out our existing data
	while (popup_list.firstChild) {
		popup_list.removeChild(popup_list.firstChild);
	}
	var breadcrumbs = document.getElementById('popup_breadcrumbs');
	while (breadcrumbs.firstChild) {
		breadcrumbs.removeChild(breadcrumbs.firstChild);
	}
	
	if (data.upload) {
		const dropArea = document.getElementById('popup_list');
		dropArea.addEventListener('dragover', (event) => {
			event.stopPropagation();
			event.preventDefault();
			// Style the drag-and-drop as a "copy file" operation.
			event.dataTransfer.dropEffect = 'copy';
		});

		dropArea.addEventListener('drop', (event) => {
			event.stopPropagation();
			event.preventDefault();
			const fileList = event.dataTransfer.files;
			for (file of fileList) {
				reader = new FileReader();
				reader.onload = function (event) {
					socket.emit("upload_file", {'filename': file.name, "data": event.target.result, 'upload_no_save': true});
				};
				reader.readAsArrayBuffer(file);
			}
		});
	} else {
		
	}
	
	openPopup("file-browser");
	
	//adjust accept button
	if (data.call_back == "") {
		document.getElementById("popup_load_cancel").classList.add("hidden");
	} else {
		document.getElementById("popup_load_cancel").classList.remove("hidden");
		var accept = document.getElementById("popup_accept");
		accept.classList.add("disabled");
		accept.setAttribute("emit", data.call_back);
		accept.setAttribute("selected_value", "");
		accept.onclick = function () {
								socket.emit(this.getAttribute("emit"), this.getAttribute("selected_value"));
								closePopups();
						  };
	}
					  
}

function redrawPopup() {
	// This is its own function as it's used to show newly-sorted rows as well.

	// Clear old items, not anything else
	$(".item").remove();

	// Create lines
	for (const row of popup_rows) {
		let tr = document.createElement("div");
		tr.classList.add("item");
		tr.setAttribute("folder", row.isFolder);
		tr.setAttribute("valid", row.isValid);
		tr.style = popup_style;

		let icon_area = document.createElement("span");
		icon_area.style = "grid-area: icons;";
		
		// Create the folder icon
		let folder_icon = document.createElement("span");
		folder_icon.classList.add("folder_icon");
		if (row.isFolder) {
			folder_icon.classList.add("oi");
			folder_icon.setAttribute('data-glyph', "folder");
		}
		icon_area.append(folder_icon);
		
		// Create the edit icon
		let edit_icon = document.createElement("span");
		edit_icon.classList.add("edit_icon");
		if (popup_editable && !row.isValid) {
			edit_icon.classList.add("oi");
			edit_icon.setAttribute('data-glyph', "spreadsheet");
			edit_icon.setAttribute("tooltip", "Edit");
			edit_icon.id = row.path;
			edit_icon.onclick = function () {
				socket.emit("popup_edit", this.id);
			};
		}
		icon_area.append(edit_icon);
		
		// Create the rename icon
		let rename_icon = document.createElement("span");
		rename_icon.classList.add("rename_icon");
		if (popup_renameable && !row.isFolder) {
			rename_icon.classList.add("oi");
			rename_icon.setAttribute('data-glyph', "pencil");
			rename_icon.setAttribute("tooltip", "Rename");
			rename_icon.id = row.path;
			rename_icon.setAttribute("filename", row.fileName);
			rename_icon.onclick = function () {
				let new_name = prompt("Please enter new filename for \n"+ this.getAttribute("filename"));
				if (new_name != null) {
					socket.emit(rename_return_emit_name, {"file": this.id, "new_name": new_name});
				}
			};
		}
		icon_area.append(rename_icon);
		
		// Create the delete icon
		let delete_icon = document.createElement("span");
		delete_icon.classList.add("delete_icon");
		if (popup_deleteable) {
			delete_icon.classList.add("oi");
			delete_icon.setAttribute('data-glyph', "x");
			delete_icon.setAttribute("tooltip", "Delete");
			delete_icon.id = row.path;
			delete_icon.setAttribute("folder", row.isFolder);
			delete_icon.onclick = function () {
				const message = this.getAttribute("folder") == "true" ?  "Do you really want to delete this folder and ALL files under it?" : "Do you really want to delete this file?";
				const delId = this.id;

				deleteConfirmation(
					[{text: message}],
					confirmText="Go for it.",
					denyText="I've changed my mind!",
					confirmCallback=function() {
						socket.emit("popup_delete", delId);
					}
				);
			};
		}
		icon_area.append(delete_icon);
		tr.append(icon_area);
		
		//create the actual item
		let gridIndex = 0;
		if (row.showFilename) {
			let popup_item = document.createElement("span");
			popup_item.style = `overflow-x: hidden; grid-area: file;`;

			popup_item.id = row.path;
			popup_item.setAttribute("folder", row.isFolder);
			popup_item.setAttribute("valid", row.isValid);
			popup_item.textContent = row.fileName;
			popup_item.onclick = function () {
				let accept = document.getElementById("popup_accept");

				if (this.getAttribute("valid") == "true") {
					accept.classList.remove("disabled");
					accept.disabled = false;
					accept.setAttribute("selected_value", this.id);
				} else {
					accept.setAttribute("selected_value", "");
					accept.classList.add("disabled");
					accept.disabled = true;
					if (this.getAttribute("folder") == "true") {
						socket.emit("popup_change_folder", this.id);
					}
				}

				let popup_list = document.getElementById('popup_list').getElementsByClassName("selected");
				for (const item of popup_list) {
					item.classList.remove("selected");
				}
				this.parentElement.classList.add("selected");
			};
			tr.append(popup_item);
		}
		
		let dataIndex = -1;
		for (const columnName of Object.keys(row.data)) {
			const dataValue = row.data[columnName];

			let td = document.createElement("span");
			td.style = `overflow-x: hidden; grid-area: p${gridIndex};`;

			gridIndex += 1;
			dataIndex++;

			td.id = row.path;
			td.setAttribute("folder", row.isFolder);
			td.setAttribute("valid", row.isValid);

			if (columnName === "Last Loaded") {
				let timestamp = parseInt(dataValue);

				if (timestamp) {
					// Date expects unix timestamps to be in milligilliaseconds or something
					const date = new Date(timestamp * 1000)
					td.textContent = date.toLocaleString();
				}
			} else {
				td.textContent = dataValue;
			}

			td.onclick = function () {
				let accept = document.getElementById("popup_accept");
				if (this.getAttribute("valid") == "true") {
					accept.classList.remove("disabled");
					accept.disabled = false;
					accept.setAttribute("selected_value", this.id);
				} else {
					accept.setAttribute("selected_value", "");
					accept.classList.add("disabled");
					accept.disabled = true;
					if (this.getAttribute("folder") == "true") {
						socket.emit("popup_change_folder", this.id);
					}
				}

				let popup_list = document.getElementById('popup_list').getElementsByClassName("selected");
				for (item of popup_list) {
					item.classList.remove("selected");
				}
				this.parentElement.classList.add("selected");
			};
			tr.append(td);
		}

		popup_list.append(tr);
	}
}

function sortPopup(key) {
	// Nullify the others
	for (const sKey of Object.keys(popup_sort)) {
		if (sKey === key) continue;
		popup_sort[sKey] = null;

		document.getElementById(`sort-icon-${sKey.toLowerCase().replaceAll(" ", "-")}`).innerText = "filter_list";
	}

	// True is asc, false is asc
	let sortState = !popup_sort[key];

	popup_sort[key] = sortState;

	popup_rows.sort(function(x, y) {
		let xDat = x.data[key];
		let yDat = y.data[key];

		if (typeof xDat == "string" && typeof yDat == "string") return xDat.toLowerCase().localeCompare(yDat.toLowerCase());

		if (xDat < yDat) return -1;
		if (xDat > yDat) return 1;
		return 0;
	});
	if (sortState) popup_rows.reverse();
	
	// Change icons
	let icon = document.getElementById(`sort-icon-${key.toLowerCase().replaceAll(" ", "-")}`);
	icon.innerText = sortState ? "arrow_drop_up" : "arrow_drop_down";

	redrawPopup();
}

function popup_items(data) {
	// The data as we recieve it is not very fit for sorting, let's do some cleaning.
	popup_rows = [];
	popup_sort = {};

	for (const name of data.column_names) {
		popup_sort[name] = null;
	}

	for (const item of data.items) {
		let itemData = item[4];
		let dataRow = {data: {}};

		for (const i in data.column_names) {
			dataRow.data[data.column_names[i]] = itemData[i];
		}

		dataRow.isFolder = item[0];
		dataRow.path = item[1];
		dataRow.fileName = item[2];
		dataRow.isValid = item[3];
		dataRow.showFilename = data.show_filename;

		popup_rows.push(dataRow);
	}


	var popup_list = document.getElementById('popup_list');
	//first, let's clear out our existing data
	while (popup_list.firstChild) {
		popup_list.removeChild(popup_list.firstChild);
	}
	document.getElementById('popup_upload_input').value = "";
	
	//create the column widths
	popup_style = 'display: grid; grid-template-areas: "icons';
	if (data.show_filename) {
		popup_style += ' file';
	}
	for (let i=0; i < data.column_widths.length; i++) {
		popup_style += ` p${i}`;
	}

	popup_style += '"; grid-template-columns: 50px';
	if (data.show_filename) {
		popup_style += ' 100px';
	}
	for (const column_width of data.column_widths) {
		popup_style += " "+column_width;
	}
	popup_style += ';';
	
	//create titles
	var tr = document.createElement("div");
	tr.style = popup_style;
	tr.classList.add("header");
	//icon area
	var td = document.createElement("span");
	td.style = "grid-area: icons;";
	tr.append(td)
	
	//add dynamic columns
	var i = 0;
	if (data.show_filename) {
		td = document.createElement("span");
		td.textContent = "File Name";
		td.classList.add("table-header-container")
		td.style = "overflow-x: hidden; grid-area: file;";
		tr.append(td)
	}

	for (const columnName of data.column_names) {
		const container = document.createElement("div");
		const td = document.createElement("span");
		const icon = document.createElement("span");

		container.addEventListener("click", function(c) {
			return (function() {
				sortPopup(c);
			});
		}(columnName));
		container.classList.add("table-header-container")
		container.style = `overflow-x: hidden; grid-area: p${i};`;

		td.classList.add("table-header-label");
		td.textContent = columnName;

		// TODO: Better unsorted icon
		icon.id = `sort-icon-${columnName.toLowerCase().replaceAll(" ", "-")}`;
		icon.innerText = "filter_list";
		icon.classList.add("material-icons-outlined");
		icon.classList.add("table-header-sort-icon");


		container.appendChild(document.createElement("spacer-dummy"));
		container.appendChild(td);
		container.appendChild(icon);

		tr.append(container);

		i++;
	}
	popup_list.append(tr);
	
	redrawPopup();
	
}

function popup_breadcrumbs(data) {
	var breadcrumbs = document.getElementById('popup_breadcrumbs')
	while (breadcrumbs.firstChild) {
		breadcrumbs.removeChild(breadcrumbs.firstChild);
	}
	
	for (item of data) {
		var button = document.createElement("button");
		button.id = item[0];
		button.textContent = item[1];
		button.classList.add("breadcrumbitem");
		button.onclick = function () {
							socket.emit("popup_change_folder", this.id);
					  };
		breadcrumbs.append(button);
		var span = document.createElement("span");
		span.textContent = "\\";
		breadcrumbs.append(span);
	}
}

function popup_edit_file(data) {
	var popup_list = document.getElementById('popup_list');
	//first, let's clear out our existing data
	while (popup_list.firstChild) {
		popup_list.removeChild(popup_list.firstChild);
	}
	var accept = document.getElementById("popup_accept");
	accept.setAttribute("selected_value", "");
	accept.onclick = function () {
							var textarea = document.getElementById("filecontents");
							socket.emit("popup_change_file", {"file": textarea.getAttribute("filename"), "data": textarea.value});
							closePopups();
					  };
	
	var textarea = document.createElement("textarea");
	textarea.classList.add("fullwidth");
	textarea.rows = 25;
	textarea.id = "filecontents"
	textarea.setAttribute("filename", data.file);
	textarea.value = data.text;
	textarea.onblur = function () {
						var accept = document.getElementById("popup_accept");
						accept.classList.remove("disabled");
					};
	popup_list.append(textarea);
	
}

function error_popup(data) {
	alert(data);
}

function oai_engines(data) {
	var oaimodel = document.getElementById("oaimodel")
	oaimodel.classList.remove("hidden")
	selected_item = 0;
	length = oaimodel.options.length;
	for (let i = 0; i < length; i++) {
		oaimodel.options.remove(1);
	}
	for (item of data.data) {
		var option = document.createElement("option");
		option.value = item[0];
		option.text = item[1];
		if(data.online_model == item[0]) {
			option.selected = true;
		}
		oaimodel.appendChild(option);
	}
}

function getModelParameterCount(modelName) {
	if (!modelName) return null;

	// The "T" and "K" may be a little optimistic...
	let paramsString = modelName.toUpperCase().match(/[\d.]+[TBMK]/)
	if (!paramsString) return null;
	paramsString = paramsString[0];

	let base = parseFloat(paramsString);
	let multiplier = {T: 1_000_000_000_000, B: 1_000_000_000, M: 1_000_000, K: 1_000}[paramsString[paramsString.length - 1]];

	return base * multiplier;
}

function show_model_menu(data) {
	//clear out the loadmodelsettings
	var loadmodelsettings = document.getElementById('loadmodelsettings')
	while (loadmodelsettings.firstChild) {
		loadmodelsettings.removeChild(loadmodelsettings.firstChild);
	}
	//Clear out plugin selector
	var model_plugin = document.getElementById('modelplugin');
	while (model_plugin.firstChild) {
		model_plugin.removeChild(model_plugin.firstChild);
	}
	model_plugin.classList.add("hidden");
	var accept = document.getElementById("btn_loadmodelaccept");
	accept.disabled = false;
	
	//clear out the breadcrumbs
	var breadcrumbs = document.getElementById('loadmodellistbreadcrumbs')
	while (breadcrumbs.firstChild) {
		breadcrumbs.removeChild(breadcrumbs.firstChild);
	}
	
	//add breadcrumbs
	if ('breadcrumbs' in data) {
		for (item of data.breadcrumbs) {
			var button = document.createElement("button");
			button.classList.add("breadcrumbitem");
			button.setAttribute("model", data.menu);
			button.setAttribute("folder", item[0]);
			button.textContent = item[1];
			button.onclick = function () {
						socket.emit('select_model', {'menu': "", 'name': this.getAttribute("model"), 'path': this.getAttribute("folder")});
					};
			breadcrumbs.append(button);
			var span = document.createElement("span");
			span.textContent = "\\";
			breadcrumbs.append(span);
		}
	}
	//clear out the items
	var model_list = document.getElementById('loadmodellistcontent')
	while (model_list.firstChild) {
		model_list.removeChild(model_list.firstChild);
	}
	//add items
	for (item of data.items) {
		var list_item = document.createElement("span");
		list_item.classList.add("model_item");
		
		//create the folder icon
		var folder_icon = document.createElement("span");
		folder_icon.classList.add("material-icons-outlined");
		folder_icon.classList.add("cursor");

		let isModel = !(
			item.isMenu ||
			item.label === "Load a model from its directory" ||
			item.label === "Load an old GPT-2 model (eg CloverEdition)"
		);

		folder_icon.textContent = isModel ? "psychology" : "folder";
		list_item.append(folder_icon);
		
		
		//create the actual item
		var popup_item = document.createElement("span");
		popup_item.classList.add("model");
		for (const key in item) {
			if (key == "name") {
				popup_item.id = item[key];
			} 
			popup_item.setAttribute(key, item[key]);
		}
		
		popup_item.onclick = function() { 
			var attributes = this.attributes;
			var obj = {};

			for (var i = 0, len = attributes.length; i < len; i++) {
				obj[attributes[i].name] = attributes[i].value;
			}
			//put the model data on the accept button so we can send it to the server when you accept
			var accept = document.getElementById("popup_accept");
			selected_model_data = obj;
			//send the data to the server so it can figure out what data we need from the user for the model
			socket.emit('select_model', obj); 
			
			//clear out the selected item and select this one visually
			for (const element of document.getElementsByClassName("model_menu_selected")) {
				element.classList.remove("model_menu_selected");
			}
			this.closest(".model_item").classList.add("model_menu_selected");
		}
		
		//name text
		var text = document.createElement("span");
		text.style="grid-area: item;";
		text.textContent = item.label;
		popup_item.append(text);
		//model size text
		var text = document.createElement("span");
		text.textContent = item.size;
		text.style="grid-area: gpu_size;padding: 2px;";
		popup_item.append(text);

		(function() {
			// Anon function to avoid unreasonable indentation
			if (!isModel) return;

			let parameterCount = getModelParameterCount(item.label);
			if (!parameterCount) return;

			let warningText = "";

			if (parameterCount > 25_000_000_000) warningText = "This is a very high-end model and will likely not run without a specialized setup."; // 25B
			if (parameterCount < 2_000_000_000) warningText = "This is a lower-end model and may perform poorly.";			// 2B
			if (parameterCount < 1_000_000_000) warningText = "This is a very low-end model and may perform incoherently.";	// 1B

			if (!warningText) return;
			$e("span", list_item, {
				classes: ["material-icons-outlined", "model-size-warning"],
				innerText: "warning",
				"style.grid-area": "warning_icon",
				tooltip: warningText
			});

		})();

		(function() {
			// Anon function to avoid unreasonable indentation
			if (!item.isDownloaded) return;
			if (!isModel) return;

			$e("span", list_item, {
				classes: ["material-icons-outlined", "model-download-notification"],
				innerText: "download_done",
				"style.grid-area": "downloaded_icon",
				tooltip: "This model is already downloaded."
			});
		})();
		
		list_item.append(popup_item);
		model_list.append(list_item);
	}
	
	
	openPopup("load-model");
	
}

function getOptions(id){
  let selectElement = document.getElementById(id);
  let optionNames = [...selectElement.options].map(o => o.text);
  return optionNames;
}

function model_settings_checker() {
	//get check value:
	missing_element = false;
	if (this.check_data != null) {
		if ('sum' in this.check_data) {
			check_value = 0
			for (const temp of this.check_data['sum']) {
				if (document.getElementById(this.id.split("|")[0] +"|"  + temp + "_value")) {
					check_value += parseInt(document.getElementById(this.id.split("|")[0] +"|"  + temp + "_value").value);
				} else {
					missing_element = true;
				}
			}
		} else {
			check_value = this.value
		}
		if (this.check_data['check'] == "=") {
			valid = (check_value == this.check_data['value']);
		} else if (this.check_data['check'] == "!=") {
			valid = (check_value != this.check_data['value']);
		} else if (this.check_data['check'] == ">=") {
			valid = (check_value >= this.check_data['value']);
		} else if (this.check_data['check'] == "<=") {	
			valid = (check_value <= this.check_data['value']);
		} else if (this.check_data['check'] == "<=") {	
			valid = (check_value > this.check_data['value']);
		} else if (this.check_data['check'] == "<=") {	
			valid = (check_value < this.check_data['value']);
		}
		if (valid || missing_element) {
			//if we are supposed to refresh when this value changes we'll resubmit
			if ((this.getAttribute("refresh_model_inputs") == "true") && !missing_element && !this.noresubmit) {
				//get an object of all the input settings from the user
				data = {}
				settings_area = document.getElementById(document.getElementById("modelplugin").value + "_settings_area");
				if (settings_area) {
					for (const element of settings_area.querySelectorAll(".model_settings_input:not(.hidden)")) {
						var element_data = element.value;
						if (element.getAttribute("data_type") == "int") {
							element_data = parseInt(element_data);
						} else if (element.getAttribute("data_type") == "float") {
							element_data = parseFloat(element_data);
						} else if (element.getAttribute("data_type") == "bool") {
							element_data = element.checked;
						}
						data[element.id.split("|")[1].replace("_value", "")] = element_data;
					}
				}
				data = {...data, ...selected_model_data};
				
				data['plugin'] = document.getElementById("modelplugin").value;
				data['valid_backends'] = getOptions("modelplugin");
				
				socket.emit("resubmit_model_info", data);
			}
			if ('sum' in this.check_data) {
				for (const temp of this.check_data['sum']) {
					if (document.getElementById(this.id.split("|")[0] +"|"  + temp + "_value")) {
						document.getElementById(this.id.split("|")[0] +"|"  + temp + "_value").closest(".setting_container_model").classList.remove('input_error');
						document.getElementById(this.id.split("|")[0] +"|"  + temp + "_value").closest(".setting_container_model").removeAttribute("tooltip");
					}
				}
			} else {
				this.closest(".setting_container_model").classList.remove('input_error');
				this.closest(".setting_container_model").removeAttribute("tooltip");
			}
		} else {
			if ('sum' in this.check_data) {
				for (const temp of this.check_data['sum']) {
					if (document.getElementById(this.id.split("|")[0] +"|"  + temp + "_value")) {
						document.getElementById(this.id.split("|")[0] +"|"  + temp + "_value").closest(".setting_container_model").classList.add('input_error');
						if (this.check_data['check_message']) {
							document.getElementById(this.id.split("|")[0] +"|"  + temp + "_value").closest(".setting_container_model").setAttribute("tooltip", this.check_data['check_message']);
						} else {
							document.getElementById(this.id.split("|")[0] +"|"  + temp + "_value").closest(".setting_container_model").removeAttribute("tooltip");
						}
					}
				}
			} else {
				this.closest(".setting_container_model").classList.add('input_error');
				if (this.check_data['check_message']) {
					this.closest(".setting_container_model").setAttribute("tooltip", this.check_data['check_message']);
				} else {
					this.closest(".setting_container_model").removeAttribute("tooltip");
				}
			}
		}
	}
	var accept = document.getElementById("btn_loadmodelaccept");
	ok_to_load = true;
	for (const item of document.getElementsByClassName("input_error")) {
		if (item.classList.contains("input_error") && !item.closest(".model_plugin_settings_area").classList.contains("hidden")) {
			ok_to_load = false;
			break;
		}
	}
	
	if (ok_to_load) {
		accept.classList.remove("disabled");
		accept.disabled = false;
	} else {
		accept.classList.add("disabled");
		accept.disabled = true;
	}
	
	
	//We now have valid display boxes potentially. We'll go through them and update the display
	for (const item of document.querySelectorAll(".model_settings_valid_display:not(#blank_model_settings_valid_display)")) {
		check_value = 0
		missing_element = false;
		for (const temp of item.check_data['sum']) {
			if (document.getElementById(item.id.split("|")[0] +"|"  + temp + "_value")) {
				check_value += parseInt(document.getElementById(item.id.split("|")[0] +"|"  + temp + "_value").value);
			} else {
				missing_element = true;
			}
		}
		if (!missing_element) {
			item.innerText = item.original_text.replace("%1", check_value);
		}
		
		
	}
}

function set_toggle(id) {
	$('#'+id).bootstrapToggle({size: "mini", onstyle: "success", toggle: "toggle"});
}

var temp;
function selected_model_info(sent_data) {
	temp = sent_data;
	const data = sent_data['model_backends'];
	//clear out the loadmodelsettings
	var loadmodelsettings = document.getElementById('loadmodelsettings')
	while (loadmodelsettings.firstChild) {
		loadmodelsettings.removeChild(loadmodelsettings.firstChild);
	}
	//Clear out plugin selector
	var model_plugin = document.getElementById('modelplugin');
	while (model_plugin.firstChild) {
		model_plugin.removeChild(model_plugin.firstChild);
	}
	
	var accept = document.getElementById("btn_loadmodelaccept");
	accept.disabled = false;
	
	modelplugin = document.getElementById("modelplugin");
	modelplugin.classList.remove("hidden");
	modelplugin.onchange = function () {
		for (const area of document.getElementsByClassName("model_plugin_settings_area")) {
				area.classList.add("hidden");
		}
		if (document.getElementById(this.value + "_settings_area")) {
			document.getElementById(this.value + "_settings_area").classList.remove("hidden");
		}
		model_settings_checker()
	}
	//create the content
	for (const [loader, items] of Object.entries(data)) {
		model_area = document.createElement("DIV");
		model_area.id = loader + "_settings_area";
		model_area.classList.add("model_plugin_settings_area");
		model_area.classList.add("hidden");
		modelpluginoption = document.createElement("option");
		modelpluginoption.innerText = loader;
		modelpluginoption.value = loader;
		modelplugin.append(modelpluginoption);
		
		//create the user input for each requested input
		for (item of items) {
			let new_setting = document.getElementById('blank_model_settings').cloneNode(true);
			new_setting.id = loader;
			new_setting.classList.remove("hidden");
			new_setting.querySelector('#blank_model_settings_label').innerText = item['label'];
			new_setting.querySelector('#blank_model_settings_tooltip').setAttribute("tooltip", item['tooltip']);
			
			onchange_event = model_settings_checker;
			if (item['uitype'] == "slider") {
				var slider_number = new_setting.querySelector('#blank_model_settings_value_slider_number');
				slider_number.value = item['default'];
				slider_number.id = loader + "|" + item['id'] + "_value_text";
				slider_number.onchange = function() { document.getElementById(this.id.replace("_text", "")).value = this.value;};

				var slider = new_setting.querySelector('#blank_model_settings_slider');
				slider.value = item['default'];
				slider.min = item['min'];
				slider.max = item['max'];
				slider.setAttribute("data_type", item['unit']);
				slider.id = loader + "|" + item['id'] + "_value";
				if ('check' in item) {
					slider.check_data = item['check'];
					slider_number.check_data = item['check'];
				} else {
					slider.check_data = null;
					slider_number.check_data = null;
				}
				slider.oninput = function() { document.getElementById(this.id+"_text").value = this.value;};
				slider.onchange = onchange_event;
				slider.setAttribute("refresh_model_inputs", item['refresh_model_inputs']);
				new_setting.querySelector('#blank_model_settings_min_label').innerText = item['min'];
				new_setting.querySelector('#blank_model_settings_max_label').innerText = item['max'];
				slider.noresubmit = true;
				slider.onchange();
				slider.noresubmit = false;
			} else {
				new_setting.querySelector('#blank_model_settings_slider').remove();
			}
			if (item['uitype'] == "toggle") {
				toggle = document.createElement("input");
				toggle.type='checkbox';
				toggle.classList.add("setting_item_input");
				toggle.classList.add("blank_model_settings_input");
				toggle.classList.add("model_settings_input");
				toggle.id = loader + "|" + item['id'] + "_value";
				toggle.checked = item['default'];
				toggle.onclick = onchange_event;
				toggle.setAttribute("data_type", item['unit']);
				toggle.setAttribute("refresh_model_inputs", item['refresh_model_inputs']);
				if ('check' in item) {
					toggle.check_data = item['check'];
				} else {
					toggle.check_data = null;
				}
				new_setting.querySelector('#blank_model_settings_toggle').append(toggle);
				setTimeout(set_toggle, 200, loader + "\\|" + item['id'] + "_value");
				toggle.noresubmit = true;
				toggle.onclick();
				toggle.noresubmit = false;
			} else {
				new_setting.querySelector('#blank_model_settings_toggle').remove();
			}
			if (item['uitype'] == "dropdown") {
				var select_element = new_setting.querySelector('#blank_model_settings_dropdown');
				select_element.id = loader + "|" + item['id'] + "_value";
				for (const dropdown_value of item['children']) {
					new_option = document.createElement("option");
					new_option.value = dropdown_value['value'];
					new_option.innerText = dropdown_value['text'];
					select_element.append(new_option);
				}
				select_element.value = item['default'];
				select_element.setAttribute("data_type", item['unit']);
				select_element.onchange = onchange_event;
				select_element.setAttribute("refresh_model_inputs", item['refresh_model_inputs']);
				if (('multiple' in item) && (item['multiple'])) {
					select_element.multiple = true;
					select_element.size = 10;
				}
				if ('check' in item) {
					select_element.check_data = item['check'];
				} else {
					select_element.check_data = null;
				}
				select_element.noresubmit = true;
				select_element.onchange();
				select_element.noresubmit = false;
			} else {
				new_setting.querySelector('#blank_model_settings_dropdown').remove();
			}
			if (item['uitype'] == "password") {
				var password_item = new_setting.querySelector('#blank_model_settings_password');
				password_item.id = loader + "|" + item['id'] + "_value";
				password_item.value = item['default'];
				password_item.setAttribute("data_type", item['unit']);
				password_item.onchange = onchange_event;
				password_item.setAttribute("refresh_model_inputs", item['refresh_model_inputs']);
				if ('check' in item) {
					password_item.check_data = item['check'];
				} else {
					password_item.check_data = null;
				}
				password_item.noresubmit = true;
				password_item.onchange();
				password_item.noresubmit = false;
			} else {
				new_setting.querySelector('#blank_model_settings_password').remove();
			}
			if (item['uitype'] == "text") {
				var text_item = new_setting.querySelector('#blank_model_settings_text');
				text_item.id = loader + "|" + item['id'] + "_value";
				text_item.value = item['default'];
				text_item.onchange = onchange_event;
				text_item.setAttribute("data_type", item['unit']);
				text_item.setAttribute("refresh_model_inputs", item['refresh_model_inputs']);
				if ('check' in item) {
					text_item.check_data = item['check'];
				} else {
					text_item.check_data = null;
				}
				text_item.noresubmit = true;
				text_item.onchange();
				text_item.noresubmit = false;
			} else {
				new_setting.querySelector('#blank_model_settings_text').remove();
			}
			
			if (item['uitype'] == "Valid Display") {
				new_setting = document.createElement("DIV");
				new_setting.classList.add("model_settings_valid_display");
				new_setting.id = loader + "|" + item['id'] + "_value";
				new_setting.innerText = item['label'];
				new_setting.check_data = item['check'];
				new_setting.original_text = item['label'];
			}
			
			model_area.append(new_setting);
			loadmodelsettings.append(model_area);
		}
	}
	
	if ('selected_model_backend' in sent_data) {
		document.getElementById("modelplugin").value = sent_data['selected_model_backend'];
	}
	
	//unhide the first plugin settings
	if (document.getElementById(document.getElementById("modelplugin").value + "_settings_area")) {
		document.getElementById(document.getElementById("modelplugin").value + "_settings_area").classList.remove("hidden");
	}
	
	model_settings_checker()
	
}

function update_gpu_layers() {
	var gpu_layers
	gpu_layers = 0;
	for (let i=0; i < document.getElementById("gpu_count").value; i++) {
		gpu_layers += parseInt(document.getElementById("gpu_layers_"+i).value);
	}
	if (document.getElementById("disk_layers")) {
		gpu_layers += parseInt(document.getElementById("disk_layers").value);
	}
	if (gpu_layers > parseInt(document.getElementById("gpu_layers_max").textContent)) {
		document.getElementById("gpu_layers_current").textContent = gpu_layers;
		document.getElementById("gpu_layers_current").classList.add("text_red");
		var accept = document.getElementById("btn_loadmodelaccept");
		accept.classList.add("disabled");
	} else {
		var accept = document.getElementById("btn_loadmodelaccept");
		accept.classList.remove("disabled");
		document.getElementById("gpu_layers_current").textContent = gpu_layers;
		document.getElementById("gpu_layers_current").classList.remove("text_red");
	}
}

function load_model() {
	var accept = document.getElementById('btn_loadmodelaccept');
	settings_area = document.getElementById(document.getElementById("modelplugin").value + "_settings_area");
	
	//get an object of all the input settings from the user
	data = {}
	if (settings_area) {
		for (const element of settings_area.querySelectorAll(".model_settings_input:not(.hidden)")) {
			var element_data = element.getAttribute("data_type") === "bool" ? element.checked : element.value;
			if ((element.tagName == "SELECT") && (element.multiple)) {
				element_data = [];
				for (var i=0, iLen=element.options.length; i<iLen; i++) {
					if (element.options[i].selected) {
						element_data.push(element.options[i].value);
					}
				}
			} else {
				if (element.getAttribute("data_type") == "int") {
					element_data = parseInt(element_data);
				} else if (element.getAttribute("data_type") == "float") {
					element_data = parseFloat(element_data);
				}
			}
			data[element.id.split("|")[1].replace("_value", "")] = element_data;
		}
	}
	data = {...data, ...selected_model_data};
	
	data['plugin'] = document.getElementById("modelplugin").value;
	
	socket.emit("load_model", data);
	closePopups();
}

function world_info_entry_used_in_game(data) {
	if (!(data.uid in world_info_data)) {
		world_info_data[data.uid] = {};
	}
	world_info_data[data.uid]['used_in_game'] = data['used_in_game'];
	world_info_card = document.getElementById("world_info_"+data.uid);
	if (world_info_card) {
		if (data.used_in_game) {
			world_info_card.classList.add("used_in_game");
		} else {
			world_info_card.classList.remove("used_in_game");
		}
	}
}

function process_world_info_entry(data) {
	let temp = []
	if (Array.isArray(data)) {
		temp = data;
	} else {
		temp = [data];
	}
	for (wi of temp) {
		world_info_entry(wi);
	}
}

function world_info_entry(data) {
	world_info_data[data.uid] = data;
	
	//First let's get the id of the element we're on so we can restore it after removing the object
	var original_focus = document.activeElement.id;
	
	if (!(document.getElementById("world_info_folder_"+data.folder))) {
		folder = document.createElement("div");
		//console.log("Didn't find folder " + data.folder);
	} else {
		folder = document.getElementById("world_info_folder_"+data.folder);
	}
	
	if (document.getElementById("world_info_"+data.uid)) {
		world_info_card = document.getElementById("world_info_"+data.uid);
	} else {
		world_info_card_template = document.getElementById("world_info_");
		world_info_card = world_info_card_template.cloneNode(true);
		world_info_card.id = "world_info_"+data.uid;
		world_info_card.setAttribute("uid", data.uid);
		folder.append(world_info_card);
	}
	if (data.used_in_game) {
		world_info_card.classList.add("used_in_game");
	} else {
		world_info_card.classList.remove("used_in_game");
	}
	const title = world_info_card.querySelector('.world_info_title');
	title.id = "world_info_title_"+data.uid;
	title.textContent = data.title;
	title.setAttribute("uid", data.uid);
	title.setAttribute("original_text", data.title);
	title.setAttribute("contenteditable", true);
	title.classList.remove("pulse");
	title.ondragstart=function() {event.preventDefault();event.stopPropagation();};
	title.onblur = function () {
				this.parentElement.parentElement.setAttribute('draggable', 'true');
				this.setAttribute('draggable', 'true');
				if (this.textContent != this.getAttribute("original_text")) {
					world_info_data[this.getAttribute('uid')]['title'] = this.textContent;
					send_world_info(this.getAttribute('uid'));
					this.classList.add("pulse");
				}
			}
	world_info_card.addEventListener('dragstart', dragStart);
	world_info_card.addEventListener('dragend', dragend);
	title.addEventListener('dragenter', dragEnter)
	title.addEventListener('dragover', dragOver);
	title.addEventListener('dragleave', dragLeave);
	title.addEventListener('drop', drop);
	delete_icon = world_info_card.querySelector('.world_info_delete');
	delete_icon.id = "world_info_delete_"+data.uid;
	delete_icon.setAttribute("uid", data.uid);
	delete_icon.setAttribute("wi-title", data.title);
	delete_icon.onclick = function () {
		const wiTitle = this.getAttribute("wi-title");
		const wiUid = parseInt(this.getAttribute("uid"));
		const wiElement = this.parentElement.parentElement;
		deleteConfirmation([
				{text: "You're about to delete World Info entry "},
				{text: wiTitle, format: "bold"},
				{text: ". Are you alright with this?"},
			],
			confirmText="Go for it.",
			denyText="I've changed my mind!",
			confirmCallback=function() {
				if (wiUid < 0) {
					wiElement.remove();
				} else {
					socket.emit("delete_world_info", wiUid);
				}
			}
		);
	}

	const wiImgContainer = world_info_card.querySelector(".world_info_image_container");
	const wiImg = wiImgContainer.querySelector(".world_info_image");
	const wiImgPlaceholder = wiImgContainer.querySelector(".placeholder");
	const wiImgInput = $e("input", null, {type: "file", accept: "image/png,image/x-png,image/gif,image/jpeg"});

	wiImg.id = `world_info_image_${data.uid}`;
	wiImg.setAttribute("context-menu", "wi-img");

	wiImg.addEventListener("load", function() {
		wiImgPlaceholder.classList.add("hidden");
		wiImg.classList.remove("hidden");
		setChatPfps(title.innerText, wiImg.src);
	});

	wiImg.addEventListener("error", function() {
		wiImg.classList.add("hidden");
	});

	// Story id is used to invalidate cache from other stories
	if (data.uid > -1) wiImg.src = `/get_wi_image/${data.uid}?${story_id}`;

	wiImgContainer.addEventListener("click", function() {
		wiImgInput.click();
	});

	wiImgInput.addEventListener("change", function() {
		const file = wiImgInput.files[0];
		if (file.type.split("/")[0] !== "image") {
			reportError("Unable to upload WI image", `File type ${file.type} is not a compatible image type!`)
			return;
		}
		let objectUrl = URL.createObjectURL(file);
		wiImgPlaceholder.classList.add("hidden");
		wiImg.src = objectUrl;

		let reader = new FileReader();
		reader.addEventListener("loadend", async function() {
			let r = await fetch(`/set_wi_image/${data.uid}`, {
				method: "POST",
				body: reader.result
			});

			setChatPfps(title.innerText, reader.result);
		});
		reader.readAsDataURL(file);
	});

	const wiTypeSelector = world_info_card.querySelector(".world_info_type");

	// We may want to change the display names of these later
	wiTypeSelector.value = {
		chatcharacter: "Chat Character",
		wi: "Keywords",
		constant: "Always On",
		commentator: "Commentator",
	}[world_info_data[data.uid].type];

	wiTypeSelector.classList.remove("pulse");
	wiTypeSelector.addEventListener("change", function(event) {
		// If no change, don't do anything. Don't loop!!!
		if (world_info_data[data.uid].type === wiTypeSelector.value) {
			return;
		}

		switch (wiTypeSelector.value) {
			case "Chat Character":
				world_info_data[data.uid].constant = true;
				break;
			case "Always On":
				world_info_data[data.uid].constant = true;
				break;
			case "Keywords":
				world_info_data[data.uid].constant = false;
				break;
			case "Commentator":
				world_info_data[data.uid].constant = true;
				break;
			default:
				reportError("Error", `Unknown WI type ${wiTypeSelector.value}`);
				return;
		}
		world_info_data[data.uid].type = {
			"Chat Character": "chatcharacter",
			"Always On": "constant",
			"Keywords": "wi",
			"Commentator": "commentator",
		}[wiTypeSelector.value];
		send_world_info(data.uid);
		this.classList.add("pulse");
	})

	tags = world_info_card.querySelector('.world_info_tag_primary_area');
	tags.id = "world_info_tags_"+data.uid;
	//add tag content here
	add_tags(tags, data);
	
	secondarytags = world_info_card.querySelector('.world_info_tag_secondary_area');
	secondarytags.id = "world_info_secondtags_"+data.uid;
	//add second tag content here
	add_secondary_tags(secondarytags, data);
	//w++ toggle
	wpp_toggle_area = world_info_card.querySelector('.world_info_wpp_toggle_area');
	wpp_toggle_area.id = "world_info_wpp_toggle_area_"+data.uid;
	if (document.getElementById("world_info_wpp_toggle_"+data.uid)) {
		wpp_toggle = document.getElementById("world_info_wpp_toggle_"+data.uid);
	} else {
		wpp_toggle = document.createElement("input");
		wpp_toggle.id = "world_info_wpp_toggle_"+data.uid;
		wpp_toggle.setAttribute("type", "checkbox");
		wpp_toggle.setAttribute("uid", data.uid);
		wpp_toggle.setAttribute("data-size", "mini");
		wpp_toggle.setAttribute("data-onstyle", "success"); 
		wpp_toggle.setAttribute("data-toggle", "toggle");
		wpp_toggle.onchange = function () {
								if (this.checked) {
									document.getElementById("world_info_wpp_area_"+this.getAttribute('uid')).classList.remove("hidden");
									document.getElementById("world_info_basic_text_"+this.getAttribute('uid')).classList.add("hidden");
								} else {
									document.getElementById("world_info_wpp_area_"+this.getAttribute('uid')).classList.add("hidden");
									document.getElementById("world_info_basic_text_"+this.getAttribute('uid')).classList.remove("hidden");
								}
								
								world_info_data[this.getAttribute('uid')]['use_wpp'] = this.checked;
								send_world_info(this.getAttribute('uid'));
								this.classList.add("pulse");
							}
		wpp_toggle_area.append(wpp_toggle);
	}
	wpp_toggle.checked = data.use_wpp;
	wpp_toggle.classList.remove("pulse");
	
	//w++ data
	let last_new_value = null
	world_info_wpp_area = world_info_card.querySelector('.world_info_wpp_area');
	world_info_wpp_area.id = "world_info_wpp_area_"+data.uid;
	world_info_wpp_area.setAttribute("uid", data.uid);
	wpp_attributes_area = world_info_card.querySelector('.wpp_attributes_area');
	while (wpp_attributes_area.firstChild) { 
		wpp_attributes_area.removeChild(wpp_attributes_area.firstChild);
	}
	wpp_format = world_info_card.querySelector('.wpp_format');
	wpp_format.id = "wpp_format_"+data.uid;
	wpp_format.setAttribute("uid", data.uid);
	wpp_format.setAttribute("data_type", "format");
	wpp_format.onchange = function () {
							do_wpp(this.parentElement);
						}
	if (data.wpp.format == "W++") {
		wpp_format.selectedIndex = 0;
	} else {
		wpp_format.selectedIndex = 1;
	}
	wpp_type = world_info_card.querySelector('.wpp_type');
	wpp_type.id = "wpp_type_"+data.uid;
	wpp_type.setAttribute("uid", data.uid);
	wpp_type.setAttribute("data_type", "type");
	wpp_type.value = data.wpp.type;
	wpp_name = world_info_card.querySelector('.wpp_name');
	wpp_name.id = "wpp_name_"+data.uid;
	wpp_name.setAttribute("uid", data.uid);
	wpp_name.setAttribute("data_type", "name");
	if ("wpp" in data) {
		wpp_name.value = data.wpp.name;
	}
	if ('attributes' in data.wpp) {
		i = -1;
		for (const [attribute, values] of Object.entries(data.wpp.attributes)) {
			if (attribute != '') {
				i += 1;
				attribute_area = document.createElement("div");
				let label = document.createElement("span");
				label.textContent = "\xa0\xa0\xa0\xa0Attribute: ";
				attribute_area.append(label);
				input = document.createElement("input");
				input.setAttribute("contenteditable", true);
				input.ondragstart=function() {event.preventDefault();event.stopPropagation();};
				input.onfocus=function() {this.parentElement.parentElement.parentElement.setAttribute('draggable', 'false');this.setAttribute('draggable', 'false');};
				input.onblur=function() {this.parentElement.parentElement.parentElement.setAttribute('draggable', 'true');this.setAttribute('draggable', 'true');};
				input.ondragstart=function() {event.preventDefault();event.stopPropagation();};
				input.value = attribute;
				input.type = "text";
				input.setAttribute("uid", data.uid);
				input.setAttribute("data_type", "attribute");
				input.id = "wpp_"+data.uid+"_attr_"+i
				input.onchange = function() {do_wpp(this.parentElement.parentElement.parentElement)};
				attribute_area.append(input);
				wpp_attributes_area.append(attribute_area);
				j=-1;
				for (value of values) {
					j+=1;
					value_area = document.createElement("div");
					label = document.createElement("span");
					label.textContent = "\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0Value: ";
					value_area.append(label);
					input = document.createElement("input");
					input.type = "text";
					input.setAttribute("contenteditable", true);
					input.ondragstart=function() {event.preventDefault();event.stopPropagation();};
					input.onchange = function() {do_wpp(this.parentElement.parentElement.parentElement)};
					input.onfocus=function() {this.parentElement.parentElement.parentElement.setAttribute('draggable', 'false');this.setAttribute('draggable', 'false');};
					input.onblur=function() {this.parentElement.parentElement.parentElement.setAttribute('draggable', 'true');this.setAttribute('draggable', 'true');};
					input.ondragstart=function() {event.preventDefault();event.stopPropagation();};
					input.value = value;
					input.setAttribute("uid", data.uid);
					input.setAttribute("data_type", "value");
					input.id = "wpp_"+data.uid+"_value_"+i+"_"+j;
					value_area.append(input);
					wpp_attributes_area.append(value_area);
				}
				value_area = document.createElement("div");
				label = document.createElement("span");
				label.textContent = "\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0Value: ";
				value_area.append(label);
				input = document.createElement("input");
				input.type = "text";
				input.setAttribute("contenteditable", true);
				input.ondragstart=function() {event.preventDefault();event.stopPropagation();};
				input.onfocus=function() {this.parentElement.parentElement.parentElement.setAttribute('draggable', 'false');this.setAttribute('draggable', 'false');};
				input.onblur=function() {this.parentElement.parentElement.parentElement.setAttribute('draggable', 'true');this.setAttribute('draggable', 'true');};
				input.ondragstart=function() {event.preventDefault();event.stopPropagation();};
				input.setAttribute("uid", data.uid);
				input.setAttribute("data_type", "value");
				input.id = "wpp_"+data.uid+"_value_"+i+"_blank";
				last_new_value = input;
				input.onchange = function() {if (this.value != "") {on_new_wi_item = this.id;do_wpp(this.parentElement.parentElement.parentElement)}};
				value_area.append(input);
				wpp_attributes_area.append(value_area);
			}
		}
	}
	attribute_area = document.createElement("div");
	let label = document.createElement("span");
	label.textContent = "\xa0\xa0\xa0\xa0Attribute: ";
	attribute_area.append(label);
	input = document.createElement("input");
	input.value = "";
	input.type = "text";
	input.setAttribute("contenteditable", true);
	input.ondragstart=function() {event.preventDefault();event.stopPropagation();};
	input.onfocus=function() {this.parentElement.parentElement.parentElement.setAttribute('draggable', 'false');this.setAttribute('draggable', 'false');};
	input.onblur=function() {this.parentElement.parentElement.parentElement.setAttribute('draggable', 'true');this.setAttribute('draggable', 'true');};
	input.setAttribute("uid", data.uid);
	input.setAttribute("value_num", i);
	input.setAttribute("data_type", "attribute");
	input.id = "wpp_"+data.uid+"_attr_blank";
	input.onchange = function() {if (this.value != "") {on_new_wi_item=this.id;do_wpp(this.parentElement.parentElement.parentElement)}};
	attribute_area.append(input);
	wpp_attributes_area.append(attribute_area);
	
	
	
	//regular data
	manual_text_area = world_info_card.querySelector('.world_info_basic_text_area');
	manual_text_area.id = "world_info_basic_text_"+data.uid;
	manual_text = world_info_card.querySelector('.world_info_entry_text');
	manual_text.id = "world_info_entry_text_"+data.uid;
	manual_text.setAttribute("uid", data.uid);
	manual_text.value = data.manual_text;
	manual_text.onchange = function () {
							world_info_data[this.getAttribute('uid')]['manual_text'] = this.value;
							send_world_info(this.getAttribute('uid'));
							this.classList.add("pulse");
						}
	manual_text.classList.remove("pulse");
	comment = world_info_card.querySelector('.world_info_comment');
	comment.id = "world_info_comment_"+data.uid;
	comment.setAttribute("uid", data.uid);
	comment.value = data.comment;
	comment.onchange = function () {
							world_info_data[this.getAttribute('uid')]['comment'] = this.textContent;
							send_world_info(this.getAttribute('uid'));
							this.classList.add("pulse");
						}
	comment.classList.remove("pulse");
						
	//Let's figure out the order to insert this card
	var found = false;
	var moved = false;
	for (var i = 0; i < world_info_folder_data[data.folder].length; i++) {
		//first find where our current card is in the list
		if (!(found)) {
			if (world_info_folder_data[data.folder][i] == data.uid) {
				found = true;
			}
		} else {
			//We have more folders, so let's see if any of them exist so we can insert before that
			if (document.getElementById("world_info_"+world_info_folder_data[data.folder][i])) {
				moved = true;
				folder.insertBefore(world_info_card, document.getElementById("world_info_"+world_info_folder_data[data.folder][i]));
				break;
			}
		}
	}
	if (!(found) | !(moved)) {
		folder.append(world_info_card);
	}
	
	//hide keys if constant set
	if (data.constant || data.type === "commentator") {
		document.getElementById("world_info_tags_"+data.uid).classList.add("hidden");
		document.getElementById("world_info_secondtags_"+data.uid).classList.add("hidden");
	} else {
		document.getElementById("world_info_tags_"+data.uid).classList.remove("hidden");
		document.getElementById("world_info_secondtags_"+data.uid).classList.remove("hidden");
	}

	const genTypeInput = world_info_card.querySelector(".world_info_item_type");
	const generateDescButton = world_info_card.querySelector(".wi-lc-text > .generate-button");
	generateDescButton.addEventListener("click", function() {
		if (generating_summary) return;
		let type = genTypeInput.innerText;

		if (!type) {
			genTypeInput.classList.add("bad-input");
			return;
		} else {
			genTypeInput.classList.remove("bad-input");
		}

		// TODO: Make type input element
		let genAmount = parseInt($el("#user_wigen_amount").value);
		generateWIData(data.uid, "desc", title.innerText, type, null, genAmount);
		this.innerText = "autorenew";
		this.classList.add("spinner");
		manual_text.classList.add("disabled");
	});

	genTypeInput.addEventListener("focus", function() {
		this.classList.remove("bad-input");
	});

	genTypeInput.addEventListener("keydown", function(event) {
		if (event.key === "Enter") {
			event.preventDefault();
			this.blur();
		}
	});

	genTypeInput.addEventListener("blur", function() {
		this.innerText = this.innerText.trim();

		if (this.innerText == this.getAttribute("old-text")) return;
		this.setAttribute("old-text", this.innerText);

		world_info_data[data.uid].object_type = this.innerText;
		send_world_info(data.uid);
	});

	genTypeInput.innerText = data.object_type;
	
	//$('#world_info_constant_'+data.uid).bootstrapToggle();
	//$('#world_info_wpp_toggle_'+data.uid).bootstrapToggle();
	setup_wi_toggles.push(data.uid);
	
	//hide/unhide w++
	if (wpp_toggle.checked) {
		document.getElementById("world_info_wpp_area_"+wpp_toggle.getAttribute('uid')).classList.remove("hidden");
		document.getElementById("world_info_basic_text_"+wpp_toggle.getAttribute('uid')).classList.add("hidden");
	} else {
		document.getElementById("world_info_wpp_area_"+wpp_toggle.getAttribute('uid')).classList.add("hidden");
		document.getElementById("world_info_basic_text_"+wpp_toggle.getAttribute('uid')).classList.remove("hidden");
	}
	
	//resize comments/text boxes
	autoResize(comment, 60);
	autoResize(manual_text, 60);
	
	//put focus back where it was
	if (original_focus && document.getElementById(original_focus)) {
		if (document.getElementById(original_focus).tagName != "BUTTON") {
			//check if we were on a new line
			if ((on_new_wi_item != null) && (document.getElementById(on_new_wi_item))) {
				//if we're on a new wpp attribute, we want to move to the new value not the new attribute, so let's fix that
				if (on_new_wi_item.includes('wpp_') && on_new_wi_item.includes('_attr_blank') && (last_new_value != null)) { 
					on_new_wi_item = last_new_value.id;
				}
				original_focus = on_new_wi_item;
			}
			on_new_wi_item = null;
			//for some reason we have to wrap this in a timmer
			setTimeout(function() {document.getElementById(original_focus.replace("-1", data.uid)).click();document.getElementById(original_focus.replace("-1", data.uid)).focus()}, 0);
		}
	}
	
	assign_world_info_to_action(null, data.uid);
	
	update_token_lengths();
	
	clearTimeout(setup_missing_wi_toggles_timeout);
	setup_missing_wi_toggles_timeout = setTimeout(setup_missing_wi_toggles, 10);
	
	return world_info_card;
}

function setup_missing_wi_toggles() {
	for (item of setup_wi_toggles) {
		$('#world_info_constant_'+item).bootstrapToggle();
		$('#world_info_wpp_toggle_'+item).bootstrapToggle();
	}
	setup_wi_toggles = [];
}

function world_info_folder(data) {
	//console.log(data);
	world_info_folder_data = data;
	var folders = Object.keys(data)
	for (var i = 0; i < folders.length; i++) {
		folder_name = folders[i];
		//check to see if folder exists
		if (!(document.getElementById("world_info_folder_"+folder_name))) {
			var folder = document.createElement("span");
			folder.id = "world_info_folder_"+folder_name;
			folder.classList.add("WI_Folder");
			title = document.createElement("h3");
			title.addEventListener('dragenter', dragEnter)
			title.addEventListener('dragover', dragOver);
			title.addEventListener('dragleave', dragLeave);
			title.addEventListener('drop', drop);
			title.classList.add("WI_Folder_Header");
			collapse_icon = document.createElement("span");
			collapse_icon.id = "world_info_folder_collapse_"+folder_name;
			collapse_icon.classList.add("wi_folder_collapser");
			collapse_icon.classList.add("material-icons-outlined");
			collapse_icon.setAttribute("folder", folder_name);
			collapse_icon.textContent = "expand_more";
			collapse_icon.onclick = function () {
								hide_wi_folder(this.getAttribute("folder"));
								document.getElementById('world_info_folder_expand_'+this.getAttribute("folder")).classList.remove('hidden');
								this.classList.add("hidden");
							};
			collapse_icon.classList.add("expand")
			title.append(collapse_icon);
			expand_icon = document.createElement("span");
			expand_icon.id = "world_info_folder_expand_"+folder_name;
			expand_icon.classList.add("wi_folder_collapser");
			expand_icon.classList.add("material-icons-outlined");
			expand_icon.setAttribute("folder", folder_name);
			expand_icon.textContent = "chevron_right";
			expand_icon.onclick = function () {
								unhide_wi_folder(this.getAttribute("folder"));
								document.getElementById('world_info_folder_collapse_'+this.getAttribute("folder")).classList.remove('hidden');
								this.classList.add("hidden");
							};
			expand_icon.classList.add("expand")
			expand_icon.classList.add("hidden");
			title.append(expand_icon);
			icon = document.createElement("span");
			icon.classList.add("material-icons-outlined");
			icon.setAttribute("folder", folder_name);
			icon.textContent = "folder";
			icon.classList.add("folder");
			title.append(icon);
			title_text = document.createElement("span");
			title_text.classList.add("wi_title");
			title_text.setAttribute("contenteditable", true);
			title_text.setAttribute("original_text", folder_name);
			title_text.textContent = folder_name;
			title_text.onblur = function () {
				if (this.textContent != this.getAttribute("original_text")) {
					//Need to check if the new folder name is already in use
					socket.emit("Rename_World_Info_Folder", {"old_folder": this.getAttribute("original_text"), "new_folder": this.textContent});
				}
			}
			title_text.classList.add("title");
			title.append(title_text);
			
			//create delete button
			delete_button = document.createElement("span");
			delete_button.classList.add("material-icons-outlined");
			delete_button.classList.add("cursor");
			delete_button.setAttribute("folder", folder_name);
			delete_button.textContent = "delete";
			delete_button.onclick = function () {
				const folderName = this.getAttribute("folder");
				deleteConfirmation([
						{text: "You're about to delete World Info folder "},
						{text: folderName, format: "bold"},
						{text: " and the "},
						{text: countWIFolderChildren(folderName), format: "bold"},
						{text: " entries inside it. Are you sure?"},
					],
					confirmText="Go for it.",
					denyText="I've changed my mind!",
					confirmCallback=function() { socket.emit("delete_wi_folder", folderName); }
				);
			};
			delete_button.classList.add("delete");
			title.append(delete_button);
			
			//create download button
			download = document.createElement("span");
			download.classList.add("material-icons-outlined");
			download.classList.add("cursor");
			download.setAttribute("folder", folder_name);
			download.textContent = "file_download";
			download.onclick = function () {
								document.getElementById('download_iframe').src = 'export_world_info_folder?folder='+this.getAttribute("folder");
							};
			download.classList.add("download");
			title.append(download);
			
			//upload element
			upload_element = document.createElement("input");
			upload_element.id = "wi_upload_element_"+folder_name;
			upload_element.type = "file";
			upload_element.setAttribute("folder", folder_name);
			upload_element.classList.add("upload_box");
			upload_element.onchange = function () {
											var fileList = this.files;
											for (file of fileList) {
												reader = new FileReader();
												reader.folder = this.getAttribute("folder");
												reader.onload = function (event) {
													socket.emit("upload_world_info_folder", {'folder': event.target.folder, 'filename': file.name, "data": event.target.result});
												};
												reader.readAsArrayBuffer(file);
												
											}
										};
			title.append(upload_element);
			
			//create upload button
			upload = document.createElement("span");
			upload.classList.add("material-icons-outlined");
			upload.classList.add("cursor");
			upload.setAttribute("folder", folder_name);
			upload.textContent = "file_upload";
			upload.onclick = function () {
								document.getElementById('wi_upload_element_'+this.getAttribute("folder")).click();
								//document.getElementById('download_iframe').src = 'export_world_info_folder?folder='+this.getAttribute("folder");
							};
			upload.classList.add("upload");
			title.append(upload);
			folder.append(title);
			
			//create add button
			new_icon = document.createElement("span");
			new_icon.classList.add("wi_add_button");
			add_icon = document.createElement("span");
			add_icon.classList.add("material-icons-outlined");
			add_icon.textContent = "post_add";
			new_icon.append(add_icon);
			add_text = document.createElement("span");
			add_text.textContent = "Add World Info Entry";
			add_text.classList.add("wi_add_text");
			add_text.setAttribute("folder", folder_name);
			new_icon.onclick = function() {
				create_new_wi_entry(this.querySelector(".wi_add_text").getAttribute("folder"));
			}
			new_icon.append(add_text);
			folder.append(new_icon);
			
			//We want to insert this folder before the next folder
			if (i+1 < folders.length) {
				//We have more folders, so let's see if any of them exist so we can insert before that
				var found = false;
				for (var j = i+1; j < folders.length; j++) {
					if (document.getElementById("world_info_folder_"+folders[j])) {
						found = true;
						document.getElementById("WI_Area").insertBefore(folder, document.getElementById("world_info_folder_"+folders[j]));
						break;
					}
				}
				if (!(found)) {
					if (document.getElementById("new_world_info_button")) {
						document.getElementById("WI_Area").insertBefore(folder, document.getElementById("new_world_info_button"));
					} else {
						document.getElementById("WI_Area").append(folder);
					}
				}
			} else {
				if (document.getElementById("new_world_info_button")) {
					document.getElementById("WI_Area").insertBefore(folder, document.getElementById("new_world_info_button"));
				} else {
					document.getElementById("WI_Area").append(folder);
				}
			}
		} else {
			folder = document.getElementById("world_info_folder_"+folder_name);
		}
		for (uid of world_info_folder_data[folder_name]) {
			if (document.getElementById("world_info_"+uid)) {
				item = document.getElementById("world_info_"+uid);
				item.classList.remove("pulse");
				if (item.parentElement != folder) {
					item.classList.remove("hidden");
					folder.append(item);
				}
			}
		}
	}
	//Delete unused folders
	for (item of document.getElementsByClassName("WI_Folder")) {
		if (!(item.id.replace("world_info_folder_", "") in world_info_folder_data)) {
			item.parentNode.removeChild(item);
		}
	}
	
	//Add new world info folder button
	if (!(document.getElementById("new_world_info_button"))) {
		add_folder = document.createElement("div");
		add_folder.id = "new_world_info_button";
		temp = document.createElement("h3");
		add_icon = document.createElement("span");
		icon = document.createElement("span");
		icon.classList.add("material-icons-outlined");
		icon.textContent = "create_new_folder";
		add_icon.append(icon);
		text_span = document.createElement("span");
		text_span.textContent = "Add World Info Folder";
		text_span.classList.add("wi_title");
		add_icon.onclick = function() {
										socket.emit("create_world_info_folder", {});
									  }
		add_icon.append(text_span);
		temp.append(add_icon);
		add_folder.append(temp);
		document.getElementById("WI_Area").append(add_folder);
	}
}

function show_error_message(data) {
	const error_box_data = $el("#error-popup").querySelector("#popup_list_area");
	//clear out the error box
	while (error_box_data.firstChild) {
		error_box_data.removeChild(error_box_data.firstChild);
	}
	if (Array.isArray(data)) {
		for (item of data) {
			$e("div", error_box_data, {'innerHTML': item, 'classes': ['console_text']});
			$e("br", error_box_data);
		}
	} else {
		//console.log(item);
		$e("div", error_box_data, {'innerHTML': data, 'classes': ['console_text']});
	}
	openPopup("error-popup");
}

function show_message(data) {
	const message_box_data = $el("#message-popup").querySelector("#popup_list_area");
	const message_box_title = $el("#message-popup").querySelector("#popup_title");
	const message_box_ok = $el("#message-popup").querySelector("#ok");
	//clear out the error box
	while (message_box_data.firstChild) {
		message_box_data.removeChild(message_box_data.firstChild);
	}
	$e("div", message_box_data, {'innerHTML': data['message'], 'classes': ['console_text']})
	message_box_title.innerText = data['title'];
	message_box_ok.setAttribute("message_id", data['id'])
	
	openPopup("message-popup");
}

function do_wpp(wpp_area) {
	wpp = {};
	wpp['attributes'] = {};
	uid = wpp_area.getAttribute("uid");
	attribute = "";
	wpp['format'] = document.getElementById("wpp_format_"+uid).value;
	for (input of wpp_area.querySelectorAll('input')) {
		if (input.getAttribute("data_type") == "name") {
			wpp['name'] = input.value;
		} else if (input.getAttribute("data_type") == "type") {
			wpp['type'] = input.value;
		} else if (input.getAttribute("data_type") == "attribute") {
			attribute = input.value;
			if (!(input.value in wpp['attributes']) && (input.value != "")) {
				wpp['attributes'][input.value] = [];
			} 
			
		} else if ((input.getAttribute("data_type") == "value") && (attribute != "")) {
			if (input.value != "") {
				wpp['attributes'][attribute].push(input.value);
			}
		}
	}
	world_info_data[uid]['wpp'] = wpp;
	send_world_info(uid);
}

function load_cookies(data) {
	colab_cookies = data;
	for (const cookie of Object.keys(colab_cookies)) {
		setCookie(cookie, colab_cookies[cookie]);
	}
	colab_cookies = null;
	wait_for_tweaks_load();
}

function wait_for_tweaks_load() {
	if (document.readyState === 'complete') {
		process_cookies();
	} else {
		clearTimeout(colab_cookie_timeout);
		colab_cookie_timeout = setTimeout(wait_for_tweaks_load, 1000);
	}
}

function process_log_message(full_data) {
	debug_info['aiserver errors'] = []
	for (data of full_data) {
		let level = data['record']['level']['name'];
		let message = data['record']['message'];
		let time = data['record']['time']['repr'];
		let full_log = data['text'];
		//log.push({'level': level, 'message': message, 'time': time, 'full_log': full_log});
		if (level == 'ERROR') {
			show_error_message(data['html']);
		}
		
		
		let temp = JSON.parse(JSON.stringify(data.record));
		debug_info['aiserver errors'].push(temp);
		
		//put log message in log popup
		const log_popup = document.getElementById('log-popup');
		const log_popup_data = log_popup.querySelector("#popup_list_area")
		//clear out the error box
		for (item of data['html']) {
			$e("div", log_popup_data, {'innerHTML': item, 'classes': ['console_text']})
			$e("br", log_popup_data)
		}
	}
}

//--------------------------------------------UI to Server Functions----------------------------------
function create_new_softprompt() {
	socket.emit("create_new_softprompt", {"sp_title": document.getElementById("sp_title").value,
										  "sp_prompt": document.getElementById("sp_prompt").value,
										  "sp_dataset": document.getElementById("sp_dataset").value,
										  "sp_author": document.getElementById("sp_author").value,
										  "sp_description": document.getElementById("sp_description").value
										});
	closePopups();
}

async function download_story_to_json() {
	//document.getElementById('download_iframe').src = 'json';
	downloaded = false;
}

async function download_story() {
	if (socket.connected) {
		try {
			let name = $el(".var_sync_story_story_name").innerText;
			let r = await fetch("story_download");
			downloadBlob(await r.blob(), `${name}.kaistory`);
			return;
		}
		catch(err) {
			console.error("Error in online download");
			console.error(err);
		}
	}

	console.warn("Online download failed! Using offline download...")

	/* Offline Download - Compile JSON file from what we have in ram */
	
	//first we're going to find all the var_sync_story_ classes used in the document.
	let allClasses = [];
	const allElements = document.querySelectorAll('*');

	for (let i = 0; i < allElements.length; i++) {
		let classes = allElements[i].classList;
		for (let j = 0; j < classes.length; j++) {
		if (!(allClasses.includes(classes[j].replace("var_sync_story_", ""))) && (classes[j].includes("var_sync_story_"))) {
			allClasses.push(classes[j].replace("var_sync_story_", ""));
		}
		}
	}
	
	//OK, now we're going to go through each of those classes and get the values from the elements
	let j = {}
	for (class_name of allClasses) {
		for (item of document.getElementsByClassName("var_sync_story_"+class_name)) {
			if (['INPUT', 'TEXTAREA', 'SELECT'].includes(item.tagName)) {
				if ((item.tagName == 'INPUT') && (item.type == "checkbox")) {
					j[class_name] = item.checked;
				} else {
					j[class_name] = item.value;
				}
			} else {
				j[class_name] = item.textContent;
			}
			break;
		}
	}
	
	//We'll add actions and world info data next
	let temp = JSON.parse(JSON.stringify(actions_data));
	delete temp[-1];
	j['actions'] = {'action_count': document.getElementById('action_count').textContent, 'actions': temp};
	j['worldinfo_v2'] = {'entries': world_info_data, 'folders': world_info_folder_data};
	
	//Biases
	let bias = {};
	for (item of document.getElementsByClassName('bias')) {
		let bias_phrase = item.querySelector(".bias_phrase").children[0].value;
		let bias_score = parseInt(item.querySelector(".bias_score").querySelector(".bias_slider_cur").textContent);
		let bias_comp_threshold = parseInt(item.querySelector(".bias_comp_threshold").querySelector(".bias_slider_cur").textContent);
		if (bias_phrase != "") {
			bias[bias_phrase] = [bias_score, bias_comp_threshold];
		}
	}
	j['biases'] = bias;
	
	//substitutions
	substitutions = [];
	for (item of document.getElementsByClassName('substitution-card')) {
		let target = item.children[0].querySelector(".target").value;
		let sub = item.children[1].querySelector(".target").value;
		let enabled = (item.children[1].querySelector(".material-icons-outlined").getAttribute("title") == 'Enabled');
		substitutions.push({'target': target, 'substitution': sub, 'enabled': enabled});
	}
	j['substitutions'] = substitutions;
	
	j['file_version'] = 2;
	j['gamestarted'] = true;
	
	downloadString(JSON.stringify(j), j['story_name']+".json")
}

function unload_userscripts() {
	files_to_unload = document.getElementById('loaded_userscripts');
	for (var i=0; i<files_to_unload.options.length; i++) {
		if (files_to_unload.options[i].selected) {
			socket.emit("unload_userscripts", files_to_unload.options[i].value);
		}
	}
}

function save_theme() {
	var [cssVars, rules] = getAllCSSVariableNames();
	for (const [key, value] of Object.entries(cssVars)) {
		if (document.getElementById(key)) {
			if (document.getElementById(key+"_select").value == "") {
				cssVars[key] = document.getElementById(key).value;
			} else {
				cssVars[key] = "var(--"+document.getElementById(key+"_select").value+")";
			}
			
		}
	}
	for (item of document.getElementsByClassName("Theme_Input")) {
		cssVars["--"+item.id] = item.value;
	}
	socket.emit("theme_change", {"name": document.getElementById("save_theme_name").value, "theme": cssVars, 'special_rules': rules});
	document.getElementById("save_theme_name").value = "";
	socket.emit('theme_list_refresh', '');
}

function move_sample(direction) {
	var previous = null;
	//console.log(direction);
	for (const [index, temp] of Array.from(document.getElementsByClassName("sample_order")).entries()) {
		if (temp.classList.contains("selected")) {
			if ((direction == 'up') && (index > 0)) {
				temp.parentElement.insertBefore(temp, previous);
				break;
			} else if ((direction == 'down') && (index+1 < Array.from(document.getElementsByClassName("sample_order")).length)) {
				temp.parentElement.insertBefore(temp, Array.from(document.getElementsByClassName("sample_order"))[index+2]);
				break;
			}
		}
		previous = temp;
	}
	var sample_order = []
	for (item of document.getElementsByClassName("sample_order")) {
		sample_order.push(map1.get(item.textContent));
	}
	socket.emit("var_change", {"ID": 'model_sampler_order', "value": sample_order});
}

function new_story() {
	//check if the story is saved
	if (document.getElementById('save_story').getAttribute('story_gamesaved') == "false") {
		//ask the user if they want to continue
		deleteConfirmation([
				{text: "You asked for a new story but your current story has not been saved. If you continue you will loose your changes."},
			],
			confirmText="Go for it.",
			denyText="I've changed my mind!",
			confirmCallback=function() {
				socket.emit('new_story', '');
			}
		);
	} else {
		socket.emit('new_story', '');
	}
}

function save_as_story(response) {
	if (response === "overwrite?") openPopup("save-confirm");
}

function save_bias() {
	var biases = {};
	//get all of our biases

	for (const biasCard of document.getElementsByClassName("bias_card")) {
		//phrase
		var phrase = biasCard.querySelector(".bias_phrase").value;
		if (!phrase) continue;
		
		//Score
		var score = parseFloat(biasCard.querySelector(".bias_score input").value);
		
		//completion threshold
		var compThreshold = parseInt(biasCard.querySelector(".bias_comp_threshold input").value);
		
		biases[phrase] = [score, compThreshold];
	}

	// Because of course JS couldn't just support comparison in a core type
	// that would be silly and foolish
	if (JSON.stringify(biases) === JSON.stringify(biases_data)) {
		// No changes. :(
		return;
	}

	biases_data = biases;
	console.info("saving biases", biases)

	//send the biases to the backend
	socket.emit("phrase_bias_update", biases);
}

function sync_to_server(item) {
	//get value
	let value = null;
	let name = null;

	if ((item.tagName.toLowerCase() === 'checkbox') || (item.tagName.toLowerCase() === 'input') || (item.tagName.toLowerCase() === 'select') || (item.tagName.toLowerCase() == 'textarea')) {
		if (item.getAttribute("type") == "checkbox") {
			value = item.checked;
		} else {
			value = item.value;
			if (item.classList.contains("sync_as_float")) value = parseFloat(value);
		}
	} else {
		value = item.innerText;
	}
	
	//get name
	for (classlist_name of item.classList) {
		if (!classlist_name.includes("var_sync_alt_") && classlist_name.includes("var_sync_")) {
			name = classlist_name.replace("var_sync_", "");
		}
	}
	
	if (name != null) {
		item.classList.add("pulse");
		//send to server with ack
		socket.emit("var_change", {"ID": name, "value": value}, (response) => {
			if ('status' in response) {
				if (response['status'] == 'Saved') {
					for (item of document.getElementsByClassName("var_sync_"+response['id'])) {
						item.classList.remove("pulse");
					}
				}
			}
		});
	}
}

function upload_file(file_box) {
	var fileList = file_box.files;
	for (file of fileList) {
		reader = new FileReader();
		reader.onload = function (event) {
			socket.emit("upload_file", {'filename': file.name, "data": event.target.result, 'upload_no_save': false});
		};
		reader.readAsArrayBuffer(file);
	}
}

function upload_file_without_save(file_box) {
	var fileList = file_box.files;
	for (file of fileList) {
		reader = new FileReader();
		reader.onload = function (event) {
			socket.emit("upload_file", {'filename': file.name, "data": event.target.result, 'upload_no_save': true});
		};
		reader.readAsArrayBuffer(file);
	}
}

function send_world_info(uid) {
	socket.emit("edit_world_info", world_info_data[uid]);
}

function save_tweaks() {
	let out = [];

	for (const tweakContainer of document.getElementsByClassName("tweak-container")) {
		let toggle = tweakContainer.querySelector("input");
		let path = tweakContainer.getAttribute("tweak-path");
		if (toggle.checked) out.push(path);
	}
	setCookie("enabledTweaks", JSON.stringify(out));
}

function load_tweaks() {
	
	let enabledTweaks = JSON.parse(getCookie("enabledTweaks", "[]"));

	for (const tweakContainer of document.getElementsByClassName("tweak-container")) {
		let toggle = tweakContainer.querySelector("input");
		let path = tweakContainer.getAttribute("tweak-path");
		if (enabledTweaks.includes(path)) $(toggle).bootstrapToggle("on");
	}
}

function toggle_adventure_mode(button) {
	if (button.textContent == "Mode: Story") {
		button.childNodes[1].textContent = "Adventure";
		var actionmode = 1
	} else {
		button.childNodes[1].textContent = "Story";
		var actionmode = 0
	}
	button.classList.add("pulse");
	socket.emit("var_change", {"ID": "story_actionmode", "value": actionmode}, (response) => {
			if ('status' in response) {
				if (response['status'] == 'Saved') {
					document.getElementById("adventure_mode").classList.remove("pulse");
				}
			}
		});
	
}

function set_edit(event) {
	//get the element sitting on
	var game_text = document.getElementById("Selected Text");
	if ((event.key === undefined) || (event.key == 'ArrowDown') || (event.key == 'ArrowUp') || (event.key == 'ArrowLeft') || (event.key == 'ArrowRight')) {
		var chunk = window.getSelection().anchorNode;
		while (chunk != game_text) {
			if ((chunk instanceof HTMLElement) && (chunk.hasAttribute("chunk"))) {
				break;
			}
			chunk = chunk.parentNode;
		}
		for (item of document.getElementsByClassName("editing")) {
			item.classList.remove("editing");
		}
		if (chunk != game_text) {
			chunk.classList.add("editing");
		}
	}
	return true;
}

function check_game_after_paste() {
	setTimeout(function() {savegametextchanges();}, 500);
}

function gametextwatcher(records) {
	//Here we want to take care of two possible events
	//User deleted an action. For this we'll restore the action and set it's text to "" and mark it as dirty
	//User changes text. For this we simply mark it as dirty
	var game_text = document.getElementById("Selected Text");
	for (const record of records) {
		if ((record.type === "childList") && (record.removedNodes.length > 0)) {
			for (const chunk of record.removedNodes) {
				//we've deleted a node. Let's find the chunk span and put it back
				//Skip over deletes that are not chunks
				if ((chunk instanceof HTMLElement) && (chunk.hasAttribute("chunk"))) {
					if (!document.getElementById("Selected Text Chunk " + chunk.getAttribute("chunk"))) {
						//Node was actually deleted. 
						if (!dirty_chunks.includes(chunk.getAttribute("chunk"))) {
							dirty_chunks.push(chunk.getAttribute("chunk"));
							//Stupid firefox sometimes looses focus as you type after deleting stuff. Fix that here
							var sel = window.getSelection();
							if (sel.anchorNode instanceof HTMLElement) {
								sel.anchorNode.focus();
							} else {
								game_text.focus();
							}
						}
					}
				}
			}
		}
		//get the actual chunk rather than the sub-node
		//console.log(record);
		var chunk = record.target;
		var found_chunk = false;
		while (chunk != game_text) {
			if (chunk) {
				if ((chunk instanceof HTMLElement) && (chunk.hasAttribute("chunk"))) {
					found_chunk = true;
					break;
				}
				chunk = chunk.parentNode;
			} else {
				break;
			}
		}
		if ((found_chunk) && (chunk.original_text != chunk.innerText)) {;
			if (!dirty_chunks.includes(chunk.getAttribute("chunk"))) {
				dirty_chunks.push(chunk.getAttribute("chunk"));
			}
		} else if ((record.addedNodes.length > 0) && !(found_chunk) && !(record.addedNodes[0] instanceof HTMLElement)) {
			if (!dirty_chunks.includes("game_text")) {
				dirty_chunks.push("game_text");
			}
		}
	}
}

function fix_dirty_game_text() {
	//This should get fired if we have deleted chunks or have added text outside of a node.
	//We wait until after the game text has lost focus to fix things otherwise it messes with typing
	var game_text = document.getElementById("Selected Text");
	//Fix missing story prompt
	if (dirty_chunks.includes("-1")) {
		if (!document.getElementById("story_prompt")) {
			story_prompt = document.createElement("span");
			story_prompt.id = "story_prompt";
			story_prompt.classList.add("var_sync_story_prompt");
			story_prompt.classList.add("var_sync_alt_story_prompt_in_ai");
			story_prompt.classList.add("rawtext");
			story_prompt.setAttribute("chunk", "-1");
			game_text.prepend(story_prompt);
		}
	}
	if (dirty_chunks.includes("game_text")) {
		dirty_chunks = dirty_chunks.filter(item => item != "game_text");
		console.log("Firing Fix messed up text");
		//Fixing text outside of chunks
		for (node of game_text.childNodes) {
			if ((!(node instanceof HTMLElement) || !node.hasAttribute("chunk")) && (node.textContent.trim() != "")) {
				//We have a text only node. It should be moved into the previous chunk if it is marked as dirty, next node if not and it's dirty, or the previous if neither is dirty
				var node_text = ""
				if (node instanceof HTMLElement) {
					node_text = node.innerText;
				} else {
					node_text = node.data;
				}
				if (!(node.nextElementSibling) || !(dirty_chunks.includes(node.nextElementSibling.getAttribute("chunk"))) || dirty_chunks.includes(node.previousElementSibling.getAttribute("chunk"))) {
					node.previousElementSibling.innerText = node.previousElementSibling.innerText + node_text;
					if (!dirty_chunks.includes(node.previousElementSibling.getAttribute("chunk"))) {
						dirty_chunks.push(node.previousElementSibling.getAttribute("chunk"));
					}
				} else {
					node.nextElementSibling.innerText = node.nextElementSibling.innerText + node_text;
				}
				
				//Looks like sometimes it splits the parent. Let's look for that and fix it too
				if (node.nextElementSibling && (node.nextElementSibling.getAttribute("chunk") == node.previousElementSibling.getAttribute("chunk"))) {
					node.previousElementSibling.innerText = node.previousElementSibling.innerText + node.nextElementSibling.innerText;
					node.nextElementSibling.remove();
				}
				node.remove();
			}
		}
	}
}

function savegametextchanges() {
	fix_dirty_game_text();
	for (item of document.getElementsByClassName("editing")) {
		item.classList.remove("editing");
	}
	if (dirty_chunks.length > 0) {
		console.log("Firing save");
	}
	for (const chunk_id of dirty_chunks) {
		if (chunk_id == -1) {
			chunk = document.getElementById("story_prompt");
		} else {
			chunk = document.getElementById("Selected Text Chunk " + chunk_id);
		}
		if (chunk) {
			update_game_text(parseInt(chunk.getAttribute("chunk")), chunk.innerText);
		} else {
			update_game_text(parseInt(chunk_id), "");
		}
	}
	dirty_chunks = [];
}


function update_game_text(id, new_text) {
	let temp = null;
	if (id == -1) {
		if (document.getElementById("story_prompt")) {
			temp = document.getElementById("story_prompt");
			temp.original_text = new_text;
			temp.classList.add("pulse");
			sync_to_server(temp);
		} else {
			socket.emit("var_change", {"ID": 'story_prompt', "value": new_text});
		}
	} else {
		if (document.getElementById("Selected Text Chunk " + id)) {
			temp = document.getElementById("Selected Text Chunk " + id);
			temp.original_text = new_text;
			temp.classList.add("pulse");
			socket.emit("Set Selected Text", {"id": id, "text": new_text});
		} else {
			socket.emit("Set Selected Text", {"id": id, "text": ""});
		}
	}
	
}

function save_preset() {
	socket.emit("save_new_preset", {"preset": document.getElementById("new_preset_name").value, "description": document.getElementById("new_preset_description").value});
	closePopups();
}

//--------------------------------------------General UI Functions------------------------------------
function put_cursor_at_element(element) {
	var range = document.createRange();
	var sel = window.getSelection();
	if ((element.lastChild) && (element.lastChild instanceof HTMLElement)) {
		range.setStart(element.lastChild, element.lastChild.innerText.length);
	} else {
		range.setStart(element, element.innerText.length);
	}
	range.collapse(true);
	sel.removeAllRanges();
	sel.addRange(range);

}

function set_ui_level(level) {
	for (classname of ['setting_container', 'setting_container_single', 'setting_container_single_wide', 'biasing', 'palette_area', 'advanced_theme']) {
		for (element of document.getElementsByClassName(classname)) {
			if (parseInt(element.getAttribute('ui_level')) <= level) {
				element.classList.remove("hidden");
			} else {
				element.classList.add("hidden");
			}
		}
	}
	for (category of document.getElementsByClassName('collapsable_header')) {
		hide = true;
		for (element of category.nextElementSibling.children) {
			if ((!element.classList.contains('help_text')) && (!element.classList.contains('hidden'))) {
				hide = false;
				break;
			}
		}
		if (hide) {
			category.classList.add("hidden");
			category.nextElementSibling.classList.add("hidden");
		} else {
			category.classList.remove("hidden");
			category.nextElementSibling.classList.remove("hidden");
		}
	}
}

function update_story_picture(chunk_id) {
	const image = document.getElementsByClassName("action_image")[0];
	if (!image) return;
	image.src = "/action_image?id=" + chunk_id + "&ignore="+new Date().getTime();
	image.setAttribute("chunk", chunk_id);
}

function privacy_mode(enabled) {
	if (enabled) {
		document.getElementById('SideMenu').classList.add("superblur");
		document.getElementById('main-grid').classList.add("superblur");
		document.getElementById('rightSideMenu').classList.add("superblur");
		openPopup("privacy_mode");
	} else {
		document.getElementById('SideMenu').classList.remove("superblur");
		document.getElementById('main-grid').classList.remove("superblur");
		document.getElementById('rightSideMenu').classList.remove("superblur");
		if (!$el("#privacy_mode").classList.contains("hidden")) closePopups();
		document.getElementById('privacy_password').value = "";
	}
}

function set_font_size(element) {
	new_font_size = element.value;
	var r = document.querySelector(':root');
	r.style.setProperty("--game_screen_font_size_adjustment", new_font_size);
	clearTimeout(font_size_cookie_timout);
	font_size_cookie_timout = setTimeout(function() {setCookie("font_size", new_font_size)}, 2000);
}

function push_selection_to_memory() {
	document.getElementById("memory").value += "\n" + getSelectionText();
	document.getElementById("memory").onchange();
}

function push_selection_to_world_info() {
	let menu = document.getElementById("rightSideMenu");
	if ((~menu.classList.contains("open")) && (~menu.classList.contains("pinned"))) {
		menu.classList.add("open");
	}
	document.getElementById("story_flyout_tab_wi").onclick();
	
	if (~("root" in world_info_folder_data)) {
		world_info_folder_data["root"] = [];
		world_info_folder(world_info_folder_data);
	}
	create_new_wi_entry("root");
	document.getElementById("world_info_entry_text_-1").value = getSelectionText();
}

function push_selection_to_phrase_bias() {
	let menu = document.getElementById("SideMenu");
	if ((~menu.classList.contains("open")) && (~menu.classList.contains("pinned"))) {
		menu.classList.add("open");
	}
	document.getElementById("settings_flyout_tab_settings").onclick();
	document.getElementById("empty_bias_phrase").value = getSelectionText();
	document.getElementById("empty_bias_phrase").scrollIntoView(false)
	document.getElementById("empty_bias_phrase").onchange()
}

function retry_from_here() {
	// TODO: Make this from the caret position (get_caret_position()) instead
	// of per action. Actions may start out well, but go off the rails later, so
	// we should be able to retry from any position.
	let chunk = null;
	for (element of document.getElementsByClassName("editing")) {
		if (element.id == 'story_prompt') {
			chunk = -1
		} else {
			chunk = parseInt(element.id.split(" ").at(-1));
		}
		element.classList.remove("editing");
	}
	if (chunk != null) {
		action_count = parseInt(document.getElementById("action_count").textContent);
		//console.log(chunk);
		for (let i = 0; i < (action_count-chunk); i++) {
			storyBack();
		}
		socket.emit('submit', {'data': "", 'theme': ""});
		document.getElementById('input_text').value = '';
		document.getElementById('themetext').value = '';
	}
}

function speak_audio(summonEvent) {
	let action_id = null;
	if (summonEvent.target.parentElement.id == "story_prompt") {
		action_id = -1;
	} else {
		action_id = summonEvent.target.parentElement.id.split(" ").at(-1);
	}
	if (action_id != null) {
		//console.log(chunk);
		document.getElementById("reader").src = "/audio?id="+action_id;
		document.getElementById("reader").setAttribute("action_id", action_id);
		document.getElementById("reader").play();
		document.getElementById("play_tts").textContent = "pause";
	}
}

function play_pause_tts() {
	if (document.getElementById("reader").paused) {
		document.getElementById("reader").play();
		document.getElementById("play_tts").textContent = "pause";
	} else {
		document.getElementById("reader").pause();
		document.getElementById("play_tts").textContent = "play_arrow";
	}
}

function stop_tts() {
	document.getElementById("reader").src="";
	document.getElementById("reader").src="/audio";
	document.getElementById("play_tts").textContent = "play_arrow";
	for (item of document.getElementsByClassName("tts_playing")) {
		item.classList.remove("tts_playing");
	}
}

function finished_tts() {
	next_action = parseInt(document.getElementById("reader").getAttribute("action_id"))+1;
	action_count = parseInt(document.getElementById("action_count").textContent);
	if (next_action-1 == "-1") {
		action = document.getElementById("story_prompt");
	} else {
		action = document.getElementById("Selected Text Chunk "+(next_action-1));
	}
	if (action) {
		action.classList.remove("tts_playing");
	}
	if (next_action <= action_count) {
		document.getElementById("reader").src = "/audio?id="+next_action;
		document.getElementById("reader").setAttribute("action_id", next_action);
		document.getElementById("reader").play();
		document.getElementById("play_tts").textContent = "pause";
	} else {
		document.getElementById("play_tts").textContent = "play_arrow";
	}
}

function tts_playing() {
	action_id = document.getElementById("reader").getAttribute("action_id");
	if (action_id == "-1") {
		action = document.getElementById("story_prompt");
	} else {
		action = document.getElementById("Selected Text Chunk "+action_id);
	}
	if (action) {
		action.classList.add("tts_playing");
	}
}

function view_selection_probabilities() {
	// Not quite sure how this should work yet. Probabilities are obviously on
	// the token level, which we have no UI representation of. There are other
	// token-level visualization features I'd like to implement (like something
	// for self-attention), so if that works out it might be best to have a
	// modifier key (i.e. alt) enter a "token selection mode" when held.
	console.log("Not implemented! :(");
}

function copy() {
	document.execCommand("copy");
}

function paste() {
	document.execCommand("paste");
}

function cut() {
	document.execCommand("cut");
}

function getSelectionText() {
    var text = "";
    var activeEl = document.activeElement;
    var activeElTagName = activeEl ? activeEl.tagName.toLowerCase() : null;
    if (
      (activeElTagName == "textarea") || (activeElTagName == "input" &&
      /^(?:text|search|password|tel|url)$/i.test(activeEl.type)) &&
      (typeof activeEl.selectionStart == "number")
    ) {
        text = activeEl.value.slice(activeEl.selectionStart, activeEl.selectionEnd);
    } else if (window.getSelection) {
        text = window.getSelection().toString();
    }
    return text;
}

function get_caret_position(target) {
	if (
		document.activeElement !== target &&
		!$.contains(target, document.activeElement)
	) return null;

	return getSelection().focusOffset;
}

function show_save_preset() {
	openPopup("save-preset");
}

function autoResize(element, min_size=200) {
	//console.log(min_size);
	element.style.height = 'auto';
	if (min_size > element.scrollHeight) {
		element.style.height = min_size + "px";
	} else {
		element.style.height = (element.scrollHeight + 5) + 'px';
	}
}


function calc_token_usage(
	soft_prompt_length,
	genre_length,
	memory_length,
	authors_note_length,
	prompt_length,
	game_text_length,
	world_info_length,
	submit_length
) {
	let total_tokens = parseInt(document.getElementById('model_max_length_cur').value);
	let unused_token_count = total_tokens - memory_length - authors_note_length - world_info_length - prompt_length - game_text_length - submit_length;

	const data = [
		{id: "soft_prompt_tokens", tokenCount: soft_prompt_length, label: "Soft Prompt"},
		{id: "genre_tokens", tokenCount: genre_length, label: "Genre"},
		{id: "memory_tokens", tokenCount: memory_length, label: "Memory"},
		{id: "authors_notes_tokens", tokenCount: authors_note_length, label: "Author's Note"},
		{id: "world_info_tokens", tokenCount: world_info_length, label: "World Info"},
		{id: "prompt_tokens", tokenCount: prompt_length, label: "Prompt"},
		{id: "game_text_tokens", tokenCount: game_text_length, label: "Game Text"},
		{id: "submit_tokens", tokenCount: submit_length, label: "Submit Text"},
		{id: "unused_tokens", tokenCount: unused_token_count, label: "Remaining"},
	]

	for (const dat of data) {
		const el = document.getElementById(dat.id);
		el.style.width = ((dat.tokenCount / total_tokens) * 100) + "%";
		el.setAttribute("tooltip", `${dat.label}: ${dat.tokenCount}`);
	}
}

function Change_Theme(theme) {
	setCookie("theme", theme);
	var elements_to_change = document.getElementsByClassName("var_sync_system_theme_list");
	for (item of elements_to_change) {
		for (element of item.childNodes) {
			if (element.value == theme) {
				element.selected = true;
			}
		}
	}

	const Acss  = document.getElementById("CSSTheme_A");
	const Bcss  = document.getElementById("CSSTheme_B");
	let new_css = 'CSSTheme_B';
	if (Bcss) {
		new_css = 'CSSTheme_A';
	}

	const css = $e(
		"link",
		document.head,
		{
			id: new_css,
			rel: "stylesheet",
			href: `/themes/${theme}.css`
		}
		
	);

	// We must wait for the style to load before we read it
	css.onload = function() {
		//Delete the old CSS item
		if (new_css == 'CSSTheme_A') {
			if (Bcss) {
				Bcss.remove();
			}
		} else {
			if (Acss) {
				Acss.remove();
			}
		}
		recolorTokens();
		create_theming_elements();
	}
}

function palette_color(item) {
	var r = document.querySelector(':root');
	r.style.setProperty("--"+item.id, item.value);
}

function getAllCSSVariableNames(styleSheets = document.styleSheets){
	let cssVars = {};
	let rules = [];
	// loop each stylesheet
	for(let i = 0; i < styleSheets.length; i++){
		// loop stylesheet's cssRules
		if ((styleSheets[i].href != null) && (styleSheets[i].href.includes("/themes/"))) {
		  //if we're in the theme css, grab all the non root variables in case there are som
		  for( let j = 0; j < styleSheets[i].cssRules.length; j++){
				if (styleSheets[i].cssRules[j].selectorText != ":root") {
					rules.push(styleSheets[i].cssRules[j].cssText);
				}
			}
		}
		try{ // try/catch used because 'hasOwnProperty' doesn't work
			for( let j = 0; j < styleSheets[i].cssRules.length; j++){
				try{
					// loop stylesheet's cssRules' style (property names)
					for(let k = 0; k < styleSheets[i].cssRules[j].style.length; k++){
						let name = styleSheets[i].cssRules[j].style[k];
						// test name for css variable signiture and uniqueness
						if(name.startsWith('--') && (styleSheets[i].ownerNode.id.includes("CSSTheme"))){
							let value = styleSheets[i].cssRules[j].style.getPropertyValue(name);
							value.replace(/(\r\n|\r|\n){2,}/g, '$1\n');
							value = value.replaceAll("\t", "").trim();
							cssVars[name] = value;
						}
					}
				} catch (error) {}
			}
		} catch (error) {}
	}
	return [cssVars, rules];
}

function create_theming_elements() {
	//console.log("Running theme editor");
	var [cssVars, rules] = getAllCSSVariableNames();
	palette_table = document.createElement("table");
	advanced_table = document.getElementById("advanced_theme_editor_table");
	theme_area = document.getElementById("Palette");
	theme_area.append(palette_table);
	
	//clear advanced_table
	while (advanced_table.firstChild) {
		advanced_table.removeChild(advanced_table.firstChild);
	}
	
	for (const [css_item, css_value] of Object.entries(cssVars)) {
		if (css_item.includes("_palette")) {
			if (document.getElementById(css_item.replace("--", ""))) {
				input = document.getElementById(css_item.replace("--", ""));
				input.setAttribute("title", css_item.replace("--", "").replace("_palette", ""));
				input.value = css_value;
			}
		} else {
			tr = document.createElement("tr");
			tr.style = "width:100%;";
			title = document.createElement("td");
			title.textContent = css_item.replace("--", "").replace("_palette", "");
			tr.append(title);
			select = document.createElement("select");
			select.style = "width: 150px; color:black;";
			var option = document.createElement("option");
			option.value="";
			option.text="User Value ->";
			select.append(option);
			select.id = css_item+"_select";
			select.onchange = function () {
									var r = document.querySelector(':root');
									r.style.setProperty(this.id.replace("_select", ""), this.value);
								}
			for (const [css_item2, css_value2] of Object.entries(cssVars)) {
			   if (css_item2 != css_item) {
					var option = document.createElement("option");
					option.value=css_item2;
					option.text=css_item2.replace("--", "");
					if (css_item2 == css_value.replace("var(", "").replace(")", "")) {
						option.selected = true;
					}
					select.append(option);
			   }
			}
			select_td = document.createElement("td");
			select_td.append(select);
			tr.append(select_td);
			td = document.createElement("td");
			tr.append(td);
			if (css_value.includes("#")) {
				input = document.createElement("input");
				input.setAttribute("type", "color");
				input.id = css_item;
				input.setAttribute("title", css_item.replace("--", "").replace("_palette", ""));
				input.value = css_value;
				input.onchange = function () {
									var r = document.querySelector(':root');
									r.style.setProperty(this.id, this.value);
								}
				td.append(input);
			} else {
			   input = document.createElement("input");
				input.setAttribute("type", "text");
				input.id = css_item;
				input.setAttribute("title", css_item.replace("--", "").replace("_palette", ""));
				if (select.value != css_value.replace("var(", "").replace(")", "")) {
					input.value = css_value;
				}
				input.onchange = function () {
									var r = document.querySelector(':root');
									r.style.setProperty(this.id, this.value);
								}
				td.append(input);
			}
			
			advanced_table.append(tr);
		}
	}
}

function select_sample(item) {
	for (temp of document.getElementsByClassName("sample_order")) {
		temp.classList.remove("selected");
	}
	item.classList.add("selected");
}

function toggle_setting_category(element) {
	item = element.nextSibling.nextSibling;
	if (item.classList.contains('hidden')) {
		item.classList.remove("hidden");
		element.firstChild.nextSibling.firstChild.textContent = "expand_more";
	} else {
		item.classList.add("hidden");
		element.firstChild.nextSibling.firstChild.textContent = "navigate_next";
	}
}

function preserve_game_space(preserve) {
	var r = document.querySelector(':root');
	if (preserve) {
		setCookie("preserve_game_space", "true");
		r.style.setProperty('--setting_menu_closed_width_no_pins_width', '0px');
		if (!(document.getElementById('preserve_game_space_setting').checked)) {
			//not sure why the bootstrap-toggle won't respect a standard item.checked = true/false, so....
			document.getElementById('preserve_game_space_setting').parentNode.click();
		}
		document.getElementById('preserve_game_space_setting').checked = true;
	} else {
		setCookie("preserve_game_space", "false");
		r.style.setProperty('--setting_menu_closed_width_no_pins_width', 'var(--flyout_menu_width)');
		if (document.getElementById('preserve_game_space_setting').checked) {
			//not sure why the bootstrap-toggle won't respect a standard item.checked = true/false, so....
			document.getElementById('preserve_game_space_setting').parentNode.click();
		}
		document.getElementById('preserve_game_space_setting').checked = false;
	}
}

function options_on_right(data) {
	var r = document.querySelector(':root');
	//console.log("Setting cookie to: "+data);
	if (data) {
		setCookie("options_on_right", "true");
		r.style.setProperty('--story_pinned_areas', 'var(--story_pinned_areas_right)');
		r.style.setProperty('--story_pinned_area_widths', 'var(--story_pinned_area_widths_right)');
		document.getElementById('preserve_game_space_setting').checked = true;
	} else {
		setCookie("options_on_right", "false");
		r.style.setProperty('--story_pinned_areas', 'var(--story_pinned_areas_left)');
		r.style.setProperty('--story_pinned_area_widths', 'var(--story_pinned_area_widths_left)');
		document.getElementById('preserve_game_space_setting').checked = false;
	}
}

function makeBiasCard(phrase="", score=0, compThreshold=10) {
	function updateBias(origin, input, save=true) {
		const textInput = input.closest(".bias_slider").querySelector(".bias_slider_cur");
		let value = (origin === "slider") ? input.value : parseFloat(textInput.innerText);
		textInput.innerText = value;
		input.value = value;

		// Only save on "commitful" actions like blur or mouseup to not spam
		// the poor server
		if (save) save_bias();
	}

	const biasContainer = $el("#bias-container");
	const biasCard = $el(".bias_card.template").cloneNode(true);
	biasCard.classList.remove("template");

	const closeButton = biasCard.querySelector(".close_button");
	closeButton.addEventListener("click", function(event) {
		biasCard.remove();

		// We just deleted the last bias, we probably don't want to keep seeing
		// them pop up
		if (!biasContainer.firstChild) biasContainer.setAttribute(
			"please-stop-adding-biases-whenever-i-delete-them",
			"i mean it"
		);
		save_bias();
	});

	const phraseInput = biasCard.querySelector(".bias_phrase");
	phraseInput.addEventListener("change", save_bias);

	const scoreInput = biasCard.querySelector(".bias_score input");
	const compThresholdInput = biasCard.querySelector(".bias_comp_threshold input");

	phraseInput.value = phrase;
	scoreInput.value = score;
	compThresholdInput.value = compThreshold;

	for (const input of [scoreInput, compThresholdInput]) {
		// Init sync
		updateBias("slider", input, false);

		// Visual update on each value change
		input.addEventListener("input", function() { updateBias("slider", this, false) });

		// Only when we leave do we sync to server
		input.addEventListener("change", function() { updateBias("slider", this) });

		// Personally I don't want to press a key 100 times to add one
		const nudge = parseFloat(input.getAttribute("keyboard-step") ?? input.getAttribute("step"));
		const min = parseFloat(input.getAttribute("min"));
		const max = parseFloat(input.getAttribute("max"));

		const currentHitbox = input.closest(".hitbox");
		const currentLabel = input.closest(".bias_slider").querySelector(".bias_slider_cur");

		// TODO: Prevent paste of just non-number characters
		currentLabel.addEventListener("paste", function(event) { event.preventDefault(); })

		currentLabel.addEventListener("keydown", function(event) {
			// Nothing special for numbers
			if (
				[".", "-", "ArrowLeft", "ArrowRight", "Backspace", "Delete"].includes(event.key)
				|| event.ctrlKey
				|| (parseInt(event.key) || parseInt(event.key) === 0)
			) return;

			// Either we are special keys or forbidden keys
			event.preventDefault();

			switch (event.key) {
				case "Enter":
					currentLabel.blur();
					break;
				// This feels very nice :^)
				case "ArrowDown":
				case "ArrowUp":
					let delta = (event.key === "ArrowUp") ? nudge : -nudge;
					let currentValue = parseFloat(currentLabel.innerText);

					event.preventDefault();
					if (!currentValue && currentValue !== 0) return;

					// toFixed because 1.1 + 0.1 !== 1.2 yay rounding errors.
					// Although the added decimal place(s) look cleaner now
					// that I think about it.
					let value = Math.min(max, Math.max(min, currentValue + delta));
					currentLabel.innerText = value.toFixed(2);

					updateBias("text", input, false);
					break;
			}
		});

		currentHitbox.addEventListener("wheel", function(event) {
			// Only when focused! (May drop this requirement later, browsers seem to behave when scrolling :] )
			if (currentLabel !== document.activeElement) return;
			if (event.deltaY === 0) return;

			let delta = (event.deltaY > 0) ? -nudge : nudge;
			let currentValue = parseFloat(currentLabel.innerText);

			event.preventDefault();
			if (!currentValue && currentValue !== 0) return;
			let value = Math.min(max, Math.max(min, currentValue + delta));
			currentLabel.innerText = value.toFixed(2);

			updateBias("text", input, false);
		});

		currentLabel.addEventListener("blur", function(event) {
			updateBias("text", input);
		});
	}

	biasContainer.appendChild(biasCard);
	return biasCard;
}
$el("#bias-add").addEventListener("click", function(event) {
	const card = makeBiasCard();
	card.querySelector(".bias_phrase").focus();
});

function do_biases(data) {
	console.info("Taking inventory of biases")
	biases_data = data.value;

	// Clear out our old bias cards, weird recursion because remove sometimes
	// doesn't work (???)
	const biasContainer = $el("#bias-container");
	for (let i=0;i<10000;i++) {
		if (!biasContainer.firstChild) break;
		biasContainer.firstChild.remove();
	}
	if(biasContainer.firstChild) reportError("We are doomed", "Undead zombie bias, please report this");

	//add our bias lines
	for (const [key, value] of Object.entries(data.value)) {
		makeBiasCard(key, value[0], value[1]);
	}

	// Add seed card if we have no bias cards and we didn't just delete the
	// last bias card
	if (
		!biasContainer.firstChild &&
		!biasContainer.getAttribute("please-stop-adding-biases-whenever-i-delete-them")
	) makeBiasCard();
}


function distortColor(rgb) {
	// rgb are 0..255, NOT NORMALIZED!!!!!!
	const brightnessTamperAmplitude = 0.1;
	const psuedoHue = 12;

	let brightnessDistortion = Math.random() * (255 * brightnessTamperAmplitude);
	rgb = rgb.map(x => x + brightnessDistortion);

	// Cheap hack to imitate hue rotation
	rgb = rgb.map(x => x += (Math.random() * psuedoHue * 2) - psuedoHue);

	// Clamp and round
	rgb = rgb.map(x => Math.round(Math.max(0, Math.min(255, x))));
	return rgb;
}

function dec2Hex2(number) {
	// Two padded hex number hack
	let x = number.toString(16);
	if (x.length === 1) return `0${x}`;
	return x;
}

function recolorTokens() {
	for (const contextContainer of document.querySelectorAll(".context-block")) {
		let rgb = window.getComputedStyle(contextContainer)["background-color"].match(/(\d+), (\d+), (\d+)/).slice(1, 4).map(Number);
		for (const tokenEl of contextContainer.querySelectorAll(".context-token")) {
			let tokenColor = distortColor(rgb);
			tokenColor = "#" + (tokenColor.map(dec2Hex2).join(""));
			tokenEl.style.backgroundColor = tokenColor;
		}
	}
}

function update_context(data) {
	$(".context-block").remove();

	let memory_length = 0;
	let genre_length = 0;
	let authors_notes_length = 0;
	let prompt_length = 0;
	let game_text_length = 0;
	let world_info_length = 0;
	let soft_prompt_length = 0;
	let submit_length = 0;
	
	//clear out within_max_length class
	for (action of document.getElementsByClassName("within_max_length")) {
		action.classList.remove("within_max_length");
	}
	for (wi of document.getElementsByClassName("used_in_game")) {
		wi.classList.remove("used_in_game");
	}
	

	for (const entry of data) {
		let contextClass = "context-" + ({
			soft_prompt: "sp",
			prompt: "prompt",
			world_info: "wi",
			genre: "genre",
			memory: "memory",
			authors_note: "an",
			action: "action",
			submit: 'submit'
		}[entry.type]);

		let el = $e(
			"span",
			$el("#context-container"),
			{classes: ["context-block", contextClass]}
		);

		let rgb = window.getComputedStyle(el)["background-color"].match(/(\d+), (\d+), (\d+)/).slice(1, 4).map(Number);

		for (const [tokenId, token] of entry.tokens) {
			let tokenColor = distortColor(rgb);
			tokenColor = "#" + (tokenColor.map(dec2Hex2).join(""));

			let tokenEl = $e("span", el, {
				classes: ["context-token"],
				"tooltip": tokenId === -1 ? "Soft" : tokenId,
				innerText: token.replaceAll(String.fromCharCode(0), '<span class="material-icons-outlined context-symbol">dangerous</span>'),
				"style.backgroundColor": tokenColor,
			});

			tokenEl.innerHTML = tokenEl.innerHTML.replaceAll("<br>", '<span class="material-icons-outlined context-symbol">keyboard_return</span>');
		}
		document.getElementById("context-container").appendChild(el);
		
		switch (entry.type) {
			case 'soft_prompt':
				soft_prompt_length += entry.tokens.length;
				break;
			case 'prompt':
				const promptEl = document.getElementById('story_prompt');
				prompt_length += entry.tokens.length;
				if (prompt_length > 0 && promptEl) {
					promptEl.classList.add("within_max_length");
				}
				break;
			case 'world_info':
				world_info_length += entry.tokens.length;
				if (document.getElementById('world_info_'+entry.uid)) {
					document.getElementById('world_info_'+entry.uid).classList.add("used_in_game");
				}
				break;
			case 'genre':
				genre_length += entry.tokens.length;
				break;
			case 'memory':
				memory_length += entry.tokens.length;
				break;
			case 'authors_note':
				authors_notes_length += entry.tokens.length;
				break;
			case 'action':
				game_text_length += entry.tokens.length;
				if ('action_ids' in entry) {
					for (action_id of entry.action_ids) {
						if (document.getElementById('Selected Text Chunk '+action_id)) {
							document.getElementById('Selected Text Chunk '+action_id).classList.add("within_max_length");
						}
					}
				}
				break;
			case 'submit':
				submit_length += entry.tokens.length;
				break;
		}
	}

	calc_token_usage(
		soft_prompt_length,
		genre_length,
		memory_length,
		authors_notes_length,
		prompt_length,
		game_text_length,
		world_info_length,
		submit_length
	);


}

function save_model_settings(settings = saved_settings) {
	for (item of document.getElementsByClassName('setting_item_input')) {
		if (item.id.includes("model")) {
			if ((item.tagName.toLowerCase() === 'checkbox') || (item.tagName.toLowerCase() === 'input') || (item.tagName.toLowerCase() === 'select') || (item.tagName.toLowerCase() == 'textarea')) {
				if (item.getAttribute("type") == "checkbox") {
					value = item.checked;
				} else {
					value = item.value;
				}
			} else {
				value = item.textContent;
			}
			settings[item.id] = value;
		}
	}
	for (item of document.getElementsByClassName('settings_select')) {
		if (item.id.includes("model")) {
			settings[item.id] = item.value;
		}
	}
}

function restore_model_settings(settings = saved_settings) {
	for (const [key, value] of Object.entries(settings)) {
		item = document.getElementById(key);
		if ((item.tagName.toLowerCase() === 'input') || (item.tagName.toLowerCase() === 'select')) {
			if (item.getAttribute("type") == "checkbox") {
				if (item.checked != value) {
					//not sure why the bootstrap-toggle won't respect a standard item.checked = true/false, so....
					item.parentNode.click();
				}
			} else {
				item.value = fix_text(value);
			}
		} else {
			item.textContent = fix_text(value);
		}
		if (typeof item.onclick == "function") {
			item.onclick.apply(item);
		}
		if (typeof item.onblur == "function") {
			item.onblur.apply(item);
		}
		if (typeof item.onchange == "function") {
			item.onchange.apply(item);
		}
		if (typeof item.oninput == "function") {
			item.oninput.apply(item);
		}
	}
}

function removeA(arr) {
    var what, a = arguments, L = a.length, ax;
    while (L > 1 && arr.length) {
        what = a[--L];
        while ((ax= arr.indexOf(what)) !== -1) {
            arr.splice(ax, 1);
        }
    }
    return arr;
}

function add_tags(tags, data) {
	while (tags.firstChild) { 
		tags.removeChild(tags.firstChild);
	}
	for (tag of data.key) {
		tag_item = document.createElement("span");
		tag_item.classList.add("tag");
		x = document.createElement("span");
		x.textContent = "x ";
		x.classList.add("delete_icon");
		x.setAttribute("uid", data.uid);
		x.setAttribute("tag", tag);
		x.onclick = function () {
						removeA(world_info_data[this.getAttribute('uid')]['key'], this.getAttribute('tag'));
						send_world_info(this.getAttribute('uid'));
						this.classList.add("pulse");
					};
		text = document.createElement("span");
		text.textContent = tag;
		text.setAttribute("contenteditable", true);
		text.setAttribute("uid", data.uid);
		text.setAttribute("tag", tag);
		text.id = "world_info_tags_text_"+data.uid+"_"+tag;
		text.ondragstart=function() {event.preventDefault();event.stopPropagation();};
		text.setAttribute("draggable", "true");
		text.onfocus=function() {this.parentElement.parentElement.parentElement.setAttribute('draggable', 'false');this.setAttribute('draggable', 'false');};
		text.onblur = function () {
						this.parentElement.parentElement.parentElement.setAttribute('draggable', 'true');
						this.setAttribute('draggable', 'true');
						for (var i = 0; i < world_info_data[this.getAttribute('uid')]['key'].length; i++) {
							if (world_info_data[this.getAttribute('uid')]['key'][i] == this.getAttribute("tag")) {
								world_info_data[this.getAttribute('uid')]['key'][i] = this.textContent;
							}
						}
						send_world_info(this.getAttribute('uid'));
						this.classList.add("pulse");
					};
		tag_item.append(x);
		tag_item.append(text);
		tag_item.id = "world_info_tags_"+data.uid+"_"+tag;
		tags.append(tag_item);
	}
	//add the blank tag
	tag_item = document.createElement("span");
	tag_item.classList.add("tag");
	x = document.createElement("span");
	x.textContent = "+ ";
	tag_item.append(x);
	text = document.createElement("span");
	text.classList.add("rawtext");
	text.textContent = "    ";
	text.setAttribute("uid", data.uid);
	text.setAttribute("contenteditable", true);
	text.id = "world_info_tags_text_"+data.uid+"_blank";
	text.ondragstart=function() {event.preventDefault();event.stopPropagation();};
	text.setAttribute("draggable", "true");
	text.onfocus=function() {this.parentElement.parentElement.parentElement.setAttribute('draggable', 'false');this.setAttribute('draggable', 'false');};
	text.onblur = function () {
					this.parentElement.parentElement.parentElement.setAttribute('draggable', 'true');
					this.setAttribute('draggable', 'true');
					if (this.textContent.trim() != "") {
						//console.log(this.textContent);
						on_new_wi_item = this.id;
						world_info_data[this.getAttribute('uid')]['key'].push(this.textContent);
						send_world_info(this.getAttribute('uid'));
						this.classList.add("pulse");
					} else {
						this.textContent = "    ";
					}
				};
	text.onclick = function () {
					this.textContent = "";
				};
	tag_item.append(text);
	tag_item.id = "world_info_secondtags_"+data.uid+"_new";
	tags.append(tag_item);
}

function add_secondary_tags(tags, data) {
	while (tags.firstChild) { 
		tags.removeChild(tags.firstChild);
	}
	for (tag of data.keysecondary) {
		tag_item = document.createElement("span");
		tag_item.classList.add("tag");
		x = document.createElement("span");
		x.textContent = "x ";
		x.classList.add("delete_icon");
		x.setAttribute("uid", data.uid);
		x.setAttribute("tag", tag);
		x.onclick = function () {
						removeA(world_info_data[this.getAttribute('uid')]['keysecondary'], this.getAttribute('tag'));
						send_world_info(this.getAttribute('uid'));
						this.classList.add("pulse");
					};
		text = document.createElement("span");
		text.textContent = tag;
		text.setAttribute("contenteditable", true);
		text.setAttribute("uid", data.uid);
		text.setAttribute("tag", tag);
		text.id = "world_info_secondtags_text_"+data.uid+"_"+tag;
		text.ondragstart=function() {event.preventDefault();event.stopPropagation();};
		text.setAttribute("draggable", "true");
		text.onfocus=function() {this.parentElement.parentElement.parentElement.setAttribute('draggable', 'false');this.setAttribute('draggable', 'false');};
		text.onblur = function () {
						this.parentElement.parentElement.parentElement.setAttribute('draggable', 'true');
						this.setAttribute('draggable', 'true');
						for (var i = 0; i < world_info_data[this.getAttribute('uid')]['keysecondary'].length; i++) {
							if (world_info_data[this.getAttribute('uid')]['keysecondary'][i] == this.getAttribute("tag")) {
								world_info_data[this.getAttribute('uid')]['keysecondary'][i] = this.textContent;
							}
						}
						send_world_info(this.getAttribute('uid'));
						this.classList.add("pulse");
					};
		tag_item.append(x);
		tag_item.append(text);
		tag_item.id = "world_info_secondtags_"+data.uid+"_"+tag;
		tags.append(tag_item);
	}
	//add the blank tag
	tag_item = document.createElement("span");
	tag_item.classList.add("tag");
	x = document.createElement("span");
	x.textContent = "+ ";
	tag_item.append(x);
	text = document.createElement("span");
	text.classList.add("rawtext");
	text.textContent = "    ";
	text.setAttribute("uid", data.uid);
	text.setAttribute("contenteditable", true);
	text.id = "world_info_secondtags_text_"+data.uid+"_blank";
	text.ondragstart=function() {event.preventDefault();event.stopPropagation();};
	text.setAttribute("draggable", "true");
	text.onfocus=function() {this.parentElement.parentElement.parentElement.setAttribute('draggable', 'false');this.setAttribute('draggable', 'false');};
	text.onblur = function () {
					this.parentElement.parentElement.parentElement.setAttribute('draggable', 'true');
					this.setAttribute('draggable', 'true');
					if (this.textContent.trim() != "") {
						on_new_wi_item = this.id;
						world_info_data[this.getAttribute('uid')]['keysecondary'].push(this.textContent);
						send_world_info(this.getAttribute('uid'));
						this.classList.add("pulse");
					} else {
						this.textContent = "    ";
					}
				};
	text.onclick = function () {
					this.textContent = "";
				};
	tag_item.append(text);
	tag_item.id = "world_info_secondtags_"+data.uid+"_new";
	tags.append(tag_item);
}
	
function create_new_wi_entry(folder) {
	var uid = -1;
	for (item of document.getElementsByClassName('world_info_card')) {
		if (parseInt(item.getAttribute("uid")) <= uid) {
			uid = parseInt(item.getAttribute("uid")) - 1;
		}
	}
	data = {"uid": uid,
                                    "title": "New World Info Entry",
                                    "key": [],
                                    "keysecondary": [],
                                    "folder": folder,
                                    "constant": false,
                                    "content": "",
									"manual_text": "",
                                    "comment": "",
                                    "token_length": 0,
                                    "selective": false,
									"wpp": {'name': "", 'type': "", 'format': 'W++', 'attributes': {}},
									'use_wpp': false,
									"object_type": null,
                                    };
	var card = world_info_entry(data);
	//card.scrollIntoView(false);
	clearTimeout(world_info_scroll_timeout);
	world_info_scroll_timeout = setTimeout(function() {card.scrollIntoView(false);}, 200);
	
}

function hide_wi_folder(folder) {
	if (document.getElementById("world_info_folder_"+folder)) {
		folder_item = document.getElementById("world_info_folder_"+folder);
		for (card of folder_item.children) {
			if (card.tagName != "H3") {
				card.classList.add("hidden");
			}
		}
	}
}

function unhide_wi_folder(folder) {
	if (document.getElementById("world_info_folder_"+folder)) {
		folder_item = document.getElementById("world_info_folder_"+folder);
		for (card of folder_item.children) {
			if (card.tagName != "H3") {
				card.classList.remove("hidden");
			}
		}
	}
}

function dragStart(e) {
    drag_id = e.target.id;
	e.dataTransfer.dropEffect = "move";
    setTimeout(() => {
        e.target.classList.add('hidden');
    }, 0);
}

function find_wi_container(e) {
	
	while (true) {
		if (e.parentElement == document) {
			return e;
		} else if (e.classList.contains('WI_Folder')) {
			return e;
		} else if (e.tagName == 'H2') {
			return e.parentElement;
		} else if (typeof e.id == 'undefined') {
			e = e.parentElement;
		} else if (e.id.replace(/[^a-z_]/gi, '') == 'world_info_') {
			return e
		} else {
			e = e.parentElement;
		}
	}
}

function dragEnter(e) {
    e.preventDefault();
	element = find_wi_container(e.target);
    element.classList.add('drag-over');
}

function dragOver(e) {
    e.preventDefault();
	//console.log(e.target);
	element = find_wi_container(e.target);
    element.classList.add('drag-over');
}

function dragLeave(e) {
	element = find_wi_container(e.target);
    element.classList.remove('drag-over');
}

function drop(e) {
	e.preventDefault();
    // get the drop element
	element = find_wi_container(e.target);
    element.classList.remove('drag-over');

    // get the draggable element
    const id = drag_id;
    const draggable = document.getElementById(id);
	//console.log(id);
	dragged_id = draggable.id.split("_").slice(-1)[0];
	drop_id = element.id.split("_").slice(-1)[0];

	
	//check if we're droping on a folder, and then append it to the folder
	if (element.classList.contains('WI_Folder')) {
		//element.append(draggable);
		socket.emit("wi_set_folder", {'dragged_id': dragged_id, 'folder': drop_id});
	} else {
		//insert the draggable element before the drop element
		console.log(element);
		element.parentElement.insertBefore(draggable, element);
		draggable.classList.add("pulse");

		// display the draggable element
		draggable.classList.remove('hidden');
		
		if (drop_id == "-1") {
			folder = element.parentElement.id.split("_").slice(-1)[0];
			socket.emit("wi_set_folder", {'dragged_id': dragged_id, 'folder': folder});
		} else if (element.getAttribute("folder") == draggable.getAttribute("folder")) {
			socket.emit("move_wi", {'dragged_id': dragged_id, 'drop_id': drop_id, 'folder': null});
		} else {
			socket.emit("move_wi", {'dragged_id': dragged_id, 'drop_id': drop_id, 'folder': element.getAttribute("folder")});
		}
	}
}

function dragend(e) {
	// get the draggable element
    const id = drag_id;
    const draggable = document.getElementById(id);
	// display the draggable element
	if (draggable) {
		draggable.classList.remove('hidden');
	}
	e.preventDefault();
}

function checkifancestorhasclass(element, classname) {
    if (element.classList.contains(classname)) {
		return true;
	} else {
		return hasSomeParentTheClass(element.parentNode, classname);
	}
}

function assign_world_info_to_action(action_item, uid) {
	return
	if (Object.keys(world_info_data).length > 0) {
		if (uid != null) {
			var worldinfo_to_check = {};
			worldinfo_to_check[uid] = world_info_data[uid];
		} else {
			var worldinfo_to_check = world_info_data;
		}
		if (action_item != null) {
			var actions = {};
			actions[action_item] = actions_data[action_item]
		} else {
			var actions = actions_data;
		}
		
		for (const [action_id, action] of  Object.entries(actions)) {
			//First check to see if we have a key in the text
			for (const [key, worldinfo] of  Object.entries(worldinfo_to_check)) {
				//remove any world info tags on the overall chunk
				if (worldinfo['constant'] == false) {
					//for (tag of action.getElementsByClassName("tag_uid_"+uid)) {
					//	tag.classList.remove("tag_uid_"+uid);
					//	tag.removeAttribute("title");
					//	current_ids = tag.parentElement.getAttribute("world_info_uids").split(",");
					//	removeA(current_ids, uid);
					//	tag.parentElement.setAttribute("world_info_uids", current_ids.join(","));
					//}
					for (keyword of worldinfo['key']) {
						if (action['WI Search Text'].includes(keyword)) {
							//Ok we have a key match, but we need to check for secondary keys if applicable
							if (worldinfo['keysecondary'].length > 0) {
								for (second_key of worldinfo['keysecondary']) {
									if (action['WI Search Text'].includes(second_key)) {
										highlight_world_info_text_in_chunk(action_id, worldinfo);
										break;
									}
								}
							} else {
								highlight_world_info_text_in_chunk(action_id, worldinfo);
								break;
							}
							
						}
					}
				}
			}
		}
	}
}

function highlight_world_info_text_in_chunk(action_id, wi) {
	//First let's assign our world info id to the action so we know to count the tokens for the world info
	let uid = wi['uid'];
	let action = undefined;
	if (action_id < 0) {
		action = document.getElementById("story_prompt");
	} else {
		action = document.getElementById("Selected Text Chunk "+action_id);
	}
	let words = action.innerText.split(" ");
	current_ids = action.getAttribute("world_info_uids")?action.getAttribute("world_info_uids").split(','):[];
	if (!(current_ids.includes(uid))) {
		current_ids.push(uid);
	}
	action.setAttribute("world_info_uids", current_ids.join(","));
	//OK we have the phrase in our action. 
	//First let's find the largest key that matches
	let largest_key = "";
	for (keyword of wi['key']) {
		if ((keyword.length > largest_key.length) && (action.getAttribute('WI_Search_Text').includes(keyword))) {
			largest_key = keyword;
		}
	}
	//console.log(largest_key);
	
	
	//Let's see if we can identify the word(s) that are triggering
	var len_of_keyword = largest_key.split(" ").length;
	//go through each word to see where we get a match
	for (var i = 0; i < words.length; i++) {
		//get the words from the ith word to the i+len_of_keyword. Get rid of non-letters/numbers/'/"
		var to_check = words.slice(i, i+len_of_keyword).join(" ").replace(/[^0-9a-z \'\"]/gi, '').trim();
		if (largest_key == to_check) {
			var start_word = i;
			var end_word = i+len_of_keyword-1;
			var passed_words = 0;
			//console.log("Finding "+to_check);
			for (span of action.childNodes) {
				//console.log(span);
				//console.log("passed_words("+passed_words+")+span("+(span.textContent.trim().split(" ").length)+")<start_word("+start_word+"): "+(passed_words + span.textContent.trim().split(" ").length < start_word));
				if (passed_words + span.innerText.trim().split(" ").length < start_word+1) {
					passed_words += span.innerText.trim().split(" ").length;
				} else if (passed_words <= end_word) {
					//OK, we have text that matches, let's do the highlighting
					//we can skip the highlighting if it's already done though
					//console.log(span.textContent.trim().split(" "));
					//console.log("start_word: "+start_word+" end_word: "+end_word+" passed_words: "+passed_words);
					//console.log(span.textContent.trim().split(" ").slice(start_word-passed_words, end_word-passed_words+1).join(" "));
					if (~(span.classList.contains('wi_match'))) {
						var span_text = span.innerText.trim().split(" ");
						//console.log(span_text);
						if (start_word-passed_words == 0) {
							var before_highlight_text = "";
						} else {
							var before_highlight_text = span_text.slice(0, start_word-passed_words).join(" ")+" ";
						}
						var highlight_text = span_text.slice(start_word-passed_words, end_word-passed_words+1).join(" ");
						if (end_word-passed_words-1 <= span_text.length) {
							highlight_text += " ";
						}
						var after_highlight_text = span_text.slice((end_word-passed_words+1)).join(" ")+" ";
						if (after_highlight_text[0] == ' ') {
							after_highlight_text = after_highlight_text.substring(1);
						}
						if (before_highlight_text != "") {
							//console.log("Before Text:'"+before_highlight_text+"'");
							var before_span = document.createElement("span");
							before_span.innerText = before_highlight_text;
							action.insertBefore(before_span, span);
						}
						//console.log("Highlight Text: '"+highlight_text+"'");
						var highlight_span = document.createElement("span");
						highlight_span.classList.add("wi_match");
						highlight_span.innerText = highlight_text;
						highlight_span.setAttribute("tooltip", wi['content']);
						highlight_span.setAttribute("wi-uid", wi.uid);
						action.insertBefore(highlight_span, span);
						if (after_highlight_text != "") {
							//console.log("After Text: '"+after_highlight_text+"'");
							var after_span = document.createElement("span");
							after_span.innerText = after_highlight_text;
							action.insertBefore(after_span, span);
						}
						//console.log("Done");
						span.remove();
					}
					passed_words += span.innerText.trim().split(" ").length;
				}
			}
		}
	}
}

function update_token_lengths() {
	clearTimeout(calc_token_usage_timeout);
	calc_token_usage_timeout = setTimeout(function() {socket.emit("update_tokens", document.getElementById("input_text").value);}, 500);
}

String.prototype.toHHMMSS = function () {
    var sec_num = parseInt(this, 10); // don't forget the second param
    var hours   = Math.floor(sec_num / 3600);
    var minutes = Math.floor((sec_num - (hours * 3600)) / 60);
    var seconds = sec_num - (hours * 3600) - (minutes * 60);

    if (hours   < 10) {hours   = "0"+hours;}
    if (minutes < 10) {minutes = "0"+minutes;}
    if (seconds < 10) {seconds = "0"+seconds;}
    return hours+':'+minutes+':'+seconds;
}

function close_menus() {
	//close settings menu
	document.getElementById("setting_menu_icon").classList.remove("change");
	document.getElementById("SideMenu").classList.remove("open");
	document.getElementById("main-grid").classList.remove("menu-open");
	
	//close story menu
	document.getElementById("story_menu_icon").classList.remove("change");
	document.getElementById("rightSideMenu").classList.remove("open");
	document.getElementById("main-grid").classList.remove("story_menu-open");
	
	//close popup menus
	closePopups();
	
	//unselect sampler items
	for (temp of document.getElementsByClassName("sample_order")) {
		temp.classList.remove("selected");
	}
}

function toggle_flyout(x) {
	if (document.getElementById("SideMenu").classList.contains("open")) {
		x.classList.remove("change");
		document.getElementById("SideMenu").classList.remove("open");
		document.getElementById("main-grid").classList.remove("menu-open");
	} else {
		x.classList.add("change");
		document.getElementById("SideMenu").classList.add("open");
		document.getElementById("main-grid").classList.add("menu-open");
		document.getElementById("menu_pin").classList.remove("hidden");
	}
}

function toggle_flyout_right(x) {
	if (document.getElementById("rightSideMenu").classList.contains("open")) {
		x.classList.remove("change");
		document.getElementById("rightSideMenu").classList.remove("open");
		document.getElementById("main-grid").classList.remove("story_menu-open");
	} else {
		x.classList.add("change");
		document.getElementById("rightSideMenu").classList.add("open");
		document.getElementById("main-grid").classList.add("story_menu-open");
		document.getElementById("story_menu_pin").classList.remove("hidden");
	}
}

function toggle_settings_pin_flyout() {
	if (document.getElementById("SideMenu").classList.contains("pinned")) {
		settings_unpin();
	} else {
		settings_pin();
	}
}

function settings_pin() {
	setCookie("Settings_Pin", "true");
	document.getElementById("SideMenu").classList.remove("open");
	document.getElementById("main-grid").classList.remove("menu-open");
	document.getElementById("setting_menu_icon").classList.remove("change");
	document.getElementById("setting_menu_icon").classList.add("hidden");
	document.getElementById("SideMenu").classList.add("pinned");
	document.getElementById("main-grid").classList.add("settings_pinned");
}

function settings_unpin() {
	setCookie("Settings_Pin", "false");
	document.getElementById("SideMenu").classList.remove("pinned");
	document.getElementById("main-grid").classList.remove("settings_pinned");
	document.getElementById("setting_menu_icon").classList.remove("hidden");
}	

function toggle_story_pin_flyout() {
	if (document.getElementById("rightSideMenu").classList.contains("pinned")) {
		story_unpin();
	} else {
		story_pin();
	}
}

function story_pin() {
	setCookie("Story_Pin", "true");
	document.getElementById("rightSideMenu").classList.remove("open");
	document.getElementById("main-grid").classList.remove("story_menu-open");
	document.getElementById("rightSideMenu").classList.add("pinned");
	document.getElementById("main-grid").classList.add("story_pinned");
	document.getElementById("story_menu_icon").classList.remove("change");
	document.getElementById("story_menu_icon").classList.add("hidden");
}

function story_unpin() {
	setCookie("Story_Pin", "false");
	document.getElementById("rightSideMenu").classList.remove("pinned");
	document.getElementById("main-grid").classList.remove("story_pinned");
	document.getElementById("story_menu_icon").classList.remove("hidden");
}

function setCookie(cname, cvalue, exdays=60) {
  const d = new Date();
  d.setTime(d.getTime() + (exdays * 24 * 60 * 60 * 1000));
  let expires = "expires="+d.toUTCString();
  if (document.getElementById("on_colab").textContent == "true") {
	socket.emit("save_cookies", {[cname]: cvalue});
  }
  document.cookie = cname + "=" + cvalue + ";" + expires + ";";
}

function getCookie(cname, default_return=null) {
  let name = cname + "=";
  let ca = document.cookie.split(';');
  for(let i = 0; i < ca.length; i++) {
	let c = ca[i];
	while (c.charAt(0) == ' ') {
	  c = c.substring(1);
	}
	if (c.indexOf(name) == 0) {
	  return c.substring(name.length, c.length);
	}
  }
  return default_return;
}

function detect_enter_submit(e) {
	if (((e.code == "Enter") || (e.code == "NumpadEnter")) && !(shift_down)) {
		if (typeof e.stopPropagation != "undefined") {
			e.stopPropagation();
		} else {
			e.cancelBubble = true;
		}
		//console.log("submitting");
		document.getElementById("btnsubmit").onclick();
		setTimeout(function() {document.getElementById('input_text').value = '';}, 1);
	}
}

function detect_enter_text(e) {
	if (((e.code == "Enter") || (e.code == "NumpadEnter")) && !(shift_down)) {
		if (typeof e.stopPropagation != "undefined") {
			e.stopPropagation();
		} else {
			e.cancelBubble = true;
		}
		//get element
		//console.log("Doing Text Enter");
		//console.log(e.currentTarget.activeElement);
		if (e.currentTarget.activeElement != undefined) {
			var item = $(e.currentTarget.activeElement);
			item.onchange();
		}
	}
}

function detect_key_down(e) {
	if ((e.code == "ShiftLeft") || (e.code == "ShiftRight")) {
		shift_down = true;
	} else if (e.code == "Escape") {
		close_menus();
	}
}

function detect_key_up(e) {
	if ((e.code == "ShiftLeft") || (e.code == "ShiftRight")) {
		shift_down = false;
	}
}

function selectTab(tab) {
	let tabTarget = document.getElementById(tab.getAttribute("tab-target"));
	let tabClass = Array.from(tab.classList).filter((c) => c.startsWith("tab-"))[0];
	let targetClass = Array.from(tabTarget.classList).filter((c) => c.startsWith("tab-target-"))[0];
	
	$(`.${tabClass}`).removeClass("selected");
	tab.classList.add("selected");
	
	$(`.${targetClass}`).addClass("hidden");
	tabTarget.classList.remove("hidden");
}

function beep() {
    var snd = new Audio("data:audio/wav;base64,//uQRAAAAWMSLwUIYAAsYkXgoQwAEaYLWfkWgAI0wWs/ItAAAGDgYtAgAyN+QWaAAihwMWm4G8QQRDiMcCBcH3Cc+CDv/7xA4Tvh9Rz/y8QADBwMWgQAZG/ILNAARQ4GLTcDeIIIhxGOBAuD7hOfBB3/94gcJ3w+o5/5eIAIAAAVwWgQAVQ2ORaIQwEMAJiDg95G4nQL7mQVWI6GwRcfsZAcsKkJvxgxEjzFUgfHoSQ9Qq7KNwqHwuB13MA4a1q/DmBrHgPcmjiGoh//EwC5nGPEmS4RcfkVKOhJf+WOgoxJclFz3kgn//dBA+ya1GhurNn8zb//9NNutNuhz31f////9vt///z+IdAEAAAK4LQIAKobHItEIYCGAExBwe8jcToF9zIKrEdDYIuP2MgOWFSE34wYiR5iqQPj0JIeoVdlG4VD4XA67mAcNa1fhzA1jwHuTRxDUQ//iYBczjHiTJcIuPyKlHQkv/LHQUYkuSi57yQT//uggfZNajQ3Vmz+Zt//+mm3Wm3Q576v////+32///5/EOgAAADVghQAAAAA//uQZAUAB1WI0PZugAAAAAoQwAAAEk3nRd2qAAAAACiDgAAAAAAABCqEEQRLCgwpBGMlJkIz8jKhGvj4k6jzRnqasNKIeoh5gI7BJaC1A1AoNBjJgbyApVS4IDlZgDU5WUAxEKDNmmALHzZp0Fkz1FMTmGFl1FMEyodIavcCAUHDWrKAIA4aa2oCgILEBupZgHvAhEBcZ6joQBxS76AgccrFlczBvKLC0QI2cBoCFvfTDAo7eoOQInqDPBtvrDEZBNYN5xwNwxQRfw8ZQ5wQVLvO8OYU+mHvFLlDh05Mdg7BT6YrRPpCBznMB2r//xKJjyyOh+cImr2/4doscwD6neZjuZR4AgAABYAAAABy1xcdQtxYBYYZdifkUDgzzXaXn98Z0oi9ILU5mBjFANmRwlVJ3/6jYDAmxaiDG3/6xjQQCCKkRb/6kg/wW+kSJ5//rLobkLSiKmqP/0ikJuDaSaSf/6JiLYLEYnW/+kXg1WRVJL/9EmQ1YZIsv/6Qzwy5qk7/+tEU0nkls3/zIUMPKNX/6yZLf+kFgAfgGyLFAUwY//uQZAUABcd5UiNPVXAAAApAAAAAE0VZQKw9ISAAACgAAAAAVQIygIElVrFkBS+Jhi+EAuu+lKAkYUEIsmEAEoMeDmCETMvfSHTGkF5RWH7kz/ESHWPAq/kcCRhqBtMdokPdM7vil7RG98A2sc7zO6ZvTdM7pmOUAZTnJW+NXxqmd41dqJ6mLTXxrPpnV8avaIf5SvL7pndPvPpndJR9Kuu8fePvuiuhorgWjp7Mf/PRjxcFCPDkW31srioCExivv9lcwKEaHsf/7ow2Fl1T/9RkXgEhYElAoCLFtMArxwivDJJ+bR1HTKJdlEoTELCIqgEwVGSQ+hIm0NbK8WXcTEI0UPoa2NbG4y2K00JEWbZavJXkYaqo9CRHS55FcZTjKEk3NKoCYUnSQ0rWxrZbFKbKIhOKPZe1cJKzZSaQrIyULHDZmV5K4xySsDRKWOruanGtjLJXFEmwaIbDLX0hIPBUQPVFVkQkDoUNfSoDgQGKPekoxeGzA4DUvnn4bxzcZrtJyipKfPNy5w+9lnXwgqsiyHNeSVpemw4bWb9psYeq//uQZBoABQt4yMVxYAIAAAkQoAAAHvYpL5m6AAgAACXDAAAAD59jblTirQe9upFsmZbpMudy7Lz1X1DYsxOOSWpfPqNX2WqktK0DMvuGwlbNj44TleLPQ+Gsfb+GOWOKJoIrWb3cIMeeON6lz2umTqMXV8Mj30yWPpjoSa9ujK8SyeJP5y5mOW1D6hvLepeveEAEDo0mgCRClOEgANv3B9a6fikgUSu/DmAMATrGx7nng5p5iimPNZsfQLYB2sDLIkzRKZOHGAaUyDcpFBSLG9MCQALgAIgQs2YunOszLSAyQYPVC2YdGGeHD2dTdJk1pAHGAWDjnkcLKFymS3RQZTInzySoBwMG0QueC3gMsCEYxUqlrcxK6k1LQQcsmyYeQPdC2YfuGPASCBkcVMQQqpVJshui1tkXQJQV0OXGAZMXSOEEBRirXbVRQW7ugq7IM7rPWSZyDlM3IuNEkxzCOJ0ny2ThNkyRai1b6ev//3dzNGzNb//4uAvHT5sURcZCFcuKLhOFs8mLAAEAt4UWAAIABAAAAAB4qbHo0tIjVkUU//uQZAwABfSFz3ZqQAAAAAngwAAAE1HjMp2qAAAAACZDgAAAD5UkTE1UgZEUExqYynN1qZvqIOREEFmBcJQkwdxiFtw0qEOkGYfRDifBui9MQg4QAHAqWtAWHoCxu1Yf4VfWLPIM2mHDFsbQEVGwyqQoQcwnfHeIkNt9YnkiaS1oizycqJrx4KOQjahZxWbcZgztj2c49nKmkId44S71j0c8eV9yDK6uPRzx5X18eDvjvQ6yKo9ZSS6l//8elePK/Lf//IInrOF/FvDoADYAGBMGb7FtErm5MXMlmPAJQVgWta7Zx2go+8xJ0UiCb8LHHdftWyLJE0QIAIsI+UbXu67dZMjmgDGCGl1H+vpF4NSDckSIkk7Vd+sxEhBQMRU8j/12UIRhzSaUdQ+rQU5kGeFxm+hb1oh6pWWmv3uvmReDl0UnvtapVaIzo1jZbf/pD6ElLqSX+rUmOQNpJFa/r+sa4e/pBlAABoAAAAA3CUgShLdGIxsY7AUABPRrgCABdDuQ5GC7DqPQCgbbJUAoRSUj+NIEig0YfyWUho1VBBBA//uQZB4ABZx5zfMakeAAAAmwAAAAF5F3P0w9GtAAACfAAAAAwLhMDmAYWMgVEG1U0FIGCBgXBXAtfMH10000EEEEEECUBYln03TTTdNBDZopopYvrTTdNa325mImNg3TTPV9q3pmY0xoO6bv3r00y+IDGid/9aaaZTGMuj9mpu9Mpio1dXrr5HERTZSmqU36A3CumzN/9Robv/Xx4v9ijkSRSNLQhAWumap82WRSBUqXStV/YcS+XVLnSS+WLDroqArFkMEsAS+eWmrUzrO0oEmE40RlMZ5+ODIkAyKAGUwZ3mVKmcamcJnMW26MRPgUw6j+LkhyHGVGYjSUUKNpuJUQoOIAyDvEyG8S5yfK6dhZc0Tx1KI/gviKL6qvvFs1+bWtaz58uUNnryq6kt5RzOCkPWlVqVX2a/EEBUdU1KrXLf40GoiiFXK///qpoiDXrOgqDR38JB0bw7SoL+ZB9o1RCkQjQ2CBYZKd/+VJxZRRZlqSkKiws0WFxUyCwsKiMy7hUVFhIaCrNQsKkTIsLivwKKigsj8XYlwt/WKi2N4d//uQRCSAAjURNIHpMZBGYiaQPSYyAAABLAAAAAAAACWAAAAApUF/Mg+0aohSIRobBAsMlO//Kk4soosy1JSFRYWaLC4qZBYWFRGZdwqKiwkNBVmoWFSJkWFxX4FFRQWR+LsS4W/rFRb/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////VEFHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAU291bmRib3kuZGUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMjAwNGh0dHA6Ly93d3cuc291bmRib3kuZGUAAAAAAAAAACU=");  
    snd.play();
}

function downloadString(string, fileName) {
	let a = document.createElement("a");
	a.setAttribute("download", fileName);
	a.href = URL.createObjectURL(new Blob([string]));
	a.click();
}

function downloadBlob(blob, fileName) {
	const a = $e("a", null, {
		href: URL.createObjectURL(blob),
		download: fileName
	});
	a.click();
}

function getRedactedValue(value) {
	if (typeof value === "string") return `[Redacted string with length ${value.length}]`;
	if (value instanceof Array) return `[Redacted array with length ${value.length}]`;

	if (typeof value === "object") {
		if (value === null) return null;

		let built = {};
		for (const key of Object.keys(value)) {
			built[getRedactedValue(key)] = getRedactedValue(value[key]);
		}

		return built;
	}

	return "[Redacted value]"
}

async function downloadDebugFile(redact=true) {
	let r = await fetch("/vars");
	let varsData = await r.json();
	
	r = await fetch("/get_log");
	let aiserver_log = await r.json();
	console.log(aiserver_log);
	
	debug_info['aiserver errors'] = []
	for (data of aiserver_log.aiserver_log) {
		let temp = JSON.parse(JSON.stringify(data.record));
		temp = {"level": temp.level.name, 'message': temp.message, 'record': temp};
		delete temp.record.message;
		debug_info['aiserver errors'].push(temp);
	}

	// Redact sensitive user info

	// [redacted string n characters long]
	// [redacted array with n elements]

	let redactables = [
		"model_settings.apikey",
		"model_settings.colaburl",
		"model_settings.oaiapikey",
		"system_settings.story_loads",
		"user_settings.username",
		"system_settings.savedir", // Can reveal username
		"story_settings.last_story_load",
	];
	
	if (redact) {
		// TODO: genseqs, splist(?)
		redactables = redactables.concat([
			"story_settings.authornote",
			"story_settings.chatname",
			"story_settings.lastact",
			"story_settings.lastctx",
			"story_settings.memory",
			"story_settings.notes",
			"story_settings.prompt",
			"story_settings.story_name",
			"story_settings.submission",
			"story_settings.biases",
			"story_settings.genseqs",

			// System
			"system_settings.spfilename",
			"system_settings.spname",
		]);
		

		// Redact more complex things

		// wifolders_d - name
		for (const key of Object.keys(varsData.story_settings.wifolders_d)) {
			varsData.story_settings.wifolders_d[key].name = getRedactedValue(varsData.story_settings.wifolders_d[key].name);
		}

		// worldinfo - comment, content, key, keysecondary
		for (const key of Object.keys(varsData.story_settings.worldinfo)) {
			for (const redactKey of ["comment", "content", "key", "keysecondary"]) {
				varsData.story_settings.worldinfo[key][redactKey] = getRedactedValue(varsData.story_settings.worldinfo[key][redactKey]);
			}
		}

		// worldinfo_i - comment, content, key, keysecondary
		for (const key of Object.keys(varsData.story_settings.worldinfo_i)) {
			for (const redactKey of ["comment", "content", "key", "keysecondary"]) {
				varsData.story_settings.worldinfo_i[key][redactKey] = getRedactedValue(varsData.story_settings.worldinfo_i[key][redactKey]);
			}
		}

		// worldinfo_u - comment, content, key, keysecondary
		for (const key of Object.keys(varsData.story_settings.worldinfo_u)) {
			for (const redactKey of ["comment", "content", "key", "keysecondary"]) {
				varsData.story_settings.worldinfo_u[key][redactKey] = getRedactedValue(varsData.story_settings.worldinfo_u[key][redactKey]);
			}
		}

		// worldinfo_v2 entries - comment, content, folder, key, keysecondary, manual_text, title, wpp
		for (const key of Object.keys(varsData.story_settings.worldinfo_v2.entries)) {
			for (const redactKey of ["comment", "content", "folder", "key", "keysecondary", "manual_text", "title", "wpp"]) {
				varsData.story_settings.worldinfo_v2.entries[key][redactKey] = getRedactedValue(varsData.story_settings.worldinfo_v2.entries[key][redactKey]);
			}
		}

		varsData.story_settings.worldinfo_v2.folders = getRedactedValue(varsData.story_settings.worldinfo_v2.folders);

		// actions - "Selected Text", Options, Probabilities
		for (const key of Object.keys(varsData.story_settings.actions.actions)) {
			for (const redactKey of ["Selected Text", "Options", "Probabilities", "Original Text"]) {
				varsData.story_settings.actions.actions[key][redactKey] = getRedactedValue(varsData.story_settings.actions.actions[key][redactKey]);
			}
		}

	}

	for (const varPath of redactables) {
		let ref = varsData;
		const parts = varPath.split(".");

		for (const part of parts.slice(0, -1)) {
			ref = ref[part];
		}

		const lastPart = parts[parts.length - 1];

		ref[lastPart] = getRedactedValue(ref[lastPart]);
	}

	debug_info.currentVars = varsData;
	
	//redact aiserver log messages
	for (log_item of debug_info['aiserver errors']) {
		if (['PROMPT', 'GENERATION'].includes(log_item.level)) {
			log_item.message = getRedactedValue(log_item.message);
		}
	}
	
	console.log(debug_info);

	downloadString(JSON.stringify(debug_info, null, 4), "kobold_debug.json");
}

function configurePrompt(placeholderData) {
	openPopup("prompt-config");

	const placeholders = document.querySelector("#prompt-config-placeholders");

	for (const phData of placeholderData) {
		let placeholder = $e("div", placeholders, {classes: ["prompt-config-ph"]});


		// ${character.name} is an AI Dungeon thing, although I believe NAI
		// supports it as well. Many prompts use it. I think this is the only
		// hardcoded thing like this.
		let titleText = phData.title || phData.id;
		if (titleText === "character.name") titleText = "Character Name";

		let title = $e("span", placeholder, {classes: ["prompt-config-title"], innerText: titleText});

		if (phData.description) $e("span", placeholder, {
			classes: ["prompt-config-desc", "help_text"],
			innerText: phData.description
		});

		let input = $e("input", placeholder, {
			classes: ["prompt-config-value"],
			value: phData.default || "",
			placeholder: phData.default || "",
			"placeholder-id": phData.id
		});
	}
}

function sendPromptConfiguration() {
	let data = {};
	for (const configInput of document.querySelectorAll(".prompt-config-value")) {
		data[configInput.getAttribute("placeholder-id")] = configInput.value;
	}

	socket.emit("configure_prompt", data);

	closePopups();
	$(".prompt-config-ph").remove();
}

async function postWI(wiData) {
	let r = await fetch("/upload_wi", {
		method: "POST",
		headers: {
			"Content-Type": "application/json"
		},
		body: JSON.stringify(wiData)
	});

	if (!r.ok) {
		reportError("WI Upload Error", `WI upload failed with status code ${r.status}. Please report this.`);
		return;
	}
}

async function loadNAILorebook(data, filename, image=null) {
	let lorebookVersion = data.lorebookVersion;
	let wi_data = {folders: {[filename]: []}, entries: {}};
	console.log(`Loading NAI lorebook version ${lorebookVersion}`);

	let base = Math.max(...Object.keys(world_info_data).map(Number)) + 1;
	if (base < 0) base = 0;

	let i = 0;
	for (const entry of data.entries) {
		// contextConfig: Object { suffix: "\n", tokenBudget: 2048, reservedTokens: 0, … }
		// displayName: "Aboleth"
		// enabled: true
		// forceActivation: false
		// keys: Array [ "Aboleth" ]
		// lastUpdatedAt: 1624443329051
		// searchRange: 1000
		// text

		wi_data.entries[i.toString()] = {
			"uid": i,
			"title": entry.displayName,
			"key": entry.keys,
			"keysecondary": [],
			"folder": filename,
			"constant": entry.forceActivation,
			"content": "",
			"manual_text": entry.text,
			"comment": "",
			"token_length": 0,
			"selective": false,
			"wpp": {"name": "", "type": "", "format": "W++", "attributes": {}},
			"use_wpp": false,
		};
		wi_data.folders[filename].push(i);


		i++;
	}

	await postWI(wi_data);

	if (image) {
		for (let offset=0;offset<i;offset++) {
			let uid = base+offset;

			const imageEl = $el(`#world_info_image_${uid}`);
			if (imageEl) imageEl.src = image;

			let r = await fetch(`/set_wi_image/${uid}`, {
				method: "POST",
				body: image
			});
		}
	}
}

async function loadKoboldJSON(data, filename) {
	if (data.gamestarted !== undefined) {
		// Story
		socket.emit("upload_file", {
			filename: filename,
			data: new Blob([JSON.stringify(data)]),
			upload_no_save: true
		});
		socket.emit("load_story_list", "");
		load_story_list();
	} else if (data.folders !== undefined && data.entries !== undefined) {
		// World Info Folder
		await postWI(data);
	} else {
		// Bad data
		reportError("Error loading file", `Unable to detect ${filename} as a valid KoboldAI file.`);
		console.error("Bad data!");
		return;
	}
}

async function blob2Base64(blob) {
	return new Promise(function(resolve, reject) {
		const reader = new FileReader();
		reader.readAsDataURL(blob);
		reader.onload = () => resolve(reader.result);
		reader.onerror = error => reject(error);
	});
}

async function blob2ArrayBuffer(blob) {
	return new Promise(function(resolve, reject) {
		const reader = new FileReader();
		reader.readAsArrayBuffer(blob);
		reader.onload = () => resolve(reader.result);
		reader.onerror = error => reject(error);
	});
}

async function readLoreCard(file) {
	// "naidata"
	const magicNumber = new Uint8Array([0x6e, 0x61, 0x69, 0x64, 0x61, 0x74, 0x61]);
	
	let filename = file.name;
	let reader = new FileReader();

	let bin = new Uint8Array(await blob2ArrayBuffer(file))

	// naidata is prefixed with magic number
	let offset = bin.findIndex(function(item, possibleIndex, array) {
		for (let i=0;i<magicNumber.length;i++) {
			if (bin[i + possibleIndex] !== magicNumber[i]) return false;
		}
		return true;
	});

	if (offset === null) {
		reportError("Error reading Lorecard", "Unable to find NAIDATA offset. Is this a valid Lorecard?");
		throw Error("Couldn't find offset!");
	}
	
	let lengthBytes = bin.slice(offset - 8, offset - 4);
	let length = 0;
	
	for (const byte of lengthBytes) {
		length = (length << 8) + byte;
	}
	
	let binData = bin.slice(offset + 8, offset + length);
	
	// Encoded in base64
	let data = atob(new TextDecoder().decode(binData));
	let j = JSON.parse(data);
	let b64Image = await blob2Base64(file);
	loadNAILorebook(j, filename, b64Image);
}

async function processDroppedFile(file) {
	let extension = /.*\.(.*)/.exec(file.name)[1];
	console.log("file is", file)
	let data;

	switch (extension) {
		case "png":
			// NovelAI lorecard, a png with a lorebook file embedded inside it.
			readLoreCard(file);
			break;
		case "json":
			// KoboldAI file (old story, etc)
			data = JSON.parse(await file.text());
			loadKoboldJSON(data, file.name);
			break;
		case "kaistory":
			// KoboldAI story file
			let r = await fetch(`/upload_kai_story/${file.name}`, {
				method: "POST",
				body: file
			});
			break;
		case "lorebook":
			// NovelAI lorebook, JSON encoded.
			data = JSON.parse(await file.text());
			loadNAILorebook(data, file.name);
			break;
		case "css":
			console.warn("TODO: THEME");
			reportError("Unsupported", "Theme drag and drop is not implemented yet. Check back later!");
			break;
		case "lua":
			console.warn("TODO: USERSCRIPT");
			reportError("Unsupported", "Userscript drag and drop is not implemented yet. Check back later!");
			break
	}
}

function highlightEl(element) {
	if (typeof element === "string") element = document.querySelector(element);
	if (!element) {
		console.error("Bad jump!")
		return;
	}
	
	const area = $(element).closest(".tab-target")[0];
	if (area) {
		// If we need to click a tab to make the element visible, do so.
		let tab = Array.from($(".tab")).filter((c) => c.getAttribute("tab-target") === area.id)[0];
		tab.click();
	}

	element.scrollIntoView();
	return element;
}

function focusEl(element) {
	const el = highlightEl(element);
	if (el) el.focus();
}

function addSearchListing(action, highlight) {
	const finder = document.getElementById("finder");

	let result = document.createElement("div");
	result.classList.add("finder-result");
	result.addEventListener("click", function(event) {
		closePopups();
		action.func();
	});

	let textblock = document.createElement("div");
	textblock.classList.add("result-textbox");
	result.appendChild(textblock);

	let titleEl = document.createElement("span");
	titleEl.classList.add("result-title");
	titleEl.innerText = action.name;

	// TODO: Sanitation
	titleEl.innerHTML = titleEl.innerHTML.replace(
		new RegExp(`(${highlight})`, "i"),
		'<span class="result-highlight">$1</span>'
	);
	textblock.appendChild(titleEl);

	if (action.desc) {
		let descriptionEl = document.createElement("span");
		descriptionEl.classList.add("result-details");
		descriptionEl.innerText = action.desc;
		descriptionEl.innerHTML = descriptionEl.innerHTML.replace(
			new RegExp(`(${highlight})`, "i"),
			'<span class="result-highlight">$1</span>'
		);

		// It can get cut off by CSS, so let's add a tooltip.
		descriptionEl.setAttribute("title", action.desc);
		textblock.appendChild(descriptionEl);
	}

	let icon = document.createElement("span");
	icon.classList.add("result-icon");
	icon.classList.add("material-icons-outlined");

	// TODO: Change depending on what pressing enter does
	icon.innerText = action.icon;
	result.appendChild(icon)

	finder.appendChild(result);

	return result;
}

function updateStandardSearchListings(query) {
	const maxResultCount = 5;
	const actionMatches = {name: [], desc: []};

	for (const action of finder_actions) {
		if (action.name.toLowerCase().includes(query)) {
			actionMatches.name.push(action);
		} else if (action.desc && action.desc.toLowerCase().includes(query)) {
			actionMatches.desc.push(action);
		}
	}

	// Title matches over desc matches
	const matchingActions = actionMatches.name.concat(actionMatches.desc);


	for (let i=0;i<maxResultCount && i<matchingActions.length;i++) {
		let action = matchingActions[i];
		addSearchListing(action, query);
	}
}

function $e(tag, parent, attributes, insertionLocation=null) {
	// Small helper function for dynamic UI creation

	let element = document.createElement(tag);

	if (!attributes) attributes = {};

	if ("classes" in attributes) {
		if (!Array.isArray(attributes.classes)) throw Error("Classes was not array!");
		for (const className of attributes.classes) {
			element.classList.add(className);
		}
		delete attributes.classes;
	}


	for (const [attribute, value] of Object.entries(attributes)) {
		if (attribute.includes(".")) {
			let ref = element;
			const parts = attribute.split(".");

			for (const part of parts.slice(0, -1)) {
				ref = ref[part];
			}

			ref[parts[parts.length - 1]] = value;
			continue;
		}

		if (attribute in element) {
			element[attribute] = value;
		} else {
			element.setAttribute(attribute, value);
		}
	}

	if (!parent) return element;

	if (insertionLocation && Object.keys(insertionLocation).length) {
		let [placement, target] = Object.entries(insertionLocation)[0];
		if (placement === "before") {
			parent.insertBefore(element, target);
		} else if (placement === "after") {
			parent.insertBefore(element, target.nextSibling);
		} else {
			throw Error(`I have no clue what placement ${placement} is`);
		}
	} else {
		parent.appendChild(element);
	}

	return element;
}

function makeFinderWITag(name, container, isPrimary, uid) {
	let wiTag = $e("span", container, {classes: ["tag"]});
	let wiTagIcon = $e("span", wiTag, {classes: ["finder-wi-tag-icon", "material-icons-outlined"], innerText: "close"});
	let wiTagText = $e("span", wiTag, {innerText: name, contenteditable: true});

	wiTagIcon.addEventListener("click", function(e) {
		socket.emit(
			"update_wi_keys",
			{uid: parseInt(uid), key: name, is_secondary: !isPrimary, operation: "remove"}
		);
		wiTag.remove();
	});
}

function updateWIInfo(event) {
	// Should be a change event or something similar. This WILL send an update
	// packet on each unfocus in some cases. It's not that big of a deal, right? :p

	let key = event.target.getAttribute("wi-sync");

	if ("checked" in event.target) {
		// checkbox / toggle
		value = event.target.checked;
	} else if ("value" in event.target) {
		// standard inputs
		value = event.target.value;
	} else {
		// contenteditable
		value = event.target.innerText
	}

	let uid = $(event.target).closest(".finder-wi-block")[0].getAttribute("wi-uid");
	socket.emit("update_wi_attribute", {uid: parseInt(uid), key: key, value: value});
}

function updateWISearchListings(data) {
	wi_finder_offset = 0;
	wi_finder_data = Object.values(data).flat();
	renderWISearchListings();
}

function renderWISearchListings() {
	const wiCarousel = document.getElementById("finder-wi-carousel");
	$(".finder-wi-block").remove();

	let data = Array.from(wi_finder_data);

	// No need for excessive shifting
	let realOffset = wi_finder_offset % data.length;

	// Make first be central
	if (data.length > 2) realOffset--;

	// Wrap around
	if (realOffset < 0) realOffset += data.length;

	// Actual data wrap
	for (let i=0;i<realOffset;i++) {
		data.push(data.shift());
	}
	let entries = data.slice(0, 3);

	// Visual spacing-- this kinda sucks
	if (entries.length == 1) entries = [null, entries[0], null];
	if (entries.length == 2) entries = [null, ...entries];

	for (const [i, entry] of entries.entries()) {
		let wiBlock = $e("div", wiCarousel, {classes: ["finder-wi-block"], "wi-uid": entry ? entry.uid : "null"});

		// Spacer hack
		if (!entry) {
			wiBlock.style.visibility = "hidden";
			continue;
		}

		// The "position" relative to others.
		let current = "center";

		if (entries.length == 3) {
			if (i !== 1) current = (i == 0) ? "left" : "right";
		} else if (entries.length == 2) {
			if (i === 1) current = "right";
		}

		if (current !== "center") {
			let blanket = $e("div", wiBlock, {classes: ["finder-wi-blanket"]});
		}

		if (current === "left") {
			wiBlock.addEventListener("click", function(event) {
				wi_finder_offset--;
				renderWISearchListings();
				event.preventDefault();
			});
		} else if (current === "right") {
			wiBlock.addEventListener("click", function(event) {
				wi_finder_offset++;
				renderWISearchListings();
				event.preventDefault();
			});
		} else if (current === "center") {
			// Focus is the center highlighted one. If there is 3 entries (max),
			// the important one is at the center. Otherwise, the important one
			// is in the front.
			wiBlock.classList.add("finder-wi-focus");
		}


		let wiTitle = $e("span", wiBlock, {
			classes: ["finder-wi-title"],
			innerText: entry.title,
			contenteditable: true,
			"data-placeholder": "Entry",
			"wi-sync": "title",
		});
		wiTitle.addEventListener("keydown", function(e) {
			if (e.key === "Enter") {
				e.preventDefault();
				wiTitle.blur();
			}
		});
		wiTitle.addEventListener("blur", updateWIInfo);

		let wiTextLabel = $e("h3", wiBlock, {innerText: "Info", "style.margin": "10px 0px 5px 0px"});
		let wiContent = $e("textarea", wiBlock, {
			classes: ["finder-wi-content"],
			value: entry.content,
			placeholder: "Write your World Info here!",
			"wi-sync": "content",
		});
		wiContent.addEventListener("blur", updateWIInfo);

		let wiComment = $e("textarea", wiBlock, {
			placeholder: "Comment",
			value: entry.comment,
			"wi-sync": "comment",
		});
		wiComment.addEventListener("blur", updateWIInfo);

		let wiActivationHeaderContainer = $e("div", wiBlock, {classes: ["finder-wi-activation-header-container"]});
		let wiActivationLabel = $e("h3", wiActivationHeaderContainer, {innerText: "Activation", "style.display": "inline"});
		let wiAlwaysContainer = $e("div", wiActivationHeaderContainer, {classes: ["finder-wi-always-container"]});
		let wiAlwaysLabel = $e("span", wiAlwaysContainer, {innerText: "Always Activate"});

		let wiActivationHelp = $e("span", wiBlock, {classes: ["help_text"], innerText: "Change when the AI reads this World Info entry"})
		let wiTagActivationContainer = $e("div", wiBlock);

		let wiAlways = $e("input", wiAlwaysContainer, {
			type: "checkbox",
			"wi-sync": "constant",
		});
		$(wiAlways).change(function(e) {
			updateWIInfo(e);
			if (this.checked) {
				wiTagActivationContainer.classList.add("disabled");
			} else {
				wiTagActivationContainer.classList.remove("disabled");
			}
		});
		$(wiAlways).bootstrapToggle({
			size: "mini",
			onstyle: "success",
		});
		$(wiAlways).bootstrapToggle(entry.constant ? "on" : "off");

		for (const isPrimary of [true, false]) {
			let wiTagLabel = $e("span", wiTagActivationContainer, {
				"style.display": "block",
				innerText: isPrimary ? "Requires one of:" : "And (if present):"
			});

			let wiTagContainer = $e("div", wiTagActivationContainer, {
				id: isPrimary ? "finder-wi-required-keys" : "finder-wi-secondary-keys",
				classes: ["finder-wi-keys"]
			});
			let wiAddedTagContainer = $e("div", wiTagContainer, {classes: ["finder-wi-added-keys"]});

			// Existing keys
			for (const key of entry.key) {
				makeFinderWITag(key, wiAddedTagContainer, isPrimary, entry.uid);
			}

			// The "fake key" add button
			let wiNewTag = $e("span", wiTagContainer, {classes: ["tag"]});
			let wiNewTagIcon = $e("span", wiNewTag, {classes: ["finder-wi-tag-icon", "material-icons-outlined"], innerText: "add"});
			let wiNewTagText = $e("span", wiNewTag, {classes: ["tag-text"], contenteditable: true, "data-placeholder": "Key"});

			function newTag() {
				let tagName = wiNewTagText.innerText;
				wiNewTagText.innerText = "";
				if (!tagName.trim()) return;
				makeFinderWITag(tagName, wiAddedTagContainer, isPrimary, entry.uid)

				socket.emit(
					"update_wi_keys",
					{uid: parseInt(entry.uid), key: tagName, is_secondary: !isPrimary, operation: "add"}
				);
			}

			wiNewTagText.addEventListener("blur", newTag);
			wiNewTagText.addEventListener("keydown", function(e) {
				if (e.key === "Enter") {
					newTag();
					e.preventDefault();
				}
			});
		}
	}
}

function recieveScratchpadResponse(data) {
	const scratchpadResponse = document.querySelector("#finder-scratchpad-response");

	clearInterval(finder_waiting_id);
	finder_waiting_id = null;

	scratchpadResponse.innerText = data;
}

function sendScratchpadPrompt(prompt) {
	// Already waiting on prompt...
	if (finder_waiting_id) return;

	const scratchpad = document.querySelector("#finder-scratchpad");
	const scratchpadPrompt = document.querySelector("#finder-scratchpad-prompt");
	const scratchpadResponse = document.querySelector("#finder-scratchpad-response");

	scratchpadPrompt.innerText = prompt;
	scratchpadResponse.innerText = "...";

	scratchpad.classList.remove("hidden");

	finder_waiting_id = setInterval(function() {
		// Little loading animation so user doesn't think nothing is happening.
		// TODO: Replace this with token streaming WHEN AVAILABLE.

		let index = scratchpadResponse.innerText.indexOf("|");
		if (index === 2) {
			scratchpadResponse.innerText = "...";
			return;
		}
		let buf = "";

		index++;

		for (let i=0;i<index;i++) buf += ".";
		buf += "|";
		for (let i=0;i<2-index;i++) buf += ".";

		scratchpadResponse.innerText = buf;
	}, 1000);

	socket.emit("scratchpad_prompt", prompt);
}

function finderSendImgPrompt(prompt) {
	closePopups();
	$el("#image-loading").classList.remove("hidden");
	socket.emit("generate_image_from_prompt", prompt);
}

function updateSearchListings() {
	if (["scratchpad", "imgPrompt"].includes(finder_mode)) return;
	if (this.value === finder_last_input) return;
	finder_last_input = this.value;
	finder_selection_index = -1;

	const wiCarousel = document.getElementById("finder-wi-carousel");
	wiCarousel.classList.add("hidden");

	let query = this.value.toLowerCase();

	// TODO: Maybe reuse the element? Would it give better performance?
	$(".finder-result").remove();

	if (!query) return;

	if (finder_mode === "wi") {
		wiCarousel.classList.remove("hidden");
		$(".finder-wi-block").remove();
		socket.emit("search_wi", {query: query});
	} else if (finder_mode === "ui") {
		updateStandardSearchListings(query)
	}
}

function updateFinderSelection() {
	let former = document.getElementsByClassName("result-selected")[0];
	if (former) former.classList.remove("result-selected");

	let newSelection = document.getElementsByClassName("finder-result")[finder_selection_index];
	newSelection.classList.add("result-selected");
}

function updateFinderMode(mode) {
	const finderIcon = document.querySelector("#finder-icon");
	const finderInput = document.querySelector("#finder-input");
	const finderScratchpad = document.querySelector("#finder-scratchpad");

	finderIcon.innerText = {ui: "search", wi: "auto_stories", scratchpad: "speaker_notes", "imgPrompt": "image"}[mode];
	finderInput.placeholder = {
		ui: "Search for something...",
		wi: "Search for a World Info entry...",
		scratchpad: "Prompt the AI...",
		imgPrompt: "Generate an image..."
	}[mode];
	finderScratchpad.classList.add("hidden");

	finder_mode = mode;
}

function cycleFinderMode() {
	// Initiated by clicking on icon
	updateFinderMode({ui: "wi", wi: "scratchpad", scratchpad: "imgPrompt", imgPrompt: "ui"}[finder_mode]);
}

function open_finder() {
	const finderInput = document.getElementById("finder-input");
	finderInput.value = "";
	$(".finder-result").remove();
	finder_selection_index = -1;
	updateFinderMode("ui");
	
	openPopup("finder");
	finderInput.focus();
}

function process_cookies() {
	if (getCookie("Settings_Pin") == "false") {
		settings_unpin();
	} else if (getCookie("Settings_Pin") == "true") {
		settings_pin();
	}
	if (getCookie("Story_Pin") == "true") {
		story_pin();
	} else if (getCookie("Story_Pin") == "false") {
		story_unpin();
	}
	if (getCookie("preserve_game_space") == "false") {
		preserve_game_space(false);
	} else if (getCookie("preserve_game_space") == "true") {
		preserve_game_space(true);
	}
	if (getCookie("options_on_right") == "false") {
		options_on_right(false);
	} else if (getCookie("options_on_right") == "true") {
		options_on_right(true);
	}
	
	Change_Theme(getCookie("theme", "Monochrome"));
	
	//set font size
	new_font_size = getCookie("font_size", 1);
	var r = document.querySelector(':root');
	r.style.setProperty("--game_screen_font_size_adjustment", new_font_size);
	document.getElementById('font_size_cur').value = new_font_size;
	document.getElementById('font_size').value = new_font_size;
	
	
	load_tweaks();
}

function position_context_menu(contextMenu, x, y) {
	// Calculate where to position context menu based on window confines and
	// menu size.

	let height = contextMenu.clientHeight;
	let width = contextMenu.clientWidth;

	let bounds = {
		top: 0,
		bottom: window.innerHeight,
		left: 0,
		right: window.innerWidth,
	};

	let farMenuBounds = {
		top: y,
		bottom: y + height,
		left: x,
		right: x + width,
	};

	if (farMenuBounds.right > bounds.right) x -= farMenuBounds.right - bounds.right;
	if (farMenuBounds.bottom > bounds.bottom) y -= farMenuBounds.bottom - bounds.bottom;

	contextMenu.style.left = `${x}px`;
	contextMenu.style.top = `${y}px`;
}

function updateTitle() {
	const titleInput = $el(".var_sync_story_story_name");
	if (!titleInput.innerText) return;
	document.title = `${titleInput.innerText} - KoboldAI Client`;
}

function openClubImport() {
	$el("#aidgpromptnum").value = "";
	openPopup("aidg-import-popup");
}

//// INIT ////

document.onkeydown = detect_key_down;
document.onkeyup = detect_key_up;
document.getElementById("input_text").onkeydown = detect_enter_submit;

/* -- Popups -- */
function openPopup(id) {
	closePopups();

	const container = $el("#popup-container");
	container.classList.remove("hidden");

	for (const popupWindow of container.children) {
		popupWindow.classList.add("hidden");
	}

	const popup = $el(`#${id}`);
	popup.classList.remove("hidden");

	// Sometimes we want to instantly focus on certain elements when a menu opens.
	for (const noticeMee of popup.getElementsByClassName("focus-on-me")) {
		noticeMee.focus();
		break;
	}
}

function closePopups() {
	const container = $el("#popup-container");
	container.classList.add("hidden");

	for (const popupWindow of container.children) {
		popupWindow.classList.add("hidden");
	}
}

$el("#popup-container").addEventListener("click", function(event) {
	if (event.target === this) closePopups();
});

/* -- Colab Cookie Handling -- */
if (colab_cookies != null) {
	for (const cookie of Object.keys(colab_cookies)) {
		setCookie(cookie, colab_cookies[cookie]);
	}	
	colab_cookies = null;
}

//create_theming_elements();

/* -- Tweak Registering -- */
for (const tweakContainer of document.getElementsByClassName("tweak-container")) {
	let toggle = tweakContainer.querySelector("input");

	$(toggle).change(function(e) {
		let path = $(this).closest(".tweak-container")[0].getAttribute("tweak-path");
		let id = `tweak-${path}`;

		if (this.checked) {
			let style = document.createElement("link");
			style.rel = "stylesheet";
			style.href = `/themes/tweaks/${path}.css`;
			style.id = id;
			document.head.appendChild(style);
		} else {
			let el = document.getElementById(id);
			if (el) el.remove();
		}

		save_tweaks();
	});
}

process_cookies();

/* -- Drag and Drop -- */
(function() {
	let lastTarget = null;

	document.body.addEventListener("drop", function(e) {
		e.preventDefault();
		$("#file-upload-notice")[0].classList.add("hidden");

		// items api
		if (e.dataTransfer.items) {
			for (const item of e.dataTransfer.items) {
				if (item.kind !== "file") continue;
				let file = item.getAsFile();
				processDroppedFile(file);
			}
		} else {
			for (const file of e.dataTransfer.files) {
				processDroppedFile(file);
			}
		}
	});

	document.body.addEventListener("dragover", function(e) {
		e.preventDefault();
	});

	document.body.addEventListener("dragenter", function(e) {
		if (!e.dataTransfer.types.includes("Files")) return;
		lastTarget = e.target;
		$("#file-upload-notice")[0].classList.remove("hidden");
	});

	document.body.addEventListener("dragleave", function(e) {
		if (!(e.target === document || e.target === lastTarget)) return;

		$("#file-upload-notice")[0].classList.add("hidden");
	});
})();

/* -- Finder -- */
(function() {
	const finderInput = document.getElementById("finder-input");
	const finderIcon = document.getElementById("finder-icon");

	// Parse settings for Finder
	for (const el of $(".setting_label")) {
		let name = el.children[0].innerText;

		let tooltipEl = el.getElementsByClassName("helpicon")[0];
		let tooltip = tooltipEl ? tooltipEl.getAttribute("tooltip") : null;

		finder_actions.push({
			name: name,
			desc: tooltip,
			icon: "open_in_new",
			type: "setting",
			func: function () { highlightEl(el.parentElement) },
		});
	}

	const themeSelector = $el("#selected_theme");

	function updateThemes() {
		finder_actions = finder_actions.filter(x => x.type !== "theme")
		// Parse themes for Finder
		for (const select of themeSelector.children) {
			let themeName = select.value;
			//console.log(themeName)
			//console.log("curve")
			finder_actions.push({
				name: themeName,
				desc: "Apply this theme to change how KoboldAI looks!",
				icon: "palette",
				type: "theme",
				func: function () {
					themeSelector.value = themeName;
					themeSelector.onchange();
				},
			});
		}
	}

	updateThemes();
	themeSelector.addEventListener("sync", updateThemes);

	for (const el of $(".collapsable_header")) {
		// https://stackoverflow.com/a/11347962
		let headerText = $(el.children[0]).contents().filter(function() {
			return this.nodeType == 3;
		}).text().trim();
		
		finder_actions.push({
			name: headerText,
			icon: "open_in_new",
			func: function () { highlightEl(el) },
		});
	}

	finderIcon.addEventListener("click", cycleFinderMode);
	finderInput.addEventListener("keyup", updateSearchListings);
	finderInput.addEventListener("keydown", function(event) {
		let delta = 0;
		const actions = document.getElementsByClassName("finder-result");

		let newMode = {">": "wi", "#": "ui", "!": "scratchpad", "?": "imgPrompt"}[event.key];
		if (newMode && !finderInput.value) {
			event.preventDefault();
			updateFinderMode(newMode);
			return;
		}

		if (event.key === "Enter") {
			if (finder_mode === "scratchpad") {
				sendScratchpadPrompt(finderInput.value);
				return;
			} else if (finder_mode === "imgPrompt") {
				finderSendImgPrompt(finderInput.value);
				return;
			} else if (finder_mode === "ui") {
				let index = finder_selection_index >= 0 ? finder_selection_index : 0;
				if (!actions[index]) return;
				actions[index].click();
			} else {
				return;
			}
		} else if (event.key === "ArrowUp") {
			delta = -1;
		} else if (event.key === "ArrowDown") {
			delta = 1
		} else if (event.key === "Tab") {
			delta = event.shiftKey ? -1 : 1;
		} else {
			return;
		}

		if (finder_mode !== "ui") return;

		const actionsCount = actions.length;
		let future = finder_selection_index + delta;

		event.preventDefault();

		if (future >= actionsCount) {
			future = 0;
		} else if (future < 0) {
			future = actionsCount - 1;
		}

		finder_selection_index = future;
		updateFinderSelection(delta);
	});
})();

/* -- Context Menu -- */
(function() {
	const contextMenu = $e("div", document.body, {id: "context-menu", classes: ["hidden"]});
	let summonEvent = null;

	for (const [key, actions] of Object.entries(context_menu_actions)) {
		for (const action of actions) {
			// Null adds horizontal rule
			if (!action) {
				$e("hr", contextMenu, {classes: [`context-menu-${key}`]});
				continue;
			}


			let item = $e("div", contextMenu, {
				classes: ["context-menu-item", "noselect", `context-menu-${key}`],
				"enabled-on": action.enabledOn,
				"cache-index": context_menu_cache.length
			});

			context_menu_cache.push({shouldShow: action.shouldShow});

			let icon = $e("span", item, {classes: ["material-icons-outlined"], innerText: action.icon});
			item.append(action.label);

			item.addEventListener("mousedown", e => e.preventDefault());
			// Expose the "summonEvent" to enable access to original context menu target.
			item.addEventListener("click", () => action.click(summonEvent));
		}
	}

	// When we make a browser context menu, close ours.
	document.addEventListener("contextmenu", function(event) {
		let target = event.target;
		while (!target.hasAttribute("context-menu")) {
			target = target.parentElement;
			if (!target) break;
		}

		// If no custom context menu or control is held, do not run our custom
		// logic or cancel the browser's.
		if (!target || event.ctrlKey) {
			contextMenu.classList.add("hidden");
			return;
		}

		summonEvent = event;

		// Show only applicable actions in the context menu
		let contextMenuType = target.getAttribute("context-menu");
		for (const contextMenuItem of contextMenu.childNodes) {
			let shouldShow = contextMenuItem.classList.contains(`context-menu-${contextMenuType}`);

			if (shouldShow) {
				let cacheIndex = parseInt(contextMenuItem.getAttribute("cache-index"));
				let cacheEntry = context_menu_cache[cacheIndex];
				if (cacheEntry && cacheEntry.shouldShow) shouldShow = cacheEntry.shouldShow();
			}

			if (shouldShow) {
				contextMenuItem.classList.remove("hidden");
			} else {
				contextMenuItem.classList.add("hidden");
			}
		}

		// Don't open browser context menu
		event.preventDefault();

		// Close if open
		if (!contextMenu.classList.contains("hidden")) {
			contextMenu.classList.add("hidden");
			return;
		}

		// Disable non-applicable items
		$(".context-menu-item").addClass("disabled");
		
		// A selection is made
		if (getSelectionText()) $(".context-menu-item[enabled-on=SELECTION]").removeClass("disabled");
		
		// The caret is placed
		if (get_caret_position(target) !== null) $(".context-menu-item[enabled-on=CARET]").removeClass("disabled");

		// The generated image is present
		if ($el(".action_image")) $(".context-menu-item[enabled-on=GENERATED-IMAGE]").removeClass("disabled");

		$(".context-menu-item[enabled-on=ALWAYS]").removeClass("disabled");

		// Make sure hr isn't first or last visible element
		let visibles = [];
		for (const item of contextMenu.children) {
			if (!item.classList.contains("hidden")) visibles.push(item);
		}
		let lastIndex = visibles.length - 1;
		if (visibles[0].tagName === "HR") visibles[0].classList.add("hidden");
		if (visibles[lastIndex].tagName === "HR") visibles[lastIndex].classList.add("hidden");

		contextMenu.classList.remove("hidden");

		// Set position to click position
		position_context_menu(contextMenu, event.x, event.y);

		// Don't let the document contextmenu catch us and close our context menu
		event.stopPropagation();
	});

	// When we click outside of our context menu, close ours.
	document.addEventListener("click", function(event) {
		contextMenu.classList.add("hidden");
	});

	window.addEventListener("blur", function(event) {
		contextMenu.classList.add("hidden");
	});
})();


/* -- WI Ctrl+Click To Jump -- */
(function() {
	document.addEventListener("keydown", function(event) {
		// Change appearance of WI when holding control
		if (event.key !== "Control") return;
		control_held = true;

		const style = ".wi_match { text-decoration: underline; cursor: pointer; }";
		$e("style", document.head, {id: "wi-link-style", innerText: style})
	});

	// Remove on up
	document.addEventListener("keyup", function(event) {
		if (event.key !== "Control") return;
		control_held = false;

		const style = document.querySelector("#wi-link-style")
		if (style) style.remove();
	});

	document.getElementById("Selected Text").addEventListener("click", function(event) {
		// Control click on WI entry to jump
		if (!event.target.classList.contains("wi_match")) return;
		if (!control_held) return;

		let uid = event.target.getAttribute("wi-uid");
		let wiCard = document.getElementById(`world_info_${uid}`);
		highlightEl(wiCard);
	});
})();

/* -- Update Tab Title on Input and Sync -- */
(function() {
	const titleInput = $el(".var_sync_story_story_name");
	titleInput.addEventListener("input", updateTitle);
	titleInput.addEventListener("sync", updateTitle);

	// Title may not have been sent by this point. Fear not; We abort if
	// there's no title. If we have missed the title sync, however, this will
	// save us.
	updateTitle();
})();

/* Substitution */
let load_substitutions;
[load_substitutions] = (function() {
	// TODO: Don't allow multiple substitutions for one target
	
	// Defaults
	let substitutions = [];
	const substitutionContainer = $el("#substitution-container");
	let charMap = [];
	
	function getTrueTarget(bareBonesTarget) {
		// If -- is converted to the 2dash, we make a "true target" so that a 2dash and - is the 3dash.
		if (!bareBonesTarget) return bareBonesTarget;

		let tries = 0;
		let whatWeGot = bareBonesTarget;
		
		// eehhhh this kinda sucks but it's the best I can think of at the moment
		while (true) {
			// Sanity check; never 100% cpu!
			tries++;
			if (tries > 2000) {
				reportError("Substitution error", "Some Substitution shenanigans are afoot; please send the developers your substitutions!");
				throw Error("Substitution shenanigans!")
				return;
			}
			
			let escape = true;
			for (const c of substitutions) {
				if (c.target === bareBonesTarget) continue;
				if (!c.enabled) continue;

				if (whatWeGot.includes(c.target)) {
					whatWeGot = whatWeGot.replaceAll(c.target, c.substitution);
					escape = false;
					break;
				}
			}
			
			if (escape) break;
		}
		
		return whatWeGot;
	}
	
	function getSubstitutionIndex(cardElement) {
		for (const i in substitutions) {
			if (substitutions[i].card === cardElement) {
				return i
			}
		}

		reportError("Substitution error", "Couldn't find substitution index from card.");
		throw Error("Didn't find substitution!");
	}
	
	function getDuplicateCards(target) {
		let duplicates = [];
		
		for (const c of substitutions) {
			if (c.target === target) duplicates.push(c.card);
		}
		
		console.log(duplicates)
		return duplicates.length > 1 ? duplicates : [];
	}
	
	function makeCard(c) {
		// How do we differentiate -- and ---? Convert stuff!
		
		let card = $e("div", substitutionContainer, {classes: ["substitution-card"]});
		let leftContainer = $e("div", card, {classes: ["card-section", "card-left"]});
		let deleteIcon = $e("span", leftContainer, {classes: ["material-icons-outlined", "cursor"], innerText: "clear"});
		let targetInput = $e("input", leftContainer, {classes: ["target"], value: c.target});
		let rightContainer = $e("div", card, {classes: ["card-section"]});
		let substitutionInput = $e("input", rightContainer, {classes: ["target"], value: c.substitution});
		
		// HACK
		let checkboxId = "sbcb" + Math.round(Math.random() * 9999).toString();
		
		let enabledCheckbox = $e("input", rightContainer, {id: checkboxId, classes: ["true-t"], type: "checkbox", checked: c.enabled});
		let initCheckTooltip = c.enabled ? "Enabled" : "Disabled";

		// HACK: We don't use in-house tooltip as it's cut off by container :(
		let enabledVisual = $e("label", rightContainer, {for: checkboxId, "title": initCheckTooltip, classes: ["material-icons-outlined"]});
		
		targetInput.addEventListener("change", function() {
			let card = this.parentElement.parentElement;
			let i = getSubstitutionIndex(card);

			substitutions[i].target = this.value;

			// Don't do a full rebake
			substitutions[i].trueTarget = getTrueTarget(this.value);
			
			for (const duplicateCard of getDuplicateCards(this.value)) {
				if (duplicateCard === card) continue;
				console.log("DUPE", duplicateCard)
				substitutions.splice(getSubstitutionIndex(duplicateCard), 1);
				duplicateCard.remove();
			}

			rebuildCharMap();
			updateSubstitutions();
		});

		substitutionInput.addEventListener("change", function() {
			let card = this.parentElement.parentElement;
			let i = getSubstitutionIndex(card);

			substitutions[i].substitution = this.value;
			// No rebaking at all is needed, that all hinges on target value; not edited here.
			updateSubstitutions();
		});
		
		deleteIcon.addEventListener("click", function() {
			let card = this.parentElement.parentElement;
			
			// Find and remove from substitution array
			substitutions.splice(getSubstitutionIndex(card), 1);
			updateSubstitutions();
			rebakeSubstitutions();
			card.remove();
		});
		
		enabledCheckbox.addEventListener("change", function() {
			let card = this.parentElement.parentElement;
			let i = getSubstitutionIndex(card);
			console.log(this.checked)

			substitutions[i].enabled = this.checked;
			enabledVisual.setAttribute("title", this.checked ? "Enabled" : "Disabled")
			rebakeSubstitutions();
			updateSubstitutions();
		});
		
		return card;
	}
	
	function updateSubstitutions() {
		let subs = substitutions.map(x => ({target: x.target, substitution: x.substitution, trueTarget: x.trueTarget, enabled: x.enabled}));
		socket.emit("substitution_update", subs);
	}
	
	function rebakeSubstitutions() {
		for (const c of substitutions) {
			c.trueTarget = getTrueTarget(c.target);
		}
		rebuildCharMap();
	}
	
	function rebuildCharMap() {
		charMap = [];
		for (const c of substitutions) {
			if (!c.enabled) continue;
			for (const char of c.target) {
				if (!charMap.includes(char)) charMap.push(char)
			}
		}
	}
	
	const newCardButton = $el("#new-sub-card");
	newCardButton.addEventListener("click", function() {
		let c = {target: "", substitution: "", enabled: true}
		substitutions.push(c);
		c.card = makeCard(c);
		//newCardButton.scrollIntoView(false);
	});
	
	// Event handler on input
	// TODO: Apply to all of gametext
	const inputText = $el("#input_text");
	inputText.addEventListener("keydown", function(event) {
		if (event.ctrlKey) return;
		if (event.ctrlKey) return;
		if (!charMap.includes(event.key)) return;

		let caretPosition = inputText.selectionStart;
		// We don't have to worry about special keys due to charMap (hopefully)
		let futureValue = inputText.value.slice(0, caretPosition) + event.key + inputText.value.slice(caretPosition);

		for (const c of substitutions) {
			if (!c.target) continue;
			if (!c.enabled) continue;

			let t = c.trueTarget;
			let preCaretPosition = caretPosition - t.length + 1;
			let bit = futureValue.slice(caretPosition - t.length + 1, caretPosition + 1)
			
			if (bit === t) {
				// We're doing it!!!!
				event.preventDefault();
				
				// Assemble the new text value
				let before = inputText.value.slice(0, caretPosition - t.length + 1);
				let after = inputText.value.slice(caretPosition);
				let newText = before + c.substitution + after;
				
				inputText.value = newText;
				
				// Move cursor back after setting text
				let sLength = c.substitution.length;
				inputText.selectionStart = preCaretPosition + sLength;
				inputText.selectionEnd = preCaretPosition + sLength;

				break;
			}
		}
	});
	
	let firstLoad = true;
	
	function load_substitutions(miniSubs) {
		// HACK: Does the same "replace all on load" thing that WI does; tab
		// support is broken and overall that kinda sucks. Would be nice to
		// make a robust system for syncing multiple entries.
		
		//console.log("load", miniSubs)

		$(".substitution-card").remove();
		// we only get target, trueTarget, and such
		for (const c of miniSubs) {
			if (!c.trueTarget) c.trueTarget = getTrueTarget(c.target);
			//if (!c.enabled) c.enabled = false;
			c.card = makeCard(c);
		}
		substitutions = miniSubs;
		rebuildCharMap();
		
		// We build trueTarget on the client, and it's not initalized on the server because I'm lazy.
		// May want to do that on the server in the future.
		if (firstLoad) updateSubstitutions();
		firstLoad = false;
	}

	return [load_substitutions];
})();

/* -- Tooltips -- */
function initalizeTooltips() {
	const tooltip = $e("span", document.body, {id: "tooltip-text", "style.display": "none"});
	let tooltipTarget = null;

	function alterTooltipState(target, specialClass=null) {
		tooltipTarget = target;
		tooltip.style.display = target ? "block" : "none";
		tooltip.className = specialClass || "";
	}

	function registerElement(el) {
		// el should have attribute "tooltip"
		let text = el.getAttribute("tooltip");

		el.addEventListener("mouseenter", function(event) {
			if (!el.hasAttribute("tooltip")) return;
			tooltip.innerText = text;
			let specialClass = "tooltip-standard";

			// Kinda lame
			if (this.classList.contains("context-token")) specialClass = "tooltip-context-token";

			alterTooltipState(el, specialClass);
		});

		el.addEventListener("mouseleave", function(event) {
			alterTooltipState(null);
		});
	}

	const xOffset = 10;
	const yOffset = 15;

	document.addEventListener("mousemove", function(event) {
		if (!tooltipTarget) return;

		let [x, y] = [event.x, event.y];

		// X + the tooltip's width is the farthest point right we will display;
		// let's account for it. If we will render outside of the window,
		// subtract accordingly.
		let xOverflow = (x + tooltip.clientWidth) - window.innerWidth;
		if (xOverflow > 0) x -= xOverflow;

		if (xOverflow + xOffset < 0) x += xOffset;

		// Same for Y!
		let yOverflow = (y + tooltip.clientHeight) - window.innerHeight;
		if (yOverflow > 0) y -= yOverflow;

		if (yOverflow + yOffset < 0) y += yOffset;

		tooltip.style.left = `${x}px`;
		tooltip.style.top = `${y}px`;
	});

	// Inital scan
	for (const element of document.querySelectorAll("[tooltip]")) {
		registerElement(element);
	}

	// Use a MutationObserver to catch future tooltips
	const observer = new MutationObserver(function(records, observer) {
		for (const record of records) {
			
			if (record.type === "attributes") {
				// Sanity check
				if (record.attributeName !== "tooltip") continue;
				registerElement(record.target);
				continue;
			}
			
			// If we remove the tooltip target, stop showing the tooltip. Maybe a little ineffecient.
			if (!document.body.contains(tooltipTarget)) alterTooltipState(null);

			for (const node of record.addedNodes) {
				if (node.nodeType !== 1) continue;

				if (node.hasAttribute("tooltip")) registerElement(node);

				// Register for descendants (Slow?)
				for (const element of node.querySelectorAll("[tooltip]")) {
					registerElement(element);
				}
			}
		}
	});
	observer.observe(document.body, {
		childList: true,
		subtree: true,
		attributeFilter: ["tooltip"],
	});
}

/* -- Shortcuts -- */
(function() {
	document.addEventListener("keydown", function(event) {
		for (const shortcut of shortcuts) {
			if (!(event.ctrlKey || event.altKey)) continue;
			if (event.ctrlKey && shortcut.mod !== "ctrl") continue;
			if (event.altKey && shortcut.mod !== "alt") continue;

			if (shortcut.key !== event.key) continue;
			if (shortcut.criteria && !shortcut.criteria()) continue;
			event.preventDefault();
			shortcut.func();
		}
	});

	// Display shortcuts in popup
	const shortcutContainer = $el("#shortcut-container");
	for (const shortcut of shortcuts) {
		const shortcutRow = $e("div", shortcutContainer, {classes: ["shortcut-item"]});
		const shortcutEl = $e("div", shortcutRow, {classes: ["shortcut-keys"]});
		const pretty = shortcut.mod[0].toUpperCase() + shortcut.mod.slice(1);

		for (const key of [pretty, shortcut.key.toUpperCase()]) {
			$e("span", shortcutEl, {classes: ["shortcut-key"], innerText: key});
		}
		const shortcutDesc = $e("div", shortcutRow, {classes: ["shortcut-desc"], innerText: shortcut.desc});
	}
})();

function showNotification(title, text, type) {
	if (!["error", "info"].includes(type)) return;
	const nContainer = $el("#notification-container");
	const notification = $e("div", nContainer, {classes: ["notification", `notification-${type}`]});
	const nTextContainer = $e("div", notification, {classes: ["notif-text"]});
	const titleEl = $e("span", nTextContainer, {classes: ["notif-title"], innerText: title});
	const bodyEl = $e("span", nTextContainer, {classes: ["notif-body"], innerText: text});
	const bar = $e("div", notification, {classes: ["notif-bar"]});
	notification.style.left = "0px";

	setTimeout(function() {
		notification.remove();
	}, 10_000);
}

function reportError(title, text) {
	// TODO: Send to server and log there?
	console.error(`${title}: ${text}`);
	showNotification(title, text, "error");
}

function canNavigateStoryHistory() {
	return !["TEXTAREA", "INPUT"].includes(document.activeElement.tagName);
}

//function to load more actions if nessisary
function infinite_scroll() {
	if (scroll_trigger_element != undefined) {
		if(scroll_trigger_element.getBoundingClientRect().bottom >= 0){
			socket.emit("get_next_100_actions", parseInt(scroll_trigger_element.getAttribute("chunk")));
			scroll_trigger_element = undefined;
		}
	}
}

function run_infinite_scroll_update(action_type, actions, first_action) {
	//console.log("first_action: "+first_action);
	const promptEl = $el("#story_prompt");
	if (!promptEl) return;

	if (action_type == "append") {
		if (document.getElementById('Selected Text Chunk '+actions[actions.length-1].id)) {
			document.getElementById('Selected Text Chunk '+actions[actions.length-1].id).scrollIntoView(false);
			document.getElementById("Selected Text").scrollBy(0, 25);
		}
		//Check to see if we need to have the scrolling in place or not
		if (promptEl.classList.contains("hidden")) {
			if (Math.min.apply(null,Object.keys(actions_data).map(Number).filter(function(x){return x>=0})) <= 0) {
				promptEl.classList.remove("hidden");
			} else {
				//console.log("Appending, but adding infinite scroll");
				//console.log(document.getElementById('Selected Text Chunk '+Math.min.apply(null,Object.keys(actions_data).map(Number).filter(function(x){return x>=0}))));
				document.getElementById("Selected Text").onscroll = infinite_scroll;
				scroll_trigger_element = document.getElementById('Selected Text Chunk '+Math.min.apply(null,Object.keys(actions_data).map(Number).filter(function(x){return x>=0})));
			}
		}
	} else if (action_type == "prepend") {
		if (Math.min.apply(null,Object.keys(actions_data).map(Number).filter(function(x){return x>=0})) == 0) {
			//We've hit our prompt, so let's unhide it, move it to the begining, and kill the infinite_scroll
			scroll_trigger_element = undefined;
			document.getElementById("Selected Text").onscroll = undefined;
			document.getElementById("Selected Text").prepend(promptEl);
			promptEl.classList.remove("hidden");
		} else {
			//we just added more text and didn't hit the prompt. Move the scroll trigger back to the first non-prompt element
			let item_in_view = false;
			if ((scroll_trigger_element != undefined) && (scroll_trigger_element)) {
				if(scroll_trigger_element.getBoundingClientRect().bottom >= 0){
					let item_in_view = true;
				}
			}
			for (id of Object.keys(actions_data).map(Number).filter(function(x){return x>0}).sort(function(a, b) {return a - b;})) {
				//console.log("Checking for "+id);
				if (document.getElementById('Selected Text Chunk '+id)) {
					scroll_trigger_element = document.getElementById('Selected Text Chunk '+id);
					break;
				}
			}
			if (document.getElementById('Selected Text Chunk '+first_action)) {
				if (item_in_view) {
					document.getElementById('Selected Text Chunk '+first_action).scrollIntoView(true);
				}
			}
			
		}
	}
	
	if (scroll_trigger_element != undefined) {
		auto_loader_timeout = setTimeout(function() {socket.emit("get_next_100_actions", parseInt(scroll_trigger_element.getAttribute("chunk")));}, 1000);
	}
}

function countWIFolderChildren(folder) {
	let count = 0;
	for (const wi of Object.values(world_info_data)) {
		if (wi.folder === folder) count += 1;
	}
	return count;
}

function sFormatted2HTML(sFormatted) {
	// "sFormatted" is a rudimentary solution to safe formatting
	let outHTML = "";

	for (const chunk of sFormatted) {
		// Expand as needed
		let format = {
			bold: "<b>%s</b>",
			italic: "<i>%s</i>"
		}[chunk.format] || "%s";
		
		// This actually sucks but apparently the best recognized way to escape
		// HTML in JavaScript is just "make an element real quick and slap some
		// text in it."
		let escaped = new Option(chunk.text).innerHTML;

		outHTML += format.replace("%s", escaped);
	}
	return outHTML;
}

function deleteConfirmation(sFormatted, confirmText, denyText, confirmCallback, denyCallback) {
	$el("#confirm-text").innerHTML = sFormatted2HTML(sFormatted);
	
	$el("#confirm-confirm-button > .text").innerText = confirmText;
	$el("#confirm-deny-button > .text").innerText = denyText;

	const confirmButton = $el("#confirm-confirm-button")
	confirmButton.onclick = function() {
		confirmCallback();
		closePopups();
		confirmButton.onclick = undefined;
	}

	const denyButton = $el("#confirm-deny-button")
	denyButton.onclick = function() {
		// No-op if no deny callback
		(denyCallback || function(){})();
		closePopups();
		confirmButton.onclick = undefined;
	}

	openPopup("confirm-delete-dialog");
}

function attemptClubLoad() {
	const input = $el("#aidgpromptnum");
	let val = input.value;
	if (!/^\d+$/.test(val)) {
		// Not an id, is it a full URL?
		const matches = val.match(/aetherroom\.club\/([0-9]+)/)
		if (!matches) {
			reportError("Malformed club import", "That doesn't look like a valid club URL or ID. Please check your input and try again.");
			return;
		}
		val = matches[1];
	}
	socket.emit("load_aidg_club", val);
	closePopups();
}


$el("#aidgpromptnum").addEventListener("keydown", function(event) {
	if (event.key !== "Enter") return;
	attemptClubLoad();
	event.preventDefault();
});

$el("#generate-image-button").addEventListener("click", function() {
	$el("#image-loading").classList.remove("hidden");
	socket.emit("generate_image", {});
});

/* -- Shiny New Chat -- */
function addMessage(author, content, actionId, afterMsgEl=null, time=null) {
	if (!time) time = Number(new Date());
	const gameScreen = $el("#gamescreen");

	let insertionLocation = afterMsgEl ? {after: afterMsgEl} : null
	const message = $e(
		"div",
		gameScreen,
		{classes: ["chat-message", "chat-style-channel"], "action-id": actionId},
		// Insertion location
		insertionLocation,
	);

	const leftContainer = $e("div", message, {classes: ["chat-left-container"]});

	const profilePicture = $e("img", leftContainer, {
		classes: ["chat-pfp"],
		src: getChatPfp(author),
		draggable: false
	});

	const addAfterButton = $e("span", leftContainer, {classes: ["chat-add", "chat-button", "material-icons-outlined"], innerText: "add"});
	const deleteButton = $e("span", leftContainer, {classes: ["chat-delete", "chat-button", "material-icons-outlined"], innerText: "delete"});

	const textContainer = $e("div", message, {classes: ["chat-text-container"]});

	const messageHeader = $e("div", textContainer, {classes: ["chat-header"]});

	const messageAuthor = $e("span", messageHeader, {classes: ["chat-author"], innerText: author, contenteditable: true, spellcheck: false, "data-placeholder": "Author"});

	// TODOB4PUSH: Better formatting
	const messageTime = $e("span", messageHeader, {classes: ["chat-timestamp", "noselect"], innerText: formatChatDate(time)});

	// TODO: In-house less intrusive spellcheck?
	const messageText = $e("span", textContainer, {classes: ["chat-text"], innerText: content, contenteditable: true, spellcheck: false, "data-placeholder": "Message"});

	// When we edit it we need to recompute the context
	// NOTE: `focusout` may not always trigger! `change` is not a thing on
	// `contenteditable` and `input` fires way to often, so we'll hope this works!

	for (const box of [messageAuthor, messageText]) {
		box.addEventListener("focusout", () => computeChatGametext(actionId));
		box.addEventListener("keydown", function(event) {
			if (event.key === "Enter") {
				event.preventDefault();
				this.blur();
			}
		});
	}

	messageAuthor.addEventListener("keyup", function() {
		profilePicture.src = getChatPfp(messageAuthor.innerText);
	})

	addAfterButton.addEventListener("click", function() {
		addMessage(null, null, actionId, message);
	});

	deleteButton.addEventListener("click", function() {
		message.remove();
		computeChatGametext(actionId);
	});

	message.scrollIntoView();
	return message;
}

function formatChatDate(unixTimestamp) {
	let date = new Date(unixTimestamp);
	let now = new Date();

	// TODO: Support 24 hour time
	let timeString = date.toLocaleString("en-US").replace(/:[0-9]+\s/, " ").split(", ").splice(-1)[0];
	let dateString = date.toLocaleString("en-US").split(", ")[0];

	let hourDelta = (now.getTime() - date.getTime()) / 1000 / 60 / 60;

	if (hourDelta >= 48) {
		return dateString;
	} else if (hourDelta >= 24) {
		return `Yesterday at ${timeString}`;
	} else {
		return `Today at ${timeString}`;
	}
}

function addInitChatMessage() {
	if (!chat.useV2) return;

	// Already exists!
	if ($el("#init-message")) return;

	let message = addMessage(null, null, -1);
	message.id = "init-message";
}

function deleteChatPromptIfEmpty() {
	if (!chat.useV2) return;

	const prompt = $el("#init-message");
	if (!prompt) return;

	let author = prompt.getElementsByClassName("chat-author")[0].innerText;
	let content = prompt.getElementsByClassName("chat-text")[0].innerText;
	if (author || content) return;

	prompt.remove();
}

function computeChatGametext(actionId) {
	// TODO: Customizable format?
	let lines = [];
	for (const message of document.querySelectorAll(`[action-id="${actionId}"]`)) {
		const name = message.getElementsByClassName("chat-author")[0].innerText;
		const text = message.getElementsByClassName("chat-text")[0].innerText;
		lines.push(`${name}: ${text}`);
	}

	let text = lines.join("\n");
	console.log(actionId, text);
	socket.emit("Set Selected Text", {id: actionId, text: text});
	chat.lastEdit = actionId;
}

function updateChatStyle() {
	const storyArea = document.getElementById("Selected Text");

	if (chat.useV2) {
		// Already v2, do nothing
		if (document.getElementsByClassName("chat-message").length) {
			return;
		}

		// Delete normal text

		while (storyArea.firstChild) {
			storyArea.removeChild(storyArea.firstChild);
		}

		let addedMessages = 0;

		for (let [chunkId, chunk] of Object.entries(actions_data).sort((a, b) => parseInt(a) > parseInt(b))) {
			chunkId = parseInt(chunkId);
			for (const message of parseChatMessages(chunk["Selected Text"])) {
				// JS Time uses milliseconds, thus the * 1000
				addMessage(message.author, message.text, chunkId, null, chunk["Time"] * 1000);
				addedMessages++;
			}
		}
		
		// If we are empty, add an init message
		if (!addedMessages) addInitChatMessage();
	} else {
		if (!storyArea.querySelectorAll(".rawtext").length) {
			for (const [chunkId, action] of Object.entries(actions_data)) {
				let item = document.createElement("span");
				item.id = 'Selected Text Chunk '+chunkId;
				item.classList.add("rawtext");
				item.setAttribute("chunk", chunkId);
				//need to find the closest element
				next_id = chunkId+1;
				if (Math.max.apply(null,Object.keys(actions_data).map(Number)) <= next_id) {
					storyArea.append(item);
				} else {
					storyArea.prepend(item);
				}

				chunk_element = document.createElement("span");
				chunk_element.innerText = action['Selected Text'];
				item.append(chunk_element);

				item.original_text = action['Selected Text'];
			}
		}

		const jQCM = $(".chat-message");
		if (jQCM.length) jQCM.remove();
	}
}

function getChatPfp(chatName) {
	if (chatName) {
		chatName = chatName.toLowerCase();
		for (const entry of Object.values(world_info_data)) {
			if (entry.type !== "chatcharacter") continue;
			if (entry.title.toLowerCase() !== chatName) continue;
			let img = $el(`#world_info_image_${entry.uid}`);

			// Not sure why this would happen, but better safe than sorry.
			if (!img) continue;
			if (!img.src) return "/static/default_pfp.png";

			return img.src;
		}
	}

	return "/static/default_pfp.png";
}

function setChatPfps(chatName, src) {
	// Refresh pfps for one user
	for (const chatEl of document.getElementsByClassName("chat-message")) {
		let author = chatEl.querySelector(".chat-author").innerText;
		if (author !== chatName) continue;

		chatEl.querySelector(".chat-pfp").src = src;
	}
}

/* -- WI Image Context Menu -- */
function wiImageView(summonEvent) {
	$el("#big-image").src = summonEvent.target.src;
	openPopup("big-image");
}

function wiImageReplace(summonEvent) {
	// This is also used for the "Upload" context menu action on the placeholder.

	// NOTE: WI image context menu stuff is pretty reliant on the current
	// element structure, be sure to update this code if that's changed.
	summonEvent.target.parentElement.click();
}

async function wiImageClear(summonEvent) {
	let uid = parseInt(summonEvent.target.id.replace("world_info_image_", ""));
	summonEvent.target.src = "";
	summonEvent.target.parentElement.querySelector(".placeholder").classList.remove("hidden");
	let r = await fetch(`/set_wi_image/${uid}`, {
		method: "POST",
		body: null
	});
}

async function wiImageUseGeneratedImage(summonEvent) {
	// summonEvent is placeholder icon
	const generatedImage = $el(".action_image");
	if (!generatedImage) return;

	let uid = parseInt(summonEvent.target.closest(".world_info_card").getAttribute("uid"));
	summonEvent.target.classList.add("hidden");

	let image = summonEvent.target.parentElement.getElementsByTagName("img")[0];
	image.src = generatedImage.src;

	let r = await fetch(`/set_wi_image/${uid}`, {
		method: "POST",
		body: generatedImage.src
	});
}

function imgGenView() {
	const image = $el(".action_image");
	if (!image) return;
	$el("#big-image").src = image.src;
	openPopup("big-image");
}

function imgGenDownload() {
	const image = $el(".action_image");
	if (!image) return;
	const a = $e("a", null, {href: image.src, download: "generated.png"});
	a.click();
}

function imgGenClear() {
	const image = $el(".action_image");
	if (!image) return;
	image.remove();

	const container = $el("#action\\ image");
	container.removeAttribute("tooltip");
	socket.emit("clear_generated_image", {});
}

function imgGenRetry() {
	const image = $el(".action_image");
	if (!image) return;
	$el("#image-loading").classList.remove("hidden");
	socket.emit("retry_generated_image");
}

/* Genres */
(async function() {
	const genreContainer = $el("#genre-container");
	const genreInput = $el("#genre-input");
	const genreSuggestionContainer = $el("#genre-suggestion-container");
	let genreData = await (await fetch("/genre_data.json")).json();
	let allGenres = genreData.list;
	let genres = genreData.init;
	let highlightIndex = -1;

	sync_hooks.push({
		class: "story",
		name: "genres",
		func: function(passedGenres) {
			genres = passedGenres;
			$(".genre").remove();
			for (const g of genres) {
				addGenreUI(g);
			}
		}
	})

	function addGenreUI(genre) {
		let div = $e("div", genreContainer, {classes: ["genre"]});
		let inner = $e("div", div, {classes: ["genre-inner"]});
		let xIcon = $e("span", inner, {innerText: "clear", classes: ["x", "material-icons-outlined"]});
		let label = $e("span", inner, {innerText: genre, classes: ["genre-label"]});

		xIcon.addEventListener("click", function() {
			div.remove();
			genres = genres.filter(x => x !== genre);
			socket.emit("var_change", {"ID": "story_genres", "value": genres});
		});
	}

	for (const initGenre of genreData.init) {
		addGenreUI(initGenre);
	}

	function addGenre(genre) {
		if (genres.includes(genre)) return;

		addGenreUI(genre);
		genreInput.value = "";
		nukeSuggestions();

		genres.push(genre);
		socket.emit("var_change", {"ID": "story_genres", "value": genres});
	}

	function nukeSuggestions() {
		genreSuggestionContainer.innerHTML = "";
		highlightIndex = -1;
	}

	document.addEventListener("click", function(event) {
		// Listening for clicks all over the document kinda sucks but blur
		// fires you can click a suggestion so...
		if (!genreSuggestionContainer.children.length) return;
		if (event.target === genreInput) return;
		if (event.target.classList.contains("genre-suggestion")) return;
		nukeSuggestions();
	});

	genreInput.addEventListener("keydown", function(event) {
		switch (event.key) {
			case "ArrowUp":
				highlightIndex--;
				break;
			case "Tab":
				highlightIndex += event.shiftKey ? -1 : 1;
				break;
			case "ArrowDown":
				highlightIndex++;
				break;
			case "Enter":
				if (highlightIndex === -1) {
					if (!genreInput.value.trim()) return;
					addGenre(genreInput.value);
				} else {
					genreSuggestionContainer.children[highlightIndex].click();
				}
				return;
			case "Escape":
				genreInput.value = "";
				nukeSuggestions();
				event.preventDefault();
				event.stopPropagation();
				return;
			default:
				return;
		}

		event.preventDefault();

		if (!genreSuggestionContainer.children.length) return;

		const oldHighlighted = $el(".genre-suggestion.highlighted");
		if (oldHighlighted) oldHighlighted.classList.remove("highlighted");

		// Wrap around
		let maxIndex = genreSuggestionContainer.children.length - 1;
		if (highlightIndex < 0) highlightIndex = maxIndex;
		if (highlightIndex > maxIndex) highlightIndex = 0;

		const highlighted = genreSuggestionContainer.children[highlightIndex];
		highlighted.classList.add("highlighted");
		highlighted.scrollIntoView({
            behavior: "auto",
            block: "center",
            inline: "center"
        });
	});

	genreInput.addEventListener("input", function() {
		let showList = [];
		let lowerMatch = genreInput.value.toLowerCase();

		nukeSuggestions();
		if (!lowerMatch) return;

		for (const genre of allGenres) {
			if (!genre.toLowerCase().includes(lowerMatch)) continue;
			showList.push(genre);
		}

		for (const genre of showList) {
			let suggestion = $e("span", genreSuggestionContainer, {
				innerText: genre,
				classes: ["genre-suggestion"]
			});

			suggestion.addEventListener("click", function() {
				addGenre(this.innerText);
			});
		}
	});


})();

(function() {
	const characterContainer = $el(".story-commentary-characters");
	const settingsContainer = $el("#story-commentary-settings");
	const storyReviewImg = $el("#story-review-img");

	async function showStoryReview(data) {
		// Story id is used to invalidate cache from other stories
		storyReviewImg.src = `/get_wi_image/${data.uid}?${story_id}`;
		$el("#story-review-author").innerText = data.who;
		$el("#story-review-content").innerText = data.review;
		
		$el("#story-review").classList.remove("hidden");
	}
	socket.on("show_story_review", showStoryReview);

	// Bootstrap toggle requires jQuery for events
	$($el("#story-commentary-enable").querySelector("input")).change(function() {
		socket.emit("var_change", {
			ID: "story_commentary_enabled",
			value: this.checked
		});
	});

	sync_hooks.push({
		class: "story",
		name: "commentary_enabled",
		func: function(commentaryEnabled) {
			this.checked = commentaryEnabled;
			if (commentaryEnabled) {
				settingsContainer.classList.remove("disabled");
			} else {
				settingsContainer.classList.add("disabled");
			}
		}
	});

	storyReviewImg.addEventListener("error", function() {
		if (storyReviewImg.src === "/static/default_pfp.png") {
			// Something has gone horribly wrong
			return;
		}
		storyReviewImg.src = "/static/default_pfp.png";
	});

	$el("#story-review-img").addEventListener
})();

for (const el of document.querySelectorAll("[sync-var]")) {
	let varName = el.getAttribute("sync-var");

	el.addEventListener("change", function() {
		sync_to_server(this);
	});



	const proxy = $el(`[sync-proxy-host="${varName}"]`);
	if (proxy) {
		el.addEventListener("input", function() {
			proxy.value = this.value;
		});

		el.addEventListener("sync", function() {
			proxy.value = this.value;
		});
	}

	let slug = varName.replaceAll(".", "_");
	el.classList.add("var_sync_" + slug);
}

for (const proxy of document.querySelectorAll("[sync-proxy-host]")) {
	let varName = proxy.getAttribute("sync-proxy-host");
	const hostEl = $el(`[sync-var="${varName}"]`);
	if (!hostEl) {
		throw Error(`Bad sync proxy host ${varName}`)
	}

	proxy.addEventListener("change", function() {
		hostEl.value = this.value;
		socket.emit("var_change", {
			ID: varName.replaceAll(".", "_"),
			value: this.value
		});
	});
}

function generateWIData(uid, field, title=null, type=null, desc=null, genAmount=80) {
	if (generating_summary) return;
	generating_summary = true;

	socket.emit("generate_wi", {
		uid: uid,
		field: field,
		genAmount: genAmount || 80,
		existing: {title: title, type: type, desc: desc}
	});
}

function showGeneratedWIData(data) {
	generating_summary = false;
	const card = $el(`.world_info_card[uid="${data.uid}"]`);
	const manualTextEl = card.querySelector(".world_info_entry_text");
	manualTextEl.classList.remove("disabled");

	// Stop spinning!
	for (const littleRobotFriend of card.querySelectorAll(".generate-button.spinner")) {
		littleRobotFriend.classList.remove("spinner");
		littleRobotFriend.innerText = "smart_toy";
	}

	if (data.field === "desc") {
		world_info_data[data.uid].manual_text = data.out;
		send_world_info(data.uid);
	}
}

$el(".gametext").addEventListener("keydown", function(event) {
	if (event.key !== "Enter") return;
	// execCommand is deprecated but until Firefox supports
	// contentEditable="plaintext-only" we're just gonna have to roll with it
	document.execCommand("insertLineBreak");
	event.preventDefault();
});

/* Screenshot */
const screenshotTarget = $el("#screenshot-target");
const screenshotImagePicker = $el("#screenshot-image-picker");
const screenshotImageContainer = $el("#screenshot-images");
const robotAttribution = $el("#robot-attribution");
const screenshotTextContainer = $el("#screenshot-text-container");

sync_hooks.push({
	class: "story",
	name: "story_name",
	func: function(title) {
		$el("#story-attribution").innerText = title;
	}
});

sync_hooks.push({
	class: "model",
	name: "model",
	func: function(modelName) {
		$el("#model-name").innerText = modelName
	}
})

sync_hooks.push({
	class: "user",
	name: "screenshot_author_name",
	func: function(name) {
		$el("#human-attribution").innerText = name;
	}
});

sync_hooks.push({
	class: "user",
	name: "screenshot_show_story_title",
	func: function(show) {
		$el("#story-title-vis").classList.toggle("hidden", !show);
		robotAttribution.scrollIntoView();
	}
});

sync_hooks.push({
	class: "user",
	name: "screenshot_show_author_name",
	func: function(show) {
		$el("#author-name-vis").classList.toggle("hidden", !show);
		$el("#screenshot-options-author-name").classList.toggle("disabled", !show);
		robotAttribution.scrollIntoView();
	}
});

sync_hooks.push({
	class: "user",
	name: "screenshot_use_boring_colors",
	func: function(boring) {
		screenshotTarget.classList.toggle("boring-colors", boring);
	}
});

async function showScreenshotWizard(actionComposition, startDebt, endDebt) {
	// startDebt is the amount we need to shave off the front, and endDebt the
	// same for the end
	
	screenshotTextContainer.innerHTML = "";
	let charCount = startDebt;
	let i = 0;
	for (const action of actionComposition) {
		for (const chunk of action) {
			// Account for debt
			if (startDebt > 0) {
				if (chunk.content.length <= startDebt) {
					startDebt -= chunk.content.length;
					continue;
				} else {
					// Slice up chunk
					chunk.content = chunk.content.slice(startDebt);
					startDebt = 0;
				}
			}

			if (charCount > endDebt) {
				break;
			} else if (charCount + chunk.content.length > endDebt) {
				let charsLeft = endDebt - charCount
				chunk.content = chunk.content.slice(0, charsLeft).trimEnd();
				endDebt = -1;
			}


			if (i == 0) chunk.content = chunk.content.trimStart();
			i++;

			charCount += chunk.content.length;

			let actionClass = {
				ai: "ai-text",
				user: "human-text",
				edit: "edit-text",
				prompt: "prompt-text",
			}[chunk.type];

			$e("span", screenshotTextContainer, {
				innerText: chunk.content,
				classes: ["action-text", actionClass]
			});
		}
	}

	let imageData = await (await fetch("/image_db.json")).json();
	screenshotImagePicker.innerHTML = "";

	for (const image of imageData) {
		if (!image) continue;

		const imgContainer = $e("div", screenshotImagePicker, {classes: ["img-container"]});
		const checkbox = $e("input", imgContainer, {type: "checkbox"});
		const imageEl = $e("img", imgContainer, {
			src: `/generated_images/${image.fileName}`,
			draggable: false,
			tooltip: image.displayPrompt
		});

		imgContainer.addEventListener("click", function(event) {
			// TODO: Preventdefault if too many images selected and checked is false
			checkbox.click();
		});

		checkbox.addEventListener("click", function(event) {
			event.stopPropagation();
			screenshotWizardUpdateShownImages();
		});
	}
	openPopup("screenshot-wizard");
}

function screenshotWizardUpdateShownImages() {
	screenshotImageContainer.innerHTML = "";

	for (const imgCont of screenshotImagePicker.children) {
		const checked = imgCont.querySelector("input").checked;
		if (!checked) continue;
		const src = imgCont.querySelector("img").src;
		$e("img", screenshotImageContainer, {src: src});
	}
}

async function downloadScreenshot() {
	// TODO: Upscale (eg transform with given ratio like 1.42 to make image
	// bigger via screenshotTarget cloning)
	const canvas = await html2canvas(screenshotTarget, {
		width: screenshotTarget.clientWidth,
		height: screenshotTarget.clientHeight - 1
	});

	canvas.style.display = "none";
	document.body.appendChild(canvas);
	$e("a", null, {download: "screenshot.png", href: canvas.toDataURL("image/png")}).click();
	canvas.remove();
}
$el("#sw-download").addEventListener("click", downloadScreenshot);

// Other side of screenshot-options hack
for (const el of document.getElementsByClassName("screenshot-setting")) {
	// yeah this really sucks but bootstrap toggle only works with this
	el.setAttribute("onchange", "sync_to_server(this);")
}

async function screenshot_selection(summonEvent) {
	// Adapted from https://stackoverflow.com/a/4220888
	let selection = window.getSelection();
	let range = selection.getRangeAt(0);
	let commonAncestorContainer = range.commonAncestorContainer;

	if (commonAncestorContainer.nodeName === "#text") commonAncestorContainer = commonAncestorContainer.parentNode;

	let rangeParentChildren = commonAncestorContainer.childNodes;
	// Array of STRING actions ids
	let selectedActionIds = [];

	for (let el of rangeParentChildren) {
		if (!selection.containsNode(el, true)) continue;
		// When selecting a portion of a singular action, el can be a text
		// node rather than an action span
		if (el.nodeName === "#text") el = el.parentNode.closest("[chunk]");
		let actionId = el.getAttribute("chunk");

		if (!actionId) continue;
		if (selectedActionIds.includes(actionId)) continue;

		selectedActionIds.push(actionId);
	}

	let actionComposition = await (await fetch(`/action_composition.json?actions=${selectedActionIds.join(",")}`)).json();

	let totalText = "";

	for (const action of actionComposition) {
		for (const chunk of action) totalText += chunk.content;
	}

	let selectionContent = selection.toString();
	let startDebt = totalText.indexOf(selectionContent);
	// lastIndexOf??
	// endDebt is distance from the end of selection.
	let endDebt = totalText.indexOf(selectionContent) + selectionContent.length;

	await showScreenshotWizard(actionComposition, startDebt=startDebt, endDebt=endDebt, totalText);
}

$el("#gamescreen").addEventListener("paste", function(event) {
	// Get rid of rich text, it messes with actions. Not a great fix since it
	// relies on execCommand but it'll have to do
	event.preventDefault();
	document.execCommand(
		"insertHTML",
		false,
		event.clipboardData.getData("text/plain")
	);
});