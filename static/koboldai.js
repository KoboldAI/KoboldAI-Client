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
socket.on('show_model_menu', function(data){show_model_menu(data);});
socket.on('selected_model_info', function(data){selected_model_info(data);});
socket.on('oai_engines', function(data){oai_engines(data);});
socket.on('buildload', function(data){buildload(data);});
socket.on('error_popup', function(data){error_popup(data);});
socket.on("world_info_entry", function(data){world_info_entry(data);});
socket.on("world_info_entry_used_in_game", function(data){world_info_entry_used_in_game(data);});
socket.on("world_info_folder", function(data){world_info_folder(data);});
socket.on("delete_new_world_info_entry", function(data){document.getElementById("world_info_-1").remove();});
socket.on("delete_world_info_entry", function(data){document.getElementById("world_info_"+data).remove();});
socket.on("delete_world_info_folder", function(data){document.getElementById("world_info_folder_"+data).remove();});
socket.on("error", function(data){show_error_message(data);});
socket.on('load_cookies', function(data){load_cookies(data);});
socket.on('load_tweaks', function(data){load_tweaks(data);});
socket.on("wi_results", updateWISearchListings);
socket.on("request_prompt_config", configurePrompt);
socket.on("log_message", function(data){process_log_message(data);});
socket.on("debug_message", function(data){console.log(data);});
socket.on("scratchpad_response", recieveScratchpadResponse);
//socket.onAny(function(event_name, data) {console.log({"event": event_name, "class": data.classname, "data": data});});

var presets = {};
var current_chunk_number = null;
var ai_busy_start = Date.now();
var popup_deleteable = false;
var popup_editable = false;
var popup_renameable = false;
var popup_rows = [];
var popup_style = "";
var popup_sort = {};
var shift_down = false;
var world_info_data = {};
var world_info_folder_data = {};
var saved_settings = {};
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
var scroll_trigger_element = undefined;
const on_colab = $el("#on_colab").textContent == "true";

// name, desc, icon, func
const finder_actions = [
	{name: "Load Model", icon: "folder_open", func: function() { socket.emit('load_model_button', {}); }},
	{name: "New Story", icon: "description", func: function() { socket.emit('new_story', ''); }},
	{name: "Load Story", icon: "folder_open", func: function() { socket.emit('load_story_list', ''); }},
	{name: "Save Story", icon: "save", func: function() { socket.emit("save_story", null, (response) => {save_as_story(response);}); }},
	{name: "Download Story", icon: "file_download", func: function() { document.getElementById('download_iframe').src = 'json'; }},

	// Locations
	{name: "Setting Presets", icon: "open_in_new", func: function() { highlightEl(".var_sync_model_selected_preset") }},
	{name: "Memory", icon: "open_in_new", func: function() { highlightEl("#memory") }},
	{name: "Author's Note", icon: "open_in_new", func: function() { highlightEl("#authors_notes") }},
	{name: "Notes", icon: "open_in_new", func: function() { highlightEl(".var_sync_story_notes") }},
	{name: "World Info", icon: "open_in_new", func: function() { highlightEl("#WI_Area") }},
	
	// TODO: Direct theme selection
	// {name: "", icon: "palette", func: function() { highlightEl("#biasing") }},
];

const context_menu_actions = [
	{label: "Cut", icon: "content_cut", visibilityCondition: "SELECTION", click: cut},
	{label: "Copy", icon: "content_copy", visibilityCondition: "SELECTION", click: copy},
	{label: "Paste", icon: "content_paste", visibilityCondition: "SELECTION", click: paste},
	// Null makes a seperation bar
	null,
	{label: "Add to Memory", icon: "assignment", visibilityCondition: "SELECTION", click: push_selection_to_memory},
	{label: "Add to World Info Entry", icon: "auto_stories", visibilityCondition: "SELECTION", click: push_selection_to_world_info},
	{label: "Add as Bias", icon: "insights", visibilityCondition: "SELECTION", click: push_selection_to_phrase_bias},
	{label: "Retry from here", icon: "refresh", visibilityCondition: "CARET", click: retry_from_here},
	// Not implemented! See view_selection_probabiltiies
	// null,
	// {label: "View Token Probabilities", icon: "assessment", visibilityCondition: "SELECTION", click: view_selection_probabilities},
	// {label: "View Token Probabilities", icon: "account_tree", visibilityCondition: "SELECTION", click: view_selection_probabilities},
];

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
var world_info_scroll_timeout;
var font_size_cookie_timout;
var setup_missing_wi_toggles_timeout;
var var_processing_time = 0;
var finder_last_input;
//-----------------------------------Server to UI  Functions-----------------------------------------------
function connect() {
	console.log("connected");
	//reset_story();
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

function reset_story() {
	console.log("Resetting story");
	clearTimeout(calc_token_usage_timeout);
	clearTimeout(game_text_scroll_timeout);
	clearTimeout(font_size_cookie_timout);
	clearTimeout(world_info_scroll_timeout);
	finder_last_input = null;
	on_new_wi_item = null;
	current_chunk_number = null;
	console.log("resetting scroll_trigger_element");
	scroll_trigger_element = undefined;
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
	dummy_span = document.createElement("div");
	dummy_span.id = "Delete Me";
	dummy_span.classList.add("noselect");
	document.getElementById("Selected Text").setAttribute("contenteditable", "false");
	text = "";
	for (i=0;i<154;i++) {
		text += "\xa0 ";
	}
	dummy_span.textContent = text;
	dummy_span.setAttribute("contenteditable", false);
	story_area.append(dummy_span);
	var option_area = document.getElementById("Select Options");
	while (option_area.firstChild) {
		option_area.removeChild(option_area.firstChild);
	}
	var world_info_area = document.getElementById("WI_Area");
	while (world_info_area.firstChild) {
		world_info_area.removeChild(world_info_area.firstChild);
	}
	world_info_data = {};
	world_info_folder({"root": []});
	document.getElementById("story_prompt").setAttribute("world_info_uids", "");
	document.getElementById('themerow').classList.remove("hidden");
	document.getElementById('input_text').placeholder = "Enter Prompt Here (shift+enter for new line)";
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
	var option_container = document.getElementById("Select Options");
	var current_chunk = parseInt(document.getElementById("action_count").textContent)+1;
	if (current_chunk != action.id.toString()) {
		return;
	}
	if (document.getElementById("Select Options Chunk " + current_chunk)) {
		//document.getElementById("Select Options Chunk " + current_chunk).classList.remove("hidden");
		document.getElementById("Select Options Chunk " + current_chunk).remove();
	}
	if (document.getElementById("Select Options Chunk " + (current_chunk-1))) {
		//document.getElementById("Select Options Chunk " + (current_chunk-1)).classList.add("hidden");
		document.getElementById("Select Options Chunk " + (current_chunk-1)).remove();
	}
	
	if (document.getElementById("Select Options Chunk "+action.id)) {
		var option_chunk = document.getElementById("Select Options Chunk "+action.id);
	} else {
		var option_chunk = document.createElement("div");
		option_chunk.id = "Select Options Chunk "+action.id;
		if (current_chunk != action.id) {
			option_chunk.classList.add("hidden");
		}
		option_container.append(option_chunk);
	}
	//first, let's clear out our existing data
	while (option_chunk.firstChild) {
		option_chunk.removeChild(option_chunk.firstChild);
	}
	var table = document.createElement("div");
	table.classList.add("sequences");
	//Add Redo options
	i=0;
	for (item of action.action.Options) {
		if ((item['Previous Selection'])) {
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
			icon.classList.add("oi");
			icon.setAttribute('data-glyph', "loop-circular");
			iconcell.append(icon);
			delete_icon = $e("span", iconcell, {"classes": ["material-icons-outlined", "cursor", 'delete_option_icon'], 
												"title": "delete option", 'option_id': i, 
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
		}
		i+=1;
	}
	//Add general options
	i=0;
	for (item of action.action.Options) {
		if (!(item.Edited) && !(item['Previous Selection'])) {
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
			icon.classList.add("oi");
			icon.setAttribute('data-glyph', "pin");
			if (!(item.Pinned)) {
				icon.setAttribute('style', "filter: brightness(50%);");
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
		}
		i+=1;
	}
	option_chunk.append(table);
	
	
	//make sure our last updated chunk is in view
	//option_chunk.scrollIntoView();
}

function do_story_text_updates(action) {
	story_area = document.getElementById('Selected Text');
	current_chunk_number = action.id;
	if (document.getElementById('Selected Text Chunk '+action.id)) {
		var item = document.getElementById('Selected Text Chunk '+action.id);
		//clear out the item first
		while (item.firstChild) { 
			item.removeChild(item.firstChild);
		}		
		span = document.createElement("span");
		span.textContent = action.action['Selected Text'];
		item.append(span);
		item.original_text = action.action['Selected Text'];
		item.setAttribute("WI_Search_Text", action.action['WI Search Text']);
		item.setAttribute("world_info_uids", "");
		item.classList.remove("pulse")
		item.classList.remove("single_pulse");
		item.classList.add("single_pulse");
		//item.scrollIntoView();
		if (item.textContent != "") {
			assign_world_info_to_action(action.id, null);
		}
	} else {
		var span = document.createElement("span");
		span.id = 'Selected Text Chunk '+action.id;
		span.classList.add("rawtext");
		span.setAttribute("chunk", action.id);
		span.original_text = action.action['Selected Text'];
		new_span = document.createElement("span");
		new_span.textContent = action.action['Selected Text'];
		span.setAttribute("WI_Search_Text", action.action['WI Search Text']);
		span.append(new_span);
		
		//need to find the closest element
		next_id = action.id+1;
		if (Math.max.apply(null,Object.keys(actions_data)) <= next_id) {
			story_area.append(span);
		} else {
			story_area.prepend(span);
		}
		span.classList.add("single_pulse");
		
		if (span.textContent != "") {
			assign_world_info_to_action(action.id, null);
		}
		//console.log("done");
	}
}

function do_prompt(data) {
	var elements_to_change = document.getElementsByClassName("var_sync_"+data.classname.replace(" ", "_")+"_"+data.name.replace(" ", "_"));
	for (item of elements_to_change) {
		//clear out the item first
		while (item.firstChild) { 
			item.removeChild(item.firstChild);
		}
		
		span = document.createElement("span");
		span.textContent = data.value;
		item.append(span);
		item.setAttribute("old_text", data.value);
		item.setAttribute("WI_Search_Text", data.value.replace(/[^0-9a-z \'\"]/gi, ''));
		item.classList.remove("pulse");
		actions_data[-1] = {'Selected Text': data.value, 'WI Search Text': data.value.replace(/[^0-9a-z \'\"]/gi, '')};
		assign_world_info_to_action(-1, null);
	}
	//if we have a prompt we need to disable the theme area, or enable it if we don't
	if (data.value != "") {
		document.getElementById('input_text').placeholder = "Enter text here (shift+enter for new line)";
		document.getElementById('themerow').classList.add("hidden");
		document.getElementById('themetext').value = "";
		if (document.getElementById("Delete Me")) {
			document.getElementById("Delete Me").remove();
			document.getElementById("Selected Text").setAttribute("contenteditable", "true");
		}
		//enable editing
		document.getElementById("Selected Text").setAttribute("contenteditable", "true");
	} else {
		document.getElementById('input_text').placeholder = "Enter Prompt Here (shift+enter for new line)";
		document.getElementById('input_text').disabled = false;
		document.getElementById('themerow').classList.remove("hidden");
	}
	
}

function do_story_text_length_updates(action) {
	if (document.getElementById('Selected Text Chunk '+action.id)) {
		document.getElementById('Selected Text Chunk '+action.id).setAttribute("token_length", action.action["Selected Text Length"]);
	} else {
		console.log('Selected Text Chunk '+action.id);
		console.log(action);
	}
	
}

function do_probabilities(action) {
	//console.log(data);
	if (document.getElementById('probabilities_'+action.id)) {
		prob_area = document.getElementById('probabilities_'+action.id)
	} else {
		probabilities = document.getElementById('probabilities');
		prob_area = document.createElement('span');
		prob_area.id = 'probabilities_'+action.id;
		probabilities.append(prob_area);
	}
	//Clear
	while (prob_area.firstChild) { 
		prob_area.removeChild(prob_area.lastChild);
	}
	//create table
	table = document.createElement("table");
	table.border=1;
	if ("Probabilities" in action.action) {
		for (token of action.action.Probabilities) {
			actual_text = document.createElement("td");
			actual_text.setAttribute("rowspan", token.length);
			actual_text.textContent = "Word Goes Here";
			for (const [index, word] of token.entries()) {
				tr = document.createElement("tr");
				if (index == 0) {
					tr.append(actual_text);
				}
				decoded = document.createElement("td");
				decoded.textContent = word.decoded;
				tr.append(decoded);
				score = document.createElement("td");
				score.textContent = (word.score*100).toFixed(2)+"%";
				tr.append(score);
				table.append(tr);
			}
		}
	}
	prob_area.append(table);
	
	//prob_area.textContent = data.value.action["Probabilities"];
	
}

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
					option.value=preset;
					option.text=preset_value.preset;
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
	for (item of percent_bar) {
		item.setAttribute("style", "width:"+percent_complete+"%");
		item.textContent = Math.round(percent_complete,1)+"%"
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

function var_changed(data) {
	//if (data.name == "sp") {
	//	console.log({"name": data.name, "data": data});
	//}
	
	if ((data.classname == 'actions') && (data.name == 'Action Count')) {
		current_action = data.value;
	}
	//Special Case for Actions
	if ((data.classname == "story") && (data.name == "actions")) {
		start_time = Date.now();
		if (document.getElementById("Selected Text").firstElementChild.id == "story_prompt") {
			first_story_element = document.getElementById("Selected Text").firstElementChild.nextElementSibling;
		} else {
			first_story_element = document.getElementById("Selected Text").firstElementChild;
		}
		if (Array.isArray(data.value)) {
			actions = data.value;
		} else {
			actions = [data.value];
		}
		if (actions.length == 0) {return;}
		for (action of actions) {
			if (action.length != 0) {
				actions_data[action.id] = action.action;
				do_story_text_updates(action);
				create_options(action);
				do_story_text_length_updates(action);
				if ('Probabilities' in action.action) {
					do_probabilities(action);
				}
				if (action.action['In AI Input']) {
					document.getElementById('Selected Text Chunk '+action.id).classList.add("within_max_length");
				} else {
					document.getElementById('Selected Text Chunk '+action.id).classList.remove("within_max_length");
				}
			}
		}
		//if we hit the top, unhide the prompt and move it to the top
		if (actions[0].id == 0) {
			prompt_span = document.getElementById("story_prompt");
			document.getElementById("Selected Text").prepend(prompt_span);
			prompt_span.classList.remove("hidden");
		} else if (actions[actions.length-1].id != current_action) {
			//we are bringing in old data (not adding to the end of the game text), so set the scroll event back up
			//and scroll so that the old top is at the top again (IE new text added is before the top of the screen)
			if (first_story_element) {
				first_story_element.scrollIntoView(true);
			}
			scroll_trigger_element = document.getElementById('Selected Text Chunk '+actions[actions.length-1].id);
		} else {
			//We are adding new game text to the screen (not past actions)
			//If this is the first add, then we need to set our scroll trigger up
			if (document.getElementsByClassName("rawtext").length == actions.length+1) {
				scroll_trigger_element = document.getElementById('Selected Text Chunk '+actions[0].id);
			}
			document.getElementById('Selected Text Chunk '+actions[actions.length-1].id).scrollIntoView(false);
		}
		//console.log("Took "+((Date.now()-start_time)/1000)+"s to process");
		
	//Special Case for Presets
	} else if ((data.classname == 'model') && (data.name == 'presets')) {
		do_presets(data);
	//Special Case for prompt
	} else if ((data.classname == 'story') && (data.name == 'prompt')) {
		do_prompt(data);
	//Special Case for phrase biasing
	} else if ((data.classname == 'story') && (data.name == 'biases')) {
		do_biases(data);
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
		while (image_area.firstChild) { 
			image_area.removeChild(image_area.firstChild);
		}
		if (data.value != "") {
			var image = new Image();
			image.src = 'data:image/png;base64,'+data.value;
			image.classList.add("action_image");
			image_area.appendChild(image);
		}
	}  else if (data.classname == "story" && data.name == "picture_prompt") {
		if (document.getElementById("action image").firstChild) {
			document.getElementById("action image").firstChild.setAttribute("title", data.value);
		}
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
	if ((data.classname == 'model') && (data.name == 'numseqs')) {
		if (data.value == 1) {
			//allow our options to collapse to 0%, but no more than 30% (in case there is a redo or the like)
			var r = document.querySelector(':root');
			r.style.setProperty('--story_options_size', 'fit-content(30%)');
		} else {
			//static 30%
			var r = document.querySelector(':root');
			r.style.setProperty('--story_options_size', '30%');
		}
	}
	
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
	var popup = document.getElementById("popup");
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
					socket.emit("upload_file", {'filename': file.name, "data": event.target.result});
				};
				reader.readAsArrayBuffer(file);
			}
		});
	} else {
		
	}
	
	popup.classList.remove("hidden");
	
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
								document.getElementById("popup").classList.add("hidden");
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
			edit_icon.title = "Edit"
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
			rename_icon.title = "Rename"
			rename_icon.id = row.path;
			rename_icon.setAttribute("filename", row.fileName);
			rename_icon.onclick = function () {
				let new_name = prompt("Please enter new filename for \n"+ this.getAttribute("filename"));
				if (new_name != null) {
					socket.emit("popup_rename", {"file": this.id, "new_name": new_name});
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
			delete_icon.title = "Delete"
			delete_icon.id = row.path;
			delete_icon.setAttribute("folder", row.isFolder);
			delete_icon.onclick = function () {
				if (this.getAttribute("folder") == "true") {
					if (window.confirm("Do you really want to delete this folder and ALL files under it?")) {
						socket.emit("popup_delete", this.id);
					}
				} else {
					if (window.confirm("Do you really want to delete this file?")) {
						socket.emit("popup_delete", this.id);
					}
				}
			};
		}
		icon_area.append(delete_icon);
		tr.append(icon_area);
		
		//create the actual item
		let gridIndex = 0;
		if (row.showFilename) {
			let popup_item = document.createElement("span");
			popup_item.style = `grid-area: p${gridIndex};`;
			gridIndex += 1;

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
			td.style = `grid-area: p${gridIndex};`;

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
		dataRow.showFilename = item.show_filename;

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
	for (let i=0; i < data.column_widths.length; i++) {
		popup_style += ` p${i}`;
	}

	if (data.show_filename) {
		popup_style += ` p${i}`;
	}

	popup_style += '"; grid-template-columns: 50px';
	for (const column_width of data.column_widths) {
		popup_style += " "+column_width;
	}
	popup_style += ';';
	
	//create titles
	var tr = document.createElement("div");
	tr.style = popup_style;
	//icon area
	var td = document.createElement("span");
	td.style = "grid-area: icons;";
	tr.append(td)
	
	//add dynamic columns
	var i = 0;
	if (data.show_filename) {
		td = document.createElement("span");
		td.textContent = "File Name";
		td.style = "grid-area: p"+i+";";
		i+=1;
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
		container.style = `grid-area: p${i};`;

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
							document.getElementById("popup").classList.add("hidden");
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
	// The "T" and "K" may be a little optimistic...
	let paramsString = modelName.toUpperCase().match(/[\d.]+[TBMK]/)
	if (!paramsString) return null;
	paramsString = paramsString[0];

	let base = parseFloat(paramsString);
	let multiplier = {T: 1_000_000_000_000, B: 1_000_000_000, M: 1_000_000, K: 1_000}[paramsString[paramsString.length - 1]];

	return base * multiplier;
}

function show_model_menu(data) {
	document.getElementById("loadmodelcontainer").classList.remove("hidden");
	
	//clear old options
	document.getElementById("modelkey").classList.add("hidden");
	document.getElementById("modelkey").value = "";
	document.getElementById("modelurl").classList.add("hidden");
	document.getElementById("use_gpu_div").classList.add("hidden");
	document.getElementById("modellayers").classList.add("hidden");
	document.getElementById("oaimodel").classList.add("hidden");
	var model_layer_bars = document.getElementById('model_layer_bars');
	while (model_layer_bars.firstChild) {
		model_layer_bars.removeChild(model_layer_bars.firstChild);
	}
	
	//clear out the breadcrumbs
	var breadcrumbs = document.getElementById('loadmodellistbreadcrumbs')
	while (breadcrumbs.firstChild) {
		breadcrumbs.removeChild(breadcrumbs.firstChild);
	}
	//add breadcrumbs
	//console.log(data.breadcrumbs);
	for (item of data.breadcrumbs) {
		var button = document.createElement("button");
		button.classList.add("breadcrumbitem");
		button.setAttribute("model", data.menu);
		button.setAttribute("folder", item[0]);
		button.textContent = item[1];
		button.onclick = function () {
					socket.emit('select_model', {'menu': "", 'model': this.getAttribute("model"), 'path': this.getAttribute("folder")});
				};
		breadcrumbs.append(button);
		var span = document.createElement("span");
		span.textContent = "\\";
		breadcrumbs.append(span);
	}
	
	//clear out the items
	var model_list = document.getElementById('loadmodellistcontent')
	while (model_list.firstChild) {
		model_list.removeChild(model_list.firstChild);
	}
	//add items
	for (item of data.data) {
		var list_item = document.createElement("span");
		list_item.classList.add("model_item");
		
		//create the folder icon
		var folder_icon = document.createElement("span");
		folder_icon.classList.add("material-icons-outlined");
		folder_icon.classList.add("cursor");
		if ((item[3]) || (item[0] == 'Load a model from its directory') || (item[0] == 'Load an old GPT-2 model (eg CloverEdition)')) {
			folder_icon.textContent = "folder";
		} else {
			folder_icon.textContent = "psychology";
		}
		list_item.append(folder_icon);
		
		
		//create the actual item
		var popup_item = document.createElement("span");
		popup_item.classList.add("model");
		popup_item.setAttribute("display_name", item[0]);
		popup_item.id = item[1];
		
		popup_item.setAttribute("Menu", data.menu)
		//name text
		var text = document.createElement("span");
		text.style="grid-area: item;";
		text.textContent = item[0];
		popup_item.append(text);
		//model size text
		var text = document.createElement("span");
		text.textContent = item[2];
		text.style="grid-area: gpu_size;padding: 2px;";
		popup_item.append(text);

		(function() {
			// Anon function to avoid unreasonable indentation
			if (folder_icon.innerText !== "psychology") return;

			let parameterCount = getModelParameterCount(item[0]);
			if (!parameterCount) return;

			let warningText = "";

			if (parameterCount > 25_000_000_000) warningText = "This is a very high-end model and will likely not run without a specialized setup."; // 25B
			if (parameterCount < 2_000_000_000) warningText = "This is a lower-end model and may perform poorly.";			// 2B
			if (parameterCount < 1_000_000_000) warningText = "This is a very low-end model and may perform incoherently.";	// 1B

			if (!warningText) return;
			$e("span", list_item, {
				classes: ["material-icons-outlined"],
				innerText: "warning",
				"style.grid-area": "warning_icon",
				title: warningText
			});
		})();
		
		popup_item.onclick = function () {
						var accept = document.getElementById("btn_loadmodelaccept");
						accept.classList.add("disabled");
						socket.emit("select_model", {"model": this.id, "menu": this.getAttribute("Menu"), "display_name": this.getAttribute("display_name")});
						var model_list = document.getElementById('loadmodellistcontent').getElementsByClassName("selected");
						for (model of model_list) {
							model.classList.remove("selected");
						}
						this.classList.add("selected");
						accept.setAttribute("selected_model", this.id);
						accept.setAttribute("menu", this.getAttribute("Menu"));
						accept.setAttribute("display_name", this.getAttribute("display_name"));
					};
		list_item.append(popup_item);
		
		
		model_list.append(list_item);
	}
	var accept = document.getElementById("btn_loadmodelaccept");
	accept.disabled = true;
	
}

function selected_model_info(data) {
	var accept = document.getElementById("btn_loadmodelaccept");
	//hide or unhide key
	if (data.key) {
		document.getElementById("modelkey").classList.remove("hidden");
		document.getElementById("modelkey").value = data.key_value;
	} else {
		document.getElementById("modelkey").classList.add("hidden");
		document.getElementById("modelkey").value = "";
	}
	//hide or unhide URL
	if  (data.url) {
		document.getElementById("modelurl").classList.remove("hidden");
	} else {
		document.getElementById("modelurl").classList.add("hidden");
	}
	
	//default URL loading
	if (data.default_url != null) {
		document.getElementById("modelurl").value = data.default_url;
	}
	
	//change model loading on url if needed
	if (data.models_on_url) {
		document.getElementById("modelurl").onchange = function () {socket.emit('get_cluster_models', {'model': document.getElementById('btn_loadmodelaccept').getAttribute('selected_model'), 'key': document.getElementById("modelkey").value, 'url': this.value});};
		document.getElementById("modelkey").onchange = function () {socket.emit('get_cluster_models', {'model': document.getElementById('btn_loadmodelaccept').getAttribute('selected_model'), 'key': this.value, 'url': document.getElementById("modelurl").value});};
	} else {
		document.getElementById("modelkey").ochange = function () {socket.emit('OAI_Key_Update', {'model': document.getElementById('btn_loadmodelaccept').getAttribute('selected_model'), 'key': this.value});};
		document.getElementById("modelurl").ochange = null;
	}
	
	//show model select for APIs
	if (data.show_online_model_select) {
		document.getElementById("oaimodel").classList.remove("hidden");
	} else {
		document.getElementById("oaimodel").classList.add("hidden");
	}
	
	//Multiple Model Select?
	if (data.multi_online_models) {
		document.getElementById("oaimodel").setAttribute("multiple", "");
		document.getElementById("oaimodel").options[0].textContent = "All"
	} else {
		document.getElementById("oaimodel").removeAttribute("multiple");
		document.getElementById("oaimodel").options[0].textContent = "Select Model(s)"
	}
	
	//hide or unhide the use gpu checkbox
	if  (data.gpu) {
		document.getElementById("use_gpu_div").classList.remove("hidden");
	} else {
		document.getElementById("use_gpu_div").classList.add("hidden");
	}
	//setup breakmodel
	if (data.breakmodel) {
		document.getElementById("modellayers").classList.remove("hidden");
		//setup model layer count
		document.getElementById("gpu_layers_current").textContent = data.break_values.reduce((a, b) => a + b, 0);
		document.getElementById("gpu_layers_max").textContent = data.layer_count;
		document.getElementById("gpu_count").value = data.gpu_count;
		
		//create the gpu load bars
		var model_layer_bars = document.getElementById('model_layer_bars');
		while (model_layer_bars.firstChild) {
			model_layer_bars.removeChild(model_layer_bars.firstChild);
		}
		
		//Add the bars
		for (let i = 0; i < data.gpu_names.length; i++) {
			var div = document.createElement("div");
			div.classList.add("model_setting_container");
			//build GPU text
			var span = document.createElement("span");
			span.classList.add("model_setting_label");
			span.textContent = "GPU " + i + " " + data.gpu_names[i] + ": "
			//build layer count box
			var input = document.createElement("input");
			input.classList.add("model_setting_value");
			input.classList.add("setting_value");
			input.inputmode = "numeric";
			input.id = "gpu_layers_box_"+i;
			input.value = data.break_values[i];
			input.onblur = function () {
								document.getElementById(this.id.replace("_box", "")).value = this.value;
								update_gpu_layers();
							}
			span.append(input);
			div.append(span);
			//build layer count slider
			var input = document.createElement("input");
			input.classList.add("model_setting_item");
			input.type = "range";
			input.min = 0;
			input.max = data.layer_count;
			input.step = 1;
			input.value = data.break_values[i];
			input.id = "gpu_layers_" + i;
			input.onchange = function () {
								document.getElementById(this.id.replace("gpu_layers", "gpu_layers_box")).value = this.value;
								update_gpu_layers();
							}
			div.append(input);
			//build slider bar #s
			//min
			var span = document.createElement("span");
			span.classList.add("model_setting_minlabel");
			var span2 = document.createElement("span");
			span2.style="top: -4px; position: relative;";
			span2.textContent = 0;
			span.append(span2);
			div.append(span);
			//max
			var span = document.createElement("span");
			span.classList.add("model_setting_maxlabel");
			var span2 = document.createElement("span");
			span2.style="top: -4px; position: relative;";
			span2.textContent = data.layer_count;
			span.append(span2);
			div.append(span);
			
			model_layer_bars.append(div);
		}
		
		//add the disk layers
		if (data.disk_break) {
			var div = document.createElement("div");
			div.classList.add("model_setting_container");
			//build GPU text
			var span = document.createElement("span");
			span.classList.add("model_setting_label");
			span.textContent = "Disk cache: "
			//build layer count box
			var input = document.createElement("input");
			input.classList.add("model_setting_value");
			input.classList.add("setting_value");
			input.inputmode = "numeric";
			input.id = "disk_layers_box";
			input.value = data.disk_break_value;
			input.onblur = function () {
								document.getElementById(this.id.replace("_box", "")).value = this.value;
								update_gpu_layers();
							}
			span.append(input);
			div.append(span);
			//build layer count slider
			var input = document.createElement("input");
			input.classList.add("model_setting_item");
			input.type = "range";
			input.min = 0;
			input.max = data.layer_count;
			input.step = 1;
			input.value = data.disk_break_value;
			input.id = "disk_layers";
			input.onchange = function () {
								document.getElementById(this.id+"_box").value = this.value;
								update_gpu_layers();
							}
			div.append(input);
			//build slider bar #s
			//min
			var span = document.createElement("span");
			span.classList.add("model_setting_minlabel");
			var span2 = document.createElement("span");
			span2.style="top: -4px; position: relative;";
			span2.textContent = 0;
			span.append(span2);
			div.append(span);
			//max
			var span = document.createElement("span");
			span.classList.add("model_setting_maxlabel");
			var span2 = document.createElement("span");
			span2.style="top: -4px; position: relative;";
			span2.textContent = data.layer_count;
			span.append(span2);
			div.append(span);
		}
		
		model_layer_bars.append(div);
		
		update_gpu_layers();
	} else {
		document.getElementById("modellayers").classList.add("hidden");
		accept.classList.remove("disabled");
	}
	accept.disabled = false;
	
	
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
	gpu_layers = []
	disk_layers = 0;
	if (!(document.getElementById("modellayers").classList.contains("hidden"))) {
		for (let i=0; i < document.getElementById("gpu_count").value; i++) {
			gpu_layers.push(document.getElementById("gpu_layers_"+i).value);
		}
		if (document.getElementById("disk_layers")) {
			disk_layers = document.getElementById("disk_layers").value;
		}
	}
	//Need to do different stuff with custom models
	if ((accept.getAttribute('menu') == 'GPT2Custom') || (accept.getAttribute('menu') == 'NeoCustom')) {
		var model = document.getElementById("btn_loadmodelaccept").getAttribute("menu");
		var path = document.getElementById("btn_loadmodelaccept").getAttribute("display_name");
	} else {
		var model = document.getElementById("btn_loadmodelaccept").getAttribute("selected_model");
		var path = "";
	}
	
	let selected_models = [];
	for (item of document.getElementById("oaimodel").selectedOptions) {
		selected_models.push(item.value);
	}
	if (selected_models == ['']) {

		selected_models = [];
	} else if (selected_models.length == 1) {
		selected_models = selected_models[0];
	}
	
	message = {'model': model, 'path': path, 'use_gpu': document.getElementById("use_gpu").checked, 
			   'key': document.getElementById('modelkey').value, 'gpu_layers': gpu_layers.join(), 
			   'disk_layers': disk_layers, 'url': document.getElementById("modelurl").value, 
			   'online_model': selected_models};
	socket.emit("load_model", message);
	document.getElementById("loadmodelcontainer").classList.add("hidden");
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

function world_info_entry(data) {

	
	world_info_data[data.uid] = data;
	
	//delete the existing world info and recreate
	var original_focus = null;
	if (document.getElementById("world_info_"+data.uid)) {
		//First let's get the id of the element we're on so we can restore it after removing the object
		original_focus = document.activeElement.id;
		//console.log("Active ID: "+original_focus);
		//console.log(document.activeElement);
		document.getElementById("world_info_"+data.uid).remove();
	}
	world_info_card_template = document.getElementById("world_info_");
	world_info_card = world_info_card_template.cloneNode(true);
	world_info_card.id = "world_info_"+data.uid;
	world_info_card.setAttribute("uid", data.uid);
	if (data.used_in_game) {
		world_info_card.classList.add("used_in_game");
	} else {
		world_info_card.classList.remove("used_in_game");
	}
	title = world_info_card.querySelector('#world_info_title_')
	title.id = "world_info_title_"+data.uid;
	title.textContent = data.title;
	title.setAttribute("uid", data.uid);
	title.setAttribute("original_text", data.title);
	title.setAttribute("contenteditable", true);
	title.onblur = function () {
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
	delete_icon = world_info_card.querySelector('#world_info_delete_');
	delete_icon.id = "world_info_delete_"+data.uid;
	delete_icon.setAttribute("uid", data.uid);
	delete_icon.setAttribute("title", data.title);
	delete_icon.onclick = function () {
		if (confirm("This will delete world info "+this.getAttribute("title"))) {
			if (parseInt(this.getAttribute("uid")) < 0) {
				this.parentElement.parentElement.remove();
			} else {
				socket.emit("delete_world_info", this.getAttribute("uid"));
			}
		}
	}
	tags = world_info_card.querySelector('#world_info_tags_');
	tags.id = "world_info_tags_"+data.uid;
	//add tag content here
	add_tags(tags, data);
	
	secondarytags = world_info_card.querySelector('#world_info_secondtags_');
	secondarytags.id = "world_info_secondtags_"+data.uid;
	//add second tag content here
	add_secondary_tags(secondarytags, data);
	//w++ toggle
	wpp_toggle_area = world_info_card.querySelector('#world_info_wpp_toggle_area_');
	wpp_toggle_area.id = "world_info_wpp_toggle_area_"+data.uid;
	wpp_toggle = document.createElement("input");
	wpp_toggle.id = "world_info_wpp_toggle_"+data.uid;
	wpp_toggle.setAttribute("type", "checkbox");
	wpp_toggle.setAttribute("uid", data.uid);
	wpp_toggle.checked = data.use_wpp;
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
	//w++ data
	let last_new_value = null
	world_info_wpp_area = world_info_card.querySelector('#world_info_wpp_area_');
	world_info_wpp_area.id = "world_info_wpp_area_"+data.uid;
	world_info_wpp_area.setAttribute("uid", data.uid);
	wpp_format = world_info_card.querySelector('#wpp_format_');
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
	wpp_type = world_info_card.querySelector('#wpp_type_');
	wpp_type.id = "wpp_type_"+data.uid;
	wpp_type.setAttribute("uid", data.uid);
	wpp_type.setAttribute("data_type", "type");
	wpp_type.value = data.wpp.type;
	wpp_name = world_info_card.querySelector('#wpp_name_');
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
				label = document.createElement("span");
				label.textContent = "\xa0\xa0\xa0\xa0Attribute: ";
				attribute_area.append(label);
				input = document.createElement("input");
				input.value = attribute;
				input.type = "text";
				input.setAttribute("uid", data.uid);
				input.setAttribute("data_type", "attribute");
				input.id = "wpp_"+data.uid+"_attr_"+i
				input.onchange = function() {do_wpp(this.parentElement.parentElement)};
				attribute_area.append(input);
				world_info_wpp_area.append(attribute_area);
				j=-1;
				for (value of values) {
					j+=1;
					value_area = document.createElement("div");
					label = document.createElement("span");
					label.textContent = "\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0Value: ";
					value_area.append(label);
					input = document.createElement("input");
					input.type = "text";
					input.onchange = function() {do_wpp(this.parentElement.parentElement)};
					input.value = value;
					input.setAttribute("uid", data.uid);
					input.setAttribute("data_type", "value");
					input.id = "wpp_"+data.uid+"_value_"+i+"_"+j;
					value_area.append(input);
					world_info_wpp_area.append(value_area);
				}
				value_area = document.createElement("div");
				label = document.createElement("span");
				label.textContent = "\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0Value: ";
				value_area.append(label);
				input = document.createElement("input");
				input.type = "text";
				input.setAttribute("uid", data.uid);
				input.setAttribute("data_type", "value");
				input.id = "wpp_"+data.uid+"_value_"+i+"_blank";
				last_new_value = input;
				input.onchange = function() {if (this.value != "") {on_new_wi_item = this.id;do_wpp(this.parentElement.parentElement)}};
				value_area.append(input);
				world_info_wpp_area.append(value_area);
			}
		}
	}
	attribute_area = document.createElement("div");
	label = document.createElement("span");
	label.textContent = "\xa0\xa0\xa0\xa0Attribute: ";
	attribute_area.append(label);
	input = document.createElement("input");
	input.value = "";
	input.type = "text";
	input.setAttribute("uid", data.uid);
	input.setAttribute("value_num", i);
	input.setAttribute("data_type", "attribute");
	input.id = "wpp_"+data.uid+"_attr_blank";
	input.onchange = function() {if (this.value != "") {on_new_wi_item=this.id;do_wpp(this.parentElement.parentElement)}};
	attribute_area.append(input);
	world_info_wpp_area.append(attribute_area);
	
	
	
	//regular data
	content_area = world_info_card.querySelector('#world_info_basic_text_');
	content_area.id = "world_info_basic_text_"+data.uid;
	manual_text = world_info_card.querySelector('#world_info_entry_text_');
	manual_text.id = "world_info_entry_text_"+data.uid;
	manual_text.setAttribute("uid", data.uid);
	manual_text.value = data.manual_text;
	manual_text.onchange = function () {
							world_info_data[this.getAttribute('uid')]['manual_text'] = this.value;
							send_world_info(this.getAttribute('uid'));
							this.classList.add("pulse");
						}
	comment = world_info_card.querySelector('#world_info_comment_');
	comment.id = "world_info_comment_"+data.uid;
	comment.setAttribute("uid", data.uid);
	comment.value = data.comment;
	comment.onchange = function () {
							world_info_data[this.getAttribute('uid')]['comment'] = this.textContent;
							send_world_info(this.getAttribute('uid'));
							this.classList.add("pulse");
						}
	constant_area = world_info_card.querySelector('#world_info_toggle_area_');
	constant_area.id = "world_info_toggle_area_"+data.uid;
	constant = document.createElement("input");
	constant.id = "world_info_constant_"+data.uid;
	constant.setAttribute("type", "checkbox");
	constant.setAttribute("uid", data.uid);
	constant.checked = data.constant;
	constant.setAttribute("data-size", "mini");
	constant.setAttribute("data-onstyle", "success"); 
	constant.setAttribute("data-toggle", "toggle");
	constant.onchange = function () {
							world_info_data[this.getAttribute('uid')]['constant'] = this.checked;
							send_world_info(this.getAttribute('uid'));
							this.classList.add("pulse");
						}
	constant_area.append(constant);
						
	if (!(document.getElementById("world_info_folder_"+data.folder))) {
		folder = document.createElement("div");
		//console.log("Didn't find folder " + data.folder);
	} else {
		folder = document.getElementById("world_info_folder_"+data.folder);
	}
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
	if (data.constant) {
		document.getElementById("world_info_tags_"+data.uid).classList.add("hidden");
		document.getElementById("world_info_secondtags_"+data.uid).classList.add("hidden");
	}
	
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
	
	//put focus back where it was
	if (document.getElementById(original_focus)) {
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
			setTimeout(function() {document.getElementById(original_focus).click();document.getElementById(original_focus).focus()}, 0);
		}
	}
	
	assign_world_info_to_action(null, data.uid);
	
	update_token_lengths();
	
	clearTimeout(calc_token_usage_timeout);
	calc_token_usage_timeout = setTimeout(calc_token_usage, 200);
	clearTimeout(setup_missing_wi_toggles_timeout);
	setup_missing_wi_toggles_timeout = setTimeout(setup_missing_wi_toggles, 200);
	
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
								if (window.confirm("Do you really want to delete this World Info folder and ALL entries under it?")) {
									socket.emit("delete_wi_folder", this.getAttribute("folder"));
								}
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
			add_text.onclick = function() {
											create_new_wi_entry(this.getAttribute("folder"));
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
	error_message_box = document.getElementById('error_message');
	error_message_box.classList.remove("hidden");
	error_box_data = error_message_box.querySelector("#popup_list_area")
	//clear out the error box
	while (error_box_data.firstChild) {
		error_box_data.removeChild(error_box_data.firstChild);
	}
	if (Array.isArray(data)) {
		for (item of data) {
			$e("div", error_box_data, {'innerHTML': item, 'classes': ['console_text']})
			$e("br", error_box_data)
		}
	} else {
		$e("div", error_box_data, {'innerHTML': item, 'classes': ['console_text']})
	}
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
	if (document.readyState === 'complete') {
		for (const cookie of Object.keys(colab_cookies)) {
			setCookie(cookie, colab_cookies[cookie]);
		}
		process_cookies();
		colab_cookies = null;
	}
}

function process_log_message(full_data) {
	for (data of full_data) {
		let level = data['record']['level']['name'];
		let message = data['record']['message'];
		let time = data['record']['time']['repr'];
		let full_log = data['text'];
		//log.push({'level': level, 'message': message, 'time': time, 'full_log': full_log});
		if (level == 'ERROR') {
			show_error_message(data['html']);
		}
		
		//put log message in log popup
		log_popup = document.getElementById('log_popup');
		log_popup_data = log_popup.querySelector("#popup_list_area")
		//clear out the error box
		for (item of data['html']) {
			$e("div", log_popup_data, {'innerHTML': item, 'classes': ['console_text']})
			$e("br", log_popup_data)
		}
	}
}

//--------------------------------------------UI to Server Functions----------------------------------
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
		if (window.confirm("You asked for a new story but your current story has not been saved. If you continue you will loose your changes.")) {
			socket.emit('new_story', '');
		}
	} else {
		socket.emit('new_story', '');
	}
}

function save_as_story(response) {
	if (response == "overwrite?") {
		document.getElementById('save-confirm').classList.remove('hidden')
	}
	
}

function save_bias(item) {
	
	var have_blank = false;
	var biases = {};
	//get all of our biases
	for (bias of document.getElementsByClassName("bias")) {
		//phrase
		var phrase = bias.querySelector(".bias_phrase").querySelector("input").value;
		
		//Score
		var percent = parseFloat(bias.querySelector(".bias_score").querySelector("input").value);
		
		//completion threshold
		var comp_threshold = parseInt(bias.querySelector(".bias_comp_threshold").querySelector("input").value);
		
		if (phrase != "") {
			biases[phrase] = [percent, comp_threshold];
		}
		bias.classList.add("pulse");
	}
	
	//send the biases to the backend
	socket.emit("phrase_bias_update", biases);
	
}

function sync_to_server(item) {
	//get value
	value = null;
	name = null;
	if ((item.tagName.toLowerCase() === 'checkbox') || (item.tagName.toLowerCase() === 'input') || (item.tagName.toLowerCase() === 'select') || (item.tagName.toLowerCase() == 'textarea')) {
		if (item.getAttribute("type") == "checkbox") {
			value = item.checked;
		} else {
			value = item.value;
		}
	} else {
		value = item.textContent;
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

function select_game_text(event) {
	if ((event == null) || (event.code == 'ArrowRight') || (event.code == 'ArrowLeft') || (event.code == 'ArrowDown') || (event.code == 'ArrowUp')) {
		let new_selected_game_chunk = null;
		if (document.selection) {
			if (document.selection.createRange().parentElement().id == 'story_prompt') {
				new_selected_game_chunk = document.selection.createRange().parentElement();
			} else if (document.selection.createRange().parentElement().id == 'gamescreen') {
				new_selected_game_chunk = null;
				console.log("Do nothing");
			} else {
				new_selected_game_chunk = document.selection.createRange().parentElement().parentElement();
			}
		} else {
			if(window.getSelection().anchorNode.parentNode) {
				if (window.getSelection().anchorNode.parentNode.id == 'story_prompt') {
					new_selected_game_chunk = window.getSelection().anchorNode.parentNode;
				} else if (window.getSelection().anchorNode.parentNode.id == "gamescreen") {
					new_selected_game_chunk = null;
					console.log("Do nothing");
				} else {
					new_selected_game_chunk = window.getSelection().anchorNode.parentNode.parentNode;
				}
			} else {
				new_selected_game_chunk = null;
			}
		}
		//if we've moved to a new game chunk we need to save the old chunk
		if (((new_selected_game_chunk != selected_game_chunk) && (selected_game_chunk != null)) || (document.activeElement != document.getElementById("Selected Text"))) {
			edit_game_text();
		}
		
		//Check to see if new selection is a game chunk or something else
		
		if ((new_selected_game_chunk == null) || (((new_selected_game_chunk.id == "story_prompt") || (new_selected_game_chunk.id.slice(0,20) == "Selected Text Chunk ")) && (document.activeElement.isContentEditable))) {
			if (new_selected_game_chunk != selected_game_chunk) {
				for (item of document.getElementsByClassName("editing")) {
					item.classList.remove("editing");
				}
				selected_game_chunk = new_selected_game_chunk;
				selected_game_chunk.classList.add("editing");
			}
			
		} else {
			selected_game_chunk = null;
			for (item of document.getElementsByClassName("editing")) {
				item.classList.remove("editing");
			}
			window.getSelection().removeAllRanges()
		}
	}
}

function edit_game_text() {
	if ((selected_game_chunk != null) && (selected_game_chunk.textContent != selected_game_chunk.original_text) && (selected_game_chunk != document.getElementById("Delete Me"))) {
		if (selected_game_chunk.id == "story_prompt") {
			sync_to_server(selected_game_chunk);
		} else {
			socket.emit("Set Selected Text", {"id": selected_game_chunk.getAttribute("chunk"), "text": selected_game_chunk.textContent});
		}
		selected_game_chunk.original_text = selected_game_chunk.textContent;
		selected_game_chunk.classList.add("pulse");
	}
}

function save_preset() {
	socket.emit("save_new_preset", {"preset": document.getElementById("new_preset_name").value, "description": document.getElementById("new_preset_description").value});
	document.getElementById('save_preset').classList.add('hidden');
}

//--------------------------------------------General UI Functions------------------------------------
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
		console.log(chunk);
		for (let i = 0; i < (action_count-chunk); i++) {
			socket.emit('back', {});
		}
		socket.emit('submit', {'data': "", 'theme': ""});
		document.getElementById('input_text').value = '';
		document.getElementById('themetext').value = '';
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
	document.getElementById("save_preset").classList.remove("hidden");
}

function autoResize(element) {
	element.style.height = 'auto';
	element.style.height = element.scrollHeight + 'px';
}

function token_length(text) {
	if (typeof encode === 'function') {
		return encode(text).length;
	} else {
		return 0;
	}
}

function calc_token_usage() {
	memory_tokens = parseInt(document.getElementById("memory").getAttribute("story_memory_length"));
    authors_notes_tokens = parseInt(document.getElementById("authors_notes").getAttribute("story_authornote_length"));
	prompt_tokens = parseInt(document.getElementById("story_prompt").getAttribute("story_prompt_length"));
    game_text_tokens = 0;
    submit_tokens = token_length(document.getElementById("input_text").value);
	total_tokens = parseInt(document.getElementById('model_max_length_cur').value);
	
	//find world info entries set to go to AI
	world_info_tokens = 0;
	for (wi of document.querySelectorAll(".world_info_card.used_in_game")) {
		if (wi.getAttribute("uid") in world_info_data) {
			world_info_tokens += world_info_data[wi.getAttribute("uid")].token_length;
		}
	}
	
	//find game text tokens
	var game_text_tokens = 0;
	var game_text = document.getElementById('Selected Text').querySelectorAll(".within_max_length");
	var game_text = Array.prototype.slice.call(game_text).reverse();
	for (item of game_text) {
		if (total_tokens - memory_tokens - authors_notes_tokens - world_info_tokens - prompt_tokens - game_text_tokens - submit_tokens > parseInt(item.getAttribute("token_length"))) {
			game_text_tokens += parseInt(item.getAttribute("token_length"));
		}
	}
	
	
	unused_tokens = total_tokens - memory_tokens - authors_notes_tokens - world_info_tokens - prompt_tokens - game_text_tokens - submit_tokens;
	
	document.getElementById("memory_tokens").style.width = (memory_tokens/total_tokens)*100 + "%";
	document.getElementById("memory_tokens").title = "Memory: "+memory_tokens;
	document.getElementById("authors_notes_tokens").style.width = (authors_notes_tokens/total_tokens)*100 + "%";
	document.getElementById("authors_notes_tokens").title = "Author's Notes: "+authors_notes_tokens
	document.getElementById("world_info_tokens").style.width = (world_info_tokens/total_tokens)*100 + "%";
	document.getElementById("world_info_tokens").title = "World Info: "+world_info_tokens
	document.getElementById("prompt_tokens").style.width = (prompt_tokens/total_tokens)*100 + "%";
	document.getElementById("prompt_tokens").title = "Prompt: "+prompt_tokens
	document.getElementById("game_text_tokens").style.width = (game_text_tokens/total_tokens)*100 + "%";
	document.getElementById("game_text_tokens").title = "Game Text: "+game_text_tokens
	document.getElementById("submit_tokens").style.width = (submit_tokens/total_tokens)*100 + "%";
	document.getElementById("submit_tokens").title = "Submit Text: "+submit_tokens
	document.getElementById("unused_tokens").style.width = (unused_tokens/total_tokens)*100 + "%";
	document.getElementById("unused_tokens").title = "Remaining: "+unused_tokens
}

function Change_Theme(theme) {
	var css = document.getElementById("CSSTheme");
    css.setAttribute("href", "/themes/"+theme+".css");
	setTimeout(() => {
		create_theming_elements();
	}, "1000")
	setCookie("theme", theme);
	select = document.getElementById("selected_theme");
	for (element of select.childNodes) {
		if (element.value == theme) {
			element.selected = true;
		} else {
			element.selected = false;
		}
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
						if(name.startsWith('--') && (styleSheets[i].ownerNode.id == "CSSTheme")){
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

function do_biases(data) {
	//console.log(data);
	//clear out our old bias lines
	let bias_list = Object.assign([], document.getElementsByClassName("bias"));
	for (item of bias_list) {
		//console.log(item);
		item.parentNode.removeChild(item);
	}
	
	//add our bias lines
	for (const [key, value] of Object.entries(data.value)) {
		bias_line = document.getElementById("empty_bias").cloneNode(true);
		bias_line.id = "";
		bias_line.classList.add("bias");
		bias_line.querySelector(".bias_phrase").querySelector("input").value = key;
		bias_line.querySelector(".bias_score").querySelector("input").value = value[0];
		update_bias_slider_value(bias_line.querySelector(".bias_score").querySelector("input"));
		bias_line.querySelector(".bias_comp_threshold").querySelector("input").value = value[1];
		update_bias_slider_value(bias_line.querySelector(".bias_comp_threshold").querySelector("input"));
		document.getElementById('biasing').append(bias_line);
	}
	
	//add another bias line if this is the phrase and it's not blank
	bias_line = document.getElementById("empty_bias").cloneNode(true);
	bias_line.id = "";
	bias_line.classList.add("bias");
	bias_line.querySelector(".bias_phrase").querySelector("input").value = "";
	bias_line.querySelector(".bias_phrase").querySelector("input").id = "empty_bias_phrase";
	bias_line.querySelector(".bias_score").querySelector("input").value = 1;
	bias_line.querySelector(".bias_comp_threshold").querySelector("input").value = 50;
	document.getElementById('biasing').append(bias_line);
}

function update_bias_slider_value(slider) {
	slider.parentElement.parentElement.querySelector(".bias_slider_cur").textContent = slider.value;
}

function update_context(data) {
	$(".context-block").remove();

	for (const entry of data) {
		//console.log(entry);
		let contextClass = "context-" + ({
			soft_prompt: "sp",
			prompt: "prompt",
			world_info: "wi",
			memory: "memory",
			authors_note: "an",
			action: "action"
		}[entry.type]);

		let el = document.createElement("span");
		el.classList.add("context-block");
		el.classList.add(contextClass);
		el.innerText = entry.text;

		el.innerHTML = el.innerHTML.replaceAll("<br>", '<span class="material-icons-outlined context-symbol">keyboard_return</span>');

		document.getElementById("context-container").appendChild(el);
	}


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
		text.onblur = function () {
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
	text.onblur = function () {
					if (this.textContent != "") {
						console.log(this.textContent);
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
		text.onblur = function () {
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
	text.onblur = function () {
					if (this.textContent != "") {
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
    e.dataTransfer.setData('text/plain', e.target.id);
	//console.log(e.target.id);
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
    const id = e.dataTransfer.getData('text/plain');
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
		element.parentElement.insertBefore(draggable, element);
		draggable.classList.add("pulse");

		// display the draggable element
		draggable.classList.remove('hidden');
		
		if (element.getAttribute("folder") == draggable.getAttribute("folder")) {
			socket.emit("move_wi", {'dragged_id': dragged_id, 'drop_id': drop_id, 'folder': null});
		} else {
			socket.emit("move_wi", {'dragged_id': dragged_id, 'drop_id': drop_id, 'folder': element.getAttribute("folder")});
		}
	}
}

function dragend(e) {
	// get the draggable element
    const id = e.dataTransfer.getData('text/plain');
    const draggable = document.getElementById(id);
	// display the draggable element
	draggable.classList.remove('hidden');
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
	let action = document.getElementById("Selected Text Chunk "+action_id);
	let words = action.textContent.split(" ");
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
				if (passed_words + span.textContent.trim().split(" ").length < start_word+1) {
					passed_words += span.textContent.trim().split(" ").length;
				} else if (passed_words <= end_word) {
					//OK, we have text that matches, let's do the highlighting
					//we can skip the highlighting if it's already done though
					//console.log(span.textContent.trim().split(" "));
					//console.log("start_word: "+start_word+" end_word: "+end_word+" passed_words: "+passed_words);
					//console.log(span.textContent.trim().split(" ").slice(start_word-passed_words, end_word-passed_words+1).join(" "));
					if (~(span.classList.contains('wi_match'))) {
						var span_text = span.textContent.trim().split(" ");
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
							before_span.textContent = before_highlight_text;
							action.insertBefore(before_span, span);
						}
						//console.log("Highlight Text: '"+highlight_text+"'");
						var highlight_span = document.createElement("span");
						highlight_span.classList.add("wi_match");
						highlight_span.textContent = highlight_text;
						highlight_span.title = wi['content'];
						highlight_span.setAttribute("wi-uid", wi.uid);
						action.insertBefore(highlight_span, span);
						if (after_highlight_text != "") {
							//console.log("After Text: '"+after_highlight_text+"'");
							var after_span = document.createElement("span");
							after_span.textContent = after_highlight_text;
							action.insertBefore(after_span, span);
						}
						//console.log("Done");
						span.remove();
					}
					passed_words += span.textContent.trim().split(" ").length;
				}
			}
		}
	}
}

function update_token_lengths() {
	clearTimeout(calc_token_usage_timeout);
	calc_token_usage_timeout = setTimeout(calc_token_usage, 200);
	return
	max_token_length = parseInt(document.getElementById("model_max_length_cur").value);
	included_world_info = [];
	//clear out the world info included tags
	for (item of document.getElementsByClassName("world_info_included")) {
		item.classList.remove("world_info_included");
	}
	//clear out the text tags
	for (item of document.getElementsByClassName("within_max_length")) {
		item.classList.remove("within_max_length");
	}
	
	//figure out memory length
	if ((document.getElementById("memory").getAttribute("story_memory_length") == null) || (document.getElementById("memory").getAttribute("story_memory_length") == "")) {
		memory_length = 0;
	} else {
		memory_length = parseInt(document.getElementById("memory").getAttribute("story_memory_length"));
	}
	//figure out and tag the length of all the constant world infos
	for (uid in world_info_data) {
		if (world_info_data[uid].constant) {
			if (world_info_data[uid].token_length != null) {
				memory_length += world_info_data[uid].token_length;
				included_world_info.push(uid);
				document.getElementById("world_info_"+uid).classList.add("world_info_included");
			}
		}
	}
	//Figure out author's notes length
	if ((document.getElementById("authors_notes").getAttribute("story_authornote_length") == null) || (document.getElementById("authors_notes").getAttribute("story_authornote_length") == "")) {
		authors_notes = 0;
	} else {
		authors_notes = parseInt(document.getElementById("authors_notes").getAttribute("story_authornote_length"));
	}
	//figure out prompt length
	if ((document.getElementById("story_prompt").getAttribute("story_prompt_length") == null) || (document.getElementById("story_prompt").getAttribute("story_prompt_length") == "")) {
		prompt_length = 0;
	} else {
		prompt_length = parseInt(document.getElementById("story_prompt").getAttribute("story_prompt_length"));
	}
	
	//prompt is truncated at 512 tokens
	if (prompt_length > 512) {
		prompt_length = 512;
	}
	
	//used token length
	token_length = memory_length + authors_notes;
	
	//add in the prompt length if it's set to always add, otherwise add it later
	always_prompt = document.getElementById("story_useprompt").value == "true";
	if (always_prompt) {
		token_length += prompt_length
		document.getElementById("story_prompt").classList.add("within_max_length");
		uids = document.getElementById("story_prompt").getAttribute("world_info_uids")
		for (uid of uids?uids.split(','):[]) {
			if (!(included_world_info.includes(uid))) {
				token_length += world_info_data[uid].token_length;
				included_world_info.push(uid);
				document.getElementById("world_info_"+uid).classList.add("world_info_included");
			}
		}
	} else {
		document.getElementById("story_prompt").classList.remove("within_max_length");
	}
	//figure out how many chunks we have
	max_chunk = -1;
	for (item of document.getElementById("Selected Text").childNodes) {
		if (item.id != undefined) {
			if (item.id != "story_prompt") {
				chunk_num = parseInt(item.id.replace("Selected Text Chunk ", ""));
				if (chunk_num > max_chunk) {
					max_chunk = chunk_num;
				}
			}
		}
	}
	
	//go backwards through the text chunks and tag them if we still have space
	passed_token_limit = false;
	for (var chunk=max_chunk;chunk >= 0;chunk--) {
		if (document.getElementById("Selected Text Chunk "+chunk).getAttribute("token_length") == null) {
			current_chunk_length = 999999999999;
		} else {
			current_chunk_length = parseInt(document.getElementById("Selected Text Chunk "+chunk).getAttribute("token_length"));
		}
		if ((current_chunk_length != 0) && (token_length+current_chunk_length < max_token_length)&& (!(passed_token_limit))) {
			token_length += current_chunk_length;
			document.getElementById("Selected Text Chunk "+chunk).classList.add("within_max_length");
			uids = document.getElementById("Selected Text Chunk "+chunk).getAttribute("world_info_uids")
			for (uid of uids?uids.split(','):[]) {
				if (!(included_world_info.includes(uid))) {
					token_length += world_info_data[uid].token_length;
					included_world_info.push(uid);
					document.getElementById("world_info_"+uid).classList.add("world_info_included");
				}
			}
		} else if (!(passed_token_limit) && (current_chunk_length != 0)) {
			passed_token_limit = true;
			document.getElementById("Selected Text Chunk "+chunk).classList.remove("within_max_length");
		} else {
			document.getElementById("Selected Text Chunk "+chunk).classList.remove("within_max_length");
		}
	}
	
	//if we don't always add prompts
	if ((!always_prompt) && (token_length+prompt_length < max_token_length)) {
		token_length += prompt_length
		document.getElementById("story_prompt").classList.add("within_max_length");
		uids = document.getElementById("story_prompt").getAttribute("world_info_uids")
		for (uid of uids?uids.split(','):[]) {
			if (!(included_world_info.includes(uid))) {
				token_length += world_info_data[uid].token_length;
				included_world_info.push(uid);
				document.getElementById("world_info_"+uid).classList.add("world_info_included");
			}
		}
	} else if (!always_prompt) {
		document.getElementById("story_prompt").classList.remove("within_max_length");
	}
	//Add token count to used_token_length tags
	for (item of document.getElementsByClassName("used_token_length")) {
		item.textContent = "Used Tokens: " + token_length;
	}
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
	document.getElementById('popup').classList.add("hidden");
	document.getElementById('loadmodelcontainer').classList.add("hidden");
	document.getElementById('save-confirm').classList.add("hidden");
	document.getElementById('error_message').classList.add("hidden");
	document.getElementById("advanced_theme_editor").classList.add("hidden");
	document.getElementById("context-viewer-container").classList.add("hidden");
	document.getElementById("save_preset").classList.add("hidden");
	document.getElementById("log_popup").classList.add("hidden");
	
	//unselect sampler items
	for (temp of document.getElementsByClassName("sample_order")) {
		temp.classList.remove("selected");
	}
	
	const finderContainer = document.getElementById("finder-container");
	finderContainer.classList.add("hidden");
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
		console.log("submitting");
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
			for (const redactKey of ["Selected Text", "Options", "Probabilities"]) {
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
	console.log(debug_info);

	downloadString(JSON.stringify(debug_info, null, 4), "kobold_debug.json");
}

function configurePrompt(placeholderData) {
	console.log(placeholderData);
	const container = document.querySelector("#prompt-config-container");
	container.classList.remove("hidden");

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

	document.querySelector("#prompt-config-container").classList.add("hidden");
	$(".prompt-config-ph").remove();
}

function loadNAILorebook(data, filename) {
	let lorebookVersion = data.lorebookVersion;
	let wi_data = {folders: {[filename]: []}, entries: {}};
	console.log(`Loading NAI lorebook version ${lorebookVersion}`);

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
	socket.emit("import_world_info", {data: wi_data});
}

async function loadKoboldData(data, filename) {
	if (data.gamestarted !== undefined) {
		// Story
		socket.emit("upload_file", {"filename": filename, "data": JSON.stringify(data)});
		socket.emit("load_story_list", "");
	} else if (data.folders !== undefined && data.entries !== undefined) {
		// World Info Folder
		socket.emit("import_world_info", {data: data});
	} else {
		// Bad data
		console.error("Bad data!");
		return;
	}
}

async function processDroppedFile(file) {
	let extension = /.*\.(.*)/.exec(file.name)[1];
	console.log("file is", file)
	let data;

	switch (extension) {
		case "png":
			// TODO: Support NovelAI's image lorebook cards. The format for those
			// is base64-encoded JSON under a TXT key called "naidata".
			console.warn("TODO: NAI LORECARDS");
			return;
		case "json":
			// KoboldAI file
			data = JSON.parse(await file.text());
			loadKoboldData(data, file.name);
			break;
		case "lorebook":
			// NovelAI lorebook, JSON encoded.
			data = JSON.parse(await file.text());
			loadNAILorebook(data, file.name);
			break;
		case "css":
			console.warn("TODO: THEME");
			break;
		case "lua":
			console.warn("TODO: USERSCRIPT");
			break
	}
}

function highlightEl(element) {
	if (typeof element === "string") element = document.querySelector(element);
	if (!element) {
		console.error("Bad jump!")
		return;
	}
	
	let area = $(element).closest(".tab-target")[0];
	
	if (!area) {
		console.error("No error? :^(");
		return;
	}
	
	let tab = Array.from($(".tab")).filter((c) => c.getAttribute("tab-target") === area.id)[0];
	tab.click();
	element.scrollIntoView();
}

function addSearchListing(action, highlight) {
	const finderContainer = document.getElementById("finder-container");
	const finder = document.getElementById("finder");

	let result = document.createElement("div");
	result.classList.add("finder-result");
	result.addEventListener("click", function(event) {
		finderContainer.classList.add("hidden");
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

function $e(tag, parent, attributes) {
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

	parent.appendChild(element);
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

function updateSearchListings() {
	if (finder_mode === "scratchpad") return;
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

	finderIcon.innerText = {ui: "search", wi: "auto_stories", scratchpad: "speaker_notes"}[mode];
	finderInput.placeholder = {ui: "Search for something...", wi: "Search for a World Info entry...", scratchpad: "Prompt the AI..."}[mode];
	finderScratchpad.classList.add("hidden");

	finder_mode = mode;
}

function cycleFinderMode() {
	// Initiated by clicking on icon
	updateFinderMode({ui: "wi", wi: "scratchpad", scratchpad: "ui"}[finder_mode]);
}

function open_finder() {
	const finderContainer = document.getElementById("finder-container");
	const finderInput = document.getElementById("finder-input");
	finderInput.value = "";
	$(".finder-result").remove();
	finder_selection_index = -1;
	updateFinderMode("ui");
	
	finderContainer.classList.remove("hidden");
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

//// INIT ////

document.onkeydown = detect_key_down;
document.onkeyup = detect_key_up;
document.getElementById("input_text").onkeydown = detect_enter_submit;

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

$("#context-viewer-close").click(function() {
	$el("#context-viewer-container").classList.add("hidden");
});

$(".token_breakdown").click(function() {
	$el("#context-viewer-container").classList.remove("hidden");
});

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
	const finderContainer = document.getElementById("finder-container");
	const finderInput = document.getElementById("finder-input");
	const finderIcon = document.getElementById("finder-icon");

	// Parse settings for Finder
	for (const el of $(".setting_label")) {
		let name = el.children[0].innerText;

		let tooltipEl = el.getElementsByClassName("helpicon")[0];
		let tooltip = tooltipEl ? tooltipEl.getAttribute("title") : null;

		finder_actions.push({
			name: name,
			desc: tooltip,
			icon: "open_in_new",
			func: function () { highlightEl(el.parentElement) },
		});
	}

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

		let newMode = {">": "wi", "#": "ui", "!": "scratchpad"}[event.key];
		if (newMode && !finderInput.value) {
			event.preventDefault();
			updateFinderMode(newMode);
			return;
		}

		if (event.key === "Enter") {
			if (finder_mode === "scratchpad") {
				sendScratchpadPrompt(finderInput.value);
				return;
			} else if (finder_mode === "ui") {
				let index = finder_selection_index >= 0 ? finder_selection_index : 0;
				actions[index].click();
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

	finderContainer.addEventListener("click", function(e) {
		if (e.target !== this) return;
		finderContainer.classList.add("hidden");
	});
})();

/* -- Debug File -- */
(function() {
	const debugContainer = document.getElementById("debug-file-container");
	// TODO: All of this generic backdrop code really sucks. There should be a
	// standardised thing for popups that adds the dimmed backdrop and standard
	// closing, etc.

	debugContainer.addEventListener("click", function(e) {
		debugContainer.classList.add("hidden");
	});
})();


/* -- Context Menu -- */
(function() {
	const contextMenu = $e("div", document.body, {id: "context-menu", classes: ["hidden"]});

	for (const action of context_menu_actions) {
		// Null adds horizontal rule
		if (!action) {
			$e("hr", contextMenu);
			continue;
		}

		let item = $e("div", contextMenu, {
			classes: ["context-menu-item", "noselect"],
			"visibility-condition": action.visibilityCondition
		});
		let icon = $e("span", item, {classes: ["material-icons-outlined"], innerText: action.icon});
		item.append(action.label);

		item.addEventListener("mousedown", (e) => (e.preventDefault()));
		item.addEventListener("click", action.click);
	}

	$("#gamescreen").contextmenu(function(event) {
		// If control is held, do not run our custom logic or cancel the browser's.
		if (event.ctrlKey) return;

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
		if (getSelectionText()) $(".context-menu-item[visibility-condition=SELECTION]").removeClass("disabled");
		
		// The caret is placed
		if (get_caret_position($("#gamescreen")[0]) !== null) $(".context-menu-item[visibility-condition=CARET]").removeClass("disabled");

		contextMenu.classList.remove("hidden");

		// Set position to click position
		position_context_menu(contextMenu, event.originalEvent.x, event.originalEvent.y);

		// Don't let the document contextmenu catch us and close our context menu
		event.stopPropagation();
	});

	// When we make a browser context menu, close ours.
	$(document).contextmenu(function(event) {
		contextMenu.classList.add("hidden");
	});

	// When we click outside of our context menu, close ours.
	$(document).click(function(event) {
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

/* -- Shortcuts -- */
document.addEventListener("keydown", function(event) {
		
	if (!event.ctrlKey) return;

	switch (event.key) {
		// TODO: Add other shortcuts
		case "k":
			open_finder()
			
			event.preventDefault();
			break;
}
});

//function to load more actions if nessisary
document.getElementById("Selected Text").onscroll = function(){
    //TOP
	if ((scroll_trigger_element != undefined) && (scroll_trigger_element != null)) {
		if(scroll_trigger_element.getBoundingClientRect().bottom >= 0){
			socket.emit("get_next_100_actions", parseInt(scroll_trigger_element.getAttribute("chunk")));
			scroll_trigger_element == null;
		}
	}
}