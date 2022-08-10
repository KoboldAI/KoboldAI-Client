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
socket.on("world_info_folder", function(data){world_info_folder(data);});
socket.on("delete_new_world_info_entry", function(data){document.getElementById("world_info_-1").remove();});
socket.on("delete_world_info_entry", function(data){document.getElementById("world_info_"+data).remove();});
socket.on("error", function(data){show_error_message(data);});
//socket.onAny(function(event_name, data) {console.log({"event": event_name, "class": data.classname, "data": data});});

var presets = {};
var current_chunk_number = null;
var ai_busy_start = Date.now();
var popup_deleteable = false;
var popup_editable = false;
var popup_renameable = false;
var shift_down = false;
var world_info_data = {};
var world_info_folder_data = {};
var saved_settings = {};
//-----------------------------------Server to UI  Functions-----------------------------------------------
function connect() {
	console.log("connected");
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
	current_chunk_number = null;
	var story_area = document.getElementById('Selected Text');
	while (story_area.lastChild.id != 'story_prompt') { 
		story_area.removeChild(story_area.lastChild);
	}
	var option_area = document.getElementById("Select Options");
	while (option_area.firstChild) {
		option_area.removeChild(option_area.firstChild);
	}
	var world_info_area = document.getElementById("story_menu_wi");
	while (world_info_area.firstChild) {
		world_info_area.removeChild(world_info_area.firstChild);
	}
	world_info_data = {};
	world_info_folder({"root": []});
	document.getElementById("story_prompt").setAttribute("world_info_uids", "");
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

function create_options(data) {
	//Set all options before the next chunk to hidden
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
	
	if (document.getElementById("Select Options Chunk "+data.value.id)) {
		var option_chunk = document.getElementById("Select Options Chunk "+data.value.id)
	} else {
		var option_chunk = document.createElement("div");
		option_chunk.id = "Select Options Chunk "+data.value.id;
		if (current_chunk != data.value.id) {
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
	for (item of data.value.options) {
		if ((item['Previous Selection'])) {
			var row = document.createElement("div");
			row.classList.add("sequence_row");
			var textcell = document.createElement("span");
			textcell.textContent = item.text;
			textcell.classList.add("sequence");
			textcell.setAttribute("option_id", i);
			textcell.setAttribute("option_chunk", data.value.id);
			var iconcell = document.createElement("span");
			iconcell.setAttribute("option_id", i);
			iconcell.setAttribute("option_chunk", data.value.id);
			iconcell.classList.add("sequnce_icon");
			var icon = document.createElement("span");
			icon.id = "Pin_"+i;
			icon.classList.add("oi");
			icon.setAttribute('data-glyph', "loop-circular");
			iconcell.append(icon);
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
	for (item of data.value.options) {
		if (!(item.Edited) && !(item['Previous Selection'])) {
			var row = document.createElement("div");
			row.classList.add("sequence_row");
			var textcell = document.createElement("span");
			textcell.textContent = item.text;
			textcell.classList.add("sequence");
			textcell.setAttribute("option_id", i);
			textcell.setAttribute("option_chunk", data.value.id);
			var iconcell = document.createElement("span");
			iconcell.setAttribute("option_id", i);
			iconcell.setAttribute("option_chunk", data.value.id);
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
	option_chunk.scrollIntoView();
}

function do_story_text_updates(data) {
	story_area = document.getElementById('Selected Text');
	current_chunk_number = data.value.id;
	if (document.getElementById('Selected Text Chunk '+data.value.id)) {
		var item = document.getElementById('Selected Text Chunk '+data.value.id);
		//clear out the item first
		while (item.firstChild) { 
			item.removeChild(item.firstChild);
		}
		if (data.value.text == null) {
			var text_array = [];
		} else {
			var text_array = data.value.text.split(" ");
		}
		text_array.forEach(function (text, i) {
			if (text != "") {
				var word = document.createElement("span");
				word.classList.add("rawtext");
				if (i == text_array.length) {
					word.textContent = text;
				} else {
					word.textContent = text+" ";
				}
				item.append(word);
			}
			
		});
		item.original_text = data.value.text;
		item.setAttribute("world_info_uids", "");
		item.classList.remove("pulse")
		item.scrollIntoView();
		assign_world_info_to_action(action_item = item);
	} else {
		var span = document.createElement("span");
		span.id = 'Selected Text Chunk '+data.value.id;
		span.classList.add("rawtext");
		span.chunk = data.value.id;
		span.original_text = data.value.text;
		span.setAttribute("contenteditable", true);
		span.onblur = function () {
			if (this.textContent != this.original_text) {
				socket.emit("Set Selected Text", {"id": this.chunk, "text": this.textContent});
				this.original_text = this.textContent;
				this.classList.add("pulse");
			}
		}
		span.onkeydown = detect_enter_text;
		var text_array = data.value.text.split(" ");
		text_array.forEach(function (text, i) {
			if (text != "") {
				var word = document.createElement("span");
				word.classList.add("rawtext");
				word.classList.add("world_info_tag");
				if (i == text_array.length) {
					word.textContent = text;
				} else {
					word.textContent = text+" ";
				}
				span.append(word);
			}
			
		});
		
		
		story_area.append(span);
		span.scrollIntoView();
		assign_world_info_to_action(action_item = span);
	}
	
	
}

function do_prompt(data) {
	var elements_to_change = document.getElementsByClassName("var_sync_"+data.classname.replace(" ", "_")+"_"+data.name.replace(" ", "_"));
	for (item of elements_to_change) {
		//clear out the item first
		while (item.firstChild) { 
			item.removeChild(item.firstChild);
		}
		
		var text_array = data.value.split(" ");
		text_array.forEach(function (text, i) {
			if (text != "") {
				var word = document.createElement("span");
				word.classList.add("rawtext");
				if (i == text_array.length) {
					word.textContent = text;
				} else {
					word.textContent = text+" ";
				}
				item.append(word);
			}
		});
		item.setAttribute("old_text", data.value)
		item.classList.remove("pulse");
	}
}

function do_story_text_length_updates(data) {
	document.getElementById('Selected Text Chunk '+data.value.id).setAttribute("token_length", data.value.length);
	
}

function do_presets(data) {
	var select = document.getElementById('presets');
	//clear out the preset list
	while (select.firstChild) {
		select.removeChild(select.firstChild);
	}
	//add our blank option
	var option = document.createElement("option");
	option.value="";
	option.text="presets";
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
		document.title = "KoboldAI Client";
	} else {
		document.title = "KoboldAI Client Generating (" + percent_complete + "%)";
	}
}

function do_ai_busy(data) {
	if (data.value) {
		ai_busy_start = Date.now();
		favicon.start_swap()
	} else {
		runtime = Date.now() - ai_busy_start;
		if (document.getElementById("Execution Time")) {
			document.getElementById("Execution Time").textContent = Math.round(runtime/1000).toString().toHHMMSS();
		}
		favicon.stop_swap()
		document.getElementById('btnsend').textContent = "Submit";
	}
}

function var_changed(data) {
	//Special Case for Story Text
	if ((data.classname == "actions") && (data.name == "Selected Text")) {
		do_story_text_updates(data);
	//Special Case for Story Options
	} else if ((data.classname == "actions") && (data.name == "Options")) {
		create_options(data);
	//Special Case for Story Text Length
	} else if ((data.classname == "actions") && (data.name == "Selected Text Length")) {
		do_story_text_length_updates(data);
	//Special Case for Presets
	} else if ((data.classname == 'model') && (data.name == 'presets')) {
		do_presets(data);
	} else if ((data.classname == 'story') && (data.name == 'prompt')) {
		do_prompt(data);
	//Special Case for phrase biasing
	} else if ((data.classname == 'story') && (data.name == 'biases')) {
		do_biases(data);
	//Basic Data Syncing
	} else {
		var elements_to_change = document.getElementsByClassName("var_sync_"+data.classname.replace(" ", "_")+"_"+data.name.replace(" ", "_"));
		for (item of elements_to_change) {
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
		//alternative syncing method
		var elements_to_change = document.getElementsByClassName("var_sync_alt_"+data.classname.replace(" ", "_")+"_"+data.name.replace(" ", "_"));
		for (item of elements_to_change) {
			item.setAttribute(data.classname.replace(" ", "_")+"_"+data.name.replace(" ", "_"), fix_text(data.value));
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

function popup_items(data) {
	var popup_list = document.getElementById('popup_list');
	//first, let's clear out our existing data
	while (popup_list.firstChild) {
		popup_list.removeChild(popup_list.firstChild);
	}
	document.getElementById('popup_upload_input').value = "";
	
	for (item of data) {
		var list_item = document.createElement("span");
		list_item.classList.add("item");
		
		//create the folder icon
		var folder_icon = document.createElement("span");
		folder_icon.classList.add("folder_icon");
		if (item[0]) {
			folder_icon.classList.add("oi");
			folder_icon.setAttribute('data-glyph', "folder");
		}
		list_item.append(folder_icon);
		
		//create the edit icon
		var edit_icon = document.createElement("span");
		edit_icon.classList.add("edit_icon");
		if ((popup_editable) && !(item[0])) {
			edit_icon.classList.add("oi");
			edit_icon.setAttribute('data-glyph', "spreadsheet");
			edit_icon.title = "Edit"
			edit_icon.id = item[1];
			edit_icon.onclick = function () {
							socket.emit("popup_edit", this.id);
					  };
		}
		list_item.append(edit_icon);
		
		//create the rename icon
		var rename_icon = document.createElement("span");
		rename_icon.classList.add("rename_icon");
		if ((popup_renameable) && !(item[0])) {
			rename_icon.classList.add("oi");
			rename_icon.setAttribute('data-glyph', "pencil");
			rename_icon.title = "Rename"
			rename_icon.id = item[1];
			rename_icon.setAttribute("filename", item[2]);
			rename_icon.onclick = function () {
							var new_name = prompt("Please enter new filename for \n"+ this.getAttribute("filename"));
							if (new_name != null) {
								socket.emit("popup_rename", {"file": this.id, "new_name": new_name});
							}
					  };
		}
		list_item.append(rename_icon);
		
		//create the delete icon
		var delete_icon = document.createElement("span");
		delete_icon.classList.add("delete_icon");
		if (popup_deleteable) {
			delete_icon.classList.add("oi");
			delete_icon.setAttribute('data-glyph', "x");
			delete_icon.title = "Delete"
			delete_icon.id = item[1];
			delete_icon.setAttribute("folder", item[0]);
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
		list_item.append(delete_icon);
		
		//create the actual item
		var popup_item = document.createElement("span");
		popup_item.classList.add("file");
		popup_item.id = item[1];
		popup_item.setAttribute("folder", item[0]);
		popup_item.setAttribute("valid", item[3]);
		popup_item.textContent = item[2];
		popup_item.onclick = function () {
						var accept = document.getElementById("popup_accept");
						if (this.getAttribute("valid") == "true") {
							accept.classList.remove("disabled");
							accept.setAttribute("selected_value", this.id);
						} else {
							accept.setAttribute("selected_value", "");
							accept.classList.add("disabled");
							if (this.getAttribute("folder") == "true") {
								socket.emit("popup_change_folder", this.id);
							}
						}
						var popup_list = document.getElementById('popup_list').getElementsByClassName("selected");
						for (item of popup_list) {
							item.classList.remove("selected");
						}
						this.classList.add("selected");
				  };
		list_item.append(popup_item);
		
		
		popup_list.append(list_item);
		
		
	}
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

function show_model_menu(data) {
	document.getElementById("loadmodelcontainer").classList.remove("hidden");
	
	//clear old options
	document.getElementById("modelkey").classList.add("hidden");
	document.getElementById("modelkey").value = "";
	document.getElementById("modelurl").classList.add("hidden");
	document.getElementById("use_gpu_div").classList.add("hidden");
	document.getElementById("modellayers").classList.add("hidden");
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
	for (item of data.breadcrumbs) {
		var button = document.createElement("button");
		button.classList.add("breadcrumbitem");
		button.id = item[0];
		button.value = item[1];
		button.onclick = function () {
					socket.emit('selectmodel', {'data': this.id, 'folder': this.value});
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
		list_item.classList.add("item");
		
		//create the folder icon
		var folder_icon = document.createElement("span");
		folder_icon.classList.add("folder_icon");
		if (item[3]) {
			folder_icon.classList.add("oi");
			folder_icon.setAttribute('data-glyph', "folder");
		}
		list_item.append(folder_icon);
		
		//create the delete icon
		//var delete_icon = document.createElement("span");
		//delete_icon.classList.add("delete_icon");
		//if (popup_deleteable) {
		//	delete_icon.classList.add("oi");
		//	delete_icon.setAttribute('data-glyph', "x");
		//	delete_icon.id = item[1];
		//	delete_icon.setAttribute("folder", item[0]);
		//	delete_icon.onclick = function () {
		//					if (this.getAttribute("folder") == "true") {
		//						if (window.confirm("Do you really want to delete this folder and ALL files under it?")) {
		//							socket.emit("popup_delete", this.id);
		//						}
		//					} else {
		//						if (window.confirm("Do you really want to delete this file?")) {
		//							socket.emit("popup_delete", this.id);
		//						}
		//					}
		//			  };
		//}
		//list_item.append(delete_icon);
		
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
	for (let i=0; i < document.getElementById("gpu_count").value; i++) {
		gpu_layers.push(document.getElementById("gpu_layers_"+i).value);
	}
	if (document.getElementById("disk_layers")) {
		disk_layers = document.getElementById("disk_layers").value;
	} else {
		disk_layers = "0";
	}
	//Need to do different stuff with custom models
	if ((accept.getAttribute('menu') == 'GPT2Custom') || (accept.getAttribute('menu') == 'NeoCustom')) {
		var model = document.getElementById("btn_loadmodelaccept").getAttribute("menu");
		var path = document.getElementById("btn_loadmodelaccept").getAttribute("display_name");
	} else {
		var model = document.getElementById("btn_loadmodelaccept").getAttribute("selected_model");
		var path = "";
	}
	
	message = {'model': model, 'path': path, 'use_gpu': document.getElementById("use_gpu").checked, 
			   'key': document.getElementById('modelkey').value, 'gpu_layers': gpu_layers.join(), 
			   'disk_layers': disk_layers, 'url': document.getElementById("modelurl").value, 
			   'online_model': document.getElementById("oaimodel").value};
	socket.emit("load_model", message);
	document.getElementById("loadmodelcontainer").classList.add("hidden");
}

function world_info_entry(data) {
	//console.log(data);
	
	world_info_data[data.uid] = data;
	
	//delete the existing world info and recreate
	if (document.getElementById("world_info_"+data.uid)) {
		document.getElementById("world_info_"+data.uid).remove();
	}
	world_info_card_template = document.getElementById("world_info_");
	world_info_card = world_info_card_template.cloneNode(true);
	world_info_card.id = "world_info_"+data.uid;
	world_info_card.setAttribute("uid", data.uid);
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
			socket.emit("delete_world_info", this.getAttribute("uid"));
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
	content = world_info_card.querySelector('#world_info_entry_text_');
	content.id = "world_info_entry_text_"+data.uid;
	content.setAttribute("uid", data.uid);
	content.value = data.content;
	content.onchange = function () {
							world_info_data[this.getAttribute('uid')]['content'] = this.textContent;
							send_world_info(this.getAttribute('uid'));
							this.classList.add("pulse");
						}
	comment = world_info_card.querySelector('#world_info_comment_');
	comment.id = "world_info_comment_"+data.uid;
	content.setAttribute("uid", data.uid);
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
	
	$('#world_info_constant_'+data.uid).bootstrapToggle();
	assign_world_info_to_action(uid=data.uid);
	
	update_token_lengths();
	
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
			title = document.createElement("h2");
			title.addEventListener('dragenter', dragEnter)
			title.addEventListener('dragover', dragOver);
			title.addEventListener('dragleave', dragLeave);
			title.addEventListener('drop', drop);
			collapse_icon = document.createElement("span");
			collapse_icon.id = "world_info_folder_collapse_"+folder_name;
			collapse_icon.classList.add("oi");
			collapse_icon.setAttribute("data-glyph", "chevron-bottom");
			collapse_icon.setAttribute("folder", folder_name);
			collapse_icon.onclick = function () {
								hide_wi_folder(this.getAttribute("folder"));
								document.getElementById('world_info_folder_expand_'+this.getAttribute("folder")).classList.remove('hidden');
								this.classList.add("hidden");
							};
			title.append(collapse_icon);
			expand_icon = document.createElement("span");
			expand_icon.id = "world_info_folder_expand_"+folder_name;
			expand_icon.classList.add("oi");
			expand_icon.setAttribute("data-glyph", "chevron-right");
			expand_icon.setAttribute("folder", folder_name);
			expand_icon.onclick = function () {
								unhide_wi_folder(this.getAttribute("folder"));
								document.getElementById('world_info_folder_collapse_'+this.getAttribute("folder")).classList.remove('hidden');
								this.classList.add("hidden");
							};
			expand_icon.classList.add("hidden");
			title.append(expand_icon);
			icon = document.createElement("span");
			icon.classList.add("oi");
			icon.setAttribute("data-glyph", "folder");
			title.append(icon);
			title_text = document.createElement("span");
			title_text.setAttribute("contenteditable", true);
			title_text.setAttribute("original_text", folder_name);
			title_text.textContent = folder_name;
			title_text.onblur = function () {
				if (this.textContent != this.getAttribute("original_text")) {
					//Need to check if the new folder name is already in use
					socket.emit("Rename_World_Info_Folder", {"old_folder": this.getAttribute("original_text"), "new_folder": this.textContent});
				}
			}
			title.append(title_text);
			folder.append(title);
			//create add button
			add_icon = document.createElement("span");
			add_icon.classList.add("oi");
			add_icon.setAttribute("data-glyph", "plus");
			add_icon.textContent = "Add World Info Entry";
			add_icon.setAttribute("folder", folder_name);
			add_icon.onclick = function() {
											create_new_wi_entry(this.getAttribute("folder"));
										  }
			folder.append(add_icon);
			
			//We want to insert this folder before the next folder
			if (i+1 < folders.length) {
				//We have more folders, so let's see if any of them exist so we can insert before that
				var found = false;
				for (var j = i+1; j < folders.length; j++) {
					if (document.getElementById("world_info_folder_"+folders[j])) {
						found = true;
						document.getElementById("story_menu_wi").insertBefore(folder, document.getElementById("world_info_folder_"+folders[j]));
						break;
					}
				}
				if (!(found)) {
					if (document.getElementById("new_world_info_button")) {
						document.getElementById("story_menu_wi").insertBefore(folder, document.getElementById("new_world_info_button"));
					} else {
						document.getElementById("story_menu_wi").append(folder);
					}
				}
			} else {
				if (document.getElementById("new_world_info_button")) {
					document.getElementById("story_menu_wi").insertBefore(folder, document.getElementById("new_world_info_button"));
				} else {
					document.getElementById("story_menu_wi").append(folder);
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
		temp = document.createElement("h2");
		add_icon = document.createElement("span");
		add_icon.classList.add("oi");
		add_icon.setAttribute("data-glyph", "plus");
		add_icon.textContent = "Add World Info Folder";
		add_icon.onclick = function() {
										socket.emit("create_world_info_folder", {});
									  }
		temp.append(add_icon);
		add_folder.append(temp);
		document.getElementById("story_menu_wi").append(add_folder);
	}
}

function show_error_message(data) {
	error_message_box = document.getElementById('error_message');
	error_message_box.classList.remove("hidden");
	error_message_box.querySelector("#popup_list_area").textContent = data;
}
//--------------------------------------------UI to Server Functions----------------------------------
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
			socket.emit("upload_file", {'filename': file.name, "data": event.target.result});
		};
		reader.readAsArrayBuffer(file);
	}
}

function send_world_info(uid) {
	socket.emit("edit_world_info", world_info_data[uid]);
}

//--------------------------------------------General UI Functions------------------------------------
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
	bias_line.querySelector(".bias_score").querySelector("input").value = 1;
	bias_line.querySelector(".bias_comp_threshold").querySelector("input").value = 50;
	document.getElementById('biasing').append(bias_line);
}

function update_bias_slider_value(slider) {
	slider.parentElement.parentElement.querySelector(".bias_slider_cur").textContent = slider.value;
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
	text.onblur = function () {
					world_info_data[this.getAttribute('uid')]['key'].push(this.textContent);
					send_world_info(this.getAttribute('uid'));
					this.classList.add("pulse");
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
	text.onblur = function () {
					world_info_data[this.getAttribute('uid')]['keysecondary'].push(this.textContent);
					send_world_info(this.getAttribute('uid'));
					this.classList.add("pulse");
				};
	text.onclick = function () {
					this.textContent = "";
				};
	tag_item.append(text);
	tag_item.id = "world_info_secondtags_"+data.uid+"_new";
	tags.append(tag_item);
}
	
function create_new_wi_entry(folder) {
	data = {"uid": -1,
                                    "title": "New World Info Entry",
                                    "key": [],
                                    "keysecondary": [],
                                    "folder": folder,
                                    "constant": false,
                                    "content": "",
                                    "comment": "",
                                    "token_length": 0,
                                    "selective": false
                                    };
	world_info_entry(data);
}

function hide_wi_folder(folder) {
	if (document.getElementById("world_info_folder_"+folder)) {
		folder_item = document.getElementById("world_info_folder_"+folder);
		for (card of folder_item.children) {
			if (card.tagName != "H2") {
				card.classList.add("hidden");
			}
		}
	}
}

function unhide_wi_folder(folder) {
	if (document.getElementById("world_info_folder_"+folder)) {
		folder_item = document.getElementById("world_info_folder_"+folder);
		for (card of folder_item.children) {
			if (card.tagName != "H2") {
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
	element = find_wi_container(e.target);
    element.classList.add('drag-over');
}

function dragLeave(e) {
	element = find_wi_container(e.target);
    element.classList.remove('drag-over');
}

function drop(e) {
	e.preventDefault();
    element = find_wi_container(e.target);
    element.classList.remove('drag-over');

    // get the draggable element
    const id = e.dataTransfer.getData('text/plain');
    const draggable = document.getElementById(id);
	//console.log(id);
	dragged_id = draggable.id.split("_").slice(-1)[0];
	drop_id = element.id.split("_").slice(-1)[0];

    // get the drop element
	element = find_wi_container(e.target);
	
	//check if we're droping on a folder, and then append it to the folder
	if (element.children[0].tagName == "H2") {
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
	element = find_wi_container(e.target);
	element.classList.remove('hidden');
	e.preventDefault();
}

function assign_world_info_to_action(uid=null, action_item=null) {
	if (Object.keys(world_info_data).length > 0) {
		if (uid != null) {
			var worldinfo_to_check = {};
			worldinfo_to_check[uid] = world_info_data[uid]
		} else {
			var worldinfo_to_check = world_info_data;
		}
		if (action_item != null) {
			var actions = [action_item]
		} else {
			var actions = document.getElementById("Selected Text").children;
		}
		for (action of actions) {
			//First check to see if we have a key in the text
			var words = Array.prototype.slice.call( action.children );
			words_text = [];
			for (word of words) {
				words_text.push(word.textContent);
			}
			for (const [key, worldinfo] of  Object.entries(worldinfo_to_check)) {
				//remove any world info tags
				for (tag of action.getElementsByClassName("tag_uid_"+uid)) {
					tag.classList.remove("tag_uid_"+uid);
					tag.removeAttribute("title");
					current_ids = tag.parentElement.getAttribute("world_info_uids").split(",");
					removeA(current_ids, uid);
					tag.parentElement.setAttribute("world_info_uids", current_ids.join(","));
				}
				for (keyword of worldinfo['key']) {
					if ((action.textContent.replace(/[^0-9a-z \'\"]/gi, '')).includes(keyword)) {
						//Ok we have a key match, but we need to check for secondary keys if applicable
						if (worldinfo['keysecondary'].length > 0) {
							for (second_key of worldinfo['keysecondary']) {
								if (action.textContent.replace(/[^0-9a-z \'\"]/gi, '').includes(second_key)) {
									//First let's assign our world info id to the action so we know to count the tokens for the world info
									current_ids = action.getAttribute("world_info_uids")?action.getAttribute("world_info_uids").split(','):[];
									if (!(current_ids.includes(uid))) {
										current_ids.push(uid);
									}
									action.setAttribute("world_info_uids", current_ids.join(","));
									//OK we have the phrase in our action. Let's see if we can identify the word(s) that are triggering
									for (var i = 0; i < words.length; i++) {
										key_words = keyword.split(" ").length;
										var to_check = words_text.slice(i, i+key_words).join("").replace(/[^0-9a-z \'\"]/gi, '').trim();
										if (keyword == to_check) {
											for (var j = i; j < key_words+i; j++) {
												words[j].title = worldinfo['content'];
												words[j].classList.add("tag_uid_"+uid);
											}
										}
									}
								}
							}
						} else {
							//First let's assign our world info id to the action so we know to count the tokens for the world info
							current_ids = action.getAttribute("world_info_uids")?action.getAttribute("world_info_uids").split(','):[];
							if (!(current_ids.includes(uid))) {
								current_ids.push(uid);
							}
							action.setAttribute("world_info_uids", current_ids.join(","));
							//OK we have the phrase in our action. Let's see if we can identify the word(s) that are triggering
							for (var i = 0; i < words.length; i++) {
								key_words = keyword.split(" ").length;
								var to_check = words_text.slice(i, i+key_words).join("").replace(/[^0-9a-z \'\"]/gi, '').trim();
								if (keyword == to_check) {
									for (var j = i; j < key_words+i; j++) {
										words[j].title = worldinfo.content;
										words[j].classList.add("tag_uid_"+uid);
									}
								}
							}
						}
						
					}
				}
			}
		}
	}
}

function update_token_lengths() {
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
		prompt_length = 999999999999;
	} else {
		prompt_length = parseInt(document.getElementById("story_prompt").getAttribute("story_prompt_length"));
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
	for (var chunk=max_chunk;chunk >= 0;chunk--) {
		if (document.getElementById("Selected Text Chunk "+chunk).getAttribute("token_length") == null) {
			current_chunk_length = 999999999999;
		} else {
			current_chunk_length = parseInt(document.getElementById("Selected Text Chunk "+chunk).getAttribute("token_length"));
		}
		if ((current_chunk_length != 0) && (token_length+current_chunk_length < max_token_length)) {
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

function toggle_flyout(x) {
	if (document.getElementById("SideMenu").classList.contains("pinned")) {
		//do nothing
	} else {
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
}

function toggle_flyout_right(x) {
	if (document.getElementById("rightSideMenu").classList.contains("pinned")) {
		//do nothing
	} else {
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
}

function toggle_settings_pin_flyout() {
	if (document.getElementById("SideMenu").classList.contains("pinned")) {
		document.getElementById("SideMenu").classList.remove("pinned");
		document.getElementById("main-grid").classList.remove("settings_pinned");
		document.getElementById("setting_menu_icon").classList.remove("hidden");
	} else {
		document.getElementById("setting_menu_icon").classList.remove("change");
		document.getElementById("setting_menu_icon").classList.add("hidden");
		document.getElementById("SideMenu").classList.add("pinned");
		document.getElementById("main-grid").classList.add("settings_pinned");
	}
}

function toggle_story_pin_flyout() {
	if (document.getElementById("rightSideMenu").classList.contains("pinned")) {
		document.getElementById("rightSideMenu").classList.remove("pinned");
		document.getElementById("main-grid").classList.remove("story_pinned");
		document.getElementById("story_menu_icon").classList.remove("hidden");
	} else {
		document.getElementById("rightSideMenu").classList.add("pinned");
		document.getElementById("main-grid").classList.add("story_pinned");
		document.getElementById("story_menu_icon").classList.remove("change");
		document.getElementById("story_menu_icon").classList.add("hidden");
	}
}

function detect_enter_submit(e) {
	if (((e.code == "Enter") || (e.code == "NumpadEnter")) && !(shift_down)) {
		if (typeof e.stopPropagation != "undefined") {
			e.stopPropagation();
		} else {
			e.cancelBubble = true;
		}
		document.getElementById("btnsend").onclick();
		document.getElementById('input_text').value = ''
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

function detect_shift_down(e) {
	if ((e.code == "ShiftLeft") || (e.code == "ShiftRight")) {
		shift_down = true;
	}
}

function detect_shift_up(e) {
	if ((e.code == "ShiftLeft") || (e.code == "ShiftRight")) {
		shift_down = false;
	}
}

$(document).ready(function(){
	document.onkeydown = detect_shift_down;
	document.onkeyup = detect_shift_up;
	document.getElementById("input_text").onkeydown = detect_enter_submit;
});