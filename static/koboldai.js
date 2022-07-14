var socket;
socket = io.connect(window.location.origin, {transports: ['polling', 'websocket'], closeOnBeforeunload: false, query:{"ui":  "2"}});

//Let's register our server communications
socket.on('connect', function(){connect();});
socket.on("disconnect", (reason, details) => {
  console.log("Lost connection from: "+reason); // "transport error"
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
//socket.onAny(function(event_name, data) {console.log({"event": event_name, "class": data.classname, "data": data});});

var backend_vars = {};
var presets = {}
var ai_busy_start = Date.now();
var popup_deleteable = false;
var popup_editable = false;
var popup_renameable = false;
var shift_down = false;
//-----------------------------------Server to UI  Functions-----------------------------------------------
function connect() {
	console.log("connected");
}

function disconnect() {
	console.log("disconnected");
}

function reset_story() {
	console.log("Resetting story");
	var story_area = document.getElementById('Selected Text');
	while (story_area.firstChild) {
		story_area.removeChild(story_area.firstChild);
	}
	var option_area = document.getElementById("Select Options");
	while (option_area.firstChild) {
		option_area.removeChild(option_area.firstChild);
	}
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
}

function do_story_text_updates(data) {
	story_area = document.getElementById('Selected Text');
	if (document.getElementById('Selected Text Chunk '+data.value.id)) {
		document.getElementById('Selected Text Chunk '+data.value.id).textContent = data.value.text;
		document.getElementById('Selected Text Chunk '+data.value.id).classList.remove("pulse")
		document.getElementById('Selected Text Chunk '+data.value.id).scrollIntoView();
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
		span.textContent = data.value.text;
		
		
		story_area.append(span);
		span.scrollIntoView();
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
		for (const [preset, preset_value] of Object.entries(value)) {
			var option = document.createElement("option");
			option.value=key+"|"+preset;
			option.text=preset;
			option.title = preset_value.description;
			option_group.append(option);
		}
		select.append(option_group);
	}
}

function selected_preset(data) {
	
	preset_key = data.value.split("|")[0];
	preset = data.value.split("|")[1];
	if ((data.value == undefined) || (presets[preset_key] == undefined)) {
		return;
	}
	if (presets[preset_key][preset] == undefined) {
		return;
	}
	for (const [key, value] of Object.entries(presets[preset_key][preset])) {
		if (key.charAt(0) != '_') {
			var elements_to_change = document.getElementsByClassName("var_sync_model_"+key);
			for (item of elements_to_change) {
				if (item.tagName.toLowerCase() === 'input') {
					item.value = value;
				} else {
					item.textContent = fix_text(value);
				}
			}
		}
	}
	socket.emit("var_change", {"ID": "model_selected_preset", "value": data.value});
}

function update_status_bar(data) {
	var total_tokens = document.getElementById('model_genamt').value;
	var percent_complete = data.value;
	var percent_bar = document.getElementsByClassName("statusbar_inner");
	for (item of percent_bar) {
		item.setAttribute("style", "width:"+percent_complete+"%");
		item.textContent = Math.round(percent_complete)+"%"
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
	} else if ((data.classname == "model") && (data.name == "selected_preset")) {
		selected_preset(data);
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

function buildload(data) {
	console.log(data);
}

//--------------------------------------------UI to Server Functions----------------------------------
function sync_to_server(item) {
	//get value
	value = null;
	name = null;
	if ((item.tagName.toLowerCase() === 'input') || (item.tagName.toLowerCase() === 'select') || (item.tagName.toLowerCase() == 'textarea')) {
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

//--------------------------------------------General UI Functions------------------------------------
function update_token_lengths() {
	max_token_length = parseInt(document.getElementById("model_max_length_cur").value);
	if ((document.getElementById("memory").getAttribute("story_memory_length") == null) || (document.getElementById("memory").getAttribute("story_memory_length") == "")) {
		memory_length = 0;
	} else {
		memory_length = parseInt(document.getElementById("memory").getAttribute("story_memory_length"));
	}
	if ((document.getElementById("authors_notes").getAttribute("story_authornote_length") == null) || (document.getElementById("authors_notes").getAttribute("story_authornote_length") == "")) {
		authors_notes = 0;
	} else {
		authors_notes = parseInt(document.getElementById("authors_notes").getAttribute("story_authornote_length"));
	}
	if ((document.getElementById("story_prompt").getAttribute("story_prompt_length") == null) || (document.getElementById("story_prompt").getAttribute("story_prompt_length") == "")) {
		prompt_length = 0;
	} else {
		prompt_length = parseInt(document.getElementById("story_prompt").getAttribute("story_prompt_length"));
	}
	
	token_length = memory_length + authors_notes;
	
	always_prompt = document.getElementById("story_useprompt").value == "true";
	if (always_prompt) {
		token_length += prompt_length
		document.getElementById("story_prompt").classList.add("within_max_length");
	} else {
		document.getElementById("story_prompt").classList.remove("within_max_length");
	}
	max_chunk = -1;
	for (item of document.getElementById("Selected Text").childNodes) {
		chunk_num = parseInt(item.id.replace("Selected Text Chunk ", ""));
		if (chunk_num > max_chunk) {
			max_chunk = chunk_num;
		}
	}
	
	for (var chunk=max_chunk;chunk >= 0;chunk--) {
		current_chunk_length = parseInt(document.getElementById("Selected Text Chunk "+chunk).getAttribute("token_length"));
		if (token_length+current_chunk_length < max_token_length) {
			token_length += current_chunk_length;
			document.getElementById("Selected Text Chunk "+chunk).classList.add("within_max_length");
		} else {
			document.getElementById("Selected Text Chunk "+chunk).classList.remove("within_max_length");
		}
	}
	
	if ((!always_prompt) && (token_length+prompt_length < max_token_length)) {
		token_length += prompt_length
		document.getElementById("story_prompt").classList.add("within_max_length");
	} else if (!always_prompt) {
		document.getElementById("story_prompt").classList.remove("within_max_length");
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
	if (document.getElementById("SideMenu").classList.contains("open")) {
		x.classList.remove("change");
		document.getElementById("SideMenu").classList.remove("open");
		document.getElementById("main-grid").classList.remove("menu-open");
		//if pinned
		if (document.getElementById("SideMenu").classList.contains("pinned")) {
			document.getElementById("menu_pin").classList.remove("hidden");
		} else {
			document.getElementById("menu_pin").classList.add("hidden");
		}
	} else {
		x.classList.add("change");
		document.getElementById("SideMenu").classList.add("open");
		document.getElementById("main-grid").classList.add("menu-open");
		document.getElementById("menu_pin").classList.remove("hidden");
	}
}

function toggle_flyout_right(x) {
	if (document.getElementById("rightSideMenu").classList.contains("open")) {
		document.getElementById("rightSideMenu").classList.remove("open");
		x.setAttribute("data-glyph", "chevron-left");
	} else {
		document.getElementById("rightSideMenu").classList.add("open");
		x.setAttribute("data-glyph", "chevron-right");
	}
}

function toggle_pin_flyout() {
	if (document.getElementById("SideMenu").classList.contains("pinned")) {
		document.getElementById("SideMenu").classList.remove("pinned");
		document.getElementById("main-grid").classList.remove("pinned");
	} else {
		document.getElementById("SideMenu").classList.add("pinned");
		document.getElementById("main-grid").classList.add("pinned");
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
		console.log("Doing Text Enter");
		console.log(e.currentTarget.activeElement);
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