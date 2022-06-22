var socket;
socket = io.connect(window.location.origin, {transports: ['polling', 'websocket'], closeOnBeforeunload: false, query:{"ui":  "2"}});

//Let's register our server communications
socket.on('connect', function(){connect();});
socket.on('disconnect', function(){disconnect();});
socket.on('reset_story', function(){reset_story();});
socket.on('var_changed', function(data){var_changed(data);});
//socket.onAny(function(event_name, data) {console.log({"event": event_name, "data": data});});

var backend_vars = {};
var presets = {}
//-----------------------------------Server to UI  Functions-----------------------------------------------
function connect() {
	console.log("connected");
}

function disconnect() {
	console.log("disconnected");
}

function reset_story() {
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
	if (document.getElementById("Select Options Chunk "+data.value.id)) {
			var option_chunk = document.getElementById("Select Options Chunk "+data.value.id)
		} else {
			var option_area = document.getElementById("Select Options");
			var option_chunk = document.createElement("div");
			option_chunk.id = "Select Options Chunk "+data.value.id;
			option_area.append(option_chunk);
		}
		//first, let's clear out our existing data
		while (option_chunk.firstChild) {
			option_chunk.removeChild(option_chunk.firstChild);
		}
		var table = document.createElement("table");
		table.classList.add("sequence");
		table.style = "border-spacing: 0;";
		//Add pins
		i=0;
		for (item of data.value.options) {
			if (item.Pinned) {
				var row = document.createElement("tr");
				row.classList.add("sequence");
				var textcell = document.createElement("td");
				textcell.textContent = item.text;
				textcell.classList.add("sequence");
				textcell.setAttribute("option_id", i);
				textcell.setAttribute("option_chunk", data.value.id);
				var iconcell = document.createElement("td");
				iconcell.setAttribute("option_id", i);
				iconcell.setAttribute("option_chunk", data.value.id);
				var icon = document.createElement("span");
				icon.id = "Pin_"+i;
				icon.classList.add("oi");
				icon.setAttribute('data-glyph', "pin");
				iconcell.append(icon);
				textcell.onclick = function () {
										socket.emit("Set Selected Text", {"chunk": this.getAttribute("option_chunk"), "option": this.getAttribute("option_id")});
								  };
				iconcell.onclick = function () {
										socket.emit("Pinning", {"chunk": this.getAttribute("option_chunk"), "option": this.getAttribute("option_id"), "set": false});
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
			if (!(item.Edited) && !(item.Pinned) && !(item['Previous Selection'])) {
				var row = document.createElement("tr");
				row.classList.add("sequence");
				var textcell = document.createElement("td");
				textcell.textContent = item.text;
				textcell.classList.add("sequence");
				textcell.setAttribute("option_id", i);
				textcell.setAttribute("option_chunk", data.value.id);
				var iconcell = document.createElement("td");
				iconcell.setAttribute("option_id", i);
				iconcell.setAttribute("option_chunk", data.value.id);
				var icon = document.createElement("span");
				icon.id = "Pin_"+i;
				icon.classList.add("oi");
				icon.setAttribute('data-glyph', "pin");
				icon.setAttribute('style', "filter: brightness(50%);");
				iconcell.append(icon);
				iconcell.onclick = function () {
										socket.emit("Pinning", {"chunk": this.getAttribute("option_chunk"), "option": this.getAttribute("option_id"), "set": true});
								   };
				textcell.onclick = function () {
										socket.emit("Set Selected Text", {"chunk": this.getAttribute("option_chunk"), "option": this.getAttribute("option_id")});
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
	} else {
		var span = document.createElement("span");
		span.id = 'Selected Text Chunk '+data.value.id;
		span.chunk = data.value.id;
		span.original_text = data.value.text;
		span.setAttribute("contenteditable", true);
		span.onblur = function () {
			if (this.textContent != this.original_text) {
				socket.emit("Set Selected Text", {"id": this.chunk, "text": this.textContent});
			}
		}
		span.textContent = data.value.text;
		
		story_area.append(span);
	}
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
	for (item of data.value) {
		presets[item.preset] = item;
		var option = document.createElement("option");
		option.value=item.preset;
		option.text=item.preset;
		select.append(option);
	}
}

function selected_preset(data) {
	if ((data.value == undefined) || (presets[data.value] == undefined)) {
		return;
	}
	for (const [key, value] of Object.entries(presets[data.value])) {
		if (key.charAt(0) != '_') {
			var elements_to_change = document.getElementsByClassName("var_sync_model_"+key);
			for (item of elements_to_change) {
				if (item.tagName.toLowerCase() === 'input') {
					item.value = fix_text(value);
				} else {
					item.textContent = fix_text(value);
				}
			}
		}
	}
}

function var_changed(data) {
	//Special Case for Story Text
	if ((data.classname == "actions") && (data.name == "Selected Text")) {
		do_story_text_updates(data);
	//Special Case for Story Options
	} else if ((data.classname == "actions") && (data.name == "Options")) {
		create_options(data);
	//Special Case for Presets
	} else if ((data.classname == 'model') && (data.name == 'presets')) {
		do_presets(data);
	} else if ((data.classname == "model") && (data.name == "selected_preset")) {
		selected_preset(data);
	//Basic Data Syncing
	} else {
		var elements_to_change = document.getElementsByClassName("var_sync_"+data.classname+"_"+data.name);
		for (item of elements_to_change) {
			if (item.tagName.toLowerCase() === 'input') {
				item.value = fix_text(data.value);
			} else {
				item.textContent = fix_text(data.value);
			}
		}
		var elements_to_change = document.getElementsByClassName("var_sync_alt_"+data.classname+"_"+data.name);
		for (item of elements_to_change) {
			item.setAttribute("server_value", fix_text(data.value));
		}
	}
}

//--------------------------------------------UI to Server Functions----------------------------------


//--------------------------------------------General UI Functions------------------------------------
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
