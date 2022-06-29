var socket;
socket = io.connect(window.location.origin, {transports: ['polling', 'websocket'], closeOnBeforeunload: false, query:{"ui":  "2"}});

//Let's register our server communications
socket.on('connect', function(){connect();});
socket.on("disconnect", (reason, details) => {
  console.log("Lost connection from: "+reason); // "transport error"
});
socket.on('reset_story', function(){reset_story();});
socket.on('var_changed', function(data){var_changed(data);});
socket.onAny(function(event_name, data) {console.log({"event": event_name, "class": data.classname, "data": data});});

var backend_vars = {};
var presets = {}
var ai_busy_start = Date.now();
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
	//Add Redo options
	i=0;
	for (item of data.value.options) {
		if ((item['Previous Selection'])) {
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
			icon.setAttribute('data-glyph', "loop-circular");
			iconcell.append(icon);
			textcell.onclick = function () {
									socket.emit("Set Selected Text", {"chunk": this.getAttribute("option_chunk"), "option": this.getAttribute("option_id")});
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
			if (!(item.Pinned)) {
				icon.setAttribute('style', "filter: brightness(50%);");
			}
			iconcell.append(icon);
			iconcell.onclick = function () {
									socket.emit("Pinning", {"chunk": this.getAttribute("option_chunk"), "option": this.getAttribute("option_id")});
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
	console.log(data);
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
	for (const [key, value] of Object.entries(data.value)) {
		presets[key] = value;
		var option = document.createElement("option");
		option.value=key;
		option.text=key;
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
		var elements_to_change = document.getElementsByClassName("var_sync_"+data.classname.replace(" ", "_")+"_"+data.name.replace(" ", "_"));
		for (item of elements_to_change) {
			if ((item.tagName.toLowerCase() === 'input') || (item.tagName.toLowerCase() === 'select')) {
				item.value = fix_text(data.value);
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
}

//--------------------------------------------UI to Server Functions----------------------------------


//--------------------------------------------General UI Functions------------------------------------
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
