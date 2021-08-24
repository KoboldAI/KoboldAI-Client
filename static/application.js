//=================================================================//
//  VARIABLES
//=================================================================//

// Socket IO Object
var socket;

// UI references for jQuery
var connect_status;
var button_newgame;
var button_rndgame;
var button_save;
var button_saveas;
var button_savetofile;
var button_load;
var button_import;
var button_importwi;
var button_impaidg;
var button_settings;
var button_format;
var button_mode;
var button_mode_label;
var button_send;
var button_actmem;
var button_actback;
var button_actretry;
var button_actwi;
var game_text;
var input_text;
var message_text;
var settings_menu;
var format_menu;
var wi_menu;
var anote_menu;
var anote_input;
var anote_labelcur;
var anote_slider;
var popup;
var popup_title;
var popup_content;
var popup_accept;
var popup_close;
var aidgpopup;
var aidgpromptnum;
var aidg_accept;
var aidg_close;
var saveaspopup;
var saveasinput;
var topic;
var saveas_accept;
var saveas_close;
var saveasoverwrite;
var loadpopup;
var	loadcontent;
var	load_accept;
var	load_close;
var nspopup;
var ns_accept;
var ns_close;
var rspopup;
var rs_accept;
var rs_close;
var seqselmenu;
var seqselcontents;

var memorymode = false;
var gamestarted = false;
var editmode = false;
var connected = false;
var newly_loaded = true;
var current_editing_chunk = null;
var chunk_conflict = false;

// Key states
var shift_down   = false;
var do_clear_ent = false;

// Display vars
var allowtoggle = false;
var formatcount = 0;
var allowedit   = true;  // Whether clicking on chunks will edit them

// Adventure
var action_mode = 0;  // 0: story, 1: action
var adventure = false;

//=================================================================//
//  METHODS
//=================================================================//

function addSetting(ob) {	
	// Add setting block to Settings Menu
	if(ob.uitype == "slider"){
		settings_menu.append("<div class=\"settingitem\">\
		<div class=\"settinglabel\">\
			<div class=\"justifyleft\">\
				"+ob.label+" <span class=\"helpicon\">?<span class=\"helptext\">"+ob.tooltip+"</span></span>\
			</div>\
			<div class=\"justifyright\" id=\""+ob.id+"cur\">\
				"+ob.default+"\
			</div>\
		</div>\
		<div>\
			<input type=\"range\" class=\"form-range airange\" min=\""+ob.min+"\" max=\""+ob.max+"\" step=\""+ob.step+"\" id=\""+ob.id+"\">\
		</div>\
		<div class=\"settingminmax\">\
			<div class=\"justifyleft\">\
				"+ob.min+"\
			</div>\
			<div class=\"justifyright\">\
				"+ob.max+"\
			</div>\
		</div>\
		</div>");
		// Set references to HTML objects
		var refin = $("#"+ob.id);
		var reflb = $("#"+ob.id+"cur");
		window["setting_"+ob.id] = refin;  // Is this still needed?
		window["label_"+ob.id]   = reflb;  // Is this still needed?
		// Add event function to input
		refin.on("input", function () {
			socket.send({'cmd': $(this).attr('id'), 'data': $(this).val()});
		});
	} else if(ob.uitype == "toggle"){
		settings_menu.append("<div class=\"settingitem\">\
			<input type=\"checkbox\" data-toggle=\"toggle\" data-onstyle=\"success\" id=\""+ob.id+"\">\
			<span class=\"formatlabel\">"+ob.label+" </span>\
			<span class=\"helpicon\">?<span class=\"helptext\">"+ob.tooltip+"</span></span>\
		</div>");
		// Tell Bootstrap-Toggle to render the new checkbox
		$("input[type=checkbox]").bootstrapToggle();
		$("#"+ob.id).on("change", function () {
			if(allowtoggle) {
				socket.send({'cmd': $(this).attr('id'), 'data': $(this).prop('checked')});
			}
			if(ob.id == "setadventure"){
				setadventure($(this).prop('checked'));
			}
		});
	}
}

function addFormat(ob) {
	// Check if we need to make a new column for this button
	if(formatcount == 0) {
		format_menu.append("<div class=\"formatcolumn\"></div>");
	}
	// Get reference to the last child column
	var ref = $("#formatmenu > div").last();
	// Add format block to Format Menu
	ref.append("<div class=\"formatrow\">\
		<input type=\"checkbox\" data-toggle=\"toggle\" data-onstyle=\"success\" id=\""+ob.id+"\">\
		<span class=\"formatlabel\">"+ob.label+" </span>\
		<span class=\"helpicon\">?<span class=\"helptext\">"+ob.tooltip+"</span></span>\
	</div>");
	// Tell Bootstrap-Toggle to render the new checkbox
	$("input[type=checkbox]").bootstrapToggle();
	// Add event to input
	$("#"+ob.id).on("change", function () {
		if(allowtoggle) {
			socket.send({'cmd': $(this).attr('id'), 'data': $(this).prop('checked')});
		}
	});
	// Increment display variable
	formatcount++;
	if(formatcount == 2) {
		formatcount = 0;
	}
}

function addImportLine(ob) {
	popup_content.append("<div class=\"popuplistitem\" id=\"import"+ob.num+"\">\
		<div>"+ob.title+"</div>\
		<div>"+ob.acts+"</div>\
		<div>"+ob.descr+"</div>\
	</div>");
	$("#import"+ob.num).on("click", function () {
		socket.send({'cmd': 'importselect', 'data': $(this).attr('id')});
		highlightImportLine($(this));
	});
}

function addWiLine(ob) {
	if(ob.init) {
		if(ob.selective){
			wi_menu.append("<div class=\"wilistitem\">\
				<div class=\"wiremove\">\
					<button type=\"button\" class=\"btn btn-primary heightfull\" id=\"btn_wi"+ob.num+"\">X</button>\
					<button type=\"button\" class=\"btn btn-success heighthalf hidden\" id=\"btn_widel"+ob.num+"\">✓</button>\
					<button type=\"button\" class=\"btn btn-danger heighthalf hidden\" id=\"btn_wican"+ob.num+"\">⮌</button>\
				</div>\
				<div class=\"wikey\">\
					<input class=\"form-control heightfull hidden\" type=\"text\" placeholder=\"Key(s)\" id=\"wikey"+ob.num+"\">\
					<input class=\"form-control heighthalf\" type=\"text\" placeholder=\"Primary Key(s)\" id=\"wikeyprimary"+ob.num+"\">\
					<input class=\"form-control heighthalf\" type=\"text\" placeholder=\"Secondary Key(s)\" id=\"wikeysecondary"+ob.num+"\">\
				</div>\
				<div class=\"wientry\">\
					<textarea class=\"form-control\" id=\"wientry"+ob.num+"\" placeholder=\"What To Remember\">"+ob.content+"</textarea>\
				</div>\
				<div class=\"wiselective\">\
					<button type=\"button\" class=\"btn btn-success heightfull hidden\" id=\"btn_wiselon"+ob.num+"\">Enable Selective Mode</button>\
					<button type=\"button\" class=\"btn btn-danger heightfull\" id=\"btn_wiseloff"+ob.num+"\">Disable Selective Mode</button>\
				</div>\
			</div>");
		} else {
			wi_menu.append("<div class=\"wilistitem\">\
				<div class=\"wiremove\">\
					<button type=\"button\" class=\"btn btn-primary heightfull\" id=\"btn_wi"+ob.num+"\">X</button>\
					<button type=\"button\" class=\"btn btn-success heighthalf hidden\" id=\"btn_widel"+ob.num+"\">✓</button>\
					<button type=\"button\" class=\"btn btn-danger heighthalf hidden\" id=\"btn_wican"+ob.num+"\">⮌</button>\
				</div>\
				<div class=\"wikey\">\
					<input class=\"form-control heightfull\" type=\"text\" placeholder=\"Key(s)\" id=\"wikey"+ob.num+"\">\
					<input class=\"form-control heighthalf hidden\" type=\"text\" placeholder=\"Primary Key(s)\" id=\"wikeyprimary"+ob.num+"\">\
					<input class=\"form-control heighthalf hidden\" type=\"text\" placeholder=\"Secondary Key(s)\" id=\"wikeysecondary"+ob.num+"\">\
				</div>\
				<div class=\"wientry\">\
					<textarea class=\"form-control\" id=\"wientry"+ob.num+"\" placeholder=\"What To Remember\">"+ob.content+"</textarea>\
				</div>\
				<div class=\"wiselective\">\
					<button type=\"button\" class=\"btn btn-success heightfull\" id=\"btn_wiselon"+ob.num+"\">Enable Selective Mode</button>\
					<button type=\"button\" class=\"btn btn-danger heightfull hidden\" id=\"btn_wiseloff"+ob.num+"\">Disable Selective Mode</button>\
				</div>\
			</div>");
		}
		// Send key value to text input
		$("#wikey"+ob.num).val(ob.key);
		$("#wikeyprimary"+ob.num).val(ob.key);
		$("#wikeysecondary"+ob.num).val(ob.keysecondary);
		// Assign delete event to button
		$("#btn_wi"+ob.num).on("click", function () {
			showWiDeleteConfirm(ob.num);
		});
	} else {
		// Show WI line item with form fields hidden (uninitialized)
		wi_menu.append("<div class=\"wilistitem\">\
			<div class=\"wiremove\">\
				<button type=\"button\" class=\"btn btn-primary heightfull\" id=\"btn_wi"+ob.num+"\">+</button>\
				<button type=\"button\" class=\"btn btn-success heighthalf hidden\" id=\"btn_widel"+ob.num+"\">✓</button>\
				<button type=\"button\" class=\"btn btn-danger heighthalf hidden\" id=\"btn_wican"+ob.num+"\">X</button>\
			</div>\
			<div class=\"wikey\">\
				<input class=\"form-control heightfull hidden\" type=\"text\" placeholder=\"Key(s)\" id=\"wikey"+ob.num+"\">\
				<input class=\"form-control heighthalf hidden\" type=\"text\" placeholder=\"Primary Key(s)\" id=\"wikeyprimary"+ob.num+"\">\
				<input class=\"form-control heighthalf hidden\" type=\"text\" placeholder=\"Secondary Key(s)\" id=\"wikeysecondary"+ob.num+"\">\
			</div>\
			<div class=\"wientry\">\
				<textarea class=\"form-control hidden\" id=\"wientry"+ob.num+"\" placeholder=\"What To Remember\"></textarea>\
			</div>\
			<div class=\"wiselective\">\
				<button type=\"button\" class=\"btn btn-success heightfull hidden\" id=\"btn_wiselon"+ob.num+"\">Enable Selective Mode</button>\
				<button type=\"button\" class=\"btn btn-danger heightfull hidden\" id=\"btn_wiseloff"+ob.num+"\">Disable Selective Mode</button>\
			</div>\
		</div>");
		// Assign function to expand WI item to button
		$("#btn_wi"+ob.num).on("click", function () {
			expandWiLine(ob.num);
		});
	}
	// Assign actions to other elements
	$("#btn_wican"+ob.num).on("click", function () {
		hideWiDeleteConfirm(ob.num);
	});
	$("#btn_widel"+ob.num).on("click", function () {
		socket.send({'cmd': 'widelete', 'data': ob.num});
	});
	$("#btn_wiselon"+ob.num).on("click", function () {
		enableWiSelective(ob.num);
	});
	$("#btn_wiseloff"+ob.num).on("click", function () {
		disableWiSelective(ob.num);
	});
}

function expandWiLine(num) {
	show([$("#wikey"+num), $("#wientry"+num), $("#btn_wiselon"+num)]);
	$("#btn_wi"+num).html("X");
	$("#btn_wi"+num).off();
	// Tell server the WI entry was initialized
	socket.send({'cmd': 'wiinit', 'data': num});
	$("#btn_wi"+num).on("click", function () {
		showWiDeleteConfirm(num);
	});
}

function showWiDeleteConfirm(num) {
	hide([$("#btn_wi"+num)]);
	show([$("#btn_widel"+num), $("#btn_wican"+num)]);
}

function hideWiDeleteConfirm(num) {
	show([$("#btn_wi"+num)]);
	hide([$("#btn_widel"+num), $("#btn_wican"+num)]);
}

function enableWiSelective(num) {
	hide([$("#btn_wiselon"+num), $("#wikey"+num)]);
	// Tell server the WI entry is now selective
	socket.send({'cmd': 'wiselon', 'data': num});
	show([$("#wikeyprimary"+num), $("#wikeysecondary"+num), $("#btn_wiseloff"+num)]);
}

function disableWiSelective(num) {
	hide([$("#btn_wiseloff"+num), $("#wikeyprimary"+num), $("#wikeysecondary"+num)]);
	// Tell server the WI entry is now non-selective
	socket.send({'cmd': 'wiseloff', 'data': num});
	show([$("#btn_wiselon"+num), $("#wikey"+num)]);
}

function highlightImportLine(ref) {
	$("#popupcontent > div").removeClass("popuplistselected");
	ref.addClass("popuplistselected");
	enableButtons([popup_accept]);
}

function enableButtons(refs) {
	for(i=0; i<refs.length; i++) {
		refs[i].prop("disabled",false);
		refs[i].removeClass("btn-secondary");
		refs[i].addClass("btn-primary");
	}
}

function disableButtons(refs) {
	for(i=0; i<refs.length; i++) {
		refs[i].prop("disabled",true);
		refs[i].removeClass("btn-primary");
		refs[i].addClass("btn-secondary");
	}
}

function enableSendBtn() {
	enableButtons([button_send])
	button_send.html("Submit");
}

function disableSendBtn() {
	disableButtons([button_send])
	button_send.html("");
}

function showMessage(msg) {
	message_text.removeClass();
	message_text.addClass("color_green");
	message_text.html(msg);
}

function errMessage(msg) {
	message_text.removeClass();
	message_text.addClass("color_red");
	message_text.html(msg);
}

function hideMessage() {
	message_text.html("");
	message_text.removeClass();
}

function showWaitAnimation() {
	$("#inputrowright").append("<img id=\"waitanim\" src=\"static/thinking.gif\"/>");
}

function hideWaitAnimation() {
	$('#waitanim').remove();
}

function hide(refs) {
	for(i=0; i<refs.length; i++) {
		refs[i].addClass("hidden");
	}
}

function show(refs) {
	for(i=0; i<refs.length; i++) {
		refs[i].removeClass("hidden");
	}
}

function popupShow(state) {
	if(state) {
		popup.removeClass("hidden");
		popup.addClass("flex");
		disableButtons([popup_accept]);
	} else {
		popup.removeClass("flex");
		popup.addClass("hidden");
	}
}

function enterEditMode() {
	editmode = true;
}

function exitEditMode() {
	editmode = false;
}

function enterMemoryMode() {
	memorymode = true;
	setmodevisibility(false);
	showMessage("Edit the memory to be sent with each request to the AI.");
	button_actmem.html("Cancel");
	hide([button_actback, button_actretry, button_actwi]);
	// Display Author's Note field
	anote_menu.slideDown("fast");
}

function exitMemoryMode() {
	memorymode = false;
	setmodevisibility(adventure);
	hideMessage();
	button_actmem.html("Memory");
	show([button_actback, button_actretry, button_actwi]);
	input_text.val("");
	// Hide Author's Note field
	anote_menu.slideUp("fast");
}

function enterWiMode() {
	showMessage("World Info will be added to memory only when the key appears in submitted text or the last action.");
	button_actwi.html("Accept");
	hide([button_actback, button_actmem, button_actretry, game_text]);
	show([wi_menu]);
	disableSendBtn();
}

function exitWiMode() {
	hideMessage();
	button_actwi.html("W Info");
	hide([wi_menu]);
	show([button_actback, button_actmem, button_actretry, game_text]);
	enableSendBtn();
}

function returnWiList(ar) {
	var list = [];
	var i;
	for(i=0; i<ar.length; i++) {
		var ob          = {"key": "", "keysecondary": "", "content": "", "num": ar[i], "selective": false};
		ob.selective    = $("#wikeyprimary"+ar[i]).css("display") != "none"
		ob.key          = ob.selective ? $("#wikeyprimary"+ar[i]).val() : $("#wikey"+ar[i]).val();
		ob.keysecondary = $("#wikeysecondary"+ar[i]).val()
		ob.content      = $("#wientry"+ar[i]).val();
		list.push(ob);
	}
	socket.send({'cmd': 'sendwilist', 'data': list});
}

function dosubmit() {
	var txt = input_text.val();
	socket.send({'cmd': 'submit', 'actionmode': adventure ? action_mode : 0, 'data': txt});
	input_text.val("");
	hideMessage();
	hidegenseqs();
}

function changemode() {
	if(gamestarted) {
		action_mode += 1;
		action_mode %= 2;  // Total number of action modes (Story and Action)
	} else {
		action_mode = 0;  // Force "Story" mode if game is not started
	}

	switch (action_mode) {
		case 0: button_mode_label.html("Story"); break;
		case 1: button_mode_label.html("Action"); break;
	}
}

function newTextHighlight(ref) {
	ref.addClass("edit-flash");
	setTimeout(function () {
		ref.addClass("colorfade");
		ref.removeClass("edit-flash");
		setTimeout(function () {
			ref.removeClass("colorfade");
		}, 1000);
	}, 50);
}

function showAidgPopup() {
	aidgpopup.removeClass("hidden");
	aidgpopup.addClass("flex");
	aidgpromptnum.focus();
}

function hideAidgPopup() {
	aidgpopup.removeClass("flex");
	aidgpopup.addClass("hidden");
}

function sendAidgImportRequest() {
	socket.send({'cmd': 'aidgimport', 'data': aidgpromptnum.val()});
	hideAidgPopup();
	aidgpromptnum.val("");
}

function showSaveAsPopup() {
	disableButtons([saveas_accept]);
	saveaspopup.removeClass("hidden");
	saveaspopup.addClass("flex");
	saveasinput.focus();
}

function hideSaveAsPopup() {
	saveaspopup.removeClass("flex");
	saveaspopup.addClass("hidden");
	saveasinput.val("");
	hide([saveasoverwrite]);
}

function sendSaveAsRequest() {
	socket.send({'cmd': 'saveasrequest', 'data': saveasinput.val()});
}

function showLoadPopup() {
	loadpopup.removeClass("hidden");
	loadpopup.addClass("flex");
}

function hideLoadPopup() {
	loadpopup.removeClass("flex");
	loadpopup.addClass("hidden");
	loadcontent.html("");
}

function buildLoadList(ar) {
	disableButtons([load_accept]);
	loadcontent.html("");
	showLoadPopup();
	var i;
	for(i=0; i<ar.length; i++) {
		loadcontent.append("<div class=\"loadlistitem\" id=\"load"+i+"\" name=\""+ar[i].name+"\">\
			<div>"+ar[i].name+"</div>\
			<div>"+ar[i].actions+"</div>\
		</div>");
		$("#load"+i).on("click", function () {
			enableButtons([load_accept]);
			socket.send({'cmd': 'loadselect', 'data': $(this).attr("name")});
			highlightLoadLine($(this));
		});
	}
}

function highlightLoadLine(ref) {
	$("#loadlistcontent > div").removeClass("popuplistselected");
	ref.addClass("popuplistselected");
}

function showNewStoryPopup() {
	nspopup.removeClass("hidden");
	nspopup.addClass("flex");
}

function hideNewStoryPopup() {
	nspopup.removeClass("flex");
	nspopup.addClass("hidden");
}

function showRandomStoryPopup() {
	rspopup.removeClass("hidden");
	rspopup.addClass("flex");
}

function hideRandomStoryPopup() {
	rspopup.removeClass("flex");
	rspopup.addClass("hidden");
}

function setStartState() {
	enableSendBtn();
	enableButtons([button_actmem, button_actwi]);
	disableButtons([button_actback, button_actretry]);
	hide([wi_menu]);
	show([game_text, button_actmem, button_actwi, button_actback, button_actretry]);
	hideMessage();
	hideWaitAnimation();
	button_actmem.html("Memory");
	button_actwi.html("W Info");
	hideAidgPopup();
	hideSaveAsPopup();
	hideLoadPopup();
	hideNewStoryPopup();
	hidegenseqs();
}

function parsegenseqs(seqs) {
	seqselcontents.html("");
	var i;
	for(i=0; i<seqs.length; i++) {
		seqselcontents.append("<div class=\"seqselitem\" id=\"seqsel"+i+"\" n=\""+i+"\">"+seqs[i].generated_text+"</div>");
		$("#seqsel"+i).on("click", function () {
			socket.send({'cmd': 'seqsel', 'data': $(this).attr("n")});
		});
	}
	$('#seqselmenu').slideDown("slow");
}

function hidegenseqs() {
	$('#seqselmenu').slideUp("slow", function() {
		seqselcontents.html("");
	});
}

function setmodevisibility(state) {
	if(state){  // Enabling
		show([button_mode]);
		$("#inputrow").addClass("show_mode");
	} else{  // Disabling
		hide([button_mode]);
		$("#inputrow").removeClass("show_mode");
	}
}

function setadventure(state) {
	adventure = state;
	if(!memorymode){
		setmodevisibility(state);
	}
}

function autofocus(event) {
	if(connected) {
		if(event.target.tagName == "CHUNK") {
			current_editing_chunk = event.target;
		}
		event.target.focus();
	} else {
		event.preventDefault();
	}
}

function chunkOnKeyDown(event) {
	// Make escape commit the changes (Originally we had Enter here to but its not required and nicer for users if we let them type freely
	// You can add the following after 27 if you want it back to committing on enter : || (!event.shiftKey && event.keyCode == 13)
	if(event.keyCode == 27) {
		setTimeout(function () {
			event.target.blur();
		}, 5);
		event.preventDefault();
		return;
	}

	// Allow left and right arrow keys (and backspace) to move between chunks
	switch(event.keyCode) {
		case 37:  // left
		case 39:  // right
			old_selection_offset = getSelection().focusOffset;
			setTimeout(function () {
				// Wait a few milliseconds and check if the caret has moved
				new_selection = getSelection();
				if(old_selection_offset != new_selection.focusOffset) {
					return;
				}
				// If it hasn't moved, we're at the beginning or end of a chunk
				// and the caret must be moved to a different chunk
				chunk = document.activeElement;
				switch(event.keyCode) {
					case 37:  // left
						if((chunk = chunk.previousSibling) && chunk.tagName == "CHUNK") {
							range = document.createRange();
							range.selectNodeContents(chunk);
							range.collapse(false);
							new_selection.removeAllRanges();
							new_selection.addRange(range);
						}
						break;

					case 39:  // right
						if((chunk = chunk.nextSibling) && chunk.tagName == "CHUNK") {
							chunk.focus();
						}
				}
			}, 2);
			return;
		
		case 8:  // backspace
			old_length = document.activeElement.innerText.length;
			setTimeout(function () {
				// Wait a few milliseconds and compare the chunk's length
				if(old_length != document.activeElement.innerText.length) {
					return;
				}
				// If it's the same, we're at the beginning of a chunk
				if((chunk = document.activeElement.previousSibling) && chunk.tagName == "CHUNK") {
					range = document.createRange();
					selection = getSelection();
					range.selectNodeContents(chunk);
					range.collapse(false);
					selection.removeAllRanges();
					selection.addRange(range);
				}
			}, 2);
			return
	}

	// Don't allow any edits if not connected to server
	if(!connected) {
		event.preventDefault();
		return;
	}

	// Prevent CTRL+B, CTRL+I and CTRL+U when editing chunks
	if(event.ctrlKey || event.metaKey) {  // metaKey is macOS's command key
		switch(event.keyCode) {
			case 66:
			case 98:
			case 73:
			case 105:
			case 85:
			case 117:
				event.preventDefault();
				return;
		}
	}
}

function submitEditedChunk(event) {
	// Don't do anything if the current chunk hasn't been edited or if someone
	// else overwrote it while you were busy lollygagging
	if(current_editing_chunk === null || chunk_conflict) {
		chunk_conflict = false;
		return;
	}

	show([$('#curtain')]);
	setTimeout(function () {
		if(document.activeElement.tagName == "CHUNK") {
			document.activeElement.blur();
		}
	}, 5);

	chunk = current_editing_chunk;
	current_editing_chunk = null;

	// Submit the edited chunk if it's not empty, otherwise delete it
	if(chunk.innerText.length) {
		socket.send({'cmd': 'inlineedit', 'chunk': chunk.getAttribute("n"), 'data': chunk.innerText});
	} else {
		socket.send({'cmd': 'inlinedelete', 'data': chunk.getAttribute("n")});
	}
}

//=================================================================//
//  READY/RUNTIME
//=================================================================//

$(document).ready(function(){
	
	// Bind UI references
	connect_status    = $('#connectstatus');
	button_newgame    = $('#btn_newgame');
	button_rndgame    = $('#btn_rndgame');
	button_save       = $('#btn_save');
	button_saveas     = $('#btn_saveas');
	button_savetofile = $('#btn_savetofile');
	button_load       = $('#btn_load');
	button_loadfrfile = $('#btn_loadfromfile');
	button_import     = $("#btn_import");
	button_importwi   = $("#btn_importwi");
	button_impaidg    = $("#btn_impaidg");
	button_settings   = $('#btn_settings');
	button_format     = $('#btn_format');
	button_mode       = $('#btnmode')
	button_mode_label = $('#btnmode_label')
	button_send       = $('#btnsend');
	button_actmem     = $('#btn_actmem');
	button_actback    = $('#btn_actundo');
	button_actretry   = $('#btn_actretry');
	button_actwi      = $('#btn_actwi');
	game_text         = $('#gametext');
	input_text        = $('#input_text');
	message_text      = $('#messagefield');
	settings_menu     = $("#settingsmenu");
	format_menu       = $('#formatmenu');
	anote_menu        = $('#anoterowcontainer');
	wi_menu           = $('#wimenu');
	anote_input       = $('#anoteinput');
	anote_labelcur    = $('#anotecur');
	anote_slider      = $('#anotedepth');
	popup             = $("#popupcontainer");
	popup_title       = $("#popuptitletext");
	popup_content     = $("#popupcontent");
	popup_accept      = $("#btn_popupaccept");
	popup_close       = $("#btn_popupclose");
	aidgpopup         = $("#aidgpopupcontainer");
	aidgpromptnum     = $("#aidgpromptnum");
	aidg_accept       = $("#btn_aidgpopupaccept");
	aidg_close        = $("#btn_aidgpopupclose");
	saveaspopup       = $("#saveascontainer");
	saveasinput       = $("#savename");
	topic             = $("#topic");
	saveas_accept     = $("#btn_saveasaccept");
	saveas_close      = $("#btn_saveasclose");
	saveasoverwrite   = $("#saveasoverwrite");
	loadpopup         = $("#loadcontainer");
	loadcontent       = $("#loadlistcontent");
	load_accept       = $("#btn_loadaccept");
	load_close        = $("#btn_loadclose");
	nspopup           = $("#newgamecontainer");
	ns_accept         = $("#btn_nsaccept");
	ns_close          = $("#btn_nsclose");
	rspopup           = $("#rndgamecontainer");
	rs_accept         = $("#btn_rsaccept");
	rs_close          = $("#btn_rsclose");
	seqselmenu        = $("#seqselmenu");
	seqselcontents    = $("#seqselcontents");
	
    // Connect to SocketIO server
    socket = io.connect(window.document.origin);
	
	socket.on('from_server', function(msg) {
        if(msg.cmd == "connected") {
			// Connected to Server Actions
			connected = true;
			connect_status.html("<b>Connected to KoboldAI Process!</b>");
			connect_status.removeClass("color_orange");
			connect_status.addClass("color_green");
			// Reset Menus
			settings_menu.html("");
			format_menu.html("");
			wi_menu.html("");
			// Set up "Allow Editing"
			$('body').on('input', autofocus).on('keydown', 'chunk', chunkOnKeyDown).on('focusout', 'chunk', submitEditedChunk);
			$('#allowediting').prop('checked', allowedit).prop('disabled', false).change().on('change', function () {
				if(allowtoggle) {
					allowedit = $(this).prop('checked')
					$("chunk").attr('contenteditable', allowedit)
				}
			});
		} else if(msg.cmd == "updatescreen") {
			_gamestarted = gamestarted;
			gamestarted = msg.gamestarted;
			if(_gamestarted != gamestarted) {
				action_mode = 0;
				changemode();
			}
			// Send game content to Game Screen
			if(allowedit && document.activeElement.tagName == "CHUNK") {
				chunk_conflict = true;
			}
			game_text.html(msg.data);
			// Make content editable if need be
			$("chunk").attr('tabindex', -1)
			$('chunk').attr('contenteditable', allowedit);
			hide([$('#curtain')]);
			// Scroll to bottom of text
			if(newly_loaded) {
				setTimeout(function () {
					$('#gamescreen').animate({scrollTop: $('#gamescreen').prop('scrollHeight')}, 1000);
				}, 5);
			}
			newly_loaded = false;
			hideMessage();
		} else if(msg.cmd == "scrolldown") {
			setTimeout(function () {
				$('#gamescreen').animate({scrollTop: $('#gamescreen').prop('scrollHeight')}, 1000);
			}, 5);
		} else if(msg.cmd == "setgamestate") {
			// Enable or Disable buttons
			if(msg.data == "ready") {
				enableSendBtn();
				enableButtons([button_actmem, button_actwi, button_actback, button_actretry]);
				hideWaitAnimation();
			} else if(msg.data == "wait") {
				disableSendBtn();
				disableButtons([button_actmem, button_actwi, button_actback, button_actretry]);
				showWaitAnimation();
			} else if(msg.data == "start") {
				setStartState();
			}
		} else if(msg.cmd == "editmode") {
			// Enable or Disable edit mode
			if(msg.data == "true") {
				enterEditMode();
			} else {
				exitEditMode();
			}
		} else if(msg.cmd == "setinputtext") {
			// Set input box text for memory mode
			if(memorymode) {
				input_text.val(msg.data);
			}
		} else if(msg.cmd == "memmode") {
			// Enable or Disable memory edit mode
			if(msg.data == "true") {
				enterMemoryMode();
			} else {
				exitMemoryMode();
			}
		} else if(msg.cmd == "errmsg") {
			// Send error message
			errMessage(msg.data);
		} else if(msg.cmd == "texteffect") {
			// Apply color highlight to line of text
			newTextHighlight($("#n"+msg.data))
		} else if(msg.cmd == "updatetemp") {
			// Send current temp value to input
			$("#settemp").val(parseFloat(msg.data));
			$("#settempcur").html(msg.data);
		} else if(msg.cmd == "updatetopp") {
			// Send current top p value to input
			$("#settopp").val(parseFloat(msg.data));
			$("#settoppcur").html(msg.data);
		} else if(msg.cmd == "updatetopk") {
			// Send current top k value to input
			$("#settopk").val(parseFloat(msg.data));
			$("#settopkcur").html(msg.data);
		} else if(msg.cmd == "updatetfs") {
			// Send current tfs value to input
			$("#settfs").val(parseFloat(msg.data));
			$("#settfscur").html(msg.data);
		} else if(msg.cmd == "updatereppen") {
			// Send current rep pen value to input
			$("#setreppen").val(parseFloat(msg.data));
			$("#setreppencur").html(msg.data);
		} else if(msg.cmd == "updateoutlen") {
			// Send current output amt value to input
			$("#setoutput").val(parseInt(msg.data));
			$("#setoutputcur").html(msg.data);
		} else if(msg.cmd == "updatetknmax") {
			// Send current max tokens value to input
			$("#settknmax").val(parseInt(msg.data));
			$("#settknmaxcur").html(msg.data);
		} else if(msg.cmd == "updateikgen") {
			// Send current max tokens value to input
			$("#setikgen").val(parseInt(msg.data));
			$("#setikgencur").html(msg.data);
		} else if(msg.cmd == "setlabeltemp") {
			// Update setting label with value from server
			$("#settempcur").html(msg.data);
		} else if(msg.cmd == "setlabeltopp") {
			// Update setting label with value from server
			$("#settoppcur").html(msg.data);
		} else if(msg.cmd == "setlabeltopk") {
			// Update setting label with value from server
			$("#settopkcur").html(msg.data);
		} else if(msg.cmd == "setlabeltfs") {
			// Update setting label with value from server
			$("#settfscur").html(msg.data);
		} else if(msg.cmd == "setlabelreppen") {
			// Update setting label with value from server
			$("#setreppencur").html(msg.data);
		} else if(msg.cmd == "setlabeloutput") {
			// Update setting label with value from server
			$("#setoutputcur").html(msg.data);
		} else if(msg.cmd == "setlabeltknmax") {
			// Update setting label with value from server
			$("#settknmaxcur").html(msg.data);
		} else if(msg.cmd == "setlabelikgen") {
			// Update setting label with value from server
			$("#setikgencur").html(msg.data);
		} else if(msg.cmd == "updateanotedepth") {
			// Send current Author's Note depth value to input
			anote_slider.val(parseInt(msg.data));
			anote_labelcur.html(msg.data);
		} else if(msg.cmd == "setlabelanotedepth") {
			// Update setting label with value from server
			anote_labelcur.html(msg.data);
		} else if(msg.cmd == "getanote") {
			// Request contents of Author's Note field
			var txt = anote_input.val();
			socket.send({'cmd': 'anote', 'data': txt});
		} else if(msg.cmd == "setanote") {
			// Set contents of Author's Note field
			anote_input.val(msg.data);
		} else if(msg.cmd == "addsetting") {
			// Add setting controls
			addSetting(msg.data);
		} else if(msg.cmd == "addformat") {
			// Add setting controls
			addFormat(msg.data);
		} else if(msg.cmd == "updatefrmttriminc") {
			// Update toggle state
			$("#frmttriminc").prop('checked', msg.data).change();
		} else if(msg.cmd == "updatefrmtrmblln") {
			// Update toggle state
			$("#frmtrmblln").prop('checked', msg.data).change();
		} else if(msg.cmd == "updatefrmtrmspch") {
			// Update toggle state
			$("#frmtrmspch").prop('checked', msg.data).change();
		} else if(msg.cmd == "updatefrmtadsnsp") {
			// Update toggle state
			$("#frmtadsnsp").prop('checked', msg.data).change();
		} else if(msg.cmd == "allowtoggle") {
			// Allow toggle change states to propagate
			allowtoggle = msg.data;
		} else if(msg.cmd == "popupshow") {
			// Show/Hide Popup
			popupShow(msg.data);
		} else if(msg.cmd == "addimportline") {
			// Add import popup entry
			addImportLine(msg.data);
		} else if(msg.cmd == "clearpopup") {
			// Clear previous contents of popup
			popup_content.html("");
		} else if(msg.cmd == "wimode") {
			// Enable or Disable WI edit mode
			if(msg.data == "true") {
				enterWiMode();
			} else {
				exitWiMode();
			}
		} else if(msg.cmd == "addwiitem") {
			// Add WI entry to WI Menu
			addWiLine(msg.data);
		} else if(msg.cmd == "clearwi") {
			// Clear previous contents of WI list
			wi_menu.html("");
		} else if(msg.cmd == "requestwiitem") {
			// Package WI contents and send back to server
			returnWiList(msg.data);
		} else if(msg.cmd == "saveas") {
			// Show Save As prompt
			showSaveAsPopup();
		} else if(msg.cmd == "hidesaveas") {
			// Hide Save As prompt
			hideSaveAsPopup();
		} else if(msg.cmd == "buildload") {
			// Send array of save files to load UI
			buildLoadList(msg.data);
		} else if(msg.cmd == "askforoverwrite") {
			// Show overwrite warning
			show([saveasoverwrite]);
		} else if(msg.cmd == "genseqs") {
			// Parse generator sequences to UI
			parsegenseqs(msg.data);
		} else if(msg.cmd == "hidegenseqs") {
			// Collapse genseqs menu
			hidegenseqs();
		} else if(msg.cmd == "setlabelnumseq") {
			// Update setting label with value from server
			$("#setnumseqcur").html(msg.data);
		} else if(msg.cmd == "updatenumseq") {
			// Send current max tokens value to input
			$("#setnumseq").val(parseInt(msg.data));
			$("#setnumseqcur").html(msg.data);
		} else if(msg.cmd == "setlabelwidepth") {
			// Update setting label with value from server
			$("#setwidepthcur").html(msg.data);
		} else if(msg.cmd == "updatewidepth") {
			// Send current max tokens value to input
			$("#setwidepth").val(parseInt(msg.data));
			$("#setwidepthcur").html(msg.data);
		} else if(msg.cmd == "updateuseprompt") {
			// Update toggle state
			$("#setuseprompt").prop('checked', msg.data).change();
		} else if(msg.cmd == "updateadventure") {
			// Update toggle state
			$("#setadventure").prop('checked', msg.data).change();
			// Update adventure state
			setadventure(msg.data);
		} else if(msg.cmd == "runs_remotely") {
			hide([button_loadfrfile, button_savetofile, button_import, button_importwi]);
		}
    });
	
	socket.on('disconnect', function() {
		connected = false;
		connect_status.html("<b>Lost connection...</b>");
		connect_status.removeClass("color_green");
		connect_status.addClass("color_orange");
	});
	
	// Bind actions to UI buttons
	button_send.on("click", function(ev) {
		dosubmit();
	});

	button_mode.on("click", function(ev) {
		changemode();
	});
	
	button_actretry.on("click", function(ev) {
		socket.send({'cmd': 'retry', 'data': ''});
		hidegenseqs();
	});
	
	button_actback.on("click", function(ev) {
		socket.send({'cmd': 'back', 'data': ''});
		hidegenseqs();
	});
	
	button_actmem.on("click", function(ev) {
		socket.send({'cmd': 'memory', 'data': ''});
	});
	
	button_savetofile.on("click", function(ev) {
		socket.send({'cmd': 'savetofile', 'data': ''});
	});
	
	button_loadfrfile.on("click", function(ev) {
		socket.send({'cmd': 'loadfromfile', 'data': ''});
	});
	
	button_import.on("click", function(ev) {
		socket.send({'cmd': 'import', 'data': ''});
	});
	
	button_importwi.on("click", function(ev) {
		socket.send({'cmd': 'importwi', 'data': ''});
	});
	
	button_settings.on("click", function(ev) {
		$('#settingsmenu').slideToggle("slow");
	});
	
	button_format.on("click", function(ev) {
		$('#formatmenu').slideToggle("slow");
	});
	
	popup_close.on("click", function(ev) {
		socket.send({'cmd': 'importcancel', 'data': ''});
	});
	
	popup_accept.on("click", function(ev) {
		socket.send({'cmd': 'importaccept', 'data': ''});
	});
	
	button_actwi.on("click", function(ev) {
		socket.send({'cmd': 'wi', 'data': ''});
	});
	
	button_impaidg.on("click", function(ev) {
		showAidgPopup();
	});
	
	aidg_close.on("click", function(ev) {
		hideAidgPopup();
	});
	
	aidg_accept.on("click", function(ev) {
		sendAidgImportRequest();
	});
	
	button_save.on("click", function(ev) {
		socket.send({'cmd': 'saverequest', 'data': ''});
	});
	
	button_saveas.on("click", function(ev) {
		showSaveAsPopup();
	});
	
	saveas_close.on("click", function(ev) {
		hideSaveAsPopup();
		socket.send({'cmd': 'clearoverwrite', 'data': ''});
	});
	
	saveas_accept.on("click", function(ev) {
		sendSaveAsRequest();
	});
	
	button_load.on("click", function(ev) {
		socket.send({'cmd': 'loadlistrequest', 'data': ''});
	});
	
	load_close.on("click", function(ev) {
		hideLoadPopup();
	});
	
	load_accept.on("click", function(ev) {
		newly_loaded = true;
		socket.send({'cmd': 'loadrequest', 'data': ''});
		hideLoadPopup();
	});
	
	button_newgame.on("click", function(ev) {
		showNewStoryPopup();
	});
	
	ns_accept.on("click", function(ev) {
		socket.send({'cmd': 'newgame', 'data': ''});
		hideNewStoryPopup();
	});
	
	ns_close.on("click", function(ev) {
		hideNewStoryPopup();
	});
	
	button_rndgame.on("click", function(ev) {
		showRandomStoryPopup();
	});
	
	rs_accept.on("click", function(ev) {
		socket.send({'cmd': 'rndgame', 'data': topic.val()});
		hideRandomStoryPopup();
	});
	
	rs_close.on("click", function(ev) {
		hideRandomStoryPopup();
	});
	
	anote_slider.on("input", function () {
		socket.send({'cmd': 'anotedepth', 'data': $(this).val()});
	});
	
	saveasinput.on("input", function () {
		if(saveasinput.val() == "") {
			disableButtons([saveas_accept]);
		} else {
			enableButtons([saveas_accept]);
		}
		hide([saveasoverwrite]);
	});
	
	// Bind Enter button to submit
	input_text.keydown(function (ev) {
		if (ev.which == 13 && !shift_down) {
			do_clear_ent = true;
			dosubmit();
		} else if(ev.which == 16) {
			shift_down = true;
		}
	});
	
	// Enter to submit, but not if holding shift
	input_text.keyup(function (ev) {
		if (ev.which == 13 && do_clear_ent) {
			input_text.val("");
			do_clear_ent = false;
		} else if(ev.which == 16) {
			shift_down = false;
		}
	});
	
	aidgpromptnum.keydown(function (ev) {
		if (ev.which == 13) {
			sendAidgImportRequest();
		}
	});
	
	saveasinput.keydown(function (ev) {
		if (ev.which == 13 && saveasinput.val() != "") {
			sendSaveAsRequest();
		}
	});
});

