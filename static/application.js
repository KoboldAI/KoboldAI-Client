//=================================================================//
//  VARIABLES
//=================================================================//

// Socket IO Object
var socket;

// UI references for jQuery
var connect_status;
var button_newgame;
var button_save;
var button_saveas;
var button_savetofile;
var button_load;
var button_import;
var button_importwi;
var button_impaidg;
var button_settings;
var button_format;
var button_send;
var button_actedit;
var button_actmem;
var button_actback;
var button_actretry;
var button_delete;
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
var seqselmenu;
var seqselcontents;

// Key states
var shift_down   = false;
var do_clear_ent = false;

// Display vars
var allowtoggle = false;
var formatcount = 0;

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
		wi_menu.append("<div class=\"wilistitem\">\
			<div class=\"wiremove\">\
				<button type=\"button\" class=\"btn btn-primary heightfull\" id=\"btn_wi"+ob.num+"\">X</button>\
				<button type=\"button\" class=\"btn btn-success heighthalf hidden\" id=\"btn_widel"+ob.num+"\">✓</button>\
				<button type=\"button\" class=\"btn btn-danger heighthalf hidden\" id=\"btn_wican"+ob.num+"\">⮌</button>\
			</div>\
			<div class=\"wikey\">\
				<input class=\"form-control\" type=\"text\" placeholder=\"Key(s)\" id=\"wikey"+ob.num+"\">\
			</div>\
			<div class=\"wientry\">\
				<textarea class=\"form-control\" id=\"wientry"+ob.num+"\" placeholder=\"What To Remember\">"+ob.content+"</textarea>\
			</div>\
		</div>");
		// Send key value to text input
		$("#wikey"+ob.num).val(ob.key);
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
				<input class=\"form-control hidden\" type=\"text\" placeholder=\"Key(s)\" id=\"wikey"+ob.num+"\">\
			</div>\
			<div class=\"wientry\">\
				<textarea class=\"form-control hidden\" id=\"wientry"+ob.num+"\" placeholder=\"What To Remember\"></textarea>\
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
}

function expandWiLine(num) {
	show([$("#wikey"+num), $("#wientry"+num)]);
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
	// Add class to each story chunk
	showMessage("Please select a story chunk to edit above.");
	button_actedit.html("Cancel");
	game_text.children('chunk').addClass("chunkhov");
	game_text.on('click', '> *', function() {
		editModeSelect($(this).attr("n"));
	});
	disableSendBtn();
	hide([button_actback, button_actmem, button_actretry, button_actwi]);
	show([button_delete]);
}

function exitEditMode() {
	// Remove class to each story chunk
	hideMessage();
	button_actedit.html("Edit");
	game_text.children('chunk').removeClass("chunkhov");
	game_text.off('click', '> *');
	enableSendBtn();
	show([button_actback, button_actmem, button_actretry, button_actwi]);
	hide([button_delete]);
	input_text.val("");
}

function editModeSelect(n) {
	socket.send({'cmd': 'editline', 'data': n});
}

function enterMemoryMode() {
	showMessage("Edit the memory to be sent with each request to the AI.");
	button_actmem.html("Cancel");
	hide([button_actback, button_actretry, button_actedit, button_delete, button_actwi]);
	// Display Author's Note field
	anote_menu.slideDown("fast");
}

function exitMemoryMode() {
	hideMessage();
	button_actmem.html("Memory");
	show([button_actback, button_actretry, button_actedit, button_actwi]);
	input_text.val("");
	// Hide Author's Note field
	anote_menu.slideUp("fast");
}

function enterWiMode() {
	showMessage("World Info will be added to memory only when the key appears in submitted text or the last action.");
	button_actwi.html("Accept");
	hide([button_actedit, button_actback, button_actmem, button_actretry, game_text]);
	show([wi_menu]);
	disableSendBtn();
}

function exitWiMode() {
	hideMessage();
	button_actwi.html("W Info");
	hide([wi_menu]);
	show([button_actedit, button_actback, button_actmem, button_actretry, game_text]);
	enableSendBtn();
}

function returnWiList(ar) {
	var list = [];
	var i;
	for(i=0; i<ar.length; i++) {
		var ob     = {"key": "", "content": "", "num": ar[i]};
		ob.key     = $("#wikey"+ar[i]).val();
		ob.content = $("#wientry"+ar[i]).val();
		list.push(ob);
	}
	socket.send({'cmd': 'sendwilist', 'data': list});
}

function dosubmit() {
	var txt = input_text.val();
	socket.send({'cmd': 'submit', 'data': txt});
	input_text.val("");
	hideMessage();
	hidegenseqs();
}

function newTextHighlight(ref) {
	ref.addClass("color_green");
	ref.addClass("colorfade");
	setTimeout(function () {
		ref.removeClass("color_green");
		setTimeout(function () {
			ref.removeClass("colorfade");
		}, 1000);
	}, 10);
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

function setStartState() {
	enableSendBtn();
	enableButtons([button_actmem, button_actwi]);
	disableButtons([button_actedit, button_actback, button_actretry]);
	hide([wi_menu, button_delete]);
	show([game_text, button_actedit, button_actmem, button_actwi, button_actback, button_actretry]);
	hideMessage();
	hideWaitAnimation();
	button_actedit.html("Edit");
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

//=================================================================//
//  READY/RUNTIME
//=================================================================//

$(document).ready(function(){
	
	// Bind UI references
	connect_status    = $('#connectstatus');
	button_newgame    = $('#btn_newgame');
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
	button_send       = $('#btnsend');
	button_actedit    = $('#btn_actedit');
	button_actmem     = $('#btn_actmem');
	button_actback    = $('#btn_actundo');
	button_actretry   = $('#btn_actretry');
	button_delete     = $('#btn_delete');
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
	seqselmenu        = $("#seqselmenu");
	seqselcontents    = $("#seqselcontents");
	
    // Connect to SocketIO server
	loc    = window.document.location;
    socket = io.connect(loc.href);
	
	socket.on('from_server', function(msg) {
        if(msg.cmd == "connected") {
			// Connected to Server Actions
			connect_status.html("<b>Connected to KoboldAI Process!</b>");
			connect_status.removeClass("color_orange");
			connect_status.addClass("color_green");
			// Reset Menus
			settings_menu.html("");
			format_menu.html("");
			wi_menu.html("");
		} else if(msg.cmd == "updatescreen") {
			// Send game content to Game Screen
			game_text.html(msg.data);
			// Scroll to bottom of text
			setTimeout(function () {
				$('#gamescreen').animate({scrollTop: $('#gamescreen').prop('scrollHeight')}, 1000);
			}, 5);
		} else if(msg.cmd == "updatechunk") {
			const {index, html, last} = msg.data;
			const existingChunk = game_text.children(`#n${index}`)
			const newChunk = $(html);
			if (existingChunk.length > 0) {
				// Update existing chunk
				existingChunk.before(newChunk);
				existingChunk.remove();
			} else {
				// Append at the end
				game_text.append(newChunk);
			}
			if(last) {
				// Scroll to bottom of text if it's the last element
				setTimeout(function () {
					$('#gamescreen').animate({scrollTop: $('#gamescreen').prop('scrollHeight')}, 1000);
				}, 5);
			}
		} else if(msg.cmd == "removechunk") {
        	let index = msg.data;
        	// Remove the chunk
        	game_text.children(`#n${index}`).remove()
			// Shift all existing chunks by 1
			index++;
        	while (true) {
        		const chunk = game_text.children(`#n${index}`)
				if(chunk.length === 0) {
					break;
				}
        		const newIndex = index - 1;
        		chunk.attr('n', newIndex.toString()).attr('id', `n${newIndex}`);
        		index++;
			}
		} else if(msg.cmd == "setgamestate") {
			// Enable or Disable buttons
			if(msg.data == "ready") {
				enableSendBtn();
				enableButtons([button_actedit, button_actmem, button_actwi, button_actback, button_actretry]);
				hideWaitAnimation();
			} else if(msg.data == "wait") {
				disableSendBtn();
				disableButtons([button_actedit, button_actmem, button_actwi, button_actback, button_actretry]);
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
			// Set input box text for edit mode
			input_text.val(msg.data);
		} else if(msg.cmd == "enablesubmit") {
			// Enables the submit button
			enableSendBtn();
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
		}
    });
	
	socket.on('disconnect', function() {
		connect_status.html("<b>Lost connection...</b>");
		connect_status.removeClass("color_green");
		connect_status.addClass("color_orange");
	});
	
	// Bind actions to UI buttons
	button_send.on("click", function(ev) {
		dosubmit();
	});
	
	button_actretry.on("click", function(ev) {
		socket.send({'cmd': 'retry', 'data': ''});
		hidegenseqs();
	});
	
	button_actback.on("click", function(ev) {
		socket.send({'cmd': 'back', 'data': ''});
		hidegenseqs();
	});
	
	button_actedit.on("click", function(ev) {
		socket.send({'cmd': 'edit', 'data': ''});
	});
	
	button_delete.on("click", function(ev) {
		socket.send({'cmd': 'delete', 'data': ''});
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

