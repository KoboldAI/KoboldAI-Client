//=================================================================//
//  VARIABLES
//=================================================================//

// Socket IO Object
var socket;

// UI references for jQuery
var button_newgame;
var button_save;
var button_load;
var button_settings;
var button_send;
var button_actedit;
var button_actmem;
var button_actback;
var button_actretry;
var button_delete;
var game_text;
var input_text;
var message_text;
var setting_temp;
var setting_topp;
var setting_reppen;
var setting_outlen;
var label_temp;
var label_topp;
var label_reppen;
var label_outlen;
var anote_menu;
var anote_input;
var anote_labelcur;
var anote_slider;

// Key states
var shift_down   = false;
var do_clear_ent = false;

//=================================================================//
//  METHODS
//=================================================================//

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

function enterEditMode() {
	// Add class to each story chunk
	showMessage("Please select a story chunk to edit above.");
	button_actedit.html("Cancel");
	game_text.children('chunk').addClass("chunkhov");
	game_text.on('click', '> *', function() {
		editModeSelect($(this).attr("n"));
	});
	disableSendBtn();
	hide([button_actback, button_actmem, button_actretry]);
	show([button_delete]);
}

function exitEditMode() {
	// Remove class to each story chunk
	hideMessage();
	button_actedit.html("Edit");
	game_text.children('chunk').removeClass("chunkhov");
	game_text.off('click', '> *');
	enableSendBtn();
	show([button_actback, button_actmem, button_actretry]);
	hide([button_delete]);
	input_text.val("");
}

function editModeSelect(n) {
	socket.send({'cmd': 'editline', 'data': n});
}

function enterMemoryMode() {
	showMessage("Edit the memory to be sent with each request to the AI.");
	button_actmem.html("Cancel");
	hide([button_actback, button_actretry, button_actedit, button_delete]);
	// Display Author's Note field
	anote_menu.slideDown("fast");
}

function exitMemoryMode() {
	hideMessage();
	button_actmem.html("Memory");
	show([button_actback, button_actretry, button_actedit]);
	input_text.val("");
	// Hide Author's Note field
	anote_menu.slideUp("fast");
}

function dosubmit() {
	var txt = input_text.val();
	socket.send({'cmd': 'submit', 'data': txt});
	input_text.val("");
	hideMessage();
}

function newTextHighlight(ref) {
	ref.addClass("color_green");
	ref.addClass("colorfade");
	setTimeout(function () {
		ref.removeClass("color_green")
		setTimeout(function () {
			ref.removeClass("colorfade")
		}, 1000);
	}, 10);
}

//=================================================================//
//  READY/RUNTIME
//=================================================================//

$(document).ready(function(){
	
	// Bind UI references
	button_newgame  = $('#btn_newgame');
	button_save     = $('#btn_save');
	button_load     = $('#btn_load');
	button_settings = $('#btn_settings');
	button_send     = $('#btnsend');
	button_actedit  = $('#btn_actedit');
	button_actmem   = $('#btn_actmem');
	button_actback  = $('#btn_actundo');
	button_actretry = $('#btn_actretry');
	button_delete   = $('#btn_delete');
	game_text       = $('#gametext');
	input_text      = $('#input_text');
	message_text    = $('#messagefield');
	setting_temp    = $('#settemp');
	setting_topp    = $('#settopp');
	setting_reppen  = $('#setreppen');
	setting_outlen  = $('#setoutput');
	label_temp      = $('#settempcur');
	label_topp      = $('#settoppcur');
	label_reppen    = $('#setreppencur');
	label_outlen    = $('#setoutputcur');
	anote_menu      = $('#anoterowcontainer');
	anote_input     = $('#anoteinput');
	anote_labelcur  = $('#anotecur');
	anote_slider    = $('#anotedepth');
	
    // Connect to SocketIO server
    socket = io.connect('http://127.0.0.1:5000');
	
	socket.on('from_server', function(msg) {
        if(msg.cmd == "connected") {
			// Connected to Server Actions
			$('#connectstatus').html("<b>Connected to KoboldAI Process!</b>");
			$('#connectstatus').removeClass("color_orange");
			$('#connectstatus').addClass("color_green");
		} else if(msg.cmd == "updatescreen") {
			// Send game content to Game Screen
			game_text.html(msg.data);
			// Scroll to bottom of text
			setTimeout(function () {
				$('#gamescreen').animate({scrollTop: $('#gamescreen').prop('scrollHeight')}, 1000);
			}, 5);
		} else if(msg.cmd == "setgamestate") {
			// Enable or Disable buttons
			if(msg.data == "ready") {
				enableSendBtn();
				enableButtons([button_actedit, button_actmem, button_actback, button_actretry]);
				hideWaitAnimation();
			} else if(msg.data == "wait") {
				disableSendBtn();
				disableButtons([button_actedit, button_actmem, button_actback, button_actretry]);
				showWaitAnimation();
			} else if(msg.data == "start") {
				enableSendBtn();
				enableButtons([button_actmem]);
				disableButtons([button_actedit, button_actback, button_actretry]);
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
			setting_temp.val(parseFloat(msg.data));
			label_temp.html(msg.data);
		} else if(msg.cmd == "updatetopp") {
			// Send current temp value to input
			setting_topp.val(parseFloat(msg.data));
			label_topp.html(msg.data);
		} else if(msg.cmd == "updatereppen") {
			// Send current temp value to input
			setting_reppen.val(parseFloat(msg.data));
			label_reppen.html(msg.data);
		} else if(msg.cmd == "updateoutlen") {
			// Send current temp value to input
			setting_outlen.val(parseInt(msg.data));
			label_outlen.html(msg.data);
		} else if(msg.cmd == "setlabeltemp") {
			// Update setting label with value from server
			label_temp.html(msg.data);
		} else if(msg.cmd == "setlabeltopp") {
			// Update setting label with value from server
			label_topp.html(msg.data);
		} else if(msg.cmd == "setlabelreppen") {
			// Update setting label with value from server
			label_reppen.html(msg.data);
		} else if(msg.cmd == "setlabeloutput") {
			// Update setting label with value from server
			label_outlen.html(msg.data);
		} else if(msg.cmd == "updatanotedepth") {
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
		}
    });	
	
	socket.on('disconnect', function() {
		$('#connectstatus').html("<b>Lost connection...</b>");
		$('#connectstatus').removeClass("color_green");
		$('#connectstatus').addClass("color_orange");
	});
	
	// Bind actions to UI buttons
	button_send.on("click", function(ev) {
		dosubmit();
	});
	
	button_actretry.on("click", function(ev) {
		socket.send({'cmd': 'retry', 'data': ''});
	});
	
	button_actback.on("click", function(ev) {
		socket.send({'cmd': 'back', 'data': ''});
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
	
	button_save.on("click", function(ev) {
		socket.send({'cmd': 'save', 'data': ''});
	});
	
	button_load.on("click", function(ev) {
		socket.send({'cmd': 'load', 'data': ''});
	});
	
	button_newgame.on("click", function(ev) {
		socket.send({'cmd': 'newgame', 'data': ''});
	});
	
	button_settings.on("click", function(ev) {
		$('#settingsmenu').slideToggle("slow");
	});
	
	// Bind settings to server calls
	$('input[type=range]').on('input', function () {
		socket.send({'cmd': $(this).attr('id'), 'data': $(this).val()});
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
	
});

