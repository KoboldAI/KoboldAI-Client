//=================================================================//
//  VARIABLES
//=================================================================//
var socket;

var button_send;
var button_actedit;
var button_actmem;
var button_actback;
var button_actretry;
var button_delete;
var game_text;
var input_text;
var message_text;

var shift_down = false;
var do_clear_ent = false;

//=================================================================//
//  METHODS
//=================================================================//

function enableButton(ref) {
	ref.prop("disabled",false);
	ref.removeClass("btn-secondary");
	ref.addClass("btn-primary");
}

function disableButton(ref) {
	ref.prop("disabled",true);
	ref.removeClass("btn-primary");
	ref.addClass("btn-secondary");
}

function enableSendBtn() {
	enableButton(button_send)
	button_send.html("Submit");
}

function disableSendBtn() {
	disableButton(button_send)
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

function hide(ref) {
	ref.addClass("hidden");
}

function show(ref) {
	ref.removeClass("hidden");
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
	hide(button_actback);
	hide(button_actmem);
	hide(button_actretry);
	show(button_delete);
}

function exitEditMode() {
	// Remove class to each story chunk
	hideMessage();
	button_actedit.html("Edit");
	game_text.children('chunk').removeClass("chunkhov");
	game_text.off('click', '> *');
	enableSendBtn();
	show(button_actback);
	show(button_actmem);
	show(button_actretry);
	hide(button_delete);
	input_text.val("");
}

function editModeSelect(n) {
	socket.send({'cmd': 'editline', 'data': n});
}

function enterMemoryMode() {
	// Add class to each story chunk
	showMessage("Edit the memory to be sent with each request to the AI.");
	button_actmem.html("Cancel");
	hide(button_actback);
	hide(button_actretry);
	hide(button_actedit);
	hide(button_delete);
}

function exitMemoryMode() {
	// Remove class to each story chunk
	hideMessage();
	button_actmem.html("Memory");
	show(button_actback);
	show(button_actretry);
	show(button_actedit);
	input_text.val("");
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
	
	// Bind references
	button_newgame  = $('#btn_newgame');
	button_save     = $('#btn_save');
	button_load     = $('#btn_load');
	button_send     = $('#btnsend');
	button_actedit  = $('#btn_actedit');
	button_actmem   = $('#btn_actmem');
	button_actback  = $('#btn_actundo');
	button_actretry = $('#btn_actretry');
	button_delete   = $('#btn_delete');
	game_text       = $('#gametext');
	input_text      = $('#input_text');
	message_text    = $('#messagefield');
	
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
		} else if(msg.cmd == "setgamestate") {
			// Enable or Disable buttons
			if(msg.data == "ready") {
				enableSendBtn();
				enableButton(button_actedit);
				enableButton(button_actmem);
				enableButton(button_actback);
				enableButton(button_actretry);
				hideWaitAnimation();
			} else if(msg.data == "wait") {
				disableSendBtn();
				disableButton(button_actedit);
				disableButton(button_actmem);
				disableButton(button_actback);
				disableButton(button_actretry);
				showWaitAnimation();
			} else if(msg.data == "start") {
				enableSendBtn();
				enableButton(button_actmem);
				disableButton(button_actedit);
				disableButton(button_actback);
				disableButton(button_actretry);
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
	
	// Bind Enter button to submit
	input_text.keydown(function (ev) {
		if (ev.which == 13 && !shift_down) {
			do_clear_ent = true;
			dosubmit();
		} else if(ev.which == 16) {
			shift_down = true;
		}
	});
	
	input_text.keyup(function (ev) {
		if (ev.which == 13 && do_clear_ent) {
			input_text.val("");
			do_clear_ent = false;
		} else if(ev.which == 16) {
			shift_down = false;
		}
	});
	
});

