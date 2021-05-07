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

// Data records
var actions = [];
var memory = "";
var prompt = "";
var authors_note = "";
var scripts = {
	"shared": "",
	"inputModifier": "",
	"contextModifier": "",
	"outputModifier": "",
};
var script_state;

var current_mode = "play"; // Tracks current input mode based off of what's been enabled / disabled. (Set based on which modes have entered/exited here, though really should probably be a variable received from the server as it changes serverside)

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
	
	current_mode = "edit";
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
	
	current_mode = "play";
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
	
	current_mode = "memory";
}

function exitMemoryMode() {
	hideMessage();
	button_actmem.html("Memory");
	show([button_actback, button_actretry, button_actedit]);
	input_text.val("");
	// Hide Author's Note field
	anote_menu.slideUp("fast");
	
	current_mode = "play";
}

function dosubmit() {
	var txt = input_text.val();
	
	if(current_mode == "play"){
		// Run the input through the input modifier script
		let [new_input, should_stop] = applyScriptModifier(scripts.inputModifier, txt);
		socket.send({'cmd': 'submit', 'data': new_input, 'stop': should_stop});	
	} else {
		socket.send({'cmd': 'submit', 'data': txt});
	}

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
//  STORY SCRIPTS
//=================================================================//

function applyScriptModifier(modifier_script, text="") {
	let modified_actions = actions.slice().splice(0, 0, prompt); // Just to be lazy, add the prompt onto the start of the actions instead of feeding it in as a separate thing
	
	// TODO: Sandbox this elsewhere so scripts can't mess with the client
	let script = new Function("text", "actions", "memory", "state", "authorsNote", `"use strict";\n${scripts.shared}\n${modifier_script}\nreturn modifier(text)`);
	
	let script_return = script(text, modified_actions, memory, script_state, authors_note);
	
	if (!script_return || (!script_return.text && !script_return.text === "")) {
		// TODO: ERRORING
	}
	
	// Should we progress to generating an AI output?
	let should_stop = false;
	if (script_return.stop) {
		should_stop = true;
	}
	
	// Update the server's record of the script state
	// (otherwise changes made to it by this script modifier won't be saved with the story!)
	socket.send({'cmd': 'recordscriptstate', 'data': script_state});
	
	return [script_return.text, should_stop];
}

// Responds to server's request for a context to be sent, kicking off the generation of an AI output (provided the contextModifer script doesn't say to abort)
function answerContextRequest(base_context) {
	// Run a base context through the contextModifer script
	let [context, should_stop]  = applyScriptModifier(scripts.contextModifier, base_context);
	
	// Provide the server with a context and trigger it to generate an output, but only if the context modifier allows it
	if (!should_stop) {
		// (Strict enforcement of tokenization / context length is performed later by the server, so no need to worry about it here)
		socket.send({'cmd': 'generate', 'data': context});
	}
}

// Receives the AI output from the server, runs it through the output modifier script, then sends it off to the server to record as the AI's output
function modifyOutput(output_text) {
	// Run through output modifier
	let [new_output] = applyScriptModifier(scripts.outputModifier, output_text);
	
	// Send the new output to the server to use as a new output
	socket.send({'cmd': 'newoutput', 'data': new_output});
}

//=================================================================//
//  READY/RUNTIME
//=================================================================//

// Replaces returns and newlines with HTML breaks
function formatForHtml(txt) {
	return txt.replace("\\r", "<br/>").replace("\\n", "<br/>").replace('\n', '<br/>').replace('\r', '<br/>');
}

// Takes actions and turns into html code for displaying
function buildGameScreen(action_list) {
	let text = '<chunk n="0" id="n0">'+prompt+'</chunk>';
	
	let position = 1;
	for (let item of actions) {
		text += '<chunk n="'+position+'" id="n'+position+'">'+item+'</chunk>';
		position += 1;
	}
	
	return formatForHtml(text);
}

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
		console.log(`Data recieved: ${JSON.stringify(msg)}`)
		
		if(msg.cmd == "connected") {
			// Connected to Server Actions
			$('#connectstatus').html("<b>Connected to KoboldAI Process!</b>");
			$('#connectstatus').removeClass("color_orange");
			$('#connectstatus').addClass("color_green");
		} else if(msg.cmd == "updatescreen") {
			// Send game content to Game Screen
			if (msg.data) {
				// Server provided its own html
				game_text.html(msg.data);
			} else {
				// Build game screen html based on what we've got
				game_text.html(buildGameScreen(actions))
			}

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
				
				current_mode = "play";
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
		} else if(msg.cmd == "updatedata") {
			// Update local records based on what the server holds
			// (Ideally we'd sync each individual change so this is only necessary on initial loads)
			if('actions' in msg) actions = msg.actions;
			if('memory' in msg) memory = msg.memory;
			if('prompt' in msg) prompt = msg.prompt;
			if('scripts' in msg) scripts = msg.scripts;
			if('authornote' in msg) authors_note = msg.authornote;
    } else if(msg.cmd == "modcontext") {
			answerContextRequest(msg.data)
		} else if(msg.cmd == "modoutput") {
			modifyOutput(msg.data)
		} else if(msg.cmd == "setscriptstate") {
			// New / loaded script state received from the server
			script_state = msg.data
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

