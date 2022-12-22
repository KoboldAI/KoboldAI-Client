function $el(selector) { return document.querySelector(selector) }

var socket = io.connect(window.location.origin, { transports: ['polling', 'websocket'], closeOnBeforeunload: false });
var question = $el("#question").value;
var model = $el("#model").value;

function send_results() {
    var answer = "";

    for (const answerId of ["A", "B", "C", "D", "E", "F"]) {
        let checkbox = document.querySelector(`[answer="${answerId}"]`).querySelector("input");
        if (checkbox.checked) {
            answer = answerId;
            break;
        }
    }

    console.log(answer);
    socket.emit("answer", { "question": question, "answer": answer, "id": document.getElementById("id").value, "model": model});
}

for (const child of $el("#choices-container").children) {
    let radioButton = child.querySelector("input");
    child.addEventListener("click", function () { radioButton.click() });
    radioButton.addEventListener("click", send_results);
}