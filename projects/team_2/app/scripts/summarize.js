var inputText = document.querySelector('#input-text');
var outputText = document.querySelector('#output-text');
var buttonTranslate = document.querySelector('#btn-translate');
var apiUrl = document.URL.toString();

function errorHandle(error) {
	// In case error occurs the errorHandle
	// function will be called
	alert('Error occurred')
	console.log("error occurred", error);
}

function clickHandler() {
	// When someone clicks on translate
	// button clickHandler will be called
	var text = inputText.value;
	var updatedUrl = apiUrl + "api/summarize/?article=" + text;
	fetch(updatedUrl).then(response =>
	response.json()).then(json =>
	outputText.innerText =
		(json.summarization)).catch(errorHandle);
}



buttonTranslate.addEventListener("click", clickHandler);
// Adding the event listener click
