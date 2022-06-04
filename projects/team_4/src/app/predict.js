
const QUERY_URL = "http://127.0.0.1:5000/predict?text=";
const QUERY_PARAMS = {
  mode: "cors",
  headers: {
    "Access-Control-Allow-Origin":"*"
  }
}
const HISTORY_SIZE = 12;

const classColorMap = {
  "non-harmful": "MediumSeaGreen",
  "cyberbullying": "SandyBrown",
  "hate-speech": "Tomato"
}

const historyContainerElement = document.getElementById("history-container");
const queryTextElement = document.getElementById("query-text");
const responseTextElement = document.getElementById("response-text");

const submit = async (_) => {
  const result = await fetch(QUERY_URL + queryTextElement.value, QUERY_PARAMS)
  const response = await result.json();
  responseTextElement.innerText = response.message;
  const color = classColorMap[response.message];
  queryTextElement.style.color = color;
  responseTextElement.style.color = color;
  return color;
}

const submitOnEnter = async (event) => {
  if (event.which === 13 && !event.shiftKey) {
    event.preventDefault();
    const color = await submit();
    const historyElement = document.createElement('a');
    historyElement.innerText = queryTextElement.value;
    historyElement.style.color = color;
    historyContainerElement.prepend(historyElement);
    styleHistoryElements();
  }
}

const styleHistoryElements = () => {
  const children = historyContainerElement.children;
  for (let i = 0; i < children.length; i++) {
    const elementMultiplier = (HISTORY_SIZE - i) / HISTORY_SIZE;
    children[i].style.opacity = elementMultiplier;
    children[i].style.fontSize = elementMultiplier * 2 + "em";
  }
  if (children.length >= HISTORY_SIZE) {
    historyContainerElement.removeChild(historyContainerElement.lastChild);
  }
}

queryTextElement.addEventListener("keypress", submitOnEnter);
