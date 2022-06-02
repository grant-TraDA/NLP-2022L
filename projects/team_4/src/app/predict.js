
const QUERY_URL = "http://127.0.0.1:5000/predict/";
const QUERY_PARAMS = {
  mode: "cors",
  headers: {
    "Access-Control-Allow-Origin":"*"
  }
}

const queryTextElement = document.getElementById("query-text");
const responseTextElement = document.getElementById("response-text");

queryTextElement.addEventListener("change", (event) => {
  fetch(QUERY_URL + event.target.value, QUERY_PARAMS)
    .then(result => result.json()).then(response => {
      console.log("fetching...")
      responseTextElement.innerText = response.message;
    });
});
