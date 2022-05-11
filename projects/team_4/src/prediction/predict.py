

from prediction_service import PredictionService

if __name__ == "__main__":
    service = PredictionService()
    while text := input("Specify text for prediction:\n"):
        prediction = service.predict(text)
        print(f"\"{text}\" is an example of {prediction}.")
