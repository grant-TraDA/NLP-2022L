FROM python:3.8

RUN apt-get update -y && \
	pip3 install --upgrade pip && \
	pip3 install --upgrade setuptools && \
    python3 -m pip install numpy pandas cython spacy sklearn fairseq sentencepiece flask flask-cors
ENV FLASK_APP=prediction/flask_service
COPY prediction/flask_service.py /prediction/
COPY prediction/predict.py /prediction/
COPY prediction/prediction_service.py /prediction/
COPY models/roberta.py /models/
COPY models/tfidf.py /models/
COPY pretrained-models/*.pickle /pretrained-models/
CMD ["flask", "run", "--host=0.0.0.0"]
