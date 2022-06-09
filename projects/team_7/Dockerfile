FROM python:latest

ADD requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /content

ADD tmp.pickle .
ADD tmp2.pickle .
ADD saved_gpt2.zip .
ADD saved_model.zip .

COPY src /content/src

ENV FLASK_APP=src/main_for_flask

CMD flask run --host poem_generation --port 5000