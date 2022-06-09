FROM python:3.9-slim

COPY app /app
COPY requirements.txt /requirements.txt

RUN apt-get update

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

WORKDIR /app
