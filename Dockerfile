FROM python:3.10-slim AS base

WORKDIR /app
COPY ./requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install gcc -y && apt-get clean 
RUN pip install -r /app/requirements.txt && rm -rf /root/.cache/pip

COPY . /app

FROM base as test
RUN pip install pytest
RUN cd app/ml/comparison && python train.py model.joblib
RUN cd /app && python -m pytest app/test_main.py
