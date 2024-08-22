FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update \
    && apt install -y curl

WORKDIR /gen_attack

COPY . ./

RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH /root/.local/bin:$PATH

RUN poetry config virtualenvs.create false \
    && poetry install
