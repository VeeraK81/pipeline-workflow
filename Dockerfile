FROM continuumio/miniconda3

WORKDIR /home
COPY . .
ENV PYTHONPATH=/home

ENV KAFKA_BROKER=kafka:9092

RUN apt-get update
RUN apt-get install nano unzip
RUN apt install curl -y

RUN curl -fsSL https://get.deta.dev/cli.sh | sh


RUN pip install -r requirements.txt

RUN pip install git+https://github.com/dpkp/kafka-python.git

CMD ["python", "app/kafka_consume_topics.py"]   