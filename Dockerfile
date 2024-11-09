FROM continuumio/miniconda3

WORKDIR /home
COPY . .
ENV PYTHONPATH=/home

ENV KAFKA_BROKER=kafka:9092

# Install necessary tools
RUN apt-get update && apt-get install -y nano unzip curl

# Install Deta CLI and Python dependencies
RUN curl -fsSL https://get.deta.dev/cli.sh | sh
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/dpkp/kafka-python.git


# Set the default command to run the consumer script
CMD ["python", "app/kafka_consume_topics.py"]