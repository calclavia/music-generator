FROM nvidia/cuda:9.1-base-ubuntu16.04

# Install Python & dependencies
RUN \
  apt-get update && \
  apt-get install -y python3 python3-dev python3-pip python3-tk fluidsynth curl lame

COPY requirements.txt /tmp/

# Install app dependencies
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

CMD ["bash"]