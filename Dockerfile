FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu

# Build args
ARG MODEL_URL=https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev
ENV MODEL_URL=${MODEL_URL}
ARG MODEL_NAME
ENV MODEL_NAME=${MODEL_NAME}
ARG MODEL_TAG=black-forest-labs/FLUX.1-Fill-dev
ENV MODEL_TAG=${MODEL_TAG}

# Update and upgrade the system packages (Worker Template)
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install --yes --no-install-recommends \
cuda-toolkit-11-8 \
build-essential \
vim \
git \
wget \
software-properties-common \
google-perftools \
curl \
bash 
RUN rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH

RUN apt-get autoremove -y && \
apt-get clean -y && \
rm -rf /var/lib/apt/lists/* && \
add-apt-repository ppa:deadsnakes/ppa -y && \
apt-get install python3.10 -y --no-install-recommends && \
ln -s /usr/bin/python3.10 /usr/bin/python && \
rm /usr/bin/python3 && \
ln -s /usr/bin/python3.10 /usr/bin/python3 && \
which python && \
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
python get-pip.py && \
rm get-pip.py && \
pip install --no-cache-dir -U pip setuptools wheel

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt && \
    rm /requirements.txt
RUN pip install flash-attn==2.7.4.post1

RUN huggingface-cli login --token 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

# Fetch the model
COPY builder/model_fetcher.py /model_fetcher.py
RUN python /model_fetcher.py --model_url=${MODEL_URL}
RUN rm /model_fetcher.py

# Add src files and data files (Worker Template)
ADD data .
ADD src .

ENV RUNPOD_DEBUG_LEVEL=INFO

CMD python -u /rp_handler.py --model_tag="$MODEL_TAG"
