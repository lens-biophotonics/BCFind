FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04 AS base

RUN set -ex \
    && apt-get update \ 
    && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
    libopenjp2-7 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg

RUN set -ex \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-wheel \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY requirements.txt /
RUN set -ex \
    # Newest setuptools 59 is needed to install PIMS from ZetaStitcher
    && pip3 install --no-cache-dir -U setuptools pip \
    && pip3 install --no-cache-dir -r requirements.txt \
    && rm requirements.txt

WORKDIR /home/

ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV CUPY_ACCELERATORS='cutensor'
ENV CUPY_CUDA_PER_THREAD_DEFAULT_STREAM=1
ENV CUPY_CACHE_DIR='/home/.cupy/kernel_cache'

COPY dist/*.whl /home/
RUN set -ex \
    && pip3 install *.whl --user\
    && rm *.whl
