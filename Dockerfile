FROM tensorflow/tensorflow:2.11.0-gpu

RUN set -ex \
    && apt-get update \ 
    && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
    libopenjp2-7 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY requirements-docker.txt /
RUN set -ex \
    && pip3 install --no-cache-dir -r requirements-docker.txt \
    && pip3 install --no-cache-dir zarr \
    && pip3 install --no-cache-dir git+https://github.com/lens-biophotonics/ZetaStitcher.git@devel

WORKDIR /home/

ENV CUPY_ACCELERATORS='cutensor'
ENV CUPY_CUDA_PER_THREAD_DEFAULT_STREAM=1
ENV CUPY_CACHE_DIR='/.cupy/kernel_cache'

COPY dist/*.whl /home/
RUN set -ex \
    && pip3 install *.whl \
    && rm *.whl 
