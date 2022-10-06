FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04 AS base

RUN set -ex \
    && apt-get update && apt-get install -y --no-install-recommends libopenjp2-7 libgl1 libglib2.0-0 libgomp1 python3-pip


FROM base AS builder
RUN set -ex \
	\
	&& apt-get install -y --no-install-recommends gcc g++ python3-dev

RUN set -ex \
    && pip install --no-cache-dir --upgrade pip \
    # Newest setuptools 59 is needed to install PIMS from ZetaStitcher
    && pip install --no-cache-dir -U setuptools

COPY requirements.txt /

RUN pip install -r requirements.txt

FROM base
WORKDIR /home/
COPY --from=builder /usr/local/lib/python3.8/ /usr/local/lib/python3.8/
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CUPY_CACHE_IN_MEMORY=1

COPY dist/*.whl /home/
RUN set -ex \
	\
    && pip install *.whl && rm *.whl
