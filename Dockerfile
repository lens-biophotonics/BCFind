FROM tensorflow/tensorflow:2.7.0-gpu AS base

RUN set -ex \
	\
    && apt-get update && apt-get install -y --no-install-recommends libopenjp2-7 libgl1 libglib2.0-0

COPY requirements.txt /requirements.txt

RUN set -ex \
    \
    # Newest setuptools 59 is needed to install PIMS from ZetaStitcher
    && pip install --no-cache-dir -U setuptools \
    && pip install --no-cache-dir -r requirements.txt
