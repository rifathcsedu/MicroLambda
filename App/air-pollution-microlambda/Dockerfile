FROM ubuntu:16.04

# Alternatively use ADD https:// (which will not be cached by Docker builder)
RUN apt-get update && apt-get install -y \
    curl apt-utils \
    python-pip python-dev python3-pip python3-dev build-essential \
    python-tk python3-tk \
    libjpeg8-dev zlib1g-dev libtiff-dev libfreetype6 libfreetype6-dev libwebp-dev libopenjp2-7-dev libopenjp2-7-dev \
    libhdf5-dev python-h5py python3-h5py
RUN python3 -m pip install --upgrade pip
RUN python2 -m pip install --upgrade pip

RUN echo "Pulling watchdog binary from Github." \
    && curl -sSL https://github.com/openfaas/faas/releases/download/0.6.9/fwatchdog > /usr/bin/fwatchdog \
    && chmod +x /usr/bin/fwatchdog

WORKDIR /home/app/

COPY PythonCode/ .
COPY requirements.txt   .
RUN pip install -r requirements.txt

ENV fprocess="python handler.py"

HEALTHCHECK --interval=3s CMD [ -e /tmp/.lock ] || exit 1

CMD ["fwatchdog"]