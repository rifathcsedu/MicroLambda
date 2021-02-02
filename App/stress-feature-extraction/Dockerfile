FROM ubuntu:18.04

ENV TZ=America/Denver
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Alternatively use ADD https:// (which will not be cached by Docker builder)
RUN apt-get update && apt-get install -y \
    curl apt-utils \
    python3-pip python3-dev build-essential python3-distutils \
    python3-tk \
    libjpeg8-dev zlib1g-dev libtiff-dev libfreetype6 libfreetype6-dev libwebp-dev libopenjp2-7-dev libopenjp2-7-dev \
    libhdf5-dev python3-h5py
RUN python3 -m pip install --upgrade pip
RUN pip3 install setuptools==44.1.1
RUN pip3 install pandas
RUN pip3 install neurokit2
RUN echo "Pulling watchdog binary from Github." \
    && curl -sSL https://github.com/openfaas/faas/releases/download/0.6.9/fwatchdog > /usr/bin/fwatchdog \
    && chmod +x /usr/bin/fwatchdog

WORKDIR /home/app/

COPY PythonCode/ .
COPY requirements.txt   .
# RUN pip3 install --ignore-installed tensorflow
RUN pip3 install -r requirements.txt

ENV fprocess="python3 handler.py"

HEALTHCHECK --interval=3s CMD [ -e /tmp/.lock ] || exit 1

CMD ["fwatchdog"]
