FROM openfaas/classic-watchdog:0.18.1 as watchdog

FROM python:3.8-alpine

COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog
RUN chmod +x /usr/bin/fwatchdog

# Allows you to add additional packages via build-arg
ARG ADDITIONAL_PACKAGE

# Alternatively use ADD https:// (which will not be cached by Docker builder)
RUN apk --no-cache add ${ADDITIONAL_PACKAGE}

# Add non root user
RUN addgroup -S app && adduser app -S -G app
RUN apk update \
    && apk upgrade \
    && apk --no-cache add --update gcc tcl apache2 ca-certificates cmake gfortran git wget curl graphicsmagick \
    apk-tools curl build-base supervisor cups-client dcron bind-tools rsync libxml2-utils libxslt
RUN apk add --no-cache jpeg-dev zlib-dev
RUN pip install Pillow
RUN pip install numpy
RUN pip install dlib
RUN pip install face_recognition
WORKDIR /home/app/

COPY PythonCode/ .
COPY requirements.txt   .
RUN pip install -r requirements.txt

ENV fprocess="python handler.py"

HEALTHCHECK --interval=3s CMD [ -e /tmp/.lock ] || exit 1

CMD ["fwatchdog"]
