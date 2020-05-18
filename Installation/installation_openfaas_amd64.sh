#!/bin/sh
which faas-cli

if [ $? -eq 0 ]
then
    faas-cli --version | grep "faas-cli version"
    if [ $? -eq 0 ]
    then
        echo "faas-cli existing"
    else
        curl -sSL https://cli.openfaas.com | sudo sh
    fi
else
    curl -sSL https://cli.openfaas.com | sudo sh
fi
#etc.
