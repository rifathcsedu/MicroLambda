#!/bin/bash
sudo apt-get update -y
sudo apt-get install build-essential cmake -y
sudo apt-get install libopenblas-dev liblapack-dev -y
sudo apt-get install libx11-dev libgtk-3-dev -y
sudo apt-get install python python-dev python-pip -y
sudo apt-get install python3 python3-dev python3-pip -y
which pip

if [ $? -eq 0 ]
then
    pip --version | grep "pip version"
    if [ $? -eq 0 ]
    then
        echo 'pip exists'
    else
        sudo apt install python-pip -y
    fi
else
    sudo apt install python-pip -y
fi


filename='required_software.txt'
n=1
while read line; do
pip install $line
done < $filename
