import numpy as np
import matplotlib.pyplot as plt
import sys
from pandas import DataFrame
import os
import time
import json
import subprocess
import glob

sys.path.append('../../Config/')
sys.path.append('../../Class/')
from configuration import *
#tshark -r Network_size2_threshold_20.0.pcap -T fields -e frame.len ip.addr==192.168.0.103
filename=sys.argv[1]
filelist=glob.glob(filename+"*.csv")
for i in filelist:
    print(i)
    data=ReadCSV(i)
    for j in data:
        del j[len(j)-1]
        del j[0]
        for k in range(len(j)):
            j[k]=j[k][:len(j[k])-1]
    #print(data)
    file=i.split("/")

    new_file=""
    for j in range(len(file)-2):
        new_file+=file[j]
        new_file+="/"
    new_file+="FilteredCPU_MEM/"
    if(os.path.exists(new_file)==False):
        os.mkdir(new_file)
    elif(os.path.exists(new_file+file[len(file)-1])==True):
        os.remove(new_file+file[len(file)-1])
        print("Previous files removed!!!")
    WriteCSV(new_file+file[len(file)-1],data)


#
