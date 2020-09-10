import os
import sys
import time
import json
import subprocess
import glob
sys.path.append('../../Config/')
from configuration import *
#tshark -r Network_size2_threshold_20.0.pcap -T fields -e frame.len ip.addr==192.168.0.103
filename=sys.argv[1]
filelist=glob.glob(filename+"*.pcap")
data=[]
for i in filelist:
    print(i)
    output=subprocess.check_output("tshark -r "+i +" -T fields -e frame.len ip.addr==192.168.0.103",shell=True)
    len_arr=output.decode("utf-8").split("\n")
    sum=0
    for j in len_arr:
        if(j!='' and j!=None):
            sum+=int(j)
        #print(i)
    temp=i[:]
    arr=[]
    num=0
    for j in range(len(temp)):
        if(temp[j]>='0' and temp[j]<='9'):
            num=num*10+int(temp[j])
            #print(num)
        elif(num!=0 and (temp[j]<'0' or temp[j]>'9')):
            arr.append(num)
            num=0
    arr.append(sum)
    print(sum)
    print(arr)
    data.append(arr)

WriteCSV(filename[:len(filename)-1]+"_network_data.csv",data)
