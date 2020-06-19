import os
import threading
import subprocess
import json
import csv
import sys

sys.path.append('../Config/')
sys.path.append('../Class/')
import subprocess
import time

from RedisPubSub import *
from configuration import *
from FaceRecognition import *
terminate_CPU_monitor=True
terminate=True
def CPU_Memory_Monitor(filename):

    cmd = "docker container stats c87b2722c705  --format  \"start,{{ .CPUPerc }},{{.MemPerc}},end\" >> hehe.csv"
    proc = subprocess.Popen([cmd], shell=True)


def subscribe_redis_monitor_app():
    p=PubSubSubscriber("MetricMonitor")
    print("Metric Monitoring PubSub Controller Started...\nWaiting for input...")
    monitor_thread=None
    proc = None
    while terminate:
        message = p.get_message()
        if message and message['data']!=1:
            print(message)
            check = json.loads(message["data"])
            print(message["data"])

            if(check["app"]=='face-app'):

                if(check["type"]=="start"):
                    print("Monitoring Face App starts...")
                    filename="Network_size"+str(check["size"])+"_threshold_"+str(check["threshold"])+".pcap"
                    cmd="sudo tshark -i wlo1 -w ../Results/PCAP/Face-App/"+filename+" &"
                    os.system(cmd)
                    filename_net = "../Results/CSV/Face-App/CPU_MEM/CPU_Memory" + str(check["size"]) + "_threshold_" + str(check["threshold"]) + ".csv"
                    #monitor_thread=threading.Thread(target=CPU_Memory_Monitor,args=(filename_net,))
                    #monitor_thread.start()
                    cmd = "docker container stats c87b2722c705  --format  \"start,{{ .CPUPerc }},{{.MemPerc}},end\" >> "+filename_net
                    proc = subprocess.Popen([cmd], shell=True)
                elif(check["type"]=="end"):
                    print("Monitoring Face App ends...")
                    cmd = "sudo killall tshark"
                    os.system(cmd)
                    print("Wireshark closed!!!")
                    # global terminate_CPU_monitor
                    # terminate_CPU_monitor=False
                    # monitor_thread.join()
                    proc.terminate()
                    print("Threading done. Saving the log data!!!!")

            elif(check["APP"]=='Air'):
                if (check["type"] == "start"):
                    print("Monitoring Face App starts...")
                    # os.system("tshark")
                    # os.system(docker ps)
                elif (check["type"] == "end"):
                    print("Monitoring Face App ends...")
                    # os.system("tshark")
                    # os.system(docker ps)
            elif (check["APP"] == 'Human'):
                if (check["type"] == "start"):
                    print("Monitoring Face App starts...")
                    # os.system("tshark")
                    # os.system(docker ps)
                elif (check["type"] == "end"):
                    print("Monitoring Face App ends...")
                    # os.system("tshark")
                    # os.system(docker ps)
            print("Done!!!")

if __name__ == '__main__':
    subscribe_redis_monitor_app()
