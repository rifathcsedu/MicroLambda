import os
import threading
import subprocess
import json
import csv
import sys
import multiprocessing
sys.path.append('../Config/')
sys.path.append('../Class/')
import subprocess
import time

from RedisPubSub import *
from configuration import *
from FaceRecognition import *
terminate_CPU_monitor=True
terminate=True
air_app_container=subprocess.check_output("docker ps -aqf \"name=air-pollution-microlambda\"",shell=True).decode("utf-8").split()[0]
face_app_container=subprocess.check_output("docker ps -aqf \"name=face-recognition-microlambda\"",shell=True).decode("utf-8").split()[0]
human_app_container=subprocess.check_output("docker ps -aqf \"name=human-activity-microlambda\"",shell=True).decode("utf-8").split()[0]
thread_run=True
def threading_CPU_MEM(cmd):
    print("Thread started!!!")
    print(cmd)
    print("measure stared!!!")
    arr=[]
    global terminate_CPU_monitor
    while(True):
        if(terminate_CPU_monitor==False):
            print("loop terminated!!")
            break
        #os.system(cmd)
        out=subprocess.check_output(cmd,shell=True)

def subscribe_redis_monitor_app():
    p=PubSubSubscriber("MetricMonitor")
    print("Metric Monitoring PubSub Controller Started...\nWaiting for input...")
    proc = None
    global terminate_CPU_monitor
    while terminate:
        message = p.get_message()
        if message and message['data']!=1:
            print(message)
            check = json.loads(message["data"])
            print(check)

            if(check["app"]=='face-app'):

                if(check["type"]=="start"):
                    print("Monitoring Face App starts...")
                    filename="Face_Network_size"+str(check["size"])+"_threshold_"+str(check["threshold"])+".pcap"
                    cmd="sudo tshark -i wlo1 -w ../Results/PCAP/Face-App/"+filename+" &"
                    os.system(cmd)

                    filename_net = "../Results/CSV/Face-App/CPU_MEM/Face_CPU_Memory" + str(check["size"]) + "_threshold_" + str(check["threshold"]) + ".csv"
                    os.system("sudo rm "+filename_net)
                    cmd = "docker container stats "+face_app_container+"  --format  \"start,{{ .CPUPerc }},{{.MemPerc}},end\"  | tee --append "+filename_net
                    print(cmd)
                    proc=multiprocessing.Process(target=threading_CPU_MEM, args=(cmd,))
                    proc.start()
                elif(check["type"]=="end"):
                    print("Monitoring Face App ends...")
                    cmd = "sudo killall tshark"
                    os.system(cmd)
                    print("Wireshark closed!!!")
                    proc.terminate()
                    print("Threading done. Saving the log data!!!!")

            elif(check["APP"]=='air-pollution-app'):
                if (check["type"] == "start"):
                    print("Monitoring Air Pollution App starts...")
                    filename = "Network_size" + str(check["size"]) + "_threshold_" + str(check["threshold"]) + ".pcap"
                    cmd = "sudo tshark -i wlo1 -w ../Results/PCAP/Air-Pollution/" + filename + " &"
                    os.system(cmd)
                    filename_net = "../Results/CSV/Air-Pollution-App/CPU_MEM/CPU_Memory" + str(
                        check["size"]) + "_threshold_" + str(check["threshold"]) + ".csv"
                    os.system("sudo rm " + filename_net)
                    cmd = "docker container stats " + air_app_container + "  --format  \"start,{{ .CPUPerc }},{{.MemPerc}},end\"  | tee --append " + filename_net
                    print(cmd)
                    proc = multiprocessing.Process(target=threading_CPU_MEM, args=(cmd, ))
                    proc.start()
                elif (check["type"] == "end"):
                    print("Monitoring Air Pollution App ends...")
                    cmd = "sudo killall tshark"
                    os.system(cmd)
                    print("Wireshark closed!!!")
                    proc.terminate()
                    print("Threading done. Saving the log data!!!!")
            elif (check["APP"] == 'human-activity-app'):
                if (check["type"] == "start"):
                    print("Monitoring Human Activity App starts...")
                    filename = "Human_Network_epoch" + str(check["size"]) + "_threshold_" + str(check["threshold"]) + ".pcap"
                    cmd = "sudo tshark -i enp1s0 -w ../Results/PCAP/Human-Activity/" + filename + " &"
                    os.system(cmd)
                    filename_net = "../Results/CSV/Human-Activity-App/CPU_MEM/Human_CPU_Memory" + str(check["size"]) + "_threshold_" + str(check["threshold"]) + ".csv"
                    os.system("sudo rm " + filename_net)
                    cmd = "docker container stats " + human_app_container + "  --format  \"start,{{ .CPUPerc }},{{.MemPerc}},end\"  | tee --append " + filename_net
                    print(cmd)
                    proc = multiprocessing.Process(target=threading_CPU_MEM, args=(cmd,))
                    proc.start()
                elif (check["type"] == "end"):
                    print("Monitoring Human Activity App ends...")
                    cmd = "sudo killall tshark"
                    os.system(cmd)
                    print("Wireshark closed!!!")
                    proc.terminate()
                    print("Threading done. Saving the log data!!!!")
            print("Done!!!")

if __name__ == '__main__':
    subscribe_redis_monitor_app()
