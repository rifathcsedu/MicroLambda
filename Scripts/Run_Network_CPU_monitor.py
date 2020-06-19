import os
import threading
import subprocess
import json
import csv
terminate=True
def subscribe_redis_monitor_app():

    p=PubSubSubscriber("MetricMonitor")
    print("Metric Monitoring PubSub Controller Started...\nWaiting for input...")
    while terminate:
        message = p.get_message()
        if message and message['data']!=1:
            print(message)
            check = json.loads(message["data"])
            print(message["data"])
            if(check["APP"]=='Face'):
                print("Monitoring Face App...")
                #os.system("tshark")
                #os.system(docker ps)
            elif(check["APP"]=='Air'):
                print("Monitoring Air App...")
                # os.system("tshark")
            elif (check["APP"] == 'Human'):
                print("Monitoring Human App...")
                # os.system("tshark")

if __name__ == '__main__':

