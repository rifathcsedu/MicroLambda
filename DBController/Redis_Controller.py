import redis
import os
import time
import json
import sys

import threading

sys.path.append('../Config/')
sys.path.append('../Class/')

from RedisPubSub import *
from configuration import *
from FaceRecognition import *

terminate=True

def pollution_app_controller():
    url = AppURL["face_app"]

    check = json.loads(message["data"])
    if (len(check["data"]) == 0):
        start_time = time.time()
        print(start_time)
    if (len(check["data"]) != check["size"] - 1):
        cmd = "curl " + url + " --data-binary " + json.dumps(message["data"])
        print(cmd)
        print(os.system(cmd))
    else:
        total_time = time.time() - start_time
        print("Computation Time: ")
        print(total_time)
        print("Computation Done!!")
        publish_redis(Topic['result_face_app'], str(json.dumps(message["data"])))

def human_app_controller():
    url = AppURL["face_app"]
    check = json.loads(message["data"])
    if (len(check["data"]) == 0):
        start_time = time.time()
        print(start_time)
    if (len(check["data"]) != check["size"] - 1):
        cmd = "curl " + url + " --data-binary " + json.dumps(message["data"])
        print(cmd)
        print(os.system(cmd))
    else:
        total_time = time.time() - start_time
        print("Computation Time: ")
        print(total_time)
        print("Computation Done!!")
        publish_redis(Topic['result_face_app'], str(json.dumps(message["data"])))

#waiting for message to trigger
def subscribe_redis_face():
    global terminate
    p=PubSubSubscriber(Topic["publish_face_app"])
    print("Face PubSub Controller Started...\nWaiting for input...")
    while terminate:
        message = p.get_message()
        if message and message['data']!=1:
            print(message)
            url = AppURL["face_app"]
            check = json.loads(message["data"])
            print(message["data"])
            if (len(check["data"]) == 0):
                start_time = time.time()
                print(start_time)
            if (len(check["data"]) != check["size"] - 1):
                cmd = "curl " + url + " --data-binary " + json.dumps(message["data"].decode('utf8').replace("'", '"'))
                print(cmd)
                print(os.system(cmd))
            else:
                total_time = time.time() - start_time
                print("Computation Time: ")
                print(total_time)
                print("Face App Computation Done!!")
                publish_redis(Topic['result_face_app'],
                              str(json.dumps(message["data"].decode('utf8').replace("'", '"'))))

#waiting for message to trigger
def subscribe_redis_pollution():
    global terminate
    p=PubSubSubscriber(Topic["publish_air_pollution_app"])
    print("Air Pollution PubSub Controller Started...\nWaiting for input...")
    while terminate:
        message = p.get_message()
        if message and message['data']!=1:
            print(message)
            url = AppURL["air_pollution_app"]
            check = json.loads(message["data"])
            print(message["data"])
            if (check["current"] == 0):
                start_time = time.time()
                print(start_time)
            if (check["current"] != check["size"]):
                cmd = "curl " + url + " --data-binary " + json.dumps(message["data"].decode('utf8').replace("'", '"'))
                print(cmd)
                print(os.system(cmd))
            else:
                total_time = time.time() - start_time
                print("Computation Time: ")
                print(total_time)
                print("Air Pollution Training Done!!")
                publish_redis(Topic['result_air_pollution_app'],
                              str(json.dumps(message["data"].decode('utf8').replace("'", '"'))))

#waiting for message to trigger
def subscribe_redis_human():
    global terminate
    p=PubSubSubscriber(Topic["input_human_activity_app"])
    print("Human Activity PubSub Controller Started...\nWaiting for input...")
    while terminate:
        message = p.get_message()
        if message and message['data']!=1:
            print(message)
            url = AppURL["human_activity_app"]
            check = json.loads(message["data"])
            print(message["data"])
            if (len(check["data"]) == 0):
                start_time = time.time()
                print(start_time)
            if (len(check["data"]) != check["size"] - 1):
                cmd = "curl " + url + " --data-binary " + json.dumps(message["data"].decode('utf8').replace("'", '"'))
                print(cmd)
                print(os.system(cmd))
            else:
                total_time = time.time() - start_time
                print("Computation Time: ")
                print(total_time)
                print("Human Activity Training Done!!")
                publish_redis(Topic['result_human_activity_app'],
                              str(json.dumps(message["data"].decode('utf8').replace("'", '"'))))

if __name__ == '__main__':
    face_thread = threading.Thread(target=subscribe_redis_face)
    face_thread.start()
    pollution_thread = threading.Thread(target=subscribe_redis_pollution)
    pollution_thread.start()
    human_thread = threading.Thread(target=subscribe_redis_human)
    human_thread.start()
    print("Type DONE to kill all thread")
    in1=input("Enter DONE to Delete: ")
    print(in1)

    if(str(in1)=="DONE"):
        print("Process killing STARTS!!")

        terminate=False
        #face_thread.terminate()
        face_thread.join()
        #pollution_thread.terminate()
        pollution_thread.join()
        #human_thread.terminate()
        human_thread.join()
        print("Process killing done!!")

