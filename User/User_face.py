import json
import pickle
import base64
import time
import csv
import sys
import face_recognition
import os
sys.path.append('../Config/')
from RedisPubSub import *
from configuration import *

def WriteCSV(path,data):
    print("Writing output and metrics in CSV...")
    with open(path, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)
    print("Writing Done!")
def load_images(arr):
    pickle_data=[]
    r = redis.Redis(host=redis_host, port=redis_port)
    for i in arr:
        data = {}
        known_image = face_recognition.load_image_file("../Dataset/Face-Recognition-Input/"+i)
        r.rpush(Topic["input_face_app"],pickle.dumps(known_image))
    publish_redis(Topic["input_face_app"],str(json.dumps({
        "data": [],
        "size": len(arr),
        "threshold":float(MicroLambda["short_lambda"])
    })))
    GetResult()
def GetResult():
    p = r.pubsub()
    p.subscribe(Topic["result_face_app"])
    print("Waiting for Result: ")
    while True:
        message = p.get_message()
        #print(message)
        if message and message["data"]!=1:
            print("Got output: "+str(json.loads(message["data"])))
            break

def UserInput():

    arr=["1.jpg","2.jpg","3.jpg","4.jpg","5.jpg","6.jpg","7.jpg","8.jpg","9.jpg","10.jpg"]
    print("Hello User! I am MR. Packetized Computation! There is your option: ")
    while(True):
        Iteration=2
        print("1. New data\n2. Exit")
        d=input()
        threshold=float(MicroLambda["short_lambda"])
        if(d=="1"):
            print("Number of Images: ")
            l=input()
            l=int(l)
            print(l)
            for i in range(Iteration):
                print("Taking Break for 15 sec!")
                time.sleep(15)
                print(Topic)
                print("Cleaning Started!")
                Cleaning(Topic["input_face_app"])
                print("Taking Break for 15 sec!")
                time.sleep(15)
                print("Iteration: "+str(i)+", Computation started for image: "+str(l))
                start=time.time()
                load_images(arr[:l])
                end=time.time()
                print("time: "+str(end-start))
                WriteCSV('interval_'+str(threshold)+'.csv',[[l,end-start]])
                print("done!")
        else:
            break

if __name__ == '__main__':
    UserInput()
