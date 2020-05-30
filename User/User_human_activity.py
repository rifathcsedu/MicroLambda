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
    arr=[]
    input_size=10
    for i in range(1,input_size+1):
        arr.append(str(i)+".jpg")
    print(arr)
    print("Hello User! I am MR. Packetized Computation! There is your option: ")
    while(True):
        Iteration=2
        print("\n\n1. New data\n2. Exit")
        d=input("Enter: ")
        threshold=float(MicroLambda["short_lambda"])
        if(d=="1"):
            print("Number of Images: (Max number: "+str(input_size)+")")
            l=input("Enter: ")
            l=int(l)
            if(l>input_size):
                print("Invalid input!!! Try again")
                continue
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
                WriteCSV('../Results/CSV/Face-App/Execution_Time.csv',[[l,threshold,end-start]])
                print("done!")
        else:
            break

if __name__ == '__main__':
    UserInput()
