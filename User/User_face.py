#system default packeages

import json
import pickle
import base64
import time
import csv
import sys
import os

#system installed packages

import face_recognition

#sys path setting

sys.path.append('../Config/')

#user defined packages
from RedisPubSub import *
from configuration import *

#upload images to redis
def load_images(arr):
    input_dir="../Dataset/Face-Recognition-Input/"
    pickle_data=[]
    r = redis.Redis(host=redis_host, port=redis_port)
    for i in arr:
        data = {}
        known_image = face_recognition.load_image_file(input_dir+i)
        r.rpush(Topic["input_face_app"],pickle.dumps(known_image))

    publish_redis(Topic["publish_face_app"],str(json.dumps({
        "data": [],
        'app': 'face-app',
        "size": len(arr),
        "threshold":float(MicroLambda["short_lambda"])
    })))
    GetResult(Topic["result_face_app"])

#user controller
def UserInput():

    #control setting
    input_size=10
    output_file = '../Results/CSV/Face-App/Execution_Time_Face_Recognition_App.csv'
    # load input in a array
    arr = []
    for i in range(1,input_size+1):
        arr.append(str(i)+".jpg")

    #user option
    print("Hello User! I am MR. Packetized Computation! There is your option: ")
    while(True):
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
                print("Taking Break for "+str(sleep_time)+" sec!")
                time.sleep(sleep_time)

                print("Cleaning Started!")
                Cleaning(Topic["input_face_app"])
                print("Taking Break for "+str(sleep_time)+" sec!")
                time.sleep(sleep_time)
                print("Iteration: "+str(i+1)+", Total Iteration "+str(Iteration)+" Computation started for image: "+str(l))

                start=time.time()
                load_images(arr[:l])
                end=time.time()
                print("time: "+str(end-start))

                WriteCSV(output_file,[[l,threshold,end-start]])
                print("done!")
        else:
            break

if __name__ == '__main__':
    UserInput()
