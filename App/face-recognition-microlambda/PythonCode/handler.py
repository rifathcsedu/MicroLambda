import os
import json
import time
import base64
import numpy as np
import pickle
import sys
#user define
from configuration import *
from FaceRecognition import *
from RedisPubSub import *

def get_stdin():
    buf = ""
    for line in sys.stdin:
        buf = buf + line
    return buf

#face_app handler
def handle(req):
    print(req)
    #load data from json
    json_req=json.loads(req)
    current=json_req["data"]
    current_length=len(current)
    results=[]
    threshold=json_req["threshold"]
    size=json_req["size"]

    start_time=time.time()

    #first image encodings
    first_image = LoadData(Topic["input_face_app"], current_length, current_length)
    biden_encoding = BidenEncoding(first_image)

    flag=0
    for d in range(current_length+1,size):
        #timeout threshold setting
        if(time.time()-start_time>threshold):
            break
        if(flag==1):
            first_image=second_image[:]
            biden_encoding=unknown_encoding[:]

        #second image encodings
        flag=1
        second_image = LoadData(Topic["input_face_app"], d, d)
        unknown_encoding = BidenEncoding(second_image)

        #image compare
        result = ImageCompare(biden_encoding,unknown_encoding)

        if(result[0]==True):
            results.append("True")
        else:
            results.append("False")

    #print(results)
    for i in results:
        current.append(i)
    print(current)

    json_req["data"]=current

    return publish_redis(Topic["publish_face_app"],json.dumps(json_req))

if __name__ == "__main__":
    st = get_stdin()
    handle(st)
