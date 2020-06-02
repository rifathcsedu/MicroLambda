import redis
import os
import time
import json
import sys
sys.path.append('../../Config/')
from RedisPubSub import *
from configuration import *
from FaceRecognition import *

#waiting for message to trigger
def subscribe_redis():
    p=PubSubSubscriber(Topic["input_face_app"])
    url=AppURL["face_app"]
    print("Controller Started...\nWaiting for input...")
    while True:
        message = p.get_message()
        if message and message["data"]!=1L:
            print(message)
            check=json.loads(message["data"])
            if(len(check["data"])==0):
                start_time=time.time()
                print(start_time)

            if(len(check["data"])!=check["size"]-1):
                cmd="curl "+url+" --data-binary "+json.dumps(message["data"])
                print(cmd)
                print(os.system(cmd))
            else:
                end=time.time()-start_time
                print("Computation Time: ")
                print(end)
                print("Computation Done!!")
                r = redis.Redis(host=redUser_face.pyis_host, port=redis_port)
                r.publish(Topic['result_face_app'],str(json.dumps(message["data"])))


if __name__ == '__main__':
    subscribe_redis()