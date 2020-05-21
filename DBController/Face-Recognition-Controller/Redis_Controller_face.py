import redis
import os
import time
import json

from ../../Config/RedisPubSub import *
from ../../Config/configuration import *
from FaceRecognition import *

def subscribe_redis():
    p = r.pubsub()
    p.subscribe(Topic["publish_face_app"])
    url=
    while True:
        message = p.get_message()
        if message and message["data"]!=1L:
            #print(message)
            check=json.loads(message["data"])
            if(len(check["data"])==0):
                start=time.time()
                print(start)
            if(len(check["data"])!=check["size"]-1):
                #start=time.time()
                #print(start)
                cmd="curl "+url+" --data-binary "+json.dumps(message["data"])
                print(cmd)
                print(os.system(cmd))

            else:
                end=time.time()-start
                print("Computation Time: ")
                print(end-start)
                print("Computation Done!!")
                r = redis.Redis(host=redis_host, port=redis_port)
                r.publish('Result',str(json.dumps(message["data"])))


if __name__ == '__main__':
    subscribe_redis()
