import redis
redis_host = "10.200.54.91"
import os
import time
import json
redis_port = 6379
redis_password = ""
def publish_redis(data):
    r = redis.Redis(host=redis_host, port=redis_port)
    r.publish('SortTrigger',str(json.dumps({
        "data": data
    })))
def subscribe_redis():
    r = redis.Redis(host=redis_host, port=redis_port)
    p = r.pubsub()
    p.subscribe('Notification')
    f=open("shortlambda_face.txt","r")
    url=f.read()
    f.close()
    url=url[0:len(url)-1]
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
