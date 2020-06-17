import redis
import json
from configuration import *
redis_host = Database["host"]
redis_port = int(Database["port"])
redis_password = Database["password"]
r = redis.Redis(host=redis_host, port=redis_port)

#publish via redis
def publish_redis(topic,data):
    r.publish(topic,data)
    print("Publish to topic "+str(topic))

#load data from redis array
def LoadData(topic,start,end):
    return r.lrange(topic, start, end)

def PubSubSubscriber(topic):
    p = r.pubsub()
    p.subscribe(topic)
    return p
#cleaning the topic
def Cleaning(topic):
    while (True):
        if (r.rpop(topic) == None):
            print("cleaning done")
            break
