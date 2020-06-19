import redis
from configuration import *
redis_host = Database["host"]
redis_port = int(Database["port"])
redis_password = Database["password"]
r = redis.Redis(host=redis_host, port=redis_port)

#save model
def RedisSaveModel(topic,data):
    r.hset(topic, topic+"1", data)

#load model
def RedisLoadModel(topic):
    return r.hget(topic, topic+"1")

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
def CleaningModel(topic):
    r.hdel(topic,topic+"1")
    print("model deleted!")

publish_redis("test",redis_host)