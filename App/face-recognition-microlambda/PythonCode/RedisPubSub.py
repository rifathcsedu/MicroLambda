import redis
import json

from configuration import *

redis_host = Database["host"]
redis_port = int(Database["port"])
redis_password = Database["password"]
r = redis.Redis(host=redis_host, port=redis_port)

#save model
def RedisSaveModel(topic,data):
    r.hset(topic, topic+"1", data)

def RedisSaveValue(topic,data):
    r.set(topic,data)

#load model
def RedisLoadModel(topic):
    return r.hget(topic, topic+"1")

def RedisLoadValue(topic):
    return r.get(topic)

ip=RedisLoadValue("ServerIPAddress").decode('utf-8')

AppURL = dict(
    face_app = 'http://'+ip+':8080/function/face-recognition-microlambda',
    air_pollution_app = 'http://'+ip+':8080/function/air-pollution-microlambda',
    human_activity_app='http://'+ip+':8080/function/human-activity-microlambda',
    mental_stress_app='http://'+ip+':8080/function/mental-stress-microlambda',
)

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
def UploadData(topic,data):
    #publish_redis("test","uploading starts")
    return r.rpush(topic, data)
#waiting for results
def GetResult(topic):
    p = r.pubsub()
    p.subscribe(topic)
    print("Waiting for Result: ")
    while True:
        message = p.get_message()
        # print(message)
        if message and message["data"] != 1:
            print("Got output: " + str(json.loads(message["data"])))
            return message["data"]

publish_redis("test",redis_host)
