import redis
import json
import pickle
import base64
import face_recognition
import time
import csv
import sys
redis_host = "10.200.54.91"
redis_port = 6379
redis_password = ""

def publish_redis(data):
    r = redis.Redis(host=redis_host, port=redis_port)
    f=open("threshold.txt","r")
    threshold=f.read()
    f.close()
    r.publish('Notification',str(json.dumps({
        "data": [],
        "size": data,
        "threshold":float(threshold)
    })))
def WriteCSV(path,data):
    with open(path, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)
def load_images(arr):
    pickle_data=[]
    r = redis.Redis(host=redis_host, port=redis_port)
    for i in arr:
        data = {}
        known_image = face_recognition.load_image_file("Input/"+i)
        r.rpush("ImageInput",pickle.dumps(known_image))

    publish_redis(len(arr))
    GetResult()
def GetResult():
    r = redis.Redis(host=redis_host, port=redis_port)
    p = r.pubsub()
    p.subscribe('Result')
    while True:
        message = p.get_message()
        if message and message["data"]!=1L:
            print(message)
            break

def UserInput():
    arr=["1.jpg","2.jpg","3.jpg","4.jpg","5.jpg","6.jpg","7.jpg","8.jpg","9.jpg","10.jpg"]
    print("Hello User! I am MR. Packetized Computation! There is your option: ")
    while(True):

        #print("Cleaning Started!")
        '''
        while(True):
            if(r.rpop("ImageInput")==None):
                print("cleaning done")
                break
        time.sleep(10)
        '''
        Iteration=int(sys.argv[1])
        print("1. New data\n2. Exit")
        d=raw_input()
        if(d=="1"):
            print("Number of Images: ")
            l=raw_input()
            l=int(l)
            #arr=[]
            #for i in range(l):
                #print("Image Name: ")
                #z=raw_input()
                #arr.append(z)
            f=open("threshold.txt","r")
            threshold=f.read()
            f.close()
            r = redis.Redis(host=redis_host, port=redis_port)
            for i in range(Iteration):
                print("Taking Break for 15 sec!")
                time.sleep(15)
                print("Cleaning Started!")
                while(True):
                    if(r.rpop("ImageInput")==None):
                        print("cleaning done")
                        break
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
