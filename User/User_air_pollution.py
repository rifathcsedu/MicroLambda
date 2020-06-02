import json
import pickle
import base64
import time
import csv
import sys
import os
from sklearn import preprocessing
from pandas import *
sys.path.append('../Config/')
sys.path.append('../Class/')
from RedisPubSub import *
from configuration import *
from AirPollution import *

def WriteCSV(path, data):
    print("Writing output and metrics in CSV...")
    with open(path, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)
    print("Writing Done!")

def load_data(filename, chunksize):
    pickle_data = []
    dataset = read_csv(filename, header=0, index_col=0)

    #data shaping
    values=DataShaping(dataset)
    start_hour = 0
    n_train_hours = 24
    # upload it to redis
    i=0
    while (True):
        if(start_hour>values.shape[0]):
            break
        oneday_data = values[start_hour:start_hour + n_train_hours, :]
        r.rpush(Topic["input_air_pollution_app"], pickle.dumps(oneday_data))
        start_hour+=n_train_hours
        i+=1

    # publish it to trigger DBController
    publish_redis(Topic["publish_air_pollution_app"], str(json.dumps({
        "data": [],
        "size": i,
        "current":0,
        "training":60,
        "testing":10,
        "threshold": float(MicroLambda["short_lambda"])
    })))
    GetResult()

def GetResult():
    p = r.pubsub()
    p.subscribe(Topic["result_air_pollution_app"])
    print("Waiting for Result: ")
    while True:
        message = p.get_message()
        # print(message)
        if message and message["data"] != 1:
            print("Got output: " + str(json.loads(message["data"])))
            break


def UserInput():
    print("Hello User! I am MR. Packetized Computation! There is your option: ")
    while (True):
        Iteration = 2
        print("\n\n1. Train Model\n2. Exit")
        d = input("Enter: ")
        threshold = float(MicroLambda["short_lambda"])
        if (str(d) == "1"):
            filename = '../Dataset/Air-Pollution-Input/pollution.csv'
            print("Loading Pollution Data from Dataset: " + filename)
            chunk_size = 1
            for i in range(Iteration):
                print("Taking Break for 15 sec!")
                time.sleep(15)
                print(Topic)
                print("Cleaning Started!")
                Cleaning(Topic["input_air_pollution_app"])
                print("Taking Break for 15 sec!")
                time.sleep(15)
                start = time.time()
                load_data(filename, chunk_size)
                end = time.time()
                print("time: " + str(end - start))
                WriteCSV('../Results/CSV/Face-App/Execution_Time.csv', [[l, threshold, end - start]])
                print("done!")
        else:
            break


if __name__ == '__main__':
    UserInput()
