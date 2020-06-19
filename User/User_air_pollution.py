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

#upload the data to Redis
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
        'app':'pollution-app',
        "size": i,
        "current":0,
        "training":chunksize,
        "threshold": float(MicroLambda["short_lambda"])
    })))
    GetResult(Topic["result_air_pollution_app"])


#user controller
def UserInput():

    #control parameter
    sleep_time=15
    Iteration=1
    input_dir='../Dataset/Air-Pollution-Input/pollution.csv'
    output_dir='../Results/CSV/Air-Pollution-App/Execution_Time_Air_Pollution.csv'
    chunk_size = 365

    #user controller
    print("Hello User! I am MR. Packetized Computation! There is your option: ")
    while (True):

        print("\n\n1. Train Model\n2. Exit")
        d = input("Enter: ")
        threshold = float(MicroLambda["short_lambda"])
        if (str(d) == "1"):
            filename = input_dir
            print("Loading Pollution Data from Dataset: " + filename)

            for i in range(Iteration):
                print("Iteration: " + str(i + 1) + ", Total Iteration " + str(Iteration))
                print("Taking Break for "+str(sleep_time)+" sec!")
                time.sleep(sleep_time)

                print("Cleaning Started!")
                Cleaning(Topic["input_air_pollution_app"])
                CleaningModel(Topic["model_air_pollution_app"])
                print("Taking Break for "+str(sleep_time)+" sec!")

                time.sleep(sleep_time)
                start = time.time()
                load_data(filename, chunk_size)
                end = time.time()
                print("time: " + str(end - start))
                WriteCSV(output_dir, [[chunk_size, threshold, end - start]])
                print("done!")
        else:
            break

if __name__ == '__main__':
    UserInput()
