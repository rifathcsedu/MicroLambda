import json
import pickle
import base64
import time
import csv
import sys
import os

import numpy as np
from sklearn import preprocessing
import pandas as pd

sys.path.append('../Config/')
sys.path.append('../Class/')

from RedisPubSub import *
from configuration import *
from AirPollution import *

#upload the data to Redis
def load_data(filename, chunksize):
    print("Data uploading started!!!")
    column_names = ['user_id', 'activity', 'timestamp', 'x_axis', 'y_axis', 'z_axis']
    df = pd.read_csv(filename, header=None, names=column_names)
    df.z_axis.replace(regex=True, inplace=True, to_replace=r';', value=r'')
    df['z_axis'] = df.z_axis.astype(np.float64)
    df.dropna(axis=0, how='any', inplace=True)

    # upload it to redis
    i=min(df['user_id'])
    while (i<=max(df['user_id'])):

        df_user = df[df['user_id'] == i]
        r.rpush(Topic["input_human_activity_app"], pickle.dumps(df_user))
        i+=1

    print("Uploading done!")
    print(i)
    # publish it to trigger DBController
    publish_redis(Topic["publish_human_activity_app"], str(json.dumps({
        "size": i,
        'app':'human-app',
        "current":0,
        "training":6,
        "testing":10,
        "threshold": float(MicroLambda["short_lambda"])
    })))
    GetResult(Topic["result_human_activity_app"])


#user controller
def UserInput():

    #control parameters
    Iteration=2
    chunk_size = 1
    input_dir='../Dataset/Human-Activity-Input/WISDM_ar_v1.1_raw.txt'
    output_dir='../Results/CSV/Human-Activity-App/Execution_Time_Human_Activity_App.csv'
    sleep_time=15

    #user controller starts
    print("Hello User! I am MR. Packetized Computation! There is your option: ")
    while (True):
        print("\n\n1. Train Model\n2. Exit")
        d = input("Enter: ")

        threshold = float(MicroLambda["short_lambda"])

        if (str(d) == "1"):
            filename = input_dir
            print("Loading Human Activity Data from Dataset: " + filename)
            for i in range(Iteration):

                print("Taking Break for "+str(sleep_time)+" sec!")
                time.sleep(sleep_time)
                print("Cleaning Started!")
                Cleaning(Topic["input_human_activity_app"])
                CleaningModel(Topic["model_human_activity_app"])
                print("Taking Break for "+str(sleep_time)+" sec!")

                time.sleep(sleep_time)
                start = time.time()
                load_data(filename, chunk_size)
                end = time.time()
                print("time: " + str(end - start))
                WriteCSV(output_dir, [[l, threshold, end - start]])
                print("done!")
        else:
            break

if __name__ == '__main__':
    UserInput()
