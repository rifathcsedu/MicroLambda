import json
import pickle
import base64
import time
import csv
import sys
import os

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix
import scipy.stats
from keras.models import model_from_json

import numpy as np
from sklearn import preprocessing
import pandas as pd

sys.path.append('../Config/')
sys.path.append('../Class/')

from RedisPubSub import *
from configuration import *
from HumanActivity import *
min_data=0
max_data=0
#upload the data to Redis
def load_data(filename, chunksize):
    global max_data
    global min_data
    print("Data uploading started!!!")
    column_names = ['user_id', 'activity', 'timestamp', 'x_axis', 'y_axis', 'z_axis']
    df = pd.read_csv(filename, header=None, names=column_names)
    df.z_axis.replace(regex=True, inplace=True, to_replace=r';', value=r'')
    df['z_axis'] = df.z_axis.astype(np.float64)
    df.dropna(axis=0, how='any', inplace=True)
    min_data=min(df['user_id'])
    max_data=max(df['user_id'])
    # upload it to redis
    i=min_data
    while (i<=max_data):

        df_user = df[df['user_id'] == i]
        r.rpush(Topic["input_human_activity_app"], pickle.dumps(df_user))
        i+=1


    print("Uploading done!")
    print(i)

def Testing(current,training_size,size):
    print("Testing Starts")
    arr=[]
    i=current
    while(i<=training_size):
        temp=LoadData(Topic["input_human_activity_app"], i, i)
        print(pickle.loads(temp[0]))
        if(i==current):
            arr=pickle.loads(temp[0])
        else:
            arr=pd.concat([arr,pickle.loads(temp[0])])
        i+=1
    df_train = arr
    arr=[]
    while(i<size):

        temp=LoadData(Topic["input_human_activity_app"], i, i)
        #print(pickle.loads(temp[0]))
        print(i)
        if(i==training_size+1):
            arr=pickle.loads(temp[0])
        else:
            arr=pd.concat([arr,pickle.loads(temp[0])])
        i+=1
    df_test=arr
    print(df_train)
    print(df_test)
    scale_columns = ['x_axis', 'y_axis', 'z_axis']

    scaler = RobustScaler()

    scaler = scaler.fit(df_train[scale_columns])

    df_train.loc[:, scale_columns] = scaler.transform(df_train[scale_columns].to_numpy())
    df_test.loc[:, scale_columns] = scaler.transform(df_test[scale_columns].to_numpy())

    TIME_STEPS = 200
    STEP = 40

    X_train, y_train = create_dataset(
        df_train[['x_axis', 'y_axis', 'z_axis']],
        df_train.activity,
        TIME_STEPS,
        STEP
    )

    X_test, y_test = create_dataset(
        df_test[['x_axis', 'y_axis', 'z_axis']],
        df_test.activity,
        TIME_STEPS,
        STEP
    )

    print(X_train.shape, y_train.shape)

    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

    enc = enc.fit(y_train)

    y_train = enc.transform(y_train)
    y_test = enc.transform(y_test)

    model = keras.Sequential()
    model.add(
        keras.layers.LSTM(
            units=50,
            input_shape=[X_train.shape[1], X_train.shape[2]]
        )
    )

    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(units=50, activation='relu'))
    model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))

    temp=RedisLoadModel(Topic["model_human_activity_app"])

    json_data = pickle.loads(temp)
    model.set_weights(json_data)
    print("Model loading done!!!")

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    #print("Model setting done and compile done!!!")
    print("Model setting done and compile done!!!")
    print("Testing Done!!!")
    return model.evaluate(X_test,y_test)
    #y_pred=model.predict(X_test)
    #print(y_pred)



#user controller
def UserInput():

    #control parameters
    chunk_size = 1
    input_dir='../Dataset/Human-Activity-Input/WISDM_ar_v1.1_raw.txt'
    output_dir='../Results/CSV/Human-Activity-App/Execution_Time_Human_Activity_App.csv'
    global max_data
    global min_data
    #user controller starts
    print("Hello User! I am MR. Packetized Computation! There is your option: ")
    print("Cleaning input Started!")
    Cleaning(Topic["input_human_activity_app"])
    #CleaningModel(Topic["model_human_activity_app"])
    print("Taking Break for "+str(sleep_time)+" sec!")

    filename = input_dir
    print("Loading Human Activity Data from Dataset: " + filename)
    upload_time=time.time()
    load_data(filename, chunk_size)
    upload_time=time.time()-upload_time
    print("Uploading Time: "+str(upload_time))
    testing_size=6
    while (True):
        print("\n\n1. Train Model\n2. Exit")
        d = input("Enter: ")
        #threshold = MicroLambda["short_lambda"]
        if (str(d) == "1"):
            #print("\n\n1. Epoch Size\n2. Exit")
            epoch_list=[20]
            for l in epoch_list:
                for threshold in MicroLambda["short_lambda"]:

                    print("threshold: "+str(threshold))
                    publish_redis("MetricMonitor", str(json.dumps({
                        'app': 'human-activity-app',
                        "type": 'start',
                        "size": l,
                        "threshold": float(threshold)
                    })))
                    for i in range(Iteration):
                        data=[]
                        print("Iteration: " + str(i + 1) + ", Total Iteration " + str(Iteration))
                        time.sleep(5)
                        print("Taking Break for "+str(sleep_time)+" sec!")
                        time.sleep(sleep_time)
                        print("Cleaning Model Started!")
                        CleaningModel(Topic["model_human_activity_app"])
                        print("Taking Break for "+str(sleep_time)+" sec!")
                        time.sleep(sleep_time)
                        start = time.time()
                        # publish it to trigger DBController
                        train=0
                        if(threshold=='1500'):
                            train=30
                        else:
                            train=10
                        publish_redis(Topic["publish_human_activity_app"], str(json.dumps({
                            "size": max_data-testing_size+1,
                            'app':'human-activity-app',
                            "current":1,
                            "training":train,
                            "epoch":int(l),
                            "threshold": int(threshold)
                        })))
                        GetResult(Topic["result_human_activity_app"])
                        end = time.time()
                        print("time: " + str(end - start))
                        acc=Testing(1,30,max_data)
                        data.append([threshold,l,acc[1],acc[0],end-start+upload_time])
                        WriteCSV(output_dir, data)
                        print("done!")

                    publish_redis("MetricMonitor", str(json.dumps({
                        'app': 'human-activity-app',
                        "type": 'end',
                        "size": l,
                        "threshold": float(threshold)
                    })))

        else:
            break

if __name__ == '__main__':
    UserInput()
