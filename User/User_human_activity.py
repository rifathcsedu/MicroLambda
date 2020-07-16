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

def Testing(current,training_size,testing_size):
    print("Testing Starts")
    arr=[]
    arr_backup=[]
    i=current
    while(i<current+training_size):
        temp=LoadData(Topic["input_human_activity_app"], i, i)
        print(pickle.loads(temp[0]))
        if(i==current):
            arr=pickle.loads(temp[0])
        else:
            arr=pd.concat([arr,pickle.loads(temp[0])])
            if(current+training_size-i==testing_size):
                arr_backup=pickle.loads(temp[0])
            if(current+training_size-i<testing_size):
                arr_backup=pd.concat([arr_backup,pickle.loads(temp[0])])
        print(arr.size)
        i+=1
    df_train = arr
    df_test=arr_backup
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
    y_pred=model.predict(X_test)
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
            l = 20
            for threshold in MicroLambda["short_lambda"]:
                data=[]
                print("threshold: "+str(threshold))
                for i in range(Iteration):
                    print("Iteration: " + str(i + 1) + ", Total Iteration " + str(Iteration))
                    print("Taking Break for "+str(sleep_time)+" sec!")
                    time.sleep(sleep_time)
                    print("Cleaning Model Started!")
                    CleaningModel(Topic["model_human_activity_app"])
                    print("Taking Break for "+str(sleep_time)+" sec!")
                    time.sleep(sleep_time)
                    start = time.time()
                    # publish it to trigger DBController
                    publish_redis(Topic["publish_human_activity_app"], str(json.dumps({
                        "size": max_data-testing_size,
                        'app':'human-app',
                        "current":1,
                        "training":10,
                        "epoch":int(l),
                        "threshold": int(threshold)
                    })))
                    GetResult(Topic["result_human_activity_app"])
                    end = time.time()
                    print("time: " + str(end - start))
                    acc=Testing(testing_size)
                    data.append([threshold,acc[1],acc[0],end-start+upload_time])
                    print("done!")
            WriteCSV(output_dir, data)
        else:
            break

if __name__ == '__main__':
    UserInput()
