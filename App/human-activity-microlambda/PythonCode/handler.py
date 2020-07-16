import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix
import scipy.stats
from HumanActivity import *
from RedisPubSub import *
from keras.models import model_from_json

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

column_names = ['user_id', 'activity', 'timestamp', 'x_axis', 'y_axis', 'z_axis']




def get_stdin():
    buf = ""
    for line in sys.stdin:
        buf = buf + line
    return buf

def handle (req):
    publish_redis("test", "Training started!!!")
    start = time.time()
    print(req)
    json_req = json.loads(req)
    print(json_req)
    current = json_req["current"]
    i = current
    training_size = json_req["training"]
    testing_size = int(training_size * 0.2)
    threshold=int(json_req["threshold"])
    size=int(json_req["size"])
    epoch=int(json_req["epoch"])
    while(time.time()-start<threshold and current<size):
        publish_redis("test", "New loop!!! current= "+str(current))
        arr=[]
        arr_backup=[]
        i=current
        while(i<current+training_size and i<size):
            publish_redis("test", "i= " + str(i))
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
        current=i
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

        if (temp != None):
            json_data = pickle.loads(temp)
            model.set_weights(json_data)
            publish_redis("test", "Model loading done!!!")
        else:
            publish_redis("test", "New Model created!!!")


        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        #print("Model setting done and compile done!!!")
        publish_redis("test", "Model setting done and compile done!!!")
        # fit network
        history = model.fit(
            X_train, y_train,
            epochs=epoch,
            batch_size=64,
            validation_split=0.1,
            shuffle=True
        )
        publish_redis("test", "Training done!!!")

        #save model to redis
        publish_redis("test", "Saving model starts...!!")

        #model_weight=pickle.dumps()

        RedisSaveModel(Topic['model_human_activity_app'], pickle.dumps(model.get_weights()))

        #print("Saving model done...!!")
        publish_redis("test", "Saving model done...!!")

    json_req["current"]=current
    publish_redis("test", json.dumps({"data":"done"}))
    return publish_redis(Topic["publish_human_activity_app"],json.dumps(json_req))

if __name__ == "__main__":
    st = get_stdin()
    handle(st)