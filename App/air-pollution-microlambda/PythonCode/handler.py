import os
import json
import sys
from RedisPubSub import *
from AirPollution import *
from math import sqrt
from numpy import concatenate
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import time

def get_stdin():
    buf = ""
    for line in sys.stdin:
        buf = buf + line
    return buf

def handle (req):
    start = time.time()
    print(req)
    json_req = json.loads(req)
    print(json_req)
    current = json_req["current"]
    training_size=json_req["training"]
    total_size=json_req["size"]
    threshold=int(json_req["threshold"])
    epoch=int(json_req["epoch"])
    batch=int(json_req["batch"])
    neoron=int(json_req["neoron"])
    while(True):
        publish_redis("test", "new loop")
        time_diff=time.time()-start
        publish_redis("test",str(time_diff))
        if(time_diff>threshold or current>=total_size):
            break
        i = current
        arr=[]
        while(i<current+training_size and i<total_size):
            temp=LoadData(Topic["input_air_pollution_app"], i, i)
            publish_redis("test","i="+str(i))
            if(i==current):
                arr=pickle.loads(temp[0])
            else:
                arr=np.concatenate((arr,pickle.loads(temp[0])))

            i+=1

        current=i
        publish_redis("test","current= "+str(current))
        n_train_hours=arr.shape[0]-24*int(json_req["training"]*0.1) # 20 percent data for testing
        train = arr[:n_train_hours, :]
        test = arr[n_train_hours:, :]
        # split into input and outputs
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        model = None
        model = Sequential()
        model.add(LSTM(neoron, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))
        temp=RedisLoadModel(Topic["model_air_pollution_app"])

        if (temp != None):
            json_data = pickle.loads(temp)
            model.set_weights(json_data)
            publish_redis("test", "Model loading done!!!")
        else:
            publish_redis("test", "New Model created!!!")

        model.compile(loss='mae', optimizer='adam')
        #print("Model setting done and compile done!!!")
        publish_redis("test", "Model setting done and compile done!!!")
        # fit network
        history = model.fit(train_X, train_y, epochs=epoch, batch_size=batch, validation_data=(test_X, test_y), verbose=2,shuffle=False)
        publish_redis("test", "Training done!!!")

        #save model to redis
        publish_redis("test", "Saving model starts...!!")

        #model_weight=pickle.dumps()

        RedisSaveModel(Topic['model_air_pollution_app'], pickle.dumps(model.get_weights()))
        #print("Saving model done...!!")
        publish_redis("test", "Saving model done...!!")
        publish_redis("test", "Current is "+str(current))

    json_req["current"]=current
    publish_redis("test", json.dumps({"data":"done","current":current,"time":str(time.time()-start)}))
    return publish_redis(Topic["publish_air_pollution_app"],json.dumps(json_req))

#main function
if __name__ == "__main__":
    st = get_stdin()
    handle(st)
    # design network
