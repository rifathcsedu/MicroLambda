import json
import pickle
import base64
import time
import csv
import sys
import os

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
from sklearn import preprocessing
from pandas import *

sys.path.append('../Config/')
sys.path.append('../Class/')

input_dir='../Dataset/Air-Pollution-Input/pollution.csv'
output_dir='../Results/CSV/Air-Pollution-App/Execution_Time_Air_Pollution.csv'

from RedisPubSub import *
from configuration import *
from AirPollution import *
min_data=0
max_data=0
#upload the data to Redis
def load_data(filename, chunksize):
    global max_data
    global min_data
    print("Data uploading started!!!")
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

    print("Uploading done!")
    print(i)
    max_data=i-1


def Testing(current,training_size,size):
    global input_dir
    global output_dir
    print("Testing Starts")
    arr=[]
    i=current
    while(i<=training_size):
        temp=LoadData(Topic["input_air_pollution_app"], i, i)
        print(pickle.loads(temp[0]))
        if(i==current):
            arr=pickle.loads(temp[0])
        else:
            arr=np.concatenate((arr,pickle.loads(temp[0])))
        i+=1
    train = arr[:]
    arr=[]
    while(i<size):

        temp=LoadData(Topic["input_air_pollution_app"], i, i)
        #print(pickle.loads(temp[0]))
        print(i)
        if(i==training_size+1):
            arr=pickle.loads(temp[0])
        else:
            arr=np.concatenate((arr,pickle.loads(temp[0])))
        i+=1
    test=arr[:]
    #n_train_hours=arr.shape[0]-24*int(json_req["training"]*0.2) # 20 percent data for testing

    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    model = None
    model = Sequential()
    model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    dataset = read_csv(input_dir, header=0, index_col=0)
    values = dataset.values
    # integer encode direction
    encoder = LabelEncoder()
    values[:,4] = encoder.fit_transform(values[:,4])
    # ensure all data is float
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    temp=RedisLoadModel(Topic["model_air_pollution_app"])
    json_data = pickle.loads(temp)
    model.set_weights(json_data)
    print("Model loading done!!!")
    model.compile(loss='mae', optimizer='adam')
    #print("Model setting done and compile done!!!")
    print("Model setting done and compile done!!!")
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    return rmse
    #y_pred=model.predict(X_test)
    #print(y_pred)
#user controller
def UserInput():

    #control parameters
    chunk_size = 1
    global max_data
    global min_data
    global input_dir
    global output_dir
    #user controller starts
    print("Hello User! I am MR. Packetized Computation! There is your option: ")
    print("Cleaning input Started!")
    Cleaning(Topic["input_air_pollution_app"])
    #CleaningModel(Topic["model_human_activity_app"])
    print("Taking Break for "+str(sleep_time)+" sec!")

    filename = input_dir
    print("Loading Air Pollution Data from Dataset: " + filename)
    upload_time=time.time()
    load_data(filename, chunk_size)
    upload_time=time.time()-upload_time
    print("Uploading Time: "+str(upload_time))
    testing_size=365
    while (True):
        print("\n\n1. Train Model\n2. Exit")
        d = input("Enter: ")
        #threshold = MicroLambda["short_lambda"]
        if (str(d) == "1"):
            #print("\n\n1. Epoch Size\n2. Exit")
            epoch_list=[25,50]
            for l in epoch_list:
                for threshold in MicroLambda["short_lambda"]:
                    print("threshold: "+str(threshold))
                    publish_redis("MetricMonitor", str(json.dumps({
                        'app': 'pollution-app',
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
                        CleaningModel(Topic["model_air_pollution_app"])
                        print("Taking Break for "+str(sleep_time)+" sec!")
                        time.sleep(sleep_time)
                        start = time.time()
                        # publish it to trigger DBController
                        train=0
                        if(threshold=='1500'):
                            train=1460
                        else:
                            train=365
                        publish_redis(Topic["publish_air_pollution_app"], str(json.dumps({
                            "size": max_data-testing_size+1,
                            'app':'pollution-app',
                            "current":0,
                            "training":train,
                            "epoch":int(l),
                            "threshold": int(threshold)
                        })))
                        GetResult(Topic["result_air_pollution_app"])
                        end = time.time()
                        print("time: " + str(end - start))
                        acc=Testing(0,1460,1825)
                        print(acc)
                        data.append([threshold,l,acc,end-start+upload_time])
                        WriteCSV(output_dir, data)
                        print("done!")

                    publish_redis("MetricMonitor", str(json.dumps({
                        'app': 'pollution-app',
                        "type": 'end',
                        "size": l,
                        "threshold": float(threshold)
                    })))

        else:
            break


if __name__ == '__main__':
    UserInput()
