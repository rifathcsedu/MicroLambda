import json
import pickle
import base64
import time
import csv
import sys
import face_recognition
import os
from sklearn import preprocessing
from pandas import *

sys.path.append('../Config/')
from RedisPubSub import *
from configuration import *


def WriteCSV(path, data):
    print("Writing output and metrics in CSV...")
    with open(path, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)
    print("Writing Done!")


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def load_data(filename, chunksize):
    pickle_data = []
    dataset = read_csv(filename, header=0, index_col=0)
    values = dataset.values
    # integer encode direction
    encoder = preprocessing.LabelEncoder()
    values[:, 4] = encoder.fit_transform(values[:, 4])
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
    print(reframed.head())

    values = reframed.values
    n_train_hours = 365 * 24
    '''
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    '''
    # upload the data to redis
    start_hour = 0
    n_train_hours = 24
    print(values.shape)
    i=0
    while (True):
        if(start_hour>values.shape[0]):
            break
        oneday_data = values[start_hour:start_hour + n_train_hours, :]
        r.rpush(Topic["input_air_pollution_app"], pickle.dumps(oneday_data))
        start_hour+=n_train_hours
        i+=1
    print(i)
    # publish it to trigger DBController
    publish_redis(Topic["publish_air_pollution_app"], str(json.dumps({
        "data": [],
        "size": i,
        "threshold": float(MicroLambda["short_lambda"])
    })))
    GetResult()


def GetResult():
    p = r.pubsub()
    p.subscribe(Topic["result_face_app"])
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
