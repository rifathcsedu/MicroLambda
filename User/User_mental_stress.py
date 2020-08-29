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

def load_control(path):
    print("Loading Control: "+path)
    with open(path) as file:
        data = read_acqknowledge(path, resample_method="numpy", impute_missing=True)
        data = data[0]
        return data

def load_stress(path):
    print("Loading Stress: " + path)
    with open(path) as file:
        data = read_acqknowledge(path, resample_method="numpy", impute_missing=True)
        data = data[0]
        return data

def read_acqknowledge(filename, sampling_rate="max", resample_method="numpy", impute_missing=True):
    """Read and format a BIOPAC's AcqKnowledge file into a pandas' dataframe.
    The function outputs both the dataframe and the sampling rate (encoded within the AcqKnowledge) file.
    Parameters
    ----------
    filename :  str
        Filename (with or without the extension) of a BIOPAC's AcqKnowledge file.
    sampling_rate : int
        Sampling rate (in Hz, i.e., samples/second). Since an AcqKnowledge file can contain signals recorded at different rates, harmonization is necessary in order to convert it to a DataFrame. Thus, if `sampling_rate` is set to 'max' (default), will keep the maximum recorded sampling rate and upsample the channels with lower rate if necessary (using the `signal_resample()` function). If the sampling rate is set to a given value, will resample the signals to the desired value. Note that the value of the sampling rate is outputted along with the data.
    resample_method : str
        Method of resampling (see `signal_resample()`).
    impute_missing : bool
        Sometimes, due to connections issues, the signal has some holes (short periods without signal). If 'impute_missing' is True, will automatically fill the signal interruptions using padding.
    Returns
    ----------
    df, sampling rate: DataFrame, int
        The AcqKnowledge file converted to a dataframe and its sampling rate.
    See Also
    --------
    signal_resample
    Example
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> data, sampling_rate = nk.read_acqknowledge('file.acq')
    """
    # Try loading bioread
    try:
        import bioread
    except ImportError:
        raise ImportError("NeuroKit error: read_acqknowledge(): the 'bioread' "
                          "module is required for this function to run. ",
                          "Please install it first (`pip install bioread`).")


    if os.path.exists(filename) is False:
        raise ValueError(
            "NeuroKit error: read_acqknowledge(): couldn't find"                          " the following file: " + filename)

    # Read file
    file = bioread.read(filename)

    # Get desired frequency
    if sampling_rate == "max":
        freq_list = []
        for channel in file.named_channels:
            freq_list.append(file.named_channels[channel].samples_per_second)
        sampling_rate = np.max(freq_list)

    # Loop through channels
    data = {}
    for channel in file.named_channels:
        signal = np.array(file.named_channels[channel].data)

        # Fill signal interruptions
        if impute_missing is True:
            if np.isnan(np.sum(signal)):
                signal = pd.Series(signal).fillna(method="pad").values

        # Resample if necessary
        if file.named_channels[channel].samples_per_second != sampling_rate:
            signal = signal_resample(signal,
                                     sampling_rate=file.named_channels[channel].samples_per_second,
                                     desired_sampling_rate=sampling_rate,
                                     method=resample_method)
        data[channel] = signal

    # Sanitize lengths
    lengths = []
    for channel in data:
        lengths += [len(data[channel])]
    if len(set(lengths)) > 1:  # If different lengths
        length = pd.Series(lengths).mode()[0]  # Find most common (target length)
        for channel in data:
            if len(data[channel]) > length:
                data[channel] = data[channel][0:length]
            if len(data[channel]) < length:
                data[channel] = np.concatenate([data[channel],
                                                np.full((length - len(data[channel])), data[channel][-1])])

    # Final dataframe
    df = pd.DataFrame(data)
    return df, sampling_rate

def getFiles(SUBJECTS):
    stress=[]
    control=[]
    #input_dir='../Dataset/Human-Activity-Input/WISDM_ar_v1.1_raw.txt'
    ROOT_PATH='../Dataset/Mental-Stress-Input/'
    for subject in SUBJECTS:
        for file in os.listdir(ROOT_PATH+"S"+str(subject)):
            if file.endswith(".acq"):
                temp=os.path.join(ROOT_PATH+"S"+str(subject), file)
            elif file.endswith(".ACQ"):
                temp=os.path.join(ROOT_PATH+"S"+str(subject), file)
            if('C'+str(subject) in temp):
                control.append(temp)
            else:
                stress.append(temp)
    return control,stress

#upload the data to Redis
def load_data(filename, chunksize):
    SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                31, 32, 33, 34, 35, 36, 36, 38, 39, 40, 41, 42, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                60]

    controlPath,stressPath=getFiles(SUBJECTS)
    print(controlPath)
    print(stressPath)

    global max_data
    global min_data
    min_data=0
    # upload it to redis
    i=min_data
    max_data=57
    while (i<max_data):
        #df_user = df[df['user_id'] == i]
        C=load_control(controlPath[i])
        S=load_stress(stressPath[i])
        temp=[C,S]
        r.rpush(Topic["input_mental_stress_app"], pickle.dumps(temp))
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
    input_dir='../Dataset/Mental-Stress-Input/'
    output_dir='../Results/CSV/Mental-Stress-App/Execution_Time_Mental_Stress_App.csv'
    global max_data
    global min_data
    #user controller starts
    print("Hello User! I am MR. Packetized Computation! There is your option: ")
    print("Cleaning input Started!")
    Cleaning(Topic["input_mental_stress_app"])
    #CleaningModel(Topic["model_human_activity_app"])
    print("Taking Break for "+str(sleep_time)+" sec!")

    filename = input_dir
    print("Loading Mental Stress Data from Dataset: " + filename)
    upload_time=time.time()
    load_data(filename, chunk_size)
    upload_time=time.time()-upload_time
    print("Uploading Time: "+str(upload_time))
    testing_size=12
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
                        'app': 'mental-stress-app',
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
                        CleaningModel(Topic["model_mental_stress_app"])
                        print("Taking Break for "+str(sleep_time)+" sec!")
                        time.sleep(sleep_time)
                        start = time.time()
                        # publish it to trigger DBController
                        train=0
                        if(threshold=='1500'):
                            train=30
                        else:
                            train=10
                        publish_redis(Topic["publish_mental_stress_app"], str(json.dumps({
                            "size": max_data-testing_size+1,
                            'app':'mental-stress-app',
                            "current":1,
                            "training":train,
                            "epoch":int(l),
                            "threshold": int(threshold)
                        })))
                        GetResult(Topic["result_mental_stress_app"])
                        end = time.time()
                        print("time: " + str(end - start))
                        #acc=Testing(1,30,max_data)
                        data.append([threshold,l,acc[1],acc[0],end-start+upload_time])
                        WriteCSV(output_dir, data)
                        print("done!")

                    publish_redis("MetricMonitor", str(json.dumps({
                        'app': 'mental-stress-app',
                        "type": 'end',
                        "size": l,
                        "threshold": float(threshold)
                    })))

        else:
            break

if __name__ == '__main__':
    UserInput()
