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
import datetime
import neurokit2 as nk
import contextlib
import io
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
sys.path.append('../Config/')
sys.path.append('../Class/')

from RedisPubSub import *
from configuration import *
from HumanActivity import *
min_data=0
max_data=0
num_features = 15
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

def getfeatures(df):
    features = [None for i in range(num_features)]
    ecg = df['ECG - ECG100C']

    ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=1000)
    instant_peaks, rpeaks, = nk.ecg_peaks(ecg_cleaned, sampling_rate=1000)
    rate = nk.ecg_rate(rpeaks, sampling_rate=1000, desired_length=len(ecg_cleaned))

    # Prepare output
    signals = pd.DataFrame({"ECG_Rate": rate})
    ecg = df['ECG - ECG100C']
    peaks, info = nk.ecg_peaks(ecg, sampling_rate=1000)
    hrv_time = nk.hrv_time(peaks, sampling_rate=1000, show=False)
    hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, show=False)

    # ECG Heart Rate Statistics:
    features[0] = signals['ECG_Rate'].mean()
    features[1] = signals['ECG_Rate'].std()
    features[2] = hrv_time['HRV_RMSSD'].to_string(header=None, index=None)
    features[3] = hrv_time['HRV_MeanNN'].to_string(header=None, index=None)
    features[4] = hrv_time['HRV_SDNN'].to_string(header=None, index=None)
    features[5] = hrv_time['HRV_CVNN'].to_string(header=None, index=None)
    features[6] = hrv_time['HRV_CVSD'].to_string(header=None, index=None)
    features[7] = hrv_time['HRV_MedianNN'].to_string(header=None, index=None)
    features[8] = hrv_time['HRV_MadNN'].to_string(header=None, index=None)
    features[9] = hrv_time['HRV_MCVNN'].to_string(header=None, index=None)
    features[10] = hrv_time['HRV_pNN50'].to_string(header=None, index=None)
    features[11] = hrv_time['HRV_pNN20'].to_string(header=None, index=None)
    features[12] = hrv_freq['HRV_HF'].to_string(header=None, index=None)
    features[13] = hrv_freq['HRV_VHF'].to_string(header=None, index=None)
    features[14] = hrv_freq['HRV_HFn'].to_string(header=None, index=None)

    return features
#upload the data to Redis
def load_data(filename, chunksize):
    SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                31, 32, 33, 34, 35, 36, 36, 38, 39, 40, 41, 42, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                60]

    #S2, S4, S5, S6, S7, S8, S9, S10, S11, S12, S12, S14, S16, S19, S20,
    # S21, S22, S23, S24, S25, S26, S27, S28, S29, S30, S31, S34, S35, S37, S38, S39, S40,
    # S41, S42, S43, S44, S45, S46, S47, S48, S52, S53, S54, S55, S56
    controlPath,stressPath=getFiles(SUBJECTS)
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

def Testing():
    print("Testing Starts")
    #testing check (will remove)
    testing_set=[0,1,3,15,17,18,32,33,36,49,50,51]
    num_features = 15
    i=0
    control=[]
    stress=[]
    while(i<len(testing_set)):
        temp=LoadData(Topic["input_mental_stress_app"], testing_set[i], testing_set[i])
        #print(pickle.loads(temp[0]))
        print(testing_set[i])
        loaded_data = pickle.loads(temp[0])
        control.append(loaded_data[0])
        stress.append(loaded_data[1])
        i+=1

    control_data = []
    stress_data = []
    for i in range(len(control)):
        publish_redis("test", "feature i= " + str(i))
        control1 = create_features(60000, control[i])
        control_data.append(control1)
        stress1 = create_features(60000, stress[i])
        stress_data.append(stress1)

    control_data = pd.concat(control_data)
    control_data = control_data.apply(pd.to_numeric)

    stress_data = pd.concat(stress_data)
    stress_data = stress_data.apply(pd.to_numeric)

    columns = ['HR_mean', 'HR_std', 'RMSSD', 'meanNN', 'HF', 'HFn']

    for f in columns:
        max_feature = control_data[f].max()
        control_data[f] = control_data[f] / max_feature
    df_con = control_data[columns]

    dfs = np.array_split(df_con, len(testing_set))

    dfs = np.split(df_con, [5], axis=0)

    for f in columns:
        max_feature = stress_data[f].max()
        stress_data[f] = stress_data[f] / max_feature
    df_str = stress_data[columns]

    dfs1 = np.array_split(df_str, len(testing_set))

    # rB,_=df_base.shape
    rC, _ = df_con.shape
    rS, _ = df_str.shape

    # y1=[0] * rB
    y2 = [0] * rC
    y3 = [1] * rS

    df_con['label'] = y2
    df_str['label'] = y3


    dfs1 = np.array_split(df_str, len(testing_set))
    dfs = np.array_split(df_con, len(testing_set))

    S=[]
    for i in range (len(testing_set)):
        S0 = pd.concat([dfs[i], dfs1[i]], ignore_index=True)
        S.append(S0)


    X1 = pd.concat(S, ignore_index=True)
    # X_test = X2[columns]
    # y_test = X2['label']
    X_test=X1[columns]
    y_test=X1['label']
    model=pickle.loads(RedisLoadModel(Topic["model_mental_stress_app"]))
    y_pred = model.predict(X_test)
    #print(y_predict_ann)
    # We can now compare the "predicted labels" for the Testing Set with its "actual labels" to evaluate the accuracy
    score_ann = accuracy_score(y_test, y_pred)
    print(score_ann)
    print("Testing Done!!!")
    return score_ann
    #y_pred=model.predict(X_test)
    #print(y_pred)
def create_features(window_size, df):
    @contextlib.contextmanager
    def nostdout():
        save_stdout = sys.stdout
        sys.stdout = io.BytesIO()
        yield
        sys.stdout = save_stdout

    with nostdout():
        time_start_features = time.time()
        feature_array = []
        total_datapoints = len(df.index)
        curr_start = 0
        while curr_start + window_size <= total_datapoints:
            curr_end = curr_start + window_size
            features = getfeatures(df[curr_start:curr_end])
            feature_array.append(features)
            curr_start = curr_start + window_size
            curr_start = int(curr_start)

            analyzed = pd.DataFrame(data=feature_array, columns=["HR_mean", "HR_std", 'RMSSD', 'meanNN', 'sdNN',
                                                                 'cvNN', 'CVSD', 'medianNN', 'madNN', 'mcvNN', 'pNN50',
                                                                 'pNN20', 'HF', 'VHF', 'HFn'])
    return analyzed
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
    #CleaningModel(Topic["model_mental_stress_app"])
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
            epoch_list=[50000]
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
                            train=45
                        else:
                            train=15
                        testing_size=12
                        publish_redis(Topic["publish_mental_stress_app"], str(json.dumps({
                            "size": max_data-testing_size,
                            'app':'mental-stress-app',
                            "current":0,
                            "training":train,
                            "epoch":int(l),
                            "threshold": int(threshold)
                        })))
                        GetResult(Topic["result_mental_stress_app"])
                        end = time.time()
                        print("time: " + str(end - start))
                        acc=Testing()
                        data.append([threshold,l,acc,end-start+upload_time, upload_time])
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
