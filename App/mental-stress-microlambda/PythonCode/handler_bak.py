import json
import pickle
import os
import datetime
import neurokit2 as nk
import contextlib
import io
import sys
import time
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from RedisPubSub import *
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def get_stdin():
    buf = ""
    for line in sys.stdin:
        buf = buf + line
    return buf

num_features = 15


# Function to extract all features(HR and HRV)
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
def Testing(model, overlap):
    print("Testing Starts")
    #testing check (will remove)
    testing_set=[0,1,3,15,17,18,32,33,36,49,50,51]
    #num_features = 15
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
        control1 = create_features(60000, control[i],overlap)
        control_data.append(control1)
        stress1 = create_features(60000, stress[i],overlap)
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
    #model=pickle.loads(RedisLoadModel(Topic["model_mental_stress_app"]))
    y_pred = model.predict(X_test)
    #print(y_predict_ann)
    # We can now compare the "predicted labels" for the Testing Set with its "actual labels" to evaluate the accuracy
    score_ann = accuracy_score(y_test, y_pred)
    publish_redis("test", "Score: "+str(score_ann))
    print(score_ann)
    print("Testing Done!!!")
    return score_ann
def create_features(window_size, df,overlap):
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
            curr_start = curr_start + window_size/overlap
            curr_start = int(curr_start)

            analyzed = pd.DataFrame(data=feature_array, columns=["HR_mean", "HR_std", 'RMSSD', 'meanNN', 'sdNN',
                                                                 'cvNN', 'CVSD', 'medianNN', 'madNN', 'mcvNN', 'pNN50',
                                                                 'pNN20', 'HF', 'VHF', 'HFn'])
    return analyzed

def handle (req):
    print("Start!!")
    #CleaningModel(Topic["model_mental_stress_app"])
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
    training_set=[2,4,5,6,7,8,9,10,11,12,12,14,16,19,20,21,22,23,24,25,26,27,28,29,30,31,34,35,37,38,39,40,41,42,43,44,45,46,47,48,52,53,54,55,56]
    #training_set=[3,5,6,7,8,9,10,12,13,14,14,16,18,21,22,23,24,25,26,27,28,29,30,31,32,33,36,36,39,40,41,42,45,46,47,48,49,50,51,52,56,57,58,59,60]


    print(len(training_set))
    print("loop starts")

    while(time.time()-start<threshold and current<size):
        publish_redis("test", "New loop!!! current= "+str(current))
        control=[]
        stress=[]
        i=current
        while(i<current+training_size and i<size):

            temp=LoadData(Topic["input_mental_stress_app"], training_set[i], training_set[i])
            #print(pickle.loads(temp[0]))
            #print(training_set[i])
            publish_redis("test", training_set[i])
            loaded_data = pickle.loads(temp[0])
            control.append(loaded_data[0])
            stress.append(loaded_data[1])
            i+=1

        current=i
        publish_redis("test", "current="+str(current))
        S_load=None
        control_data = []
        stress_data = []
        temp=RedisLoadModel(Topic["model_mental_stress_app"])
        if(temp==None):
            S_load=[]
            publish_redis("test", "empty data")
        else:
            S=pickle.loads(temp)
            control_data = S_load[0]
            stress_data = S_load[1]
            publish_redis("test", "loaded data")

        for i in range(0,len(control)):
            publish_redis("test", "feature i= " + str(i))
            control1 = create_features(60000, control[i],epoch)
            control_data.append(control1)
            stress1 = create_features(60000, stress[i],epoch)
            stress_data.append(stress1)
        publish_redis("test", "control size = " + str(len(control_data)))
        if(current==size):
            publish_redis("test","S is "+str(len(control_data)))
            #print("S is "+str(len(control_data)))
            control_data = pd.concat(control_data)
            control_data = control_data.apply(pd.to_numeric)
            stress_data = pd.concat(stress_data)
            stress_data = stress_data.apply(pd.to_numeric)

            columns = ['HR_mean', 'HR_std', 'RMSSD', 'meanNN', 'HF', 'HFn']

            for f in columns:
                max_feature = control_data[f].max()
                control_data[f] = control_data[f] / max_feature
            df_con = control_data[columns]

            dfs = np.array_split(df_con, len(training_set))

            dfs = np.split(df_con, [5], axis=0)

            for f in columns:
                max_feature = stress_data[f].max()
                stress_data[f] = stress_data[f] / max_feature
            df_str = stress_data[columns]
            dfs1 = np.array_split(df_str, len(training_set))

            # rB,_=df_base.shape
            rC, _ = df_con.shape
            rS, _ = df_str.shape

            # y1=[0] * rB
            y2 = [0] * rC
            y3 = [1] * rS

            df_con['label'] = y2
            df_str['label'] = y3


            dfs1 = np.array_split(df_str, len(training_set))
            dfs = np.array_split(df_con, len(training_set))
            S=[]
            for i in range (len(training_set)):
                S0 = pd.concat([dfs[i], dfs1[i]], ignore_index=True)
                S.append(S0)

            X1 = pd.concat(S, ignore_index=True)
            publish_redis("test", "Size of S="+str(len(S)))
            #X2 = pd.concat([S0, S1, S3, S15, S17, S18, S32, S33, S36, S49, S50, S51])

            X_train = X1[columns]
            y_train = X1["label"]
            # X_test = X2[columns]
            # y_test = X2['label']
            start=time.time()
            publish_redis("test", "training starts")
            model = MLPClassifier(hidden_layer_sizes=(4,), activation='identity',
                                       solver='lbfgs', alpha=0.1, random_state=1,
                                       learning_rate='adaptive', momentum=0.3,
                                       learning_rate_init=0.1, max_iter=100, batch_size=16)

            publish_redis("test", "New Model created!!!")
            model.fit(X_train, y_train)

            publish_redis("test","Training Done!!!")
            publish_redis("test", "Saving model starts...!!")
            Testing(mode,epoch)
            RedisSaveModel(Topic['model_mental_stress_app'], pickle.dumps(model))

            #print("Saving model done...!!")
            publish_redis("test", "Saving model done...!!")
        else:
            print("Size of S"+str(len(control_data)))
            publish_redis("test", "Intermediate state save starts")
            RedisSaveModel(Topic['model_mental_stress_app'], pickle.dumps([control_data,stress_data]))
            publish_redis("test", "Intermediate state save done!!")

    json_req["current"]=current
    publish_redis("test", json.dumps({"data":"done"}))
    return publish_redis(Topic["publish_mental_stress_app"],json.dumps(json_req))

if __name__ == "__main__":
    st = get_stdin()
    handle(st)
