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
from sklearn.metrics import accuracy_score
from RedisPubSub import *

num_features = 32

# Function to extract all features(HR,HRV,RR and RRV)
def getfeatures(df):
    features = [None for i in range(num_features)]
    #ecg = df['ECG, Y, RSPEC-R']
    rsp = df["RSP, X, RSPEC-R"]
    ecg=df['ECG, Y, RSPEC-R']
    ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=1000)
    instant_peaks, rpeaks, = nk.ecg_peaks(ecg_cleaned, sampling_rate=1000)
    rate = nk.ecg_rate(rpeaks, sampling_rate=1000, desired_length=len(ecg_cleaned))

    # Prepare output
    signals = pd.DataFrame({"ECG_Rate": rate})
    ecg=df['ECG, Y, RSPEC-R']
    peaks, info = nk.ecg_peaks(ecg, sampling_rate=1000)
    hrv_time = nk.hrv_time(peaks, sampling_rate=1000, show=False)
    hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, show=False)
    hrv_non = nk.hrv_nonlinear(peaks, sampling_rate=1000, show=False)
    rsp, info = nk.rsp_process(rsp)
    rrv = nk.rsp_rrv(rsp, show=False)
    ecg_signals, info = nk.ecg_process(df['ECG, Y, RSPEC-R'], sampling_rate=1000)
    rsp_signals, _ = nk.rsp_process(df["RSP, X, RSPEC-R"], sampling_rate=1000)
    hrv_rsa = nk.hrv_rsa(ecg_signals, rsp_signals, info, sampling_rate=1000, continuous=True)
    cleaned = nk.rsp_clean(df["RSP, X, RSPEC-R"], sampling_rate=1000)
    df, peaks_dict = nk.rsp_peaks(cleaned)
    info = nk.rsp_fixpeaks(peaks_dict)
    formatted = nk.signal_formatpeaks(info, desired_length=len(cleaned),peak_indices=info["RSP_Peaks"])
    rsp_rate = nk.rsp_rate(formatted, desired_length=None, sampling_rate=1000)

    # ECG Heart Rate Statistics:
    features[0] = signals['ECG_Rate'].mean()
    features[1] = signals['ECG_Rate'].std()
    features[2] = signals['ECG_Rate'].min()
    features[3] = signals['ECG_Rate'].max()
    features[4] = hrv_time['HRV_RMSSD'].to_string(header=None, index=None)
    features[5] = hrv_time['HRV_MeanNN'].to_string(header=None, index=None)
    features[6] = hrv_time['HRV_SDNN'].to_string(header=None, index=None)
    features[7] = hrv_time['HRV_CVNN'].to_string(header=None, index=None)
    features[8] = hrv_time['HRV_CVSD'].to_string(header=None, index=None)
    features[9] = hrv_time['HRV_MedianNN'].to_string(header=None, index=None)
    features[10]= hrv_time['HRV_MadNN'].to_string(header=None, index=None)
    features[11]= hrv_time['HRV_MCVNN'].to_string(header=None, index=None)
    features[12] = hrv_time['HRV_pNN50'].to_string(header=None, index=None)
    features[13] = hrv_time['HRV_pNN20'].to_string(header=None, index=None)
    features[14] = hrv_freq['HRV_HF'].to_string(header=None, index=None)
    features[15] = hrv_freq['HRV_VHF'].to_string(header=None, index=None)
    features[16] = hrv_freq['HRV_HFn'].to_string(header=None, index=None)
    features[17] = hrv_freq['HRV_LFHF'].to_string(header=None, index=None)
    features[18] = hrv_non['HRV_SD1'].to_string(header=None, index=None)
    features[19] = hrv_non['HRV_SD2'].to_string(header=None, index=None)
    features[20] = hrv_non['HRV_SD1SD2'].to_string(header=None, index=None)
    features[21] = rsp_rate.mean()
    features[22] = rsp_rate.std()
    features[23] = rsp_rate.min()
    features[24] = rsp_rate.max()
    features[25] = rrv['RRV_RMSSD'].to_string(header=None, index=None)
    features[26] = rrv['RRV_MeanBB'].to_string(header=None, index=None)
    features[27] = rrv['RRV_SDBB'].to_string(header=None, index=None)
    #features[28] = rrv['RRV_LF'].to_string(header=None, index=None)
    #features[29] = rrv['RRV_HF'].to_string(header=None, index=None)
    #features[30] = rrv['RRV_LFHF'].to_string(header=None, index=None)
    features[28] = rrv['RRV_SD1'].to_string(header=None, index=None)
    features[29] = rrv['RRV_SD2'].to_string(header=None, index=None)
    features[30] = rrv['RRV_SD2SD1'].to_string(header=None, index=None)
    features[31] = hrv_rsa['RSA_P2T'].mean()
    return features

columns=['HR_mean','HR_std','HR_min','HR_max','HRV_RMSSD','HRV_MeanNN','HRV_SDNN','HRV_CVNN',
                                                                'HRV_CVSD','HRV_MedianNN','HRV_MadNN','HRV_MCVNN','HRV_pNN50','HRV_pNN20'
                                                                ,'HRV_HF','HRV_VHF','HRV_HFn','HRV_LFHF','HRV_SD1','HRV_SD2','HRV_SD1SD2'
                                                                ,'RSP_mean','RSP_std','RSP_min','RSP_max','RRV_RMSSD','RRV_MeanBB','RRV_SDBB'
                                                                ,'RRV_SD1','RRV_SD2','RRV_SD2SD1','RSA','label']
def create_features(window_size, df, type, feature):

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
            publish_redis("test", "feature extraction: window= ")
            curr_end = curr_start + window_size
            features = getfeatures(df[curr_start:curr_end])
            feature_array.append(features)
            curr_start = curr_start + window_size/4
            curr_start=int(curr_start)
            analyzed= pd.DataFrame(data=feature_array, columns=columns[:len(columns)-1])
            if(type==3 or type==4):
                break
    publish_redis("test", "feature extraction: done= ")
    return analyzed

def get_stdin():
    buf = ""
    for line in sys.stdin:
        buf = buf + line
    return buf

def handle (req):
    publish_redis("test", "Analysis started!!!")
    start = time.time()
    print(req)
    json_req = json.loads(req)
    print(json_req)
    current = json_req["current"]
    threshold=int(json_req["threshold"])
    size=int(json_req["size"])
    type=int(json_req["type"])
    feature=-1
    if(type==2 or type==4):
        feature=int(json_req["feature"])
    # data_list=[1,2,3,4,5,6,7,8,12,13,14,17,20,21,22,23,24,25,26,28,29,30,31,32]
    control=[]
    stress=[]
    # while(time.time()-start<threshold and current<size):
    #     publish_redis("test", "New loop!!! current= "+str(current))
    i=current
    if(type==1 or type==3):
        temp=LoadData(Topic["input_stress_extraction_app"], current, current)
        publish_redis("test", str(current)+": Calm data")
        loaded_data = pickle.loads(temp[0])
        control.append(loaded_data[0])
        temp=LoadData(Topic["input_stress_extraction_app"], current+1, current+1)
        publish_redis("test", str(current)+": Stress data")
        loaded_data = pickle.loads(temp[0])
        #control.append(loaded_data[0])
        stress.append(loaded_data[0])

        data = control[:]
        P1C = data[0]
        publish_redis("test", "feature extraction: calm data")
        P1C = create_features(120000, P1C,type,feature)
        P1C = P1C.apply(pd.to_numeric)
        data = stress[:]
        P1S = data[0]
        publish_redis("test", "feature extraction: stress data")
        P1S = create_features(120000, P1S,type,feature)
        P1S = P1S.apply(pd.to_numeric)

    elif(type==2 or type==4):
        if(feature==0):
            temp=LoadData(Topic["input_stress_extraction_app"], current, current)
            publish_redis("test", str(current)+": Calm data")
            loaded_data = pickle.loads(temp[0])
            control.append(loaded_data[0])

            data = control[:]
            P1C = data[0]
            P1C = create_features(120000, P1C,type,feature)
            P1C = P1C.apply(pd.to_numeric)
        else:
            temp=LoadData(Topic["input_stress_extraction_app"], current+1, current+1)
            publish_redis("test", str(current)+": Stress data")
            loaded_data = pickle.loads(temp[0])
            #control.append(loaded_data[0])
            stress.append(loaded_data[0])

            data = stress[:]
            P1S = data[0]
            P1S = create_features(120000, P1S,type,feature)
            P1S = P1S.apply(pd.to_numeric)

        #rB,_=df_base.shape
        # rC,_=P1C.shape
        # rS,_=P1S.shape
        # #y1=[0] * rB
        # y1=[0] * rC
        # y2=[1] * rS
        # #df_base['label']=y1
        # P1C['label']=y1
        # P1S['label']=y2
        # S1=pd.concat([P1C,P1S], ignore_index=True)
        # for f  in columns:
        #     max_feature = S1[f].max()
        #     S1[f]= S1[f]/max_feature
        # P1= S1[columns]
        # P1['label'] = P1['label'].astype(int)

    json_req["current"]=current
    publish_redis("test", json.dumps({"data":"done"}))
    return publish_redis(Topic["publish_stress_extraction_app"],json.dumps(json_req))

if __name__ == "__main__":
    st = get_stdin()
    handle(st)
