import json
import pickle
import base64
import time
import csv
import sys
import os

import numpy as np
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
    ROOT_PATH='../Dataset/Stress_clean/'
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

# def getfeatures(df):
#     features = [None for i in range(num_features)]
#     ecg = df['ECG - ECG100C']
#
#     ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=1000)
#     instant_peaks, rpeaks, = nk.ecg_peaks(ecg_cleaned, sampling_rate=1000)
#     rate = nk.ecg_rate(rpeaks, sampling_rate=1000, desired_length=len(ecg_cleaned))
#
#     # Prepare output
#     signals = pd.DataFrame({"ECG_Rate": rate})
#     ecg = df['ECG - ECG100C']
#     peaks, info = nk.ecg_peaks(ecg, sampling_rate=1000)
#     hrv_time = nk.hrv_time(peaks, sampling_rate=1000, show=False)
#     hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, show=False)
#
#     # ECG Heart Rate Statistics:
#     features[0] = signals['ECG_Rate'].mean()
#     features[1] = signals['ECG_Rate'].std()
#     features[2] = hrv_time['HRV_RMSSD'].to_string(header=None, index=None)
#     features[3] = hrv_time['HRV_MeanNN'].to_string(header=None, index=None)
#     features[4] = hrv_time['HRV_SDNN'].to_string(header=None, index=None)
#     features[5] = hrv_time['HRV_CVNN'].to_string(header=None, index=None)
#     features[6] = hrv_time['HRV_CVSD'].to_string(header=None, index=None)
#     features[7] = hrv_time['HRV_MedianNN'].to_string(header=None, index=None)
#     features[8] = hrv_time['HRV_MadNN'].to_string(header=None, index=None)
#     features[9] = hrv_time['HRV_MCVNN'].to_string(header=None, index=None)
#     features[10] = hrv_time['HRV_pNN50'].to_string(header=None, index=None)
#     features[11] = hrv_time['HRV_pNN20'].to_string(header=None, index=None)
#     features[12] = hrv_freq['HRV_HF'].to_string(header=None, index=None)
#     features[13] = hrv_freq['HRV_VHF'].to_string(header=None, index=None)
#     features[14] = hrv_freq['HRV_HFn'].to_string(header=None, index=None)
#
#     return features
#upload the data to Redis
def load_data(filename, chunknumber):
    controlPath,stressPath=getFiles([chunknumber])
    C=load_control(controlPath[0])
    S=load_stress(stressPath[0])
    temp=[C]
    r.rpush(Topic["input_stress_extraction_app"], pickle.dumps([C]))
    r.rpush(Topic["input_stress_extraction_app"], pickle.dumps([S]))
        # i+=1
    print("Uploading done!")

def UserInput():

    #control parameters
    chunk_size = 1
    input_dir='../Dataset/Stress_clean/'
    output_dir='../Results/CSV/Stress-Feature-Extraction/Execution_Time_Stress_Extraction_App.csv'
    global max_data
    global min_data
    #user controller starts
    print("Hello User! I am MR. Packetized Computation! There is your option: ")
    print("Cleaning input Started!")
    Cleaning(Topic["input_stress_extraction_app"])
    #CleaningModel(Topic["model_stress_extraction_app"])
    print("Taking Break for "+str(sleep_time)+" sec!")

    filename = input_dir
    print("Loading Mental Stress Data from Dataset: " + filename)

    testing_size=12
    #data_number=[1,2,3,4,5,6,7,8,12,13,14,17,20,21,22,23,24,25,26,28,29,30,31,32]
    data_number=[1]
    while (True):
        print("\n\n1. Feature Extraction\n2. Exit")
        d = input("Enter: ")
        #threshold = MicroLambda["short_lambda"]
        if (str(d) == "1"):
            #print("\n\n1. Epoch Size\n2. Exit")
            epoch_list=[1,2]
            #[4,3,2,1]
            for current_data in data_number:
                print("Cleaning input Started!")
                Cleaning(Topic["input_stress_extraction_app"])
                upload_time=time.time()
                load_data(filename, current_data)
                upload_time=time.time()-upload_time
                print("Uploading Time: "+str(upload_time))
                for threshold in MicroLambda["short_lambda"]:

                    print("threshold: "+str(threshold))
                    # publish_redis("MetricMonitor", str(json.dumps({
                    #     'app': 'stress-extraction-app',
                    #     "type": 'start',
                    #     "size": l,
                    #     "threshold": float(threshold)
                    # })))
                    for i in range(Iteration):
                        for l in epoch_list:
                            data=[]
                            overlap=4
                            print("Iteration: " + str(i + 1) + ", Total Iteration " + str(Iteration))
                            time.sleep(5)
                            print("Taking Break for "+str(sleep_time)+" sec!")
                            time.sleep(sleep_time)
                            print("Cleaning Model Started!")
                            CleaningModel(Topic["model_stress_extraction_app"])
                            Cleaning(Topic["feature_stress_extraction_app"])
                            print("Taking Break for "+str(sleep_time)+" sec!")
                            time.sleep(sleep_time)
                            start = time.time()
                            # publish it to trigger DBController
                            train=1
                            publish_redis(Topic["publish_stress_extraction_app"], str(json.dumps({
                                "size": train,
                                "status":"Pending",
                                'app':'stress-extraction-app',
                                "current":0,
                                "overlap":overlap,
                                "type":int(l),
                                "threshold": int(threshold)
                            })))
                            return_data=GetResult(Topic["result_stress_extraction_app"])
                            end = time.time()
                            print("time: " + str(end - start))
                            if(l==1):
                                data.append([threshold,120000/overlap,l,end-start+upload_time, upload_time,"Serial"])
                            if(l==2):
                                data.append([threshold,120000/overlap,l,end-start+upload_time, upload_time,"Parallel"])
                            WriteCSV(output_dir, data)
                            print("done!")

                    # publish_redis("MetricMonitor", str(json.dumps({
                    #     'app': 'stress-extraction-app',
                    #     "type": 'end',
                    #     "size": l,
                    #     "threshold": float(threshold)
                    # })))

        else:
            break

if __name__ == '__main__':
    UserInput()
