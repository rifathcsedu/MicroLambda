import numpy as np
import matplotlib.pyplot as plt
import sys
from pandas import DataFrame

sys.path.append('../../Config/')
sys.path.append('../../Class/')

from configuration import *

GraphData={1:{"x_label":"Number of Images","y_label":"Execution Time (seconds)"},2:{"x_label":"Epoch Size","y_label":"Execution Time (seconds)"},3:{"x_label":"Epoch Size","y_label":"Execution Time (seconds)"}}

def FaceDataProcessing(FileName):
    data=ReadCSV(FileName)
    df = DataFrame (data,columns=['Data_Size','Threshold','Time'])
    print (df)
    unique_threshold=df.Threshold.unique()

    Average_Data=[]
    for i in unique_threshold:
        print(i)
        df_list = df[df['Threshold'] == i]
        print(df_list)
        unique_size=df_list.Data_Size.unique()
        print (unique_size)
        for j in unique_size:
            df_list_size = df_list[df_list['Data_Size'] == j]
            print(df_list_size)
            print(df_list_size["Time"].mean())
            Average_Data.append([i,j,df_list_size["Time"].mean()])
    print(Average_Data)
    df_avg = DataFrame (Average_Data,columns=['Threshold','Data_Size','Time'])
    unique_threshold=df_avg.Threshold.unique()
    unique_threshold.sort()
    unique_threshold=unique_threshold.tolist()
    x =df_avg.Data_Size.unique()
    x.sort()
    x=x.tolist()
    print(x)
    line_list={}
    for th in unique_threshold:
        df_temp=df_avg[df_avg['Threshold'] == th]
        df_temp = df_temp.drop('Threshold', 1)
        df_temp=df_temp.sort_values(by=['Data_Size'])

        df_temp = df_temp.drop('Data_Size', 1)
        df_temp=df_temp["Time"].tolist()
        #print(df_temp)
        line_list[th]=df_temp
    print(line_list)
    unique_threshold_legend=[]
    for i in unique_threshold:
        if(i!=1500):
            unique_threshold_legend.append("Short lambda ("+str(i)+" secs)")
        else:
            unique_threshold_legend.append("Long lambda")
    return x,line_list,unique_threshold_legend

def HumanDataProcessing(FileName):
    data=ReadCSV(FileName)
    print(data)
    df = DataFrame (data,columns=['Threshold','Epoch','Accuracy','Loss','Time'])
    print (df)
    unique_threshold=df.Threshold.unique()

    Average_Data=[]
    for i in unique_threshold:
        print(i)
        df_list = df[df['Threshold'] == i]
        print(df_list)
        unique_size=df_list.Epoch.unique()
        print (unique_size)
        for j in unique_size:
            df_list_size = df_list[df_list['Epoch'] == j]
            print(df_list_size)
            print(df_list_size["Time"].mean())
            Average_Data.append([i,j,df_list_size["Time"].mean()])
    print(Average_Data)
    df_avg = DataFrame (Average_Data,columns=['Threshold','Epoch','Time'])
    unique_threshold=df_avg.Threshold.unique()
    unique_threshold.sort()
    unique_threshold=unique_threshold.tolist()
    x =df_avg.Epoch.unique()
    x.sort()
    x=x.tolist()
    print(x)
    line_list={}
    for th in unique_threshold:
        df_temp=df_avg[df_avg['Threshold'] == th]
        df_temp = df_temp.drop('Threshold', 1)
        df_temp=df_temp.sort_values(by=['Epoch'])

        df_temp = df_temp.drop('Epoch', 1)
        df_temp=df_temp["Time"].tolist()
        #print(df_temp)
        line_list[th]=df_temp
    print(line_list)
    unique_threshold_legend=[]
    for i in unique_threshold:
        if(i!=1500):
            unique_threshold_legend.append("Short lambda ("+str(i)+" secs)")
        else:
            unique_threshold_legend.append("Long lambda")
    return x,line_list,unique_threshold_legend

def GetDataForApp(app_name):
    if(app_name==1):
        return FaceDataProcessing("../CSV/Face-App/Execution_Time_Face_Recognition_App.csv")
    elif(app_name==2):
        return "../CSV/Air-Pollution-App/Execution_Time_Air_Pollution.csv"
    elif(app_name==3):
        return HumanDataProcessing("../CSV/Human-Activity-App/Execution_Time_Human_Activity_App.csv")


#main_function
app_name=input("1. Face App\n2. Air Pollution App\n3. Human Activity App\nSelect App: ")
app_name=int(app_name)

x_axis,y_line_list,unique_threshold_legend=GetDataForApp(app_name)

fig, ax = plt.subplots()
for i in y_line_list:
    print (i)
    ax.plot(x_axis,y_line_list[i])
plt.legend(unique_threshold_legend)
plt.xlabel(GraphData[app_name]["x_label"])
plt.ylabel(GraphData[app_name]["y_label"])
plt.show()
