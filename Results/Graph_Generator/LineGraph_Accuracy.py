import numpy as np
import matplotlib.pyplot as plt
import sys
from pandas import DataFrame

sys.path.append('../../Config/')
sys.path.append('../../Class/')

from configuration import *

GraphData={1:{1:{"x_label":"Number of Images","y_label":"Execution Time (sec)","title":"Execution Time for Face Recognition Application"}},2:{1:{"x_label":"Epoch Size","y_label":"Execution Time (sec)","title":"Execution Time for Air Pollution Training"},2:{"x_label":"Epoch Size","y_label":"RMSE","title":"RMSE for Air Pollution Training"}},3:{1:{"x_label":"Epoch Size","y_label":"Execution Time (sec)","title":"Execution Time for Human Activity Classification Training"},2:{"x_label":"Epoch Size","y_label":"Accuracy (%)","title":"Accuracy rate for Human Activity Classification Training"},3:{"x_label":"Epoch Size","y_label":"Error (%)","title":"Error rate for Human Activity Classification Training"}}}

def AccuracyGraph(FileName):
    data=ReadCSV(FileName)
    if(len(data[0])>3):
        len_data=len(data[0])-3
        for j in data:
            for k in range(len_data):
                del j[3]
    print(data)
    df = DataFrame (data,columns=['Threshold','Epoch','Accuracy'])
    #df= df.drop('Time', 1)
    df = df.sort_values(["Threshold", "Epoch"], ascending=(True, True))
    print (df)
    unique_threshold=df.Threshold.unique()

    Average_Data=[]
    for i in unique_threshold:
        #print(i)
        df_list = df[df['Threshold'] == i]
        #print(df_list)
        unique_size=df_list.Epoch.unique()
        #print (unique_size)
        for j in unique_size:
            df_list_size = df_list[df_list['Epoch'] == j]
            #print(df_list_size)
            #print(df_list_size["Accuracy"].mean())
            Average_Data.append([i,j,df_list_size["Accuracy"].mean()])
    print(Average_Data)
    df_avg = DataFrame (Average_Data,columns=['Threshold','Epoch','Accuracy'])
    df_avg=df_avg.sort_values(["Threshold", "Epoch"], ascending=(True, True))
    print(df_avg)
    unique_threshold=df_avg.Threshold.unique()
    unique_threshold.sort()
    unique_threshold=unique_threshold.tolist()
    x =df_avg.Epoch.unique()
    x.sort()
    x=x.tolist()
    #print(x)
    line_list={}
    for th in unique_threshold:
        df_temp=df_avg[df_avg['Threshold'] == th]
        df_temp = df_temp.drop('Threshold', 1)
        #df_temp=df_temp.sort_values(by=['Epoch'])
        df_temp = df_temp.drop('Epoch', 1)
        df_temp=df_temp["Accuracy"].tolist()
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


def ErrorGraph(FileName):
    data=ReadCSV(FileName)
    len_data=len(data[0])
    if(len(data[0])>3):
        len_data=len(data[0])-3
        for j in data:
            #del j[len_data-1]
            del j[2]
            del j[3]
    print(data)
    df = DataFrame (data,columns=['Threshold','Epoch','Accuracy'])
    #df= df.drop('Time', 1)
    df = df.sort_values(["Threshold", "Epoch"], ascending=(True, True))
    print (df)
    unique_threshold=df.Threshold.unique()

    Average_Data=[]
    for i in unique_threshold:
        #print(i)
        df_list = df[df['Threshold'] == i]
        #print(df_list)
        unique_size=df_list.Epoch.unique()
        #print (unique_size)
        for j in unique_size:
            df_list_size = df_list[df_list['Epoch'] == j]
            #print(df_list_size)
            #print(df_list_size["Accuracy"].mean())
            Average_Data.append([i,j,df_list_size["Accuracy"].mean()])
    print(Average_Data)
    df_avg = DataFrame (Average_Data,columns=['Threshold','Epoch','Accuracy'])
    df_avg=df_avg.sort_values(["Threshold", "Epoch"], ascending=(True, True))
    print(df_avg)
    unique_threshold=df_avg.Threshold.unique()
    unique_threshold.sort()
    unique_threshold=unique_threshold.tolist()
    x =df_avg.Epoch.unique()
    x.sort()
    x=x.tolist()
    #print(x)
    line_list={}
    for th in unique_threshold:
        df_temp=df_avg[df_avg['Threshold'] == th]
        df_temp = df_temp.drop('Threshold', 1)
        #df_temp=df_temp.sort_values(by=['Epoch'])
        df_temp = df_temp.drop('Epoch', 1)
        df_temp=df_temp["Accuracy"].tolist()
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


def TimeGraph(FileName):
    data=ReadCSV(FileName)
    if (len(data[0]) > 3):
        len_data = len(data[0]) - 3
        for j in data:
            for k in range(len_data):
                del j[2]
    print(data)
    #print(data)
    df = DataFrame (data,columns=['Threshold','Epoch','Time'])
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



def GetDataForApp(app_name,type):
    if(app_name==1 and type==1):
        return TimeGraph("../CSV/Face-App/Execution_Time_Face_Recognition_App.csv")
    elif(app_name==3 and type==1):
        return TimeGraph("../CSV/Human-Activity-App/Execution_Time_Human_Activity_App.csv")
    elif (app_name == 3 and type == 2):
        return AccuracyGraph("../CSV/Human-Activity-App/Execution_Time_Human_Activity_App.csv")
    elif (app_name == 3 and type == 3):
        return ErrorGraph("../CSV/Human-Activity-App/Execution_Time_Human_Activity_App.csv")
    elif (app_name==2 and type==1):
        return TimeGraph("../CSV/Air-Pollution-App/Execution_Time_Air_Pollution.csv")
    elif (app_name == 2 and type == 2):
        return AccuracyGraph("../CSV/Air-Pollution-App/Execution_Time_Air_Pollution.csv")


#main_function
app_name=input("1. Face Recognition App\n2. Air Pollution App\n3. Human Activity App\nSelect App: ")
app_name=int(app_name)

type=input("1. Execution Time\n2. Accuracy/RMSE\n3. Error\nSelect Type: ")
type=int(type)

if((app_name==1 and (type==2 or type==3)) or (app_name==2 and type==3)):
    print ("Invalid Input! No graph for Air Pollution App Error/ Face Recognition Accuracy/Error graphs....!!!!\nTerminating the program..!!!\nRun Again!!!\n\n")
else:
    x_axis,y_line_list,unique_threshold_legend=GetDataForApp(app_name,type)

    fig, ax = plt.subplots()
    color_data=0.9
    for i in y_line_list:
        print (i)
        if(len(x_axis)==len(y_line_list[i])):
            ax.plot(x_axis,y_line_list[i],c=str(color_data))
        else:
            ax.plot(x_axis[:len(y_line_list[i])], y_line_list[i],c=str(color_data))
        color_data-=.15
    plt.title(GraphData[app_name][type]["title"])
    plt.legend(unique_threshold_legend)
    plt.xlabel(GraphData[app_name][type]["x_label"])
    plt.ylabel(GraphData[app_name][type]["y_label"])
    plt.show()
