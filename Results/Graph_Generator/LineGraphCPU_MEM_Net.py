import numpy as np
import matplotlib.pyplot as plt
import sys
from pandas import DataFrame
import glob
import re
sys.path.append('../../Config/')
sys.path.append('../../Class/')

from configuration import *

GraphData={
1:{1:{"x_label":"Number of Images","y_label":"CPU USAGE (%)","title":"CPU USAGE for Face Recognition Application"},2:{"x_label":"Number of Images","y_label":"Memory USAGE (%)","title":"Memory USAGE for Face Recognition Application"},3:{"x_label":"Number of Images","y_label":"Network USAGE (byte)","title":"Network USAGE for Face Recognition Application"}},
2:{1:{"x_label":"Epoch Size","y_label":"CPU USAGE (%)","title":"CPU USAGE for Air Pollution Prediction Application"},2:{"x_label":"Epoch Size","y_label":"Memory USAGE (%)","title":"Memory USAGE for Air Pollution Prediction Application"},3:{"x_label":"Epoch Size","y_label":"Network USAGE (byte)","title":"Network USAGE for Air Pollution Prediction Application"}},
3:{1:{"x_label":"Epoch Size","y_label":"CPU USAGE (%)","title":"CPU USAGE for Human Pollution Prediction Application"},2:{"x_label":"Epoch Size","y_label":"Memory USAGE (%)","title":"Memory USAGE for Air Pollution Prediction Application"},3:{"x_label":"Epoch Size","y_label":"Network USAGE (byte)","title":"Network USAGE for Air Pollution Prediction Application"}},
4:{1:{"x_label":"Epoch Size","y_label":"CPU USAGE (%)","title":"CPU USAGE for Human Pollution Prediction Application"},2:{"x_label":"Epoch Size","y_label":"Memory USAGE (%)","title":"Memory USAGE for Air Pollution Prediction Application"},3:{"x_label":"Epoch Size","y_label":"Network USAGE (byte)","title":"Network USAGE for Air Pollution Prediction Application"}},
}
App={1:"Face Recognition Application",3:"Human Activity Classification Application", 2:"Air Pollution Prediction Application", 4:"Mental Stress Classification Application"}
Label={1:"CPU Usage", 2: "Memory Usage", 3: "Network Usage"}
def GraphData(app_name,type,label):
    if(label=="y_label"):
        return Label[type]
    if(label=="title"):
        return Label[type]+" for "+ App[app_name]
    if(label=="x_label"):
        if(app_name==1):
            return "Number of Images"
        if(app_name==2 or app_name==3):
            return "Epoch Size"
        if(app_name==4):
            return "Threshold"
    if(label=="filename"):
        if(app_name==1 and type==1):
            return "face_app_cpu.pdf"
        if(app_name==1 and type==2):
            return "face_app_memory.pdf"
        if(app_name==1 and type==3):
            return "face_app_network.pdf"
        if(app_name==2 and type==1):
            return "air_app_cpu.pdf"
        if(app_name==2 and type==2):
            return "air_app_memory.pdf"
        if(app_name==2 and type==3):
            return "air_app_network.pdf"
        if(app_name==3 and type==1):
            return "human_app_cpu.pdf"
        if(app_name==3 and type==2):
            return "human_app_memory.pdf"
        if(app_name==3 and type==3):
            return "human_app_network.pdf"
        if(app_name==4 and type==1):
            return "mental_app_cpu.pdf"
        if(app_name==4 and type==2):
            return "mental_app_memory.pdf"
        if(app_name==4 and type==3):
            return "mental_app_network.pdf"


def ConvertFloat(s):
    try:
        return float(s)
    except ValueError:
        return -1

def CPUGraph(path, app_name):
    filelist=glob.glob(path+"*.csv")
    all_data=[]
    for file in filelist:
        data=ReadCSV(file)
        data_new=[]
        for dt in data:
            temp=[]
            flag=0
            for i in dt:
                z=ConvertFloat(i)
                if(z!=-1):
                    temp.append(z)
                else:
                    flag=1
                    break
            if(flag==0):
                data_new.append(temp)

        df = DataFrame (data_new,columns=['CPU','Memory'])
        df_list = df[df['CPU']>25]
        #print(df_list)
        temp=file.split("/")
        temp=temp[len(temp)-1]
        temp = re.findall(r'\d+', temp)
        res = list(map(int, temp))
        while(0 in res):
            res.remove(0)
        res.append(df_list['CPU'].mean())
        all_data.append(res)

    print(all_data)
    df_avg = DataFrame (all_data,columns=['Epoch','Threshold','CPU'])
    df_avg=df_avg.sort_values(["Epoch", "Threshold"], ascending=(True, True))
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
        df_temp=df_temp["CPU"].tolist()
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

def CPUGraphMental(path, app_name):
    filelist=glob.glob(path+"*.csv")
    all_data=[]
    for file in filelist:
        data=ReadCSV(file)
        data_new=[]
        for dt in data:
            temp=[]
            flag=0
            for i in dt:
                z=ConvertFloat(i)
                if(z!=-1):
                    temp.append(z)
                else:
                    flag=1
                    break
            if(flag==0):
                data_new.append(temp)

        df = DataFrame (data_new,columns=['CPU','Memory'])
        df_list = df[df['CPU']>25]
        #print(df_list)
        temp=file.split("/")
        temp=temp[len(temp)-1]
        temp = re.findall(r'\d+', temp)
        res = list(map(int, temp))
        while(0 in res):
            res.remove(0)
        res.append(df_list['CPU'].mean())
        res.append(df_list['CPU'].std())
        all_data.append(res)

    print(all_data)
    df_avg = DataFrame (all_data,columns=['Threshold','Epoch','CPU','STD'])
    df_avg=df_avg.sort_values(["Epoch", "Threshold"], ascending=(True, True))
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
        df_temp1=df_temp["CPU"].tolist()
        df_temp2=df_temp["STD"].tolist()
        #print(df_temp)
        line_list[th]=[df_temp1,df_temp2]
    print(line_list)
    unique_threshold_legend=[]
    for i in unique_threshold:
        if(i!=1500):
            unique_threshold_legend.append("Short lambda ("+str(i)+" secs)")
        else:
            unique_threshold_legend.append("Long lambda")
    return x,line_list,unique_threshold_legend

def MemoryGraph(path, app_name):
    filelist=glob.glob(path+"*.csv")
    all_data=[]
    for file in filelist:
        data=ReadCSV(file)
        data_new=[]
        for dt in data:
            temp=[]
            flag=0
            for i in dt:
                z=ConvertFloat(i)
                if(z!=-1):
                    temp.append(z)
                else:
                    flag=1
                    break
            if(flag==0):
                data_new.append(temp)

        df = DataFrame (data_new,columns=['CPU','Memory'])
        df_list = df[df['Memory']>1]
        #print(df_list)
        temp=file.split("/")
        temp=temp[len(temp)-1]
        temp = re.findall(r'\d+', temp)
        res = list(map(int, temp))
        while(0 in res):
            res.remove(0)
        res.append(df_list['Memory'].mean())
        all_data.append(res)

    print(all_data)
    df_avg = DataFrame (all_data,columns=['Epoch','Threshold','CPU'])
    df_avg=df_avg.sort_values(["Epoch", "Threshold"], ascending=(True, True))
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
        df_temp=df_temp["CPU"].tolist()
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

def MemoryGraphMental(path, app_name):
    filelist=glob.glob(path+"*.csv")
    all_data=[]
    for file in filelist:
        data=ReadCSV(file)
        data_new=[]
        for dt in data:
            temp=[]
            flag=0
            for i in dt:
                z=ConvertFloat(i)
                if(z!=-1):
                    temp.append(z)
                else:
                    flag=1
                    break
            if(flag==0):
                data_new.append(temp)

        df = DataFrame (data_new,columns=['CPU','Memory'])
        df_list = df[df['Memory']>1]
        #print(df_list)
        temp=file.split("/")
        temp=temp[len(temp)-1]
        temp = re.findall(r'\d+', temp)
        res = list(map(int, temp))
        while(0 in res):
            res.remove(0)
        res.append(df_list['Memory'].mean())
        res.append(df_list['Memory'].std())
        all_data.append(res)

    print(all_data)
    df_avg = DataFrame (all_data,columns=['Threshold','Epoch','CPU','STD'])
    df_avg=df_avg.sort_values(["Epoch", "Threshold"], ascending=(True, True))
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
        df_temp1=df_temp["CPU"].tolist()
        df_temp2=df_temp["STD"].tolist()
        #print(df_temp)
        line_list[th]=[df_temp1,df_temp2]
    print(line_list)
    unique_threshold_legend=[]
    for i in unique_threshold:
        if(i!=1500):
            unique_threshold_legend.append("Short lambda ("+str(i)+" secs)")
        else:
            unique_threshold_legend.append("Long lambda")
    return x,line_list,unique_threshold_legend

def NetworkGraph(path, app_name):
    data=ReadCSV(path)
    print(data)
    df = DataFrame (data,columns=['Epoch','Threshold','Network'])
    #df= df.drop('Time', 1)
    df = df.sort_values(["Epoch", "Threshold"], ascending=(True, True))
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
            Average_Data.append([i,j,df_list_size["Network"].mean()])
    print(Average_Data)
    df_avg = DataFrame (Average_Data,columns=['Threshold','Epoch','Network'])
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
        df_temp=df_temp["Network"].tolist()
        #print(df_temp)
        line_list[th]=[x / 50 for x in df_temp]
    print(line_list)
    unique_threshold_legend=[]
    for i in unique_threshold:
        if(i!=1500):
            unique_threshold_legend.append("Short lambda ("+str(i)+" secs)")
        else:
            unique_threshold_legend.append("Long lambda")
    return x,line_list,unique_threshold_legend

def NetworkGraphMental(path, app_name):
    data=ReadCSV(path)
    print(data)
    df = DataFrame (data,columns=['Epoch','Threshold','Network'])
    #df= df.drop('Time', 1)
    df = df.sort_values(["Epoch", "Threshold"], ascending=(True, True))
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
            Average_Data.append([i,j,df_list_size["Network"].mean(),0])
            #Average_Data.append([i,j,df_list_size["Network"].std()])
    print(Average_Data)
    df_avg = DataFrame (Average_Data,columns=['Epoch','Threshold','CPU','STD'])
    df_avg=df_avg.sort_values(["Epoch", "Threshold"], ascending=(True, True))
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
        df_temp1=df_temp["CPU"].tolist()
        df_temp2=df_temp["STD"].tolist()
        #print(df_temp)
        #[x / 50 for x in df_temp]
        line_list[th]=[[x / 50 for x in df_temp1],df_temp2]
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
        return CPUGraph("../CSV/Face-App/FilteredCPU_MEM/", app_name)
    elif(app_name==2 and type==1):
        return CPUGraph("../CSV/Air-Pollution-App/FilteredCPU_MEM/", app_name)
    elif (app_name == 3 and type == 1):
        return CPUGraph("../CSV/Human-Activity-App/FilteredCPU_MEM/", app_name)
    elif (app_name == 4 and type == 1):
        return CPUGraphMental("../CSV/Mental-Stress-App/FilteredCPU_MEM/", app_name)
    elif(app_name==1 and type==2):
        return MemoryGraph("../CSV/Face-App/FilteredCPU_MEM/", app_name)
    elif(app_name==2 and type==2):
        return MemoryGraph("../CSV/Air-Pollution-App/FilteredCPU_MEM/", app_name)
    elif (app_name == 3 and type == 2):
        return MemoryGraph("../CSV/Human-Activity-App/FilteredCPU_MEM/", app_name)
    elif (app_name == 4 and type == 2):
        return MemoryGraphMental("../CSV/Mental-Stress-App/FilteredCPU_MEM/", app_name)
    elif(app_name==1 and type==3):
        return NetworkGraph("../CSV/Face-App/Face-App_network_data.csv", app_name)
    elif(app_name==2 and type==3):
        return NetworkGraph("../CSV/Air-Pollution-App/Air-Pollution_network_data.csv", app_name)
    elif (app_name == 3 and type == 3):
        return NetworkGraph("../CSV/Human-Activity-App/Human-Activity-App_network_data.csv", app_name)
    elif (app_name == 4 and type == 3):
        return NetworkGraphMental("../CSV/Mental-Stress-App/Mental-Stress_network_data.csv", app_name)


#main_function
app_name=input("1. Face Recognition App\n2. Air Pollution App\n3. Human Activity App\n4. Mental Stress App\nSelect App: ")
app_name=int(app_name)

type=input("1. CPU \n2. Memory\n3. Network\nSelect Type: ")
type=int(type)

if(app_name>4 or type >3):
    print ("Invalid Input!\n")
elif(app_name==4):
    x_axis,y_line_list,unique_threshold_legend=GetDataForApp(app_name,type)

    #import matplotlib.pyplot as plt
    for y_axis in y_line_list:
        fig, ax = plt.subplots()
        langs = []
        for i in x_axis:
            if(i==1500):
                langs.append("Long")
            else:
                langs.append(str(i))
        students = y_line_list[y_axis][0]
        error= y_line_list[y_axis][1]
        ax.bar(langs, students, yerr=error, align='center', alpha=0.5, color='black', capsize=10)
        ax.set_ylabel(GraphData(app_name,type,"y_label"),fontsize='14')
        ax.set_xticks(langs)
        ax.set_xticklabels(langs, fontsize='12')
        ax.tick_params(labelsize=12)
        #ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
        ax.yaxis.grid(True)
        ymin=min(students)
        ymax=max(students)
        plt.ylim([ymin*.9,ymax*1.05])
        plt.xlabel(GraphData(app_name,type,"x_label"), fontsize='14')

        # Save the figure and show
        plt.tight_layout()
        plt.savefig('Graphs/'+GraphData(app_name,type,"filename"), format='pdf', dpi=1200,bbox_inches='tight',
                   transparent=True,
                   pad_inches=0)
        #plt.show()
        # fig = plt.figure()
        # ax = fig.add_axes([0,0,1,1])
        # langs = []
        # for i in x_axis:
        #     if(i==1500):
        #         langs.append("Long")
        #     else:
        #         langs.append(str(i))
        # students = y_line_list[y_axis][0]
        # print(langs)
        # print(students)
        # ax.bar(langs,students)
        # plt.show()
    '''
    fig, ax = plt.subplots()
    color_data=0.9
    ymin=10000000000000000
    ymax=-1000000000
    for i in y_line_list:
        print (i)
        if(len(x_axis)==len(y_line_list[i])):
            #ax.plot(x_axis,y_line_list[i],c=str(color_data)) #black-white
            ax.plot(x_axis,y_line_list[i])
        else:
            #ax.plot(x_axis[:len(y_line_list[i])], y_line_list[i],c=str(color_data))
            ax.plot(x_axis[:len(y_line_list[i])], y_line_list[i])
        ymin=min(min(y_line_list[i]),ymin)
        ymax=max(max(y_line_list[i]),ymax)
        color_data-=.15
    #plt.title(GraphData(app_name,type,"title"))
    plt.legend(unique_threshold_legend,fontsize=12)
    plt.xlabel(GraphData(app_name,type,"x_label"),fontsize=14)
    plt.ylabel(GraphData(app_name,type,"y_label"),fontsize=14)
    plt.ylim([ymin*.9,ymax*1.05])
    plt.xticks(fontsize=12, rotation=0)
    #plt.show()
    fig.savefig('Graphs/'+GraphData(app_name,type,"filename"), format='pdf', dpi=1200,bbox_inches='tight',
               transparent=True,
               pad_inches=0)
    '''
else:
    x_axis,y_line_list,unique_threshold_legend=GetDataForApp(app_name,type)

    fig, ax = plt.subplots()
    color_data=0.9
    ymin=10000000000000000
    ymax=-1000000000
    for i in y_line_list:
        print (i)
        if(len(x_axis)==len(y_line_list[i])):
            #ax.plot(x_axis,y_line_list[i],c=str(color_data)) #black-white
            ax.plot(x_axis,y_line_list[i])
        else:
            #ax.plot(x_axis[:len(y_line_list[i])], y_line_list[i],c=str(color_data))
            ax.plot(x_axis[:len(y_line_list[i])], y_line_list[i])
        ymin=min(min(y_line_list[i]),ymin)
        ymax=max(max(y_line_list[i]),ymax)
        color_data-=.15
    #plt.title(GraphData(app_name,type,"title"))
    plt.legend(unique_threshold_legend,fontsize=12)
    plt.xlabel(GraphData(app_name,type,"x_label"),fontsize=14)
    plt.ylabel(GraphData(app_name,type,"y_label"),fontsize=14)
    plt.ylim([ymin*.9,ymax*1.05])
    plt.xticks(fontsize=12, rotation=0)
    #plt.show()
    fig.savefig('Graphs/'+GraphData(app_name,type,"filename"), format='pdf', dpi=1200,bbox_inches='tight',
               transparent=True,
               pad_inches=0)
