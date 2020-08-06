import json
import csv


Iteration=50
sleep_time=5

Server=dict(
    IPAddress='ServerIPAddress',
    DBServer='DBIPAddress'
)
Database = dict(
    host = '192.168.0.107',
    port = '6379',
    password='',
)
Topic = dict(
    publish_face_app = 'ImageStateStore',
    input_face_app='ImageInput',
    result_face_app='ResultImageApp',
    publish_human_activity_app = 'HumanStateStore',
    input_human_activity_app='HumanInput',
    result_human_activity_app='ResultHumanApp',
    model_human_activity_app='ModelActivityApp',
    publish_air_pollution_app='PollutionStateStore',
    input_air_pollution_app='PollutionInput',
    result_air_pollution_app='ResultPollutionApp',
    model_air_pollution_app='ModelPollutionApp',
)

MicroLambda=dict(
    #short_lambda=['240']
    short_lambda=['1500','420','360','300','240','180']

)


#store metrics to CSV
def WriteCSV(path, data):
    print("Writing output and metrics in CSV...")
    with open(path, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)
    print("Writing Done!")




def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def RepresentsFloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def ReadCSV(path):
    data=[]
    with open(path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        #fields = csvreader.next()

        for row_data in csvreader:
            temp = []
            for item in row_data:
                if(RepresentsInt(item)):
                    new_item=int(item)
                elif(RepresentsFloat(item)):
                    new_item=float(item)
                else:
                    new_item=item
                temp.append(new_item)
            data.append(temp)
    return data
