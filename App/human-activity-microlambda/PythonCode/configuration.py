import json
import csv


Iteration=2
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
    long_lambda='1500',
    short_lambda='1500'
)


#store metrics to CSV
def WriteCSV(path, data):
    print("Writing output and metrics in CSV...")
    with open(path, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)
    print("Writing Done!")
