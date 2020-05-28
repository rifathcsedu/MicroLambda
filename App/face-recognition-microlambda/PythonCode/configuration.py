Database = dict(
    host = '10.200.100.253',
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
    publish_air_pollution_app='PollutionStateStore',
    input_air_pollution_app='PollutionInput',
    result_air_pollution_app='ResultPollutionApp',
)

MicroLambda=dict(
    long_lambda='1500',
    short_lambda='1000'
)

AppURL = dict(
    face_app = 'http://10.200.87.202:8080/function/face-recognition-microlambda',
    air_pollution_app = '',
    human_activity_app='',
)
