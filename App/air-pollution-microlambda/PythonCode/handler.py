import os
import json
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

def get_stdin():
    buf = ""
    for line in sys.stdin:
        buf = buf + line
    return buf

def handle (req):
    print(req)
    json_req = json.loads(req)
    current = json_req["current"]
    i=current
    training_size=json_req["training"]
    arr=[]
    while(i<current+training_size):
        temp=LoadData(Topic["input_air_pollution_app"], i, i)
        if(i==current):
            arr=pickle.loads(temp)
        else:
            arr=np.concatenate((arr,pickle.loads(temp)))
    print(arr.shape[0])
    '''        
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    return publish_redis(Topic["publish_air_pollution_app"],json.dumps({"size": json_req["size"],"current":json_req["current"]+json_req["training"],"training":json_req["training"],"testing":json_req["testing"],"threshold": float(json_req["threshold"])}))
    '''

#main function
if __name__ == "__main__":
    st = get_stdin()
    handle(st)
    # design network
