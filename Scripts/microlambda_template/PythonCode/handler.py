import os
import json

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
    while(time.time()-start<threshold and current<size):
        publish_redis("test", "New loop!!! current= "+str(current))
    json_req["current"]=current
    publish_redis("test", json.dumps({"data":"done"}))
    return publish_redis(Topic["publish(app_name)"],json.dumps(json_req))

if __name__ == "__main__":
    st = get_stdin()
    handle(st)