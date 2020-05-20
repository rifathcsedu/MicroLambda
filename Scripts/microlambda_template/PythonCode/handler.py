import os
import json

def get_stdin():
    buf = ""
    for line in sys.stdin:
        buf = buf + line
    return buf

if __name__ == "__main__":
    print(os.system("ls"))
    print(json.dumps("hehe"))
