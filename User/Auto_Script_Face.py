import sys
import os
import json
import time
i=2
size=10
while(i<=size):
    print("Writing New Input")
    f=open("input.txt","w")
    f.write("1\n")
    f.write(str(i)+"\n")
    f.write("2\n")
    f.close()
    os.system("python3 User_face.py")
    print("One input done!!")
    i+=1
    time.sleep(10)
