import sys
import os
f=open("command.txt","r")
cmd=f.read()
os.system(cmd)
f.close()
