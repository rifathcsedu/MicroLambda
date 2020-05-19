import sys
import os
import subprocess
cmd="faas-cli new --lang python3 "
print("Enter Function Name: ")
func=input()
manager=subprocess.check_output("docker info |grep 'Node Address: ' | tr --delete 'Node Address: '",shell=True)
manager=manager.split()[0]
cmd+=func
cmd+=" --gateway http://"+str(manager.decode("utf-8"))+str(":8080")
print(cmd)
os.chdir("../App/")
os.system(cmd)
