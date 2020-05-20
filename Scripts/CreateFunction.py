import sys
import os
import subprocess
cmd="faas-cli new --lang dockerfile "
print("Enter Function Name: ")
func=input()
#script command
manager=subprocess.check_output("docker info |grep 'Node Address: ' | tr --delete 'Node Address: '",shell=True)
manager=manager.split()[0]
cmd+=func
cmd+=" --gateway http://"+str(manager.decode("utf-8"))+str(":8080")
print(cmd)

#change directory
os.chdir("../App/")
#execute command to create function
os.system(cmd)

#copy microlambda template
cmd="cp -r ../Scripts/microlambda_template/* "+func+"/"
print(cmd)
os.system(cmd)
