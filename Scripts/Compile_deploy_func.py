import os
import sys
import glob
os.chdir("../App/")
filelist=glob.glob("*.yml")
print("\n\nHere is the existing app. Select one to build and deploy: \n\n")
for i in range(len(filelist)):
    print(str(i+1)+ ": "+str(filelist[i]))
#os.system(cmd)
print("\nEnter the choice: ")
choice=input()
cmd="faas-cli build -f ./"+str(filelist[int(choice)-1])
print(cmd)
CRED = '\33[34m'
CEND = '\033[0m'
if os.system(cmd) == 0:
    print(CRED+"\nBuild is done successfully\n"+CEND)
    cmd="faas-cli deploy -f ./"+str(filelist[int(choice)-1])
    print(cmd)
    if os.system(cmd) == 0:
        print(CRED+"\nDeploy is done successfully\n"+CEND)
else:
    print("Build failed")
