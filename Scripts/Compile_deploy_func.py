import os
import sys
import glob

#change directory and load all the functions
os.chdir("../App/")
filelist=glob.glob("*.yml")
print("\n\nHere is the existing app. Select one to build and deploy: \n\n")
for i in range(len(filelist)):
    print(str(i+1)+ ": "+str(filelist[i]))

#Terminal color set
CRED = '\33[34m'
CEND = '\033[0m'

#input desired function
print("\nEnter the choice: ")
choice=input()

#remove previous deployed function
cmd="faas-cli remove -f ./"+str(filelist[int(choice)-1])
if os.system(cmd) == 0:
    print(CRED+"\nRemoved previous deployed version successfully\n"+CEND)
else:
    print("Remove failed")

#build and deploy
cmd="faas-cli build -f ./"+str(filelist[int(choice)-1])
print(cmd)
if os.system(cmd) == 0:
    print(CRED+"\nBuild is done successfully\n"+CEND)
    cmd="faas-cli deploy -f ./"+str(filelist[int(choice)-1])
    print(cmd)
    if os.system(cmd) == 0:
        print(CRED+"\nDeploy is done successfully\n"+CEND)
else:
    print("Build failed")
