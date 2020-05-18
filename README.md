# Re-Lambda

## Docker installation for Raspberry PI
Clone the project into the each RPI and run this commands:

    sudo ./installation_docker_rpi.sh
    sudo reboot

after the reboot, check whether docker is in the groups. Command:

    groups

You will see "docker" in the list.

## OpenFaas CLI Installation
For installing OpenFaas-Cli, Run this commands for each RPI:

    sudo ./installation_openfaas_rpi.sh

after that check whether faas-cli is installed or not. Command:

    faas-cli version


## Docker Swarm Cluster
To create Swarm cluster, run the command:

    docker swarm init

After that the output will be like this:

    To add a worker to this swarm, run the following command:

    docker swarm join --token SWMTKN-1-0pk4x9k3zkc 10.200.10.56:2377

The RPI will act as a manager. Copy the command "docker swarm --token SWMTKN-1-0pk4x9k3zkc 10.200.10.56:2377" and paste it to other RPIs who will work as a worker.

After that, check the list of nodes from manager RPI:

    docker node ls
It will show the list of nodes in the cluster:

    ID                    HOSTNAME            STATUS              AVAILABILITY        MANAGER STATUS      ENGINE VERSION
    4t9jsu68v02m *        raspberrypi         Ready               Active              Leader              19.03.5
    cj2ystlqnb95jr8oa     raspberrypi         Ready               Active                                  19.03.5
    cvofgcwsubdy51vd      raspberrypi         Ready               Active                                  19.03.5

Now, using manager RPI, run those commands to install OpenFaas:

    git clone https://github.com/alexellis/faas/
    cd faas
    ./deploy_stack.sh

Output will be like this:

    Deploying OpenFaaS core services for ARM
    Creating network func_functions
    Creating config func_prometheus_config
    Creating config func_prometheus_rules
    Creating config func_alertmanager_config
    Creating service func_gateway
    Creating service func_basic-auth-plugin
    Creating service func_faas-swarm
    Creating service func_nats
    Creating service func_queue-worker
    Creating service func_prometheus
    Creating service func_alertmanager

You will also see the password if you scroll up the terminal.

## Deploy first serverless function:

Raspberry Pi uses ARM architecture which is different from other PC (In general, other PC uses x86_64 architecture). So, when you create a function in Node or Python you need to add a suffix of -armhf to use a special Docker image for the Raspberry Pi. Run this command inside of faas folder which we just cloned from GitHub.

      faas-cli new --lang python-armhf python-hello

1. Now, we will see that a folder name "python-hello"
2. a file "python-hello.yml"

First, we need to modify the python-hello.yml and stack_arm.yml (As we are using Raspberry Pi) and replace "localhost" / "127.0.0.1" with the ip address of the manager node. Now, you can change the code in "python-hello" folder and change "handler.py". Then build the code using this command:

      faas-cli build -f ./python-hello.yml

To verify the credentials, run the command with your password and manager ip:

      echo"password" | faas-cli login --password-stdin --gateway http://10.200.10.56:8080/

Then deploy the function:

      faas-cli deploy -f ./python-hello.yml

After that, you can login to the http://10.200.10.56:8080/ ID: admin and password: the password you got when you installed openfaas. The deployed function will take some time to show the invoke button because it creates replica with worker nodes. You can check it using the command.

      watch 'docker service ls'


## Deploy MicroLambda:

Run this command inside of faas folder which we just cloned from GitHub.

      faas-cli build -f ./face-recognition.yml

Then deploy the function:

      faas-cli deploy -f ./face-recognition.yml


Initially you need a Redis Database remotely. Configure a Redis Remote Database using https://www.digitalocean.com/community/questions/enable-remote-redis-connection. We use a RaspberryPi Device for Redis Database. We need a Redis Controller and we use another device for this operation. Now staring the Redis Controller, paste the serverless URL in this shortlambda_face.txt file and then Run this command to start Redis Controller.

      python Redis_Controller_face.py

For User, you can use any device. Just run the command:
      python User_face.py
