# MicroLambda: V2

## Docker installation for Ubuntu 18.04
Clone the project into the each node and run this commands:

    sudo ./installation_docker_amd64.sh
    sudo reboot

after the reboot, check whether docker is in the groups. Command:

    groups

You will see "docker" in the list.

## OpenFaas CLI Installation
For installing OpenFaas-Cli, Run this commands for each node:

    sudo ./installation_openfaas_amd64.sh

after that check whether faas-cli is installed or not. Command:

    faas-cli version

## Docker Swarm Cluster
To create Swarm cluster, run the command:

    docker swarm init

After that the output will be like this:

    To add a worker to this swarm, run the following command:

    docker swarm join --token SWMTKN-1-0pk4x9k3zkc 10.200.10.56:2377

The node will act as a manager. Copy the command "docker swarm --token SWMTKN-1-0pk4x9k3zkc 10.200.10.56:2377" and paste it to other nodes who will work as a worker.

After that, check the list of nodes from manager node:

    docker node ls
It will show the list of nodes in the cluster:

    ID                    HOSTNAME            STATUS              AVAILABILITY        MANAGER STATUS      ENGINE VERSION
    4t9jsu68v02m *        node1               Ready               Active              Leader              19.03.5
    cj2ystlqnb95jr8oa     node2               Ready               Active                                  19.03.5
    cvofgcwsubdy51vd      node3               Ready               Active                                  19.03.5

Now, using manager node, run those commands to install OpenFaas:

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

You will also see the password if you scroll up the terminal. You can save the command to Installation/command.txt and add --gateway. For example,
    echo -n baded4e467b4ea70dab8bc7a844ddfe2c517013b32ff045b6f6fd19f6e9f1e03 | faas-cli login --username=admin --password-stdin --gateway http://youip:8080


## Deploy microlambda function:

<!-- Raspberry Pi uses ARM architecture which is different from other PC (In general, other PC uses x86_64 architecture). So, when you create a function in Node or Python you need to add a suffix of -armhf to use a special Docker image for the Raspberry Pi. Run this command inside of faas folder which we just cloned from GitHub. -->
Initially, set your Redis Database IP to Config/configuration.py

      Database = dict(
          host = 'set Redis IP here',
          port = '6379',
          password='',
      )
And set Redis variable:


      set ServerIPAddress your_redis_ip

1. For creating new microlambda, go to the Scripts folder and run CreateFunction.py.

      cd Scripts/
      python3 CreateFunction.py

2. For deploying existing microlambda functions from App folder, run:

      cd Scripts/
      python3 Compile_deploy_func.py

<!-- First, we need to modify the python-hello.yml and stack_arm.yml (As we are using Raspberry Pi) and replace "localhost" / "127.0.0.1" with the ip address of the manager node. Now, you can change the code in "python-hello" folder and change "handler.py". Then build the code using this command:

      faas-cli build -f ./python-hello.yml

To verify the credentials, run the command with your password and manager ip:

      echo"password" | faas-cli login --password-stdin --gateway http://10.200.10.56:8080/

Then deploy the function:

      faas-cli deploy -f ./python-hello.yml -->

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
