#docker_cmd_93.sh
img="nvcr.io/nvidia/tensorflow:19.01-py3"
#img="padim:0.1"


docker run --gpus all  --privileged=true   --workdir /git --name "calstm"  -e DISPLAY --ipc=host -d --rm  -p 6611:4452  \
-v /raid/git/multimodal-deep-learning:/git/multimodal-deep-learning \
 -v /raid/git/datasets:/git/datasets \
 $img sleep infinity

docker exec -it calstm /bin/bash

pip list  |grep "pytorch"