#!/bin/sh
SHELL_PATH=$(pwd)
MODEL="$1"

docker run -itd --rm -u root --gpus all --shm-size=1g \
-p 9080:8080 \
-p 9081:8081 \
-p 9082:8082 \
-p 8070:7070 \
-p 8071:7071 \
-v $SHELL_PATH/resources/model-store:/home/model-server/model-store pytorch/torchserve:latest-gpu torchserve \
--start \
--model-store model-store \
--models $MODEL \
--foreground
