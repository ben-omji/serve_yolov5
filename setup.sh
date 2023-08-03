#!/bin/sh
SHELL_PATH=$(pwd)
TYPE="$1"

if [ $TYPE = "seg" ] ; then
  MODEL="yolov5s-seg"
else
  MODEL="yolov5s"  
fi

mkdir -p resources/weights resources/model-store

docker run -it --rm -v $SHELL_PATH/resources/:/usr/src/app/resources ultralytics/yolov5:v7.0-cpu python3 export.py --weights resources/weights/$MODEL.pt --include torchscript
mv resources/weights/$MODEL.torchscript resources/weights/coco_$MODEL.torchscript.pt

if [ ! -d serve ] ; then
  git clone -b v0.8.1 https://github.com/pytorch/serve.git
  cp config.properties serve/docker/
fi
cd serve/docker && ./build_image.sh -bt production -g -cv cu111 && cd $SHELL_PATH

docker run -it --rm -u root --gpus all --entrypoint '' -v $SHELL_PATH/resources:/home/model-server/resources pytorch/torchserve:latest-gpu torch-model-archiver -f \
--model-name coco_$MODEL \
--version 0.1 \
--serialized-file /home/model-server/resources/weights/coco_$MODEL.torchscript.pt \
--handler /home/model-server/resources/handler_${TYPE:-box}.py \
--requirements-file /home/model-server/resources/requirements.txt \
--extra-files /home/model-server/resources/index_to_name.json,/home/model-server/resources/handler_${TYPE:-box}.py \
--export-path /home/model-server/resources/model-store

if [ -e resources/model-store/coco_${MODEL}.mar ]; then
  echo "Model archive is generated at ./resource/model-store/coco_${MODEL}.mar"
else
  echo "Failed, Please check stdout above"
fi
