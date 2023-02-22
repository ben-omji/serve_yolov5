#!/bin/sh
SHELL_PATH=$(pwd)

mkdir -p resources/weights resources/model-store

docker run -it --rm -v $SHELL_PATH/resources/:/usr/src/app/resources ultralytics/yolov5 python export.py --weights resources/weights/yolov5s.pt --include torchscript
mv resources/weights/yolov5s.torchscript resources/weights/coco_yolov5s.torchscript.pt

if [ ! -d serve ] ; then
  git clone https://github.com/pytorch/serve.git
  cp config.properties serve/docker/
fi
cd serve/docker && ./build_image.sh -bt production -g -cv cu116 && cd $SHELL_PATH

docker run -it --rm -u root --gpus all --entrypoint '' -v $SHELL_PATH/resources:/home/model-server/resources pytorch/torchserve:latest-gpu torch-model-archiver -f \
--model-name coco_yolov5s \
--version 0.1 \
--serialized-file /home/model-server/resources/weights/coco_yolov5s.torchscript.pt \
--handler /home/model-server/resources/handler.py \
--requirements-file /home/model-server/resources/requirements.txt \
--extra-files /home/model-server/resources/index_to_name.json,/home/model-server/resources/handler.py \
--export-path /home/model-server/resources/model-store

if [ -e resources/model-store/coco_yolov5s.mar ]; then
  echo "Model archive is generated at ./resource/model-store"
else
  echo "Failed, Please check stdout above"
