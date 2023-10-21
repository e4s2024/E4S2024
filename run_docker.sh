#!/bin/bash
image_full_name=$1
if [ -z ${image_full_name} ]
then
   echo "please input a image name. eg: mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda9.0-cudnn7.6-tf1.12:latest    \n"
   exit 0
fi

cmd=$2
if [ -z "${cmd}" ]
then
   echo "please input your train command, eg: python3.6 /apdcephfs/private_YOURRTX/train/train.py  --dataset_dir=/apdcephfs/private_YOURRTX/data     \n"
   exit 0
fi

echo ${image_full_name}
echo ${cmd}
nvidia-docker run -it -e NVIDIA_VISIBLE_DEVICES=all --network=host -v /apdcephfs_cq2/:/apdcephfs_cq2/ ${image_full_name} ${cmd}

