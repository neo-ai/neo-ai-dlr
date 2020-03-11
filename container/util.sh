#!/bin/bash
set -e

MXNET_REPO=sagemaker-neo-mxnet
TF_REPO=sagemaker-neo-tensorflow
PYTORCH_REPO=sagemaker-neo-pytorch
XGBOOST_REPO=xgboost-neo
IMG_REPO=image-classification-neo

XGBOOST_TAGS=("latest" )
# Tags separated by CPU or GPU.
MXNET_TAGS_GPU=("1.3.0-gpu-py3" "0.12.1-gpu-py3" "1.1.0-gpu-py3" "1.2.1-gpu-py3"
"1.2.0-gpu-py3" "1.0.0-gpu-py3" "1.4.1-gpu-py3" "1.4-gpu-py3" "1.3-gpu-py3"
"1.2-gpu-py3" "1.4.0-gpu-py3" "1.5-gpu-py3")
MXNET_TAGS_CPU=("1.4.1-cpu-py3" "1.3-cpu-py3" "1.2-cpu-py3"
"1.1.0-cpu-py3" "1.0.0-cpu-py3" "1.4.0-cpu-py3" "1.4-cpu-py3" "1.3.0-cpu-py3"
"1.2.0-cpu-py3" "1.2.1-cpu-py3" "0.12.1-cpu-py3" "1.5-cpu-py3")
TF_TAGS_GPU=("1.9.0-gpu-py3" "1.13.0-gpu-py3" "1.8.0-gpu-py3" "1.14.0-gpu-py3"
"1.7.0-gpu-py3" "1.10.0-gpu-py3" "1.5.0-gpu-py3" "1.11.0-gpu-py3" "1.12.0-gpu-py3"
"1.6.0-gpu-py3" "1.4.1-gpu-py3" "1.15.0-gpu-py3")
TF_TAGS_CPU=("1.13.0-cpu-py3" "1.7.0-cpu-py3" "1.4.1-cpu-py3"
"1.14.0-cpu-py3" "1.6.0-cpu-py3" "1.11.0-cpu-py3" "1.10.0-cpu-py3" "1.9.0-cpu-py3"
"1.5.0-cpu-py3" "1.8.0-cpu-py3" "1.12.0-cpu-py3" "1.15.0-cpu-py3")
PYTORCH_TAGS_GPU=("0.4.0-gpu-py3" "1.0.0-gpu-py3" "1.1.0-gpu-py3" "1.2.0-gpu-py3"
"1.3.0-gpu-py3" "1.4.0-gpu-py3" )
PYTORCH_TAGS_CPU=("0.4.0-cpu-py3" "1.0.0-cpu-py3" "1.1.0-cpu-py3" "1.2.0-cpu-py3"
"1.3.0-cpu-py3" "1.4.0-cpu-py3")
IMG_TAGS_GPU=("gpu")
IMG_TAGS_CPU=("latest" "cpu")


format_repo()
{
    echo "$1.dkr.ecr.$2.amazonaws.com/$3"
}

pull_images()
{
    if [ -z "$1" ];
    then
        echo "pull_images requires the repository as the first parameter"
        exit 1
    fi
    if [ -z "$2" ];
    then
        echo "pull_images requires the tags to be pulled as the second parameter"
        exit 1
    fi
    eval "tags=\${$2[*]}"
    for tag in $tags; do
        image="$1:$tag"
        docker pull $image
        echo "pulled $image"
    done
}

tag_images()
{
    if [ -z "$1" ];
    then
        echo "tag_images requires the source repository as the first parameter"
        exit 1
    fi
    if [ -z "$2" ];
    then
        echo "tag_images requires the target repository as the second parameter"
        exit 1
    fi
    if [ -z "$3" ];
    then
        echo "tag_images requires the tags to be tagged as the third parameter"
        exit 1
    fi
    eval "tags=\${$3[*]}"
    for tag in $tags; do
        source_image="$1:$tag"
        dest_image="$2:$tag"
        docker tag $source_image $dest_image
        echo "tagged $source_image as $dest_image"
    done
}

tag_single_image_as_multiple()
{
    if [ -z "$1" ];
    then
        echo "tag_single_image_as_multiple requires the source image as the first parameter"
        exit 1
    fi
    if [ -z "$2" ];
    then
        echo "tag_single_image_as_multiple requires the target repository as the second parameter"
        exit 1
    fi
    if [ -z "$3" ];
    then
        echo "tag_single_image_as_multiple requires the tags to be tagged as the third parameter"
        exit 1
    fi
    eval "tags=\${$3[*]}"
    for tag in $tags; do
        source_image="$1"
        dest_image="$2:$tag"
        docker tag $source_image $dest_image
        echo "tagged $source_image as $dest_image"
    done
}

push_images()
{
    if [ -z "$1" ];
    then
        echo "push_images requires the repository as the first parameter"
        exit 1
    fi
    if [ -z "$2" ];
    then
        echo "push_images requires the tags to be pushed as the second parameter"
        exit 1
    fi
    eval "tags=\${$2[*]}"
    for tag in $tags; do
        image="$1:$tag"
        docker push $image
        echo "pushed $image"
    done
}

tag_and_push() {
    if [ -z "$1" ];
    then
        echo "tag_and_push requires the repository name as the first parameter"
        exit 1
    fi
    if [ -z "$2" ];
    then
        echo "tag_and_push requires the tags to be pushed as the second parameter"
        exit 1
    fi
    if [ -z "$3" ];
    then
        echo "tag_and_push requires the source image:tag as the third parameter"
        exit 1
    fi
    if [ -z "$4" ];
    then
        echo "tag_and_push requires the target regions array as the fourth parameter"
        exit 1
    fi
    if [ -z "$5" ];
    then
        echo "tag_and_push requires the target accounts array as the fifth parameter"
        exit 1
    fi
    eval "target_regions=(\${$4[*]})"
    eval "target_accounts=(\${$5[*]})"
    echo $target_regions
    source_image=$3
    for ((i=0; i<${#target_regions[@]}; ++i)); do
        target_repo=$(format_repo ${target_accounts[$i]} ${target_regions[$i]} $1)
        tag_single_image_as_multiple $source_image $target_repo $2
        push_images $target_repo $2
    done
}
