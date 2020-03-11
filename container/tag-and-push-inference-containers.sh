#!/bin/bash
source container/util.sh

TARGET_REGIONS=($AWS_DEFAULT_REGION)
TARGET_ACCOUNTS=($AWS_ACCOUNT_ID)

tag_and_push $XGBOOST_REPO "XGBOOST_TAGS" "xgboost-cpu:latest" "TARGET_REGIONS" "TARGET_ACCOUNTS"
tag_and_push $IMG_REPO "IMG_TAGS_CPU" "ic-cpu:latest" "TARGET_REGIONS" "TARGET_ACCOUNTS"
