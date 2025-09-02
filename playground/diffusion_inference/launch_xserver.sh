#!/bin/bash

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd -P)
XSCHED_PATH=$(cd ${SCRIPT_DIR}/../../csrc/3rdparty/xsched/output/bin && pwd -P)

PORT=50000

${XSCHED_PATH}/xserver HPF ${PORT}