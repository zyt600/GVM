#!/bin/bash

SCHEDULER=$1

./start_vllm_server.sh ${SCHEDULER} 2>&1 | tee vllm.log &
sleep 25
./start_diffusion.sh ${SCHEDULER} 2>&1 | tee diffusion.log &

sleep 10
if [ "${SCHEDULER}" == "gvm" ]; then
    ./config_gvm.sh 0 7
fi

sleep 10
./start_vllm_client.sh

wait
