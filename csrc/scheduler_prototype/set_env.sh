ROOT=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd -P)
export LD_PRELOAD="${ROOT}/scheduler-lib/libcuda-control.so:${ROOT}/scheduler-lib/libnvidia-ml.so.1:${ROOT}/scheduler-lib/libcuda.so.1"