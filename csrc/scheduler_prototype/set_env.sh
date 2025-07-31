LIBPATH=$(cd "$(dirname "$0")" && pwd -P)
export LD_PRELOAD="${LIBPATH}/scheduler-lib/libcuda-control.so:${LIBPATH}/scheduler-lib/libnvidia-ml.so.1:${LIBPATH}/scheduler-lib/libcuda.so.1"