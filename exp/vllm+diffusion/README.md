## Usage

### Prerequisites

Must have the corresponding scheduler (`GVM` or `xsched`) installed.


### Prepare datasets

```shell
./dowload_dataset.sh
# Prepare vidprom dataset
python csv_to_prompts.py datasets/VidProM_unique_example.csv -o datasets/vidprom_prompts.txt
```

### Before running xsched

```shell
./launch_xserver.sh
```

### Run

#### Diffusion inference

```shell
./start_diffusion.sh [gvm|xsched|none]
```

#### vLLM

```shell
./start_vllm_server.sh [gvm|xsched|none]
# Waiting for vLLM server starts...
./start_vllm_client.sh
```

### After launching apps with GVM

```shell
./config_gvm.sh
```

To config each application's compute priority and memory limit. Check the script to adjust parameters.