# RAD: Robotics After Dark

Repo for the eclectic pursuits of the RAD team.

## Requirements
* Ubuntu Computer with Nvidia GPU (preferebly 22.04 or 24.04) and ~35 GB free.

## Setup
1. Pull the submodules:
    ```bash
    git submodule update --init --recursive
    ```
### Option A: Docker Setup
This is required to run the simulation. If you don't want to use Docker, you can skip those instructions and instead install the dependencies in `Dockerfile.isaac` on your Ubuntu Computer.
Alternatively, if you just want to run the training, see the [Option B](#option-b-uv-setup) instructions below.


1. Install Docker by following the `Install using the apt repository` instruction [here](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository). We recommend also following the [post-installation steps to manage docker as non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).

2. Install Docker compose by following there `Install using the repository` [instructions here](https://docs.docker.com/compose/install/linux/#install-using-the-repository).

3. Install Nvidia Container Toolkit by following [these instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). We have tested version 1.20.0 and 1.17.8 but other versions may work (although we know for sure that version 1.12 has Vulkan issues). 
    * Make sure you complete the `Installation` section for `With apt: Ubuntu, Debian` and also the `Configuring Docker` section.
    * To check proper installation, run `sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi`. This should output a table with your Nvidia driver. If you run into `Failed to initialize NVML: Unknown Error`, reference [this post](https://stackoverflow.com/questions/72932940/failed-to-initialize-nvml-unknown-error-in-docker-after-few-hours) for the solution.

4. Launch your Docker container:
    ```bash
    # Grant screen forwarding perms
    xhost +local:

    # Compile container (~35 GB). This will take a while the first time you run because Isaacsim is huge
    docker compose -f compose.isaac.yaml build

    # Launch container
    docker compose -f compose.isaac.yaml run --rm isaac-base 
    ```

    To test your setup, run `isaacsim` and the sim window should appear. The first time you run this, the center may stay black for a while. Wait for the gridlines to show up.

### Option B: UV Setup
This installation will only let you run training/evaluation. If you want to run simulation, use [Option A](#option-a-docker-setup) above.
1. Install uv using [these instructions](https://docs.astral.sh/uv/getting-started/installation/).
2. Setup and source your venv:
    ```bash
    uv sync
    source .venv/bin/activate
    ```

## Running

Run teleop:
```bash
python3 -m sim.run_sim_teleop
```

Run success classifier:
```bash
# Test Accuracy: 94.4%, Test Avg loss: 0.172887
python3 models/train_classifier.py
```

Run Eval on whole dataset after training:
```bash
# Accuracy: 97.2%, Avg loss: 0.088487
python3 models/eval_classifier.py
```
