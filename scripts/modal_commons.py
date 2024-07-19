import os
from pathlib import PurePosixPath
from typing import Dict, Union

import modal

GPU_NAME_TO_MODAL_CLASS_MAP = {
    "H100": modal.gpu.H100,
    "A100": modal.gpu.A100,
    "A10G": modal.gpu.A10G,
}
N_GPUS = int(os.environ.get("N_GPUS", 1))
GPU_MEM = int(os.environ.get("GPU_MEM", 40))
GPU_NAME = os.environ.get("GPU_NAME", "A100")
GPU_CONFIG = GPU_NAME_TO_MODAL_CLASS_MAP[GPU_NAME](count=N_GPUS, size=str(GPU_MEM) + 'GB')

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = "/pretrained"
GENERATED_IMAGES_DIR = "/generated_images"
VOLUME_CONFIG: Dict[Union[str, PurePosixPath], modal.Volume] = {
    MODEL_DIR: modal.Volume.from_name(
        "pretrained", create_if_missing=True
    ),
    GENERATED_IMAGES_DIR: modal.Volume.from_name(
        "generated_images", create_if_missing=True
    ),
}

# Taken from https://github.com/karpathy/llm.c
cuda_image = (
    modal.Image.from_registry(
        "totallyvyom/cuda-env:latest-2",
        add_python="3.10",
    )
    .pip_install("huggingface_hub==0.20.3", "hf-transfer==0.1.5")
    .env(
        dict(
            HUGGINGFACE_HUB_CACHE=MODEL_DIR,
            HF_HUB_ENABLE_HF_TRANSFER="1",
            TQDM_DISABLE="true",
        )
    )
    .run_commands(
        "wget -q https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-Linux-x86_64.sh",
        "bash cmake-3.28.1-Linux-x86_64.sh --skip-license --prefix=/usr/local",
        "rm cmake-3.28.1-Linux-x86_64.sh",
        "ln -s /usr/local/bin/cmake /usr/bin/cmake",
    )
    .run_commands(
        "apt-get install -y --allow-change-held-packages libcudnn8 libcudnn8-dev",
        "apt-get install -y openmpi-bin openmpi-doc libopenmpi-dev kmod sudo",
        "git clone https://github.com/NVIDIA/cudnn-frontend.git /root/cudnn-frontend",
        "cd /root/cudnn-frontend && mkdir build && cd build && cmake .. && make",
    )
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
        mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
        apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub && \
        add-apt-repository \"deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /\" && \
        apt-get update"
    ).run_commands(
        "apt-get install -y nsight-systems-2023.3.3"
    )
)

image = (
    cuda_image
    .pip_install("setuptools==69.5.1", "wheel", extra_options="--force")
    .pip_install_from_requirements(
        f"{CURR_DIR}/modal_requirements.txt",
        secrets=[modal.Secret.from_name("hf-write-secret")],
    )
    # `flash-attn` requires `torch` to be installed first. Hence the order.
    .pip_install("flash-attn", extra_options="--no-cache-dir --no-build-isolation")
    .copy_local_dir(os.path.dirname(CURR_DIR), "/mmsg")
    .run_commands("pip install /mmsg[test]")
)

app = modal.App(image=image, volumes=VOLUME_CONFIG)
