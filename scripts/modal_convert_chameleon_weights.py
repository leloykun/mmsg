import os
from typing import Optional

import modal
from modal_commons import GPU_CONFIG, MODEL_DIR, VOLUME_CONFIG, app


@app.function(
    cpu=4.0,
    gpu=GPU_CONFIG,
    timeout=60*60*3,
    volumes=VOLUME_CONFIG,
    _allow_background_volume_commits=True,
    secrets=[modal.Secret.from_name("hf-write-secret")],
)
def run_convert_chameleon_weights(
    model_id: str = "GAIR/Anole-7b-v0.1",
    model_size: str = "7B",
    local_input_model_path: Optional[str] = None,
    local_output_model_path: Optional[str] = None,
    force_download: bool = False,
    force_conversion: bool = False,
    upload_to_hf: bool = False,
):
    import subprocess

    from transformers.models.chameleon.convert_chameleon_weights_to_hf import (
        NUM_SHARDS,
        write_model,
    )

    _, model_name = model_id.split("/")
    if local_input_model_path is None:
        local_input_model_path = f"{MODEL_DIR}/{model_name}"
    if local_output_model_path is None:
        local_output_model_path = f"{MODEL_DIR}/{model_name}-hf"

    continue_download = True
    if os.path.exists(local_input_model_path):
        if force_download:
            print("Removing original model...")
            subprocess.run(
                [
                    "rm",
                    "-rf",
                    local_input_model_path,
                ],
                check=True,
            )
        else:
            continue_download = False
            print(
                f"Original model already downloaded at {local_input_model_path}. Skipping redownload."
            )
    if continue_download:
        subprocess.run(
            [
                "huggingface-cli",
                "download",
                "--resume-download",
                model_id,
                "--local-dir",
                local_input_model_path,
                "--local-dir-use-symlinks",
                "False",
            ],
            check=True,
        )

    continue_conversion = True
    if os.path.exists(local_output_model_path):
        if force_conversion:
            print("Removing converted model...")
            subprocess.run(
                [
                    "rm",
                    "-rf",
                    local_output_model_path,
                ],
                check=True,
            )
        else:
            continue_conversion = False
            print(f"Converted model already exists at {local_output_model_path}. Skipping.")
    if continue_conversion:
        if model_size not in NUM_SHARDS:
            raise ValueError(
                f"Model size {model_size} not supported. Choose from {NUM_SHARDS.keys()}"
            )
        write_model(
            model_path=local_output_model_path,
            input_base_path=local_input_model_path,
            model_size=model_size,
        )

    if upload_to_hf:
        subprocess.run(
            [
                "huggingface-cli",
                "upload",
                "--token",
                os.environ["HF_TOKEN"],
                "--private",
                f"leloy/{model_name}-hf",
                local_output_model_path,
            ],
            check=True,
        )


@app.local_entrypoint()
def main(
    model_id: str = "GAIR/Anole-7b-v0.1",
    model_size: str = "7B",
    local_input_model_path: Optional[str] = None,
    local_output_model_path: Optional[str] = None,
    force_download: bool = False,
    force_conversion: bool = False,
    upload_to_hf: bool = False,
    local: bool = False,
):
    if local:
        run_convert_chameleon_weights.local(
            model_id=model_id,
            model_size=model_size,
            local_input_model_path=local_input_model_path,
            local_output_model_path=local_output_model_path,
            force_download=force_download,
            force_conversion=force_conversion,
            upload_to_hf=upload_to_hf,
        )
    else:
        run_convert_chameleon_weights.remote(
            model_id=model_id,
            model_size=model_size,
            local_input_model_path=local_input_model_path,
            local_output_model_path=local_output_model_path,
            force_download=force_download,
            force_conversion=force_conversion,
            upload_to_hf=upload_to_hf,
        )
