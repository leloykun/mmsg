import logging
import os
import subprocess
import uuid
from typing import Optional

import modal
from image_only_generation import run_image_only_generation
from modal_commons import CURR_DIR, GENERATED_IMAGES_DIR, GPU_CONFIG, MODEL_DIR, app
from structured_generation import run_structured_generation
from text_only_generation import run_text_only_generation

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()


@app.function(
    cpu=4.0,
    gpu=GPU_CONFIG,
    timeout=60 * 60 * 3,
    _allow_background_volume_commits=True,
    secrets=[modal.Secret.from_name("hf-write-secret")],
    mounts=[
        modal.Mount.from_local_file(
            local_path=f"{CURR_DIR}/text_only_generation.py",
            remote_path="text_only_generation.py",
        )
    ],
)
def run_inference_text_only(
    model_id: str = f"{MODEL_DIR}/Anole-7b-v0.1-hf",
    inference_mode: str = "text-to-text",
    prompt: Optional[str] = None,
    image_1_path: Optional[str] = None,
    image_2_path: Optional[str] = None,
    max_new_tokens: int = 40,
    model_cache_dir: str = MODEL_DIR,
    outputs_dir: str = GENERATED_IMAGES_DIR,
    seed: Optional[int] = None,
):
    run_text_only_generation(
        model_id=model_id,
        inference_mode=inference_mode,
        prompt=prompt,
        image_1_path=image_1_path,
        image_2_path=image_2_path,
        max_new_tokens=max_new_tokens,
        model_cache_dir=model_cache_dir,
        seed=seed,
    )


@app.function(
    cpu=4.0,
    gpu=GPU_CONFIG,
    timeout=60 * 60 * 3,
    _allow_background_volume_commits=True,
    secrets=[modal.Secret.from_name("hf-write-secret")],
)
def run_inference_image_only(
    model_id: str = f"{MODEL_DIR}/Anole-7b-v0.1-hf",
    inference_mode: str = "text-to-image",
    prompt: Optional[str] = None,
    image_1_path: Optional[str] = None,
    image_2_path: Optional[str] = None,
    max_new_tokens: int = 2400,
    model_cache_dir: str = MODEL_DIR,
    outputs_dir: str = GENERATED_IMAGES_DIR,
    seed: Optional[int] = None,
):
    run_image_only_generation(
        model_id=model_id,
        inference_mode=inference_mode,
        prompt=prompt,
        image_1_path=image_1_path,
        image_2_path=image_2_path,
        max_new_tokens=max_new_tokens,
        model_cache_dir=model_cache_dir,
        outputs_dir=outputs_dir,
        seed=seed,
    )


@app.function(
    cpu=4.0,
    gpu=GPU_CONFIG,
    timeout=60 * 60 * 3,
    _allow_background_volume_commits=True,
    secrets=[modal.Secret.from_name("hf-write-secret")],
)
def run_inference_structured_generation(
    inference_mode: str,
    model_id: str = f"{MODEL_DIR}/Anole-7b-v0.1-hf",
    prompt: Optional[str] = None,
    image_1_path: Optional[str] = None,
    image_2_path: Optional[str] = None,
    max_new_tokens: int = 5000,
    json_schema_path: Optional[str] = None,
    regex_pattern: Optional[str] = None,
    model_cache_dir: str = MODEL_DIR,
    outputs_dir: str = GENERATED_IMAGES_DIR,
    seed: Optional[int] = None,
):
    run_structured_generation(
        model_id=model_id,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        json_schema_path=json_schema_path,
        regex_pattern=regex_pattern,
        model_cache_dir=MODEL_DIR,
        outputs_dir=GENERATED_IMAGES_DIR,
        seed=seed,
    )


@app.function(
    cpu=4.0,
    gpu=GPU_CONFIG,
    timeout=60 * 60 * 3,
    _allow_background_volume_commits=True,
    secrets=[modal.Secret.from_name("hf-write-secret")],
)
def run_inference(
    inference_mode: str,
    model_id: str = f"{MODEL_DIR}/Anole-7b-v0.1-hf",
    prompt: Optional[str] = None,
    image_1_path: Optional[str] = None,
    image_2_path: Optional[str] = None,
    max_new_tokens: int = 2400,
    json_schema_path: Optional[str] = None,
    regex_pattern: Optional[str] = None,
    model_cache_dir: str = MODEL_DIR,
    outputs_dir: str = GENERATED_IMAGES_DIR,
    seed: Optional[int] = None,
):
    if inference_mode in ["text-to-text", "text-image-to-text", "multi-image-to-text"]:
        if json_schema_path is not None:
            raise ValueError(
                "json_schema_path is only supported for structured generation"
            )
        if regex_pattern is not None:
            raise ValueError(
                "regex_pattern is only supported for structured generation"
            )
        run_text_only_generation(
            model_id=model_id,
            inference_mode=inference_mode,
            prompt=prompt,
            image_1_path=image_1_path,
            image_2_path=image_2_path,
            max_new_tokens=max_new_tokens,
            model_cache_dir=model_cache_dir,
            seed=seed,
        )
    elif inference_mode in [
        "text-to-image",
        "text-image-to-image",
        "multi-image-to-image",
    ]:
        if json_schema_path is not None:
            raise ValueError(
                "json_schema_path is only supported for structured generation"
            )
        if regex_pattern is not None:
            raise ValueError(
                "regex_pattern is only supported for structured generation"
            )
        run_image_only_generation(
            model_id=model_id,
            inference_mode=inference_mode,
            prompt=prompt,
            image_1_path=image_1_path,
            image_2_path=image_2_path,
            max_new_tokens=max_new_tokens,
            model_cache_dir=model_cache_dir,
            outputs_dir=outputs_dir,
            seed=seed,
        )
    elif inference_mode in ["structured"]:
        run_structured_generation(
            model_id=model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            model_cache_dir=model_cache_dir,
            outputs_dir=outputs_dir,
            json_schema_path=json_schema_path,
            regex_pattern=regex_pattern,
            seed=seed,
        )
    else:
        raise ValueError(f"Invalid inference_mode: {inference_mode}")
