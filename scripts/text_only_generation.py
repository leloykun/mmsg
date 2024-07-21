import argparse
import logging
import os
from typing import Literal, Optional

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()


def run_text_only_generation(
    model_id: str = "leloy/Anole-7b-v0.1-hf",
    inference_mode: Literal[
        "text-to-text", "text-image-to-text", "multi-image-to-text"
    ] = "text-to-text",
    prompt: Optional[str] = None,
    image_1_path: Optional[str] = None,
    image_2_path: Optional[str] = None,
    max_new_tokens: int = 40,
    fast: bool = False,
    model_cache_dir: str = "/pretrained",
    seed: Optional[int] = None,
) -> str:
    import torch
    from transformers import (
        ChameleonForConditionalGeneration,
        ChameleonProcessor,
        set_seed,
    )

    from mmsg.utils import load_image

    if seed is not None:
        set_seed(seed)
    torch.set_printoptions(threshold=10_000)

    if fast:
        model = ChameleonForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            device_map="auto",
            token=os.environ.get("HF_TOKEN"),
            cache_dir=model_cache_dir,
        )
    else:
        model = ChameleonForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            token=os.environ.get("HF_TOKEN"),
            cache_dir=model_cache_dir,
        )
    processor = ChameleonProcessor.from_pretrained(
        model_id,
        token=os.environ.get("HF_TOKEN"),
        cache_dir=model_cache_dir,
    )

    if inference_mode == "text-to-text":
        logger.info("TASK: Text to Text generation")

        if prompt is None:
            prompt = "Is a banana a fruit or a vegetable? Please answer with yes or no."
        logger.info(f"Prompt: {prompt}")

        inputs = processor(prompt, return_tensors="pt").to(
            model.device, dtype=model.dtype
        )
    elif inference_mode == "text-image-to-text":
        logger.info("TASK: Text-Image to Text generation")

        if prompt is None:
            prompt = "What is in this image?"
        prompt = f"{prompt}<image>"
        logger.info(f"Prompt: {prompt}")

        if image_1_path is None:
            image_1_path = "https://nineplanets.org/wp-content/uploads/2020/12/the-big-dipper-1.jpg"
        image = load_image(image_1_path)
        logger.info("Image 1 loaded.", image_1_path)

        inputs = processor(prompt, image, return_tensors="pt").to(
            model.device, dtype=model.dtype
        )
    elif inference_mode == "multi-image-to-text":
        logger.info("TASK: Multi-Image generation")

        if prompt is None:
            prompt = "What do these two images have in common?"
        prompt = f"{prompt}<image><image>"
        logger.info(f"Prompt: {prompt}")

        if image_1_path is None:
            image_1_path = "https://nineplanets.org/wp-content/uploads/2020/12/the-big-dipper-1.jpg"
        if image_2_path is None:
            image_2_path = (
                "https://www.kxan.com/wp-content/uploads/sites/40/2020/10/ORION.jpg"
            )
        images = [load_image(image_1_path), load_image(image_2_path)]
        logger.info("Images loaded.", image_1_path, image_2_path)

        inputs = processor(
            text=prompt,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(model.device, dtype=model.dtype)
    else:
        raise ValueError(f"Invalid inference_mode: {inference_mode}")

    logger.info("Generating response...")
    with torch.inference_mode():
        output_token_ids_batch = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
        )
    logger.info(f"Finished generation.")

    response_token_ids = [
        output_token_ids[len(input_token_ids) :]
        for input_token_ids, output_token_ids in zip(
            inputs["input_ids"], output_token_ids_batch
        )
    ]
    response = processor.decode(response_token_ids[0], skip_special_tokens=True)
    logger.info(f"Response: {response}")
    return response


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate text-only content based on prompt which can include images."
    )
    parser.add_argument(
        "-m",
        "--model_id",
        type=str,
        required=False,
        default="leloy/Anole-7b-v0.1-hf",
        help="The model ID to use for generation. This could be a huggingface repo or a path to a local directory.",
    )
    parser.add_argument(
        "-i",
        "--inference_mode",
        choices=["text-to-text", "text-image-to-text", "multi-image-to-text"],
        required=False,
        default="text-to-text",
        help="",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=False,
        default=None,
        help="The prompt for generation. Will be appended by <image> or <image><image> if images are provided.",
    )
    parser.add_argument(
        "-i1",
        "--image_1_path",
        type=str,
        required=False,
        default=None,
        help="The path to the first image to be used for generation.",
    )
    parser.add_argument(
        "-i2",
        "--image_2_path",
        type=str,
        required=False,
        default=None,
        help="The path to the second image to be used for generation.",
    )
    parser.add_argument(
        "-n",
        "--max_new_tokens",
        type=int,
        required=False,
        default=40,
        help="The maximum number of tokens to generate.",
    )
    parser.add_argument(
        "-f",
        "--fast",
        type=int,
        required=False,
        default=40,
        help="Whether to convert the model to bfloat16 & use Flash Attention 2",
    )
    parser.add_argument(
        "-c",
        "--model_cache_dir",
        type=str,
        required=False,
        default="/pretrained",
        help="The directory to cache the model in.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        required=False,
        default=None,
        help="The seed to use for generation.",
    )
    args: argparse.Namespace = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    logger.info(f"Running image only generation... {args = }")
    run_text_only_generation(
        model_id=args.model_id,
        inference_mode=args.inference_mode,
        prompt=args.prompt,
        image_1_path=args.image_1_path,
        image_2_path=args.image_2_path,
        max_new_tokens=args.max_new_tokens,
        fast=args.fast,
        model_cache_dir=args.model_cache_dir,
        seed=args.seed,
    )
