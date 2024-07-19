import argparse
import json
import logging
import os
from typing import Literal, Optional

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()


def run_interleaved_generation(
    model_id: str = "leloy/Anole-7b-v0.1-hf",
    inference_mode: Literal["text-to-interleaved-text-image"] = "text-to-interleaved-text-image",
    prompt: Optional[str] = None,
    max_new_tokens: int = 2400,
    fast: bool = False,
    model_cache_dir: Optional[str] = None,
    outputs_dir: str = ".",
    seed: Optional[int] = None,
) -> str:
    import torch
    from term_image.image import from_file
    from transformers import ChameleonForCausalLM, ChameleonProcessor, set_seed
    from transformers.generation.logits_process import LogitsProcessorList

    from mmsg.integrations.chameleon_logits_processor import (
        ChameleonFSMLogitsProcessor,
        ChameleonModalityFSMGuide,
    )
    from mmsg.integrations.chameleon_utils import postprocess_token_sequence

    if seed:
        set_seed(42)
    torch.set_printoptions(threshold=10_000)

    if fast:
        model = ChameleonForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            device_map="auto",
            token=os.environ.get("HF_TOKEN"),
            cache_dir=model_cache_dir,
        )
    else:
        model = ChameleonForCausalLM.from_pretrained(
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

    if inference_mode == "text-to-interleaved-text-image":
        logger.info("TASK: Text to Interleaved Text-Image generation")
        if prompt is None:
            prompt = "Please draw an apple!"
        logger.info(f"Prompt: {prompt}")

        inputs = processor(prompt, return_tensors="pt").to(
            model.device, dtype=model.dtype
        )
    else:
        raise ValueError(f"Invalid inference_id: {inference_mode}")

    max_length = max_new_tokens + inputs["input_ids"].shape[-1]

    logits_processor = LogitsProcessorList([
        ChameleonFSMLogitsProcessor(
            fsm=ChameleonModalityFSMGuide(
                all_token_ids=model.vocabulary_mapping.vocab_map.values(),
                image_token_ids=model.vocabulary_mapping.image_token_ids,
                eos_token_id=model.config.eos_token_id,
                boi_token_id=model.vocabulary_mapping.boi_token_id,
                eoi_token_id=model.vocabulary_mapping.eoi_token_id,
                device=model.device,
                multimodal_generation_mode="interleaved-text-image",
            ),
            max_length=max_length,
        )
    ])

    logger.info("Generating response...")
    with torch.inference_mode():
        output_token_ids_batch = model.generate(
            **inputs,
            multimodal_generation_mode="free",
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
            do_sample=True,
        )
    logger.info("Finished generation.")

    output_token_ids_batch = output_token_ids_batch.to(dtype=inputs["input_ids"].dtype).detach().cpu().numpy()

    response_token_ids = output_token_ids_batch[0][len(inputs["input_ids"][0]) :]

    full_outputs_dir = os.path.abspath(outputs_dir)
    if not os.path.exists(full_outputs_dir):
        logging.info(f"Creating directory: {full_outputs_dir}")
        os.mkdir(full_outputs_dir)

    response = postprocess_token_sequence(
        response_token_ids, model, processor, full_outputs_dir, validate=True
    )

    logger.info(f"Response: {response['text']}")
    for image in response["images"]:
        if "save_path" not in image:
            continue
        logger.info(f"{image['save_path'] = }")
        terminal_image = from_file(image["save_path"])
        terminal_image.draw()

    with open(f"{full_outputs_dir}/response.json", "w") as f:
        json.dump(response, f)
    logger.info(f"Response saved to {full_outputs_dir}/response.json")

    return json.dumps(response)


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
        choices=["text-to-image", "text-image-to-image", "multi-image-to-image"],
        required=False,
        default="text-to-image",
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
    # parser.add_argument(
    #     "-i1",
    #     "--image_1_path",
    #     type=str,
    #     required=False,
    #     default=None,
    #     help="The path to the first image to be used for generation.",
    # )
    # parser.add_argument(
    #     "-i2",
    #     "--image_2_path",
    #     type=str,
    #     required=False,
    #     default=None,
    #     help="The path to the second image to be used for generation.",
    # )
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
        default=None,
        help="The directory to cache the model in.",
    )
    parser.add_argument(
        "-o",
        "--outputs_dir",
        type=str,
        required=False,
        default=".",
        help="The directory to save the generated images in.",
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
    logger.info(f"Running interleaved generation... {args = }")
    run_interleaved_generation(
        model_id=args.model_id,
        inference_mode=args.inference_mode,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        fast=args.fast,
        model_cache_dir=args.model_cache_dir,
        outputs_dir=args.outputs_dir,
        seed=args.seed,
    )
