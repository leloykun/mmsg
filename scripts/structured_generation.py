import argparse
import json
import logging
import os
from typing import Optional

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()


def run_structured_generation(
    model_id: str = "leloy/Anole-7b-v0.1-hf",
    prompt: Optional[str] = None,
    json_schema_path: Optional[str] = None,
    regex_pattern: Optional[str] = None,
    max_new_tokens: int = 2400,
    fast: bool = False,
    model_cache_dir: str = "/pretrained",
    outputs_dir: str = ".",
    seed: Optional[int] = None,
) -> str:
    import torch
    from outlines.processors import FSMLogitsProcessor
    from term_image.image import from_file
    from transformers import ChameleonForCausalLM, ChameleonProcessor, set_seed
    from transformers.generation.logits_process import LogitsProcessorList

    from mmsg.fsm.guide import RegexWithMultimodalMarkersGuide
    from mmsg.fsm.json_schema import build_regex_from_schema
    from mmsg.integrations.chameleon_utils import postprocess_token_sequence
    from mmsg.integrations.multimodal_tokenizer import MultimodalTokenizer

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

    mmsg_tokenizer = MultimodalTokenizer(
        processor.tokenizer,
        image_token_ids=set(range(4, 4 + 8192)),
        image_token="<image>",
        boi_token="<racm3:break>",
        eoi_token="<eoss>",
    )

    if prompt is None:
        prompt = (
            "Please generate a fruit along with a picture of it and related fruits."
        )

    json_schema = None

    if json_schema_path is not None and regex_pattern is not None:
        raise ValueError(
            "Both json schema and regex pattern cannot be provided at the same time."
        )
    elif json_schema_path is None and regex_pattern is None:
        json_schema = {
            "name": "Fruit Generator",
            "description": "A tool that generates details about a fruit with text and images in one go!",
            "type": "object",
            "properties": {
                "fruit_name": {
                    "type": "string",
                    # "minLength": 1,
                    # "maxLength": 20,
                    "pattern": "[a-zA-Z0-9]{1,20}",
                },
                "fruit_image": {
                    "type": "image",
                    # "maxLength": 10,
                },
                "images_of_related_fruits": {
                    "type": "array",
                    "items": {
                        "type": "image",
                        # "minLength": 1,
                    },
                    "minItems": 3,
                    "maxItems": 3,
                },
            },
            "required": ["fruit_name", "fruit_image", "images_of_related_fruits"],
        }
        logger.info(
            f"No path to json schema nor regex pattern provided. Will use the default json schema instead."
        )

    if json_schema_path is not None:
        with open(json_schema_path) as f:
            json_schema = json.load(f)
            logger.info(f"Loaded json schema from {json_schema_path}")

    if json_schema is not None:
        json_schema_str = json.dumps(json_schema)
        regex_pattern = build_regex_from_schema(json_schema_str)
        logger.info(f"Built regex pattern from json schema: {regex_pattern}")

        prompt = f"{prompt} Please follow this schema: {json.dumps(json_schema)}"
    else:
        prompt = f"{prompt} Please follow this regex pattern: {regex_pattern}"
    logger.info(f"Prompt: {prompt}")

    images = None

    logger.info("Building regex guide...")
    regex_guide = RegexWithMultimodalMarkersGuide(
        regex_pattern,
        mmsg_tokenizer,
        frozen_tokens=[
            mmsg_tokenizer.image_token,
            mmsg_tokenizer.boi_token,
            mmsg_tokenizer.eoi_token,
        ],
    )
    logger.info("Finished building regex guide.")

    logits_processor = LogitsProcessorList(
        [FSMLogitsProcessor(mmsg_tokenizer, regex_guide)]
    )

    inputs = processor(prompt, images=images, return_tensors="pt").to(model.device)

    logger.info("Starting generation...")
    with torch.inference_mode():
        output_token_ids_batch = model.generate(
            **inputs,
            multimodal_generation_mode="free",
            logits_processor=logits_processor,
            max_new_tokens=max_new_tokens,
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
    # parser.add_argument(
    #     "-i",
    #     "--inference_mode",
    #     choices=["text-to-text", "text-image-to-text", "multi-image-to-text"],
    #     required=False,
    #     default="text-to-text",
    #     help=""
    # )
    parser.add_argument(
        "-j",
        "--json_schema_path",
        type=str,
        required=False,
        default=None,
        help="The path to the json schema file to use to constrain the generation.",
    )
    parser.add_argument(
        "-r",
        "--regex_pattern",
        type=str,
        required=False,
        default=None,
        help="The regex pattern to use to constrain the generation.",
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
        "-o",
        "--outputs_dir",
        type=str,
        required=False,
        default=".",
        help="The directory to save the generated images in.",
    )
    args: argparse.Namespace = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    logger.info(f"Running image only generation... {args = }")
    run_structured_generation(
        model_id=args.model_id,
        # inference_mode=args.inference_mode,
        prompt=args.prompt,
        json_schema_path=args.json_schema_path,
        regex_pattern=args.regex_pattern,
        max_new_tokens=args.max_new_tokens,
        fast=args.fast,
        model_cache_dir=args.model_cache_dir,
        outputs_dir=args.outputs_dir,
        seed=args.seed,
    )
