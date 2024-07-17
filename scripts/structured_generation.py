import argparse
import json
import logging
import os
import uuid
from typing import Optional

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
):
    import torch
    from outlines.processors import FSMLogitsProcessor
    from term_image.image import from_file
    from transformers import ChameleonForCausalLM, ChameleonProcessor, set_seed
    from transformers.generation.logits_process import LogitsProcessorList

    from mmsg.fsm.guide import RegexWithMultimodalMarkersGuide
    from mmsg.fsm.json_schema import build_regex_from_schema
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
            token=os.environ["HF_TOKEN"],
            cache_dir=model_cache_dir,
        )
    else:
        model = ChameleonForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            token=os.environ["HF_TOKEN"],
            cache_dir=model_cache_dir,
        )
    processor = ChameleonProcessor.from_pretrained(
        model_id,
        token=os.environ["HF_TOKEN"],
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
        generated_ids = model.generate(
            **inputs,
            multimodal_generation_mode="free",
            logits_processor=logits_processor,
            max_new_tokens=max_new_tokens,
            do_sample=True,
        )
    logger.info("Finished generation.")

    response_token_ids = [
        output_token_ids[len(input_token_ids) :]
        for input_token_ids, output_token_ids in zip(inputs["input_ids"], generated_ids)
    ]

    image_tokens_list = []
    in_image_gen_mode = False
    curr_image_tokens = []
    text_token_ids = []
    for token_id in response_token_ids[0]:
        if token_id == mmsg_tokenizer.boi_token_id:
            in_image_gen_mode = True
            text_token_ids.append(mmsg_tokenizer.image_token_id)
            continue
        if token_id == mmsg_tokenizer.eoi_token_id:
            if in_image_gen_mode:
                in_image_gen_mode = False
                image_tokens_list.append(curr_image_tokens)
                curr_image_tokens = []
            continue

        if in_image_gen_mode:
            curr_image_tokens.append(token_id)
        else:
            text_token_ids.append(token_id)

    response = processor.decode(text_token_ids, skip_special_tokens=True)

    logger.info(f"Response: {response.replace(prompt, '')}")

    # normalize to 1024 tokens per image
    image_tokens_list = [
        image_tokens + [1] * (1024 - len(image_tokens))
        if len(image_tokens) <= 1024
        else image_tokens[:1024]
        for image_tokens in image_tokens_list
    ]

    image_tokens_tensor = torch.tensor(image_tokens_list).to("cuda")
    with torch.inference_mode():
        reconstructed_pixel_values = model.decode_image_tokens(image_tokens_tensor)
    reconstructed_images = processor.postprocess_pixel_values(
        reconstructed_pixel_values.float().detach().cpu().numpy()
    )

    for reconstructed_image in reconstructed_images:
        image_path = f"{outputs_dir}/test_image-{str(uuid.uuid4())}.png"
        logger.info(f"{image_path = }")
        reconstructed_image.save(image_path)
        terminal_image = from_file(image_path)
        terminal_image.draw()


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
    print(args)
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
