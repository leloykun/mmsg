import argparse
import logging
import os
import uuid
from typing import Literal, Optional

logger = logging.getLogger()


def run_image_only_generation(
    model_id: str = "leloy/Anole-7b-v0.1-hf",
    inference_mode: Literal[
        "text-to-image", "text-image-to-image", "multi-image-to-image"
    ] = "text-to-image",
    prompt: Optional[str] = None,
    image_1_path: Optional[str] = None,
    image_2_path: Optional[str] = None,
    max_new_tokens: int = 2400,
    fast: bool = False,
    model_cache_dir: Optional[str] = None,
    outputs_dir: str = ".",
    seed: Optional[int] = None,
):
    import torch
    from term_image.image import from_file
    from transformers import ChameleonForCausalLM, ChameleonProcessor, set_seed
    from transformers.generation.logits_process import LogitsProcessorList

    from mmsg.integrations.chameleon_logits_processor import (
        ChameleonFSMLogitsProcessor,
        ChameleonModalityFSMGuide,
    )
    from mmsg.integrations.multimodal_tokenizer import MultimodalTokenizer
    from mmsg.utils import load_image

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

    if inference_mode == "text-to-image":
        logger.info("TASK: Text to Image generation")
        if prompt is None:
            prompt = "Please draw an apple!"
        logger.info(f"Prompt: {prompt}")

        inputs = processor(prompt, return_tensors="pt").to(
            model.device, dtype=model.dtype
        )
    elif inference_mode == "text-image-to-image":
        logger.info("TASK: Text-Image to Image generation")

        if prompt is None:
            prompt = "Draw a variation of this image:"
        prompt = f"{prompt}<image>"
        logger.info(f"Prompt: {prompt}")

        if image_1_path is None:
            image_1_path = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
        image = load_image(image_1_path)
        logger.info("Image 1 loaded.", image_1_path)

        inputs = processor(
            text=prompt,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(model.device, dtype=model.dtype)
    elif inference_mode == "multi-image-to-image":
        logger.info("TASK: Multi-Image to Image generation")

        if prompt is None:
            prompt = "Draw an image that looks like a combination of these two images:"
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
        raise ValueError(f"Invalid inference_id: {inference_id}")

    max_length = max_new_tokens + inputs["input_ids"].shape[-1]
    print(f"{max_length = } | {max_new_tokens = } | {inputs['input_ids'].shape = }")

    logits_processor = LogitsProcessorList([
        ChameleonFSMLogitsProcessor(
            fsm=ChameleonModalityFSMGuide(
                all_token_ids=model.model.vocabulary_mapping.vocab_map.values(),
                image_token_ids=model.model.vocabulary_mapping.image_token_ids,
                eos_token_id=model.model.config.eos_token_id,
                boi_token_id=model.model.vocabulary_mapping.boi_token_id,
                eoi_token_id=model.model.vocabulary_mapping.eoi_token_id,
                device=model.device,
                multimodal_generation_mode="image-only",
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

    response_token_ids = [
        output_token_ids[len(input_token_ids) :]
        for input_token_ids, output_token_ids in zip(
            inputs["input_ids"], output_token_ids_batch
        )
    ]

    print(f"{response_token_ids = }")

    image_tokens_list = []
    in_image_gen_mode = False
    curr_image_tokens = []
    text_token_ids = []
    for token_id in response_token_ids[0]:
        if token_id == mmsg_tokenizer.boi_token_id:
            in_image_gen_mode = True
            continue
        if token_id == mmsg_tokenizer.eoi_token_id:
            if in_image_gen_mode:
                in_image_gen_mode = False
                text_token_ids.append(mmsg_tokenizer.image_token_id)
                image_tokens_list.append(curr_image_tokens)
                curr_image_tokens = []
            continue

        if in_image_gen_mode:
            curr_image_tokens.append(token_id)
        else:
            text_token_ids.append(token_id)

    response = processor.decode(text_token_ids, skip_special_tokens=True)

    logger.info(f"Response: {response}")

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
    print(args)
    run_image_only_generation(
        model_id=args.model_id,
        inference_mode=args.inference_mode,
        prompt=args.prompt,
        image_1_path=args.image_1_path,
        image_2_path=args.image_2_path,
        max_new_tokens=args.max_new_tokens,
        fast=args.fast,
        model_cache_dir=args.model_cache_dir,
        outputs_dir=args.outputs_dir,
        seed=args.seed,
    )
