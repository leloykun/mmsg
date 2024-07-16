import logging
import os
import uuid
from typing import Optional

import modal
import requests
from modal_commons import GENERATED_IMAGES_DIR, GPU_CONFIG, MODEL_DIR, app
from PIL import Image

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()


def load_image(image_path: str):
    if image_path.startswith("http"):
        image = Image.open(requests.get(image_path, stream=True).raw)
    else:
        image = Image.open(image_path)
    return image


@app.function(
    cpu=4.0,
    gpu=GPU_CONFIG,
    timeout=60*60*3,
    _allow_background_volume_commits=True,
    secrets=[modal.Secret.from_name("hf-write-secret")],
)
def run_inference_text_only(
    model_id: str = f"{MODEL_DIR}/Anole-7b-v0.1-hf",
    inference_id: int = 1,
    prompt: Optional[str] = None,
    image_1_path: Optional[str] = None,
    image_2_path: Optional[str] = None,
    max_new_tokens: int = 40,
):
    import torch
    from transformers import ChameleonProcessor, ChameleonForCausalLM, set_seed

    set_seed(0)
    torch.set_printoptions(threshold=10_000)

    model: ChameleonForCausalLM = ChameleonForCausalLM.from_pretrained(
        model_id,
        # load_in_4bit=True,
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        device_map="auto",
        token=os.environ["HF_TOKEN"],
        cache_dir=MODEL_DIR,
    )
    processor = ChameleonProcessor.from_pretrained(
        model_id,
        token=os.environ["HF_TOKEN"],
        cache_dir=MODEL_DIR,
    )

    if inference_id == 1:
        logger.info("TASK: Text to Text generation")

        if prompt is None:
            prompt = "Is a banana a fruit or a vegetable? Please answer with yes or no."
        logger.info(f"Prompt: {prompt}")

        inputs = processor(prompt, return_tensors="pt").to(
            model.device, dtype=model.dtype
        )
    if inference_id == 2:
        logger.info("TASK: Text-Image to Text generation")

        if prompt is None:
            prompt = "What is in this image?"
        prompt = f"{prompt}<image>"
        logger.info(f"Prompt: {prompt}")

        if image_1_path is None:
            image_1_path = "https://nineplanets.org/wp-content/uploads/2020/12/the-big-dipper-1.jpg"
        image = load_image(image_1_path)
        logger.info("Image 1 loaded.", image_1_path=image_1_path)

        inputs = processor(prompt, image, return_tensors="pt").to(
            model.device, dtype=model.dtype
        )
    if inference_id == 3:
        logger.info("TASK: Multi-Image generation")
        
        if prompt is None:
            prompt = "What do these two images have in common?"
        prompt = f"{prompt}<image><image>"
        logger.info(f"Prompt: {prompt}")

        if image_1_path is None:
            image_1_path = "https://nineplanets.org/wp-content/uploads/2020/12/the-big-dipper-1.jpg"
        if image_2_path is None:
            image_2_path = "https://www.kxan.com/wp-content/uploads/sites/40/2020/10/ORION.jpg"
        images = [load_image(image_1_path), load_image(image_2_path)]
        logger.info("Images loaded.", image_1_path=image_1_path, image_2_path=image_2_path)

        inputs = processor(
            text=prompt,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(model.device, dtype=model.dtype)
    else:
        raise ValueError(f"Invalid inference_id: {inference_id}")

    logger.info("Generating response...")
    with torch.inference_mode():
        output_token_ids_batch = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
        )
    logger.info(f"Finished generation.")

    response_token_ids = [
        output_token_ids[len(input_token_ids):]
        for input_token_ids, output_token_ids in zip(
            inputs["input_ids"], output_token_ids_batch
        )
    ]
    output_text = processor.decode(response_token_ids[0], skip_special_tokens=True)
    response = output_text.replace(prompt, "")
    logger.info(f"Response: {response}")


@app.function(
    cpu=4.0,
    gpu=GPU_CONFIG,
    timeout=60*60*3,
    _allow_background_volume_commits=True,
    secrets=[modal.Secret.from_name("hf-write-secret")],
)
def run_inference_image_only(
    model_id: str=f"{MODEL_DIR}/Anole-7b-v0.1-hf",
    inference_id: int=1,
    prompt: Optional[str] = None,
    image_1_path: Optional[str] = None,
    image_2_path: Optional[str] = None,
    max_new_tokens: int=2400,
):
    import torch
    from transformers import ChameleonProcessor, ChameleonForCausalLM, set_seed
    from term_image.image import from_file

    from mmsg.integrations.multimodal_tokenizer import MultimodalTokenizer

    set_seed(42)
    torch.set_printoptions(threshold=10_000)

    model: ChameleonForCausalLM = ChameleonForCausalLM.from_pretrained(
        model_id,
        # load_in_4bit=True,
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        device_map="auto",
        token=os.environ["HF_TOKEN"],
        cache_dir=MODEL_DIR,
    )
    processor = ChameleonProcessor.from_pretrained(
        model_id,
        token=os.environ["HF_TOKEN"],
        cache_dir=MODEL_DIR,
    )

    mmsg_tokenizer = MultimodalTokenizer(
        processor.tokenizer,
        image_token_ids=set(range(4, 4+8192)),
        image_token="<image>",
        image_start_token="<racm3:break>",
        image_end_token="<eoss>",
    )

    if inference_id == 1:
        logger.info("TASK: Text to Image generation")
        if prompt is None:
            prompt = "Please draw an apple!"
        logger.info(f"Prompt: {prompt}")

        inputs = processor(prompt, return_tensors="pt").to(model.device, dtype=model.dtype)
    elif inference_id == 2:
        logger.info("TASK: Text-Image to Image generation")

        if prompt is None:
            prompt = "Draw a variation of this image:"
        prompt = f"{prompt}<image>"
        logger.info(f"Prompt: {prompt}")

        if image_1_path is None:
            image_1_path = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
        image = load_image(image_1_path)
        logger.info("Image 1 loaded.", image_1_path=image_1_path)

        inputs = processor(
            text=prompt,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(model.device, dtype=model.dtype)
    elif inference_id == 3:
        logger.info("TASK: Multi-Image to Image generation")

        if prompt is None:
            prompt = "Draw what is common between these images."
        prompt = f"{prompt}<image><image>"
        logger.info(f"Prompt: {prompt}")

        if image_1_path is None:
            image_1_path = "https://nineplanets.org/wp-content/uploads/2020/12/the-big-dipper-1.jpg"
        if image_2_path is None:
            image_2_path = "https://www.kxan.com/wp-content/uploads/sites/40/2020/10/ORION.jpg"
        images = [load_image(image_1_path), load_image(image_2_path)]
        logger.info("Images loaded.", image_1_path=image_1_path, image_2_path=image_2_path)

        inputs = processor(
            text=prompt,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(model.device, dtype=model.dtype)
    else:
        raise ValueError(f"Invalid inference_id: {inference_id}")

    logger.info("Generating response...")
    model.multimodal_generation_mode="image-only"
    with torch.inference_mode():
        output_token_ids_batch = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True
        )
    logger.info("Finished generation.")

    response_token_ids = [
        output_token_ids[len(input_token_ids):]
        for input_token_ids, output_token_ids in zip(
            inputs["input_ids"], output_token_ids_batch
        )
    ]

    image_tokens_list = []
    in_image_gen_mode = False
    curr_image_tokens = []
    text_token_ids = []
    for token_id in response_token_ids[0]:
        if token_id == mmsg_tokenizer.vocabulary[
            mmsg_tokenizer.image_start_token
        ]:
            in_image_gen_mode = True
            text_token_ids.append(
                mmsg_tokenizer.vocabulary[
                    mmsg_tokenizer.image_token
                ]
            )
            continue
        if token_id == mmsg_tokenizer.vocabulary[
            mmsg_tokenizer.image_end_token
        ]:
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
        reconstructed_pixel_values = model.decode_image_tokens(
            image_tokens_tensor
        )
    reconstructed_images = processor.postprocess_pixel_values(
        reconstructed_pixel_values.float().detach().cpu().numpy()
    )

    for reconstructed_image in reconstructed_images:
        image_path = f"{GENERATED_IMAGES_DIR}/test_image-{str(uuid.uuid4())}.png"
        logger.info(f"{image_path = }")
        reconstructed_image.save(image_path)
        terminal_image = from_file(image_path)
        terminal_image.draw()


@app.function(
    cpu=4.0,
    gpu=GPU_CONFIG,
    timeout=60*60*3,
    _allow_background_volume_commits=True,
    secrets=[modal.Secret.from_name("hf-write-secret")],
)
def run_inference_structured_generation(
    model_id: str=f"{MODEL_DIR}/Anole-7b-v0.1-hf",
    prompt: Optional[str] = None,
    max_new_tokens: int=2400,
):
    import json

    import torch
    from outlines.processors import FSMLogitsProcessor
    from term_image.image import from_file
    from transformers import ChameleonProcessor, ChameleonForCausalLM, set_seed
    from transformers.generation.logits_process import LogitsProcessorList

    from mmsg.fsm.json_schema import build_regex_from_schema
    from mmsg.fsm.guide import RegexWithMultimodalMarkersGuide
    from mmsg.integrations.multimodal_tokenizer import MultimodalTokenizer

    set_seed(42)
    torch.set_printoptions(threshold=10_000)

    model: ChameleonForCausalLM = ChameleonForCausalLM.from_pretrained(
        model_id,
        # load_in_4bit=True,
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        device_map="auto",
        token=os.environ["HF_TOKEN"],
        cache_dir=MODEL_DIR,
    )
    processor = ChameleonProcessor.from_pretrained(
        model_id,
        token=os.environ["HF_TOKEN"],
        cache_dir=MODEL_DIR,
    )

    mmsg_tokenizer = MultimodalTokenizer(
        processor.tokenizer,
        image_token_ids=set(range(4, 4+8192)),
        image_token="<image>",
        image_start_token="<racm3:break>",
        image_end_token="<eoss>",
    )

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
            "fruit_image" : {
                "type": "image",
                # "maxLength": 10,
            },
            "images_of_related_fruits" : {
                "type": "array",
                "items": {
                    "type": "image",
                    # "minLength": 1,
                },
                "minItems": 3,
                "maxItems": 3,
            }
        },
        "required": ["fruit_name", "fruit_image", "images_of_related_fruits"],
    }

    if prompt is None:
        prompt = "Please generate a fruit along with a picture of it and related fruits."
    prompt = f"{prompt} Please follow this schema: {json.dumps(json_schema)}"
    logger.info(f"Prompt: {prompt}")

    images = None
    model.multimodal_generation_mode="free"

    logger.info("Building regex guide...")
    regex_str = build_regex_from_schema(json.dumps(json_schema))
    logger.info(f"{regex_str = }")

    regex_guide = RegexWithMultimodalMarkersGuide(
        regex_str,
        mmsg_tokenizer,
        frozen_tokens=[
            mmsg_tokenizer.image_token,
            mmsg_tokenizer.image_start_token,
            mmsg_tokenizer.image_end_token,
        ]
    )
    logger.info("Finished building regex guide.")

    logits_processor = LogitsProcessorList([
        FSMLogitsProcessor(mmsg_tokenizer, regex_guide)
    ])

    inputs = processor(prompt, images=images, return_tensors="pt").to(model.device)

    logger.info("Starting generation...")
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            logits_processor=logits_processor,
            max_new_tokens=max_new_tokens,
            do_sample=True,
        )
    logger.info("Finished generation.")

    response_token_ids = [
        output_token_ids[len(input_token_ids):]
        for input_token_ids, output_token_ids in zip(
            inputs["input_ids"], generated_ids
        )
    ]

    image_tokens_list = []
    in_image_gen_mode = False
    curr_image_tokens = []
    text_token_ids = []
    for token_id in response_token_ids[0]:
        if token_id == mmsg_tokenizer.vocabulary[
            mmsg_tokenizer.image_start_token
        ]:
            in_image_gen_mode = True
            text_token_ids.append(
                mmsg_tokenizer.vocabulary[
                    mmsg_tokenizer.image_token
                ]
            )
            continue
        if token_id == mmsg_tokenizer.vocabulary[
            mmsg_tokenizer.image_end_token
        ]:
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
        reconstructed_pixel_values = model.decode_image_tokens(
            image_tokens_tensor
        )
    reconstructed_images = processor.postprocess_pixel_values(
        reconstructed_pixel_values.float().detach().cpu().numpy()
    )

    for reconstructed_image in reconstructed_images:
        image_path = f"{GENERATED_IMAGES_DIR}/test_image-{str(uuid.uuid4())}.png"
        logger.info(f"{image_path = }")
        reconstructed_image.save(image_path)
        terminal_image = from_file(image_path)
        terminal_image.draw()
