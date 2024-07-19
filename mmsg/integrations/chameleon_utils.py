import os
import uuid
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, TypedDict, Union

from typing_extensions import NotRequired

from ..utils import pil_to_base64

if TYPE_CHECKING:
    import numpy as np
    from transformers import ChameleonForCausalLM, ChameleonModel, ChameleonProcessor


class ImageDataDict(TypedDict):
    base64_str: str
    save_path: NotRequired[str]


class ResponseDict(TypedDict):
    text: str
    images: List[ImageDataDict]


def split_tokens_into_segments_by_modality(
    token_ids: "np.ndarray",
    image_token_ids: List[int],
    boi_token_id: int,
    eoi_token_id: int,
    validate: bool = False,
) -> List[Tuple[Literal["text", "image"], List[int]]]:
    segments: List[Tuple[Literal["text", "image"], List[int]]] = []
    curr_sequence: List[int] = []
    modality: Literal["text", "image"] = "text"
    for idx, token_id in enumerate(token_ids):
        if token_id == boi_token_id:
            if validate and modality == "image":
                raise ValueError(
                    f"Invalid token sequence: sequence has duplicate image generation start token."
                )
            if idx > 0:
                segments.append((modality, curr_sequence))
                curr_sequence = []
            modality = "image"
            continue
        elif token_id == eoi_token_id:
            if validate and modality == "text":
                raise ValueError(
                    f"Invalid token sequence: sequence has image generation end token without start token."
                )
            segments.append((modality, curr_sequence))
            modality = "text"
            curr_sequence = []
            continue

        is_image_token = token_id in image_token_ids
        if validate and ((modality == "image") ^ is_image_token):
            raise ValueError(
                f"Invalid token sequence: sequence has invalid token {token_id} in {modality} generation mode."
            )
        curr_sequence.append(token_id)
    if curr_sequence:
        if modality == "text":
            segments.append(("text", curr_sequence))
        else:
            if validate:
                raise ValueError(
                    f"Invalid token sequence: sequence has image generation start token without end token."
                )
            segments.append(("image", curr_sequence))
    return segments


def build_response_from_segments(
    model: Union["ChameleonModel", "ChameleonForCausalLM"],
    processor: "ChameleonProcessor",
    segments: List[Tuple[Literal["text", "image"], List[int]]],
    outputs_dir: Optional[str] = None,
):
    text_tokens_list = [
        token_ids for modality, token_ids in segments if modality == "text"
    ]
    image_tokens_list = [
        token_ids for modality, token_ids in segments if modality == "image"
    ]

    text_str_list = processor.batch_decode(text_tokens_list, skip_special_tokens=True)

    pixel_values = model.decode_image_tokens(image_tokens_list)
    images = processor.postprocess_pixel_values(
        pixel_values.float().detach().cpu().numpy()
    )

    response: ResponseDict = {"text": "", "images": []}
    for modality, _ in segments:
        if modality == "text":
            response["text"] += text_str_list.pop(0)
        else:
            response["text"] += "<image>"
            image = images.pop(0)
            image_data: ImageDataDict = {
                "base64_str": f"data:image/png;base64,{pil_to_base64(image)}"
            }

            if outputs_dir is not None:
                if not os.path.exists(f"{outputs_dir}/images"):
                    os.mkdir(f"{outputs_dir}/images")
                image_save_path = f"{outputs_dir}/images/{str(uuid.uuid4())}.png"
                image.save(image_save_path)
                image_data["save_path"] = image_save_path

            response["images"].append(image_data)
    return response


def postprocess_token_sequence(
    token_ids: "np.ndarray",
    model: Union["ChameleonModel", "ChameleonForCausalLM"],
    processor: "ChameleonProcessor",
    outputs_dir: Optional[str] = None,
    validate: bool = True,
) -> ResponseDict:
    segments = split_tokens_into_segments_by_modality(
        token_ids,
        model.vocabulary_mapping.image_token_ids,
        model.vocabulary_mapping.boi_token_id,
        model.vocabulary_mapping.eoi_token_id,
        validate=validate,
    )
    return build_response_from_segments(model, processor, segments, outputs_dir)
