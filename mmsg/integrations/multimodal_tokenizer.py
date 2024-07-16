from typing import TYPE_CHECKING, Set

from outlines.models.transformers import TransformerTokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


class MultimodalTokenizer(TransformerTokenizer):
    image_token_ids: Set[int]
    image_token: str
    image_start_token: str
    image_end_token: str

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizer",
        image_token_ids: Set[int],
        image_token: str,
        image_start_token: str,
        image_end_token: str,
        **kwargs,
    ) -> None:
        super().__init__(tokenizer)
        self.image_token_ids = image_token_ids
        self.image_token = image_token
        self.image_start_token = image_start_token
        self.image_end_token = image_end_token
        # This is important. Without this, negated text tokens could
        # go directly to image tokens.
        self.special_tokens |= {
            token
            for token, token_id in self.vocabulary.items()
            if token_id in image_token_ids
        }
