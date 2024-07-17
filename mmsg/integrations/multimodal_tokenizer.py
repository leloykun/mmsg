from typing import TYPE_CHECKING, Set

from outlines.models.transformers import TransformerTokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


class MultimodalTokenizer(TransformerTokenizer):
    image_token_ids: Set[int]
    image_token: str
    boi_token: str
    eoi_token: str

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizer",
        image_token_ids: Set[int],
        image_token: str,
        boi_token: str,
        eoi_token: str,
        **kwargs,
    ) -> None:
        super().__init__(tokenizer)
        self.image_token_ids = image_token_ids
        self.image_token = image_token
        self.boi_token = boi_token
        self.eoi_token = eoi_token
        # This is important. Without this, negated text tokens could
        # go directly to image tokens.
        self.special_tokens |= {
            token
            for token, token_id in self.vocabulary.items()
            if token_id in image_token_ids
        }

    @property
    def image_token_id(self) -> int:
        return self.vocabulary[self.image_token]

    @property
    def boi_token_id(self) -> int:
        return self.vocabulary[self.boi_token]

    @property
    def eoi_token_id(self) -> int:
        return self.vocabulary[self.eoi_token]
