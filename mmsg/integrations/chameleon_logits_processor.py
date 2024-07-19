"""
[Chameleon's generator](https://github.com/facebookresearch/chameleon/blob/main/chameleon/inference/chameleon.py#L374)  # noqa E501
uses a finite state machine (FSM) with the modality of the tokens as states to
dynamically switch between text and image generation. The generator uses this FSM to
determine which tokens are allowed to be generated next based on the previously
generated tokens. Additionally, due to the limitations of the VQ-VAE model used to
tokenize & detokenize images, the generator also ensures that:

1. The image tokens are generated in a contiguous block of `image_seq_length` tokens,
    which is currently fixed at 1024.
2. The block of image tokens is preceded by the `boi_token` (`<racm3:break>`)
    and followed by the `eoi_token` (`<eoss>`) marker tokens. And that
3. The block of image tokens doesn't overshoot the maximum sequence length
    (`max_length`). It does this by simply preventing the `boi_token` from being
    generated if there isn't enough space for the rest of the tokens.

Chameleon is so dependent on this FSM that removing it leads to sever model degradation.
Removing or ignoring the invalid/nonsensical tokens as they are generated is not enough.
We _have_ to mask their logits out entirely.

Here, we implement four generation modes, each with its own FSM guide (or none thereof):

1. `text-only` (default): We mask out the logits of image tokens. The
    `boi_token` and `eoi_token` are allowed but they have to close each
    other out immediately. This is done by the `ChameleonTextOnlyLogitsProcessor` class.
2. `image-only`: We mask out the logits of non-image tokens.
3. `interleaved-text-image`: We allow interleaved generation of text and images. This is
    actually the default mode of the official Chameleon model. And
4. `free`: We remove all constraints. This is useful for when you want to use a custom
    finite state machine or no FSM at all.

Note: We decided to follow [Outline](https://github.com/outlines-dev/outlines)'s
interface to make integration easier.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Literal, Optional, Protocol

import torch


@dataclass(frozen=True)
class Instruction:
    tokens: torch.Tensor


class Guide(Protocol):
    """Base definition of a generation guide.

    A generation guide defines the behavior of a finite-state machine that guides
    a text generation procedure. Unlike the DFAs built from regular expressions
    guides can also emit a `Write` instructions which tells the model that it can
    append a sequence of tokens (or token word) instead of generating it.

    """

    def get_next_instruction(self, state: int) -> Instruction:
        ...

    def get_next_state(self, state: int, token_id: int) -> int:
        ...

    def is_final_state(self, state: int) -> bool:
        ...

    def copy(self) -> "Guide":
        ...


class ChameleonModalityFSMGuide(Guide):
    FINAL_STATE = -1
    TEXT_STATE = 0
    # TODO: handle other modalities later
    IMAGE_STATE = 1

    initial_state = TEXT_STATE

    def __init__(
        self,
        eos_token_id: int,
        all_token_ids: List[int],
        image_token_ids: List[int],
        boi_token_id: int,
        eoi_token_id: int,
        device: torch.device = torch.device("cpu"),
        multimodal_generation_mode: Literal[
            "text-only", "image-only", "interleaved-text-image", "free"
        ] = "text-only",
    ):
        self.eos_token_id = eos_token_id
        self.all_token_ids = all_token_ids
        self.image_token_ids = image_token_ids
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.multimodal_generation_mode = multimodal_generation_mode
        self.device = device

        self.text_token_ids: List[int] = [
            token_id
            for token_id in all_token_ids
            if (
                token_id not in [boi_token_id, eoi_token_id]
                and token_id not in image_token_ids
            )
        ]

        if multimodal_generation_mode == "interleaved-text-image":
            self.states_to_token_maps = {
                self.TEXT_STATE: {
                    **{token_id: self.TEXT_STATE for token_id in self.text_token_ids},
                    **{boi_token_id: self.IMAGE_STATE},
                },
                self.IMAGE_STATE: {
                    **{token_id: self.IMAGE_STATE for token_id in image_token_ids},
                    **{eoi_token_id: self.TEXT_STATE},
                },
            }
        elif multimodal_generation_mode == "text-only":
            self.states_to_token_maps = {
                self.TEXT_STATE: {
                    token_id: self.TEXT_STATE for token_id in self.text_token_ids
                }
            }
        elif multimodal_generation_mode == "image-only":
            self.states_to_token_maps = {
                # Immediately transition to the image state
                self.TEXT_STATE: {boi_token_id: self.IMAGE_STATE},
                self.IMAGE_STATE: {
                    **{token_id: self.IMAGE_STATE for token_id in image_token_ids},
                    **{eoi_token_id: self.TEXT_STATE},
                },
            }
        elif multimodal_generation_mode == "free":
            raise ValueError(
                "Unconstrained generation of text and image tokens is incompatible with this FSM."  # noqa E501
            )
        else:
            raise ValueError(
                f"Invalid mode: {multimodal_generation_mode}. Must be one of 'text-only', 'image-only', or 'interleaved-text-image'"  # noqa E501
            )

        self.states_to_token_maps[self.TEXT_STATE][self.eos_token_id] = self.FINAL_STATE

        self.states_to_token_mask = {
            state: torch.tensor(list(next_tokens_to_end_states.keys()), device=device)
            for state, next_tokens_to_end_states in self.states_to_token_maps.items()
        }

    def get_next_instruction(self, state: int) -> Instruction:
        return Instruction(
            self.states_to_token_mask.get(
                state, torch.tensor([self.eos_token_id], device=self.device)
            )
        )

    def get_next_state(self, state: int, token_id: int) -> int:
        if token_id == self.eos_token_id or state not in self.states_to_token_mask:
            return self.FINAL_STATE

        last_token_to_end_state = self.states_to_token_maps[state]
        next_state = last_token_to_end_state.get(token_id)
        if next_state is None:
            next_state = self.FINAL_STATE

        return next_state

    def is_final_state(self, state: int) -> bool:
        return state == self.FINAL_STATE

    def copy(self) -> "ChameleonModalityFSMGuide":
        return self


class ChameleonFSMLogitsProcessor:
    """Bias Chameleon generation using a finite state machine."""

    def __init__(
        self,
        fsm: ChameleonModalityFSMGuide,
        max_length: int,
        image_seq_length: int = 1024,
    ):
        """A FSM-based logits processor.

        Args
            fsm (`Guide`): The finite state machine which is used to bias the logits.
            max_length (`int`): The maximum sequence length (prompt + generated tokens).
            image_seq_length (`int`, *optional*, defaults to 1024): The number of
                discrete image tokens needed to generate an image.
        """
        self._fsm_states: Dict[int, int] = {}
        self.fsm: ChameleonModalityFSMGuide = fsm
        self._is_first_token = True
        self._seq_start_idx: Optional[int] = None
        self.image_token_ids_tensor = torch.tensor(self.fsm.image_token_ids)

        self.max_length = max_length
        self.image_seq_length = image_seq_length

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Use the FSM to bias the logits before sampling the next token.

        Args:
            input_ids (`torch.Tensor` of shape `(batch, sequence_length)`):
                The input token ids.
            scores (`torch.Tensor` of shape `(batch, vocab_size)`):
                The logits to be biased.

        Returns:
            `torch.Tensor` of shape `(batch, vocab_size)`.
        """
        sequence_states: List[int] = []  # vector of states corresponding to `input_ids`

        if self._is_first_token:
            self._is_first_token = False
            self._seq_start_idx = len(input_ids[0])

            self._fsm_states = {hash(()): self.fsm.initial_state}
            sequence_states = [self.fsm.initial_state] * len(input_ids)

        else:
            for seq_ids in input_ids.tolist():
                prev_state_key = hash(tuple(seq_ids[self._seq_start_idx : -1]))
                prev_state = self._fsm_states[prev_state_key]

                curr_state_key = hash(tuple(seq_ids[self._seq_start_idx :]))
                curr_state = self.fsm.get_next_state(prev_state, seq_ids[-1])

                self._fsm_states[curr_state_key] = curr_state
                sequence_states.append(curr_state)

        mask = torch.full_like(scores, torch.finfo(scores.dtype).min)
        for i, fsm_state in enumerate(sequence_states):
            allowed_tokens = self.fsm.get_next_instruction(fsm_state).tokens
            allowed_tokens = self._satisfy_image_seq_len_constraint(
                fsm_state, input_ids[i], allowed_tokens
            )
            mask[i, allowed_tokens] = scores[i, allowed_tokens]

        return mask

    def copy(self) -> "ChameleonFSMLogitsProcessor":
        """Return a copy of the logits processor."""
        return ChameleonFSMLogitsProcessor(
            fsm=self.fsm.copy(),
            max_length=self.max_length,
            image_seq_length=self.image_seq_length,
        )

    def _satisfy_image_seq_len_constraint(
        self,
        state: int,
        input_ids: torch.Tensor,
        allowed_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if state == self.fsm.TEXT_STATE:
            # Don't start generating image tokens if we're going to run out of space
            if len(input_ids) + self.image_seq_length + 2 > self.max_length:
                return allowed_tokens[allowed_tokens != self.fsm.boi_token_id]
        elif state == self.fsm.IMAGE_STATE:
            last_block = input_ids[-self.image_seq_length :]
            if len(last_block) < self.image_seq_length:
                return allowed_tokens[allowed_tokens != self.fsm.eoi_token_id]
            # If there are already `image_seq_length` image tokens, don't generate more
            if torch.all(
                torch.isin(last_block, self.image_token_ids_tensor.to(last_block.device))
            ):
                return torch.tensor([self.fsm.eoi_token_id])
        return allowed_tokens


class ChameleonTextOnlyLogitsProcessor:
    """Masks image tokens when generating text-only sequences."""

    def __init__(
        self,
        image_token_ids: List[int],
        max_length: int,
        boi_token_id: int,
        eoi_token_id: int,
        vocab_size: int,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Mask the logits of image tokens to prevent them from being generated. The begin
        and end image markers are allowed, but they must close each other out
        immediately.

        Args
            image_token_ids (`List[int]`): The image token ids, not including the start
                and end image marker tokens.
            max_length (`int`): The maximum sequence length (prompt + generated tokens).
            boi_token_id (`int`): The start image marker token id.
            eoi_token_id (`int`): The end image marker token id.
            vocab_size (`int`): The number of tokens in the tokenizer's vocabulary.
            device (`torch.device`, *optional*, defaults to `torch.device("cpu")`): The
                device on which to perform computations.
        """
        self.max_length = max_length
        self.image_token_ids = image_token_ids
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.vocab_size = vocab_size
        self.eoi_token_mask = torch.zeros(vocab_size, device=device, dtype=torch.bool)
        self.eoi_token_mask[self.eoi_token_id] = 1

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Masks image tokens

        Args:
            input_ids (`torch.Tensor` of shape `(batch, sequence_length)`):
                The input token ids.
            scores (`torch.Tensor` of shape `(batch, vocab_size)`):
                The logits to be biased.

        Returns:
            `torch.Tensor` of shape `(batch, vocab_size)`:  The biased scores.
        """
        scores[:, self.image_token_ids] = torch.finfo(scores.dtype).min
        # Make sure that every image start token is always followed by an image end token
        scores[
            (input_ids[:, -1] == self.boi_token_id).view(-1, 1).expand_as(scores)
            & ~self.eoi_token_mask
        ] = torch.finfo(scores.dtype).min
        return scores

    def copy(self) -> "ChameleonTextOnlyLogitsProcessor":
        """Return a copy of the logits processor."""
        return ChameleonTextOnlyLogitsProcessor(
            image_token_ids=self.image_token_ids,
            max_length=self.max_length,
            boi_token_id=self.boi_token_id,
            eoi_token_id=self.eoi_token_id,
            vocab_size=self.vocab_size,
        )


class ChameleonPrefixAllowedTokensFunc:
    """Bias transformers generation based on a simple finite state machine (FSM) with
    modality as states.

    """

    def __init__(
        self,
        fsm: ChameleonModalityFSMGuide,
        max_new_tokens: int,
        image_seq_length: int = 1024,
    ):
        self.max_new_tokens = max_new_tokens
        self.image_seq_length = image_seq_length

        self.fsm = fsm
        self._fsm_state: DefaultDict[int, int] = defaultdict(int)

        # The generated text with `transformers` include the input token IDs as well,
        # so we use this attribute to keep track of the input token IDs. This allows us
        # to reset the FSM state when the input token IDs change, as well as to only
        # apply the FSM to the generated tokens.
        self._prefix = [-1]

    def __call__(self, batch_id: int, sent: torch.Tensor) -> torch.Tensor:
        """Use the FSM to bias the logits before sampling the next token.

        Args:
            batch_id (`int`): The index of the current batch. Unused.
            sent (`torch.Tensor` of shape `(seq_len,)`): The tokens of the current
                sentence.

        Returns:
            torch.Tensor:
                The indices of the tokens that are allowed to be sampled next.
        """
        input_ids = sent.tolist()

        # If the prefix token IDs have changed we assume that we are dealing with a new
        # sample and reset the FSM state
        if input_ids[: len(self._prefix)] != self._prefix:
            self._fsm_state = defaultdict(int)
            self._prefix = input_ids
            seq_id = hash(tuple([]))
            return self.fsm.get_next_instruction(state=self._fsm_state[0]).tokens

        else:
            # Remove the prefix token IDs from the input token IDs, as the FSM should
            # only be applied to the generated tokens
            input_ids = input_ids[len(self._prefix) :]

            last_token = input_ids[-1]
            last_seq_id = hash(tuple(input_ids[:-1]))
            seq_id = hash(tuple(input_ids))
            self._fsm_state[seq_id] = self.fsm.get_next_state(
                state=self._fsm_state[last_seq_id], token_id=last_token
            )

        allowed_tokens = self.fsm.get_next_instruction(
            state=self._fsm_state[seq_id]
        ).tokens
        allowed_tokens = self._satisfy_image_seq_len_constraint(
            self._fsm_state[seq_id], input_ids, allowed_tokens
        )
        return allowed_tokens

    def _satisfy_image_seq_len_constraint(
        self,
        state: int,
        input_ids: List[int],
        allowed_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if state == self.fsm.TEXT_STATE:
            # Don't start generating image tokens if we're going to run out of space
            if len(input_ids) + self.image_seq_length + 2 > self.max_new_tokens:
                return allowed_tokens[allowed_tokens != self.fsm.boi_token_id]
        elif state == self.fsm.IMAGE_STATE:
            last_block = torch.tensor(input_ids[-self.image_seq_length :])
            if len(last_block) < self.image_seq_length:
                return allowed_tokens[allowed_tokens != self.fsm.eoi_token_id]
            # If there are already `image_seq_length` image tokens, don't generate more
            if torch.all(torch.isin(last_block, torch.tensor(self.fsm.image_token_ids))):
                return torch.tensor([self.fsm.eoi_token_id])
        return allowed_tokens
