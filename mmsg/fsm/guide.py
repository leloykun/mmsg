from functools import partial
from typing import List

import interegular
import torch
from outlines.fsm.guide import RegexGuide, create_states_mapping

from mmsg.fsm.regex import parse_pattern_with_modality_markers
from mmsg.integrations.multimodal_tokenizer import MultimodalTokenizer


class RegexWithMultimodalMarkersGuide(RegexGuide):
    """Guide to generate text in the language of a regular expression."""

    initial_state = 0

    def __init__(
        self,
        regex_string: str,
        tokenizer: MultimodalTokenizer,
        frozen_tokens: List[str],
    ):
        parse_pattern_with_modality_markers_partial = partial(
            parse_pattern_with_modality_markers, tokenizer=tokenizer
        )
        (
            self.states_to_token_maps,
            self.empty_token_ids,
            fsm_finals,
        ) = create_states_mapping(
            regex_string,
            tokenizer,
            parse_pattern_with_modality_markers_partial,
            frozen_tokens,
        )
        self.eos_token_id = tokenizer.eos_token_id
        self.final_states = fsm_finals | {-1}

        self.token_equivalence_classes = self._build_token_equivalence_classes(tokenizer)
        self.token_equivalence_classes_rev = {
            token_id: canonical_token_id
            for canonical_token_id, token_ids in self.token_equivalence_classes.items()
            for token_id in token_ids
        }

        # cache returned masks token masks
        # this increases performance of the mask substantially
        self.states_to_token_mask = self._build_states_to_token_mask(
            self.states_to_token_maps
        )

    def _build_token_equivalence_classes(self, tokenizer: MultimodalTokenizer):
        return {
            tokenizer.vocabulary[tokenizer.image_token]: [
                *tokenizer.image_token_ids,
                tokenizer.vocabulary[tokenizer.image_token],
            ]
        }

    def _build_states_to_token_mask(self, states_to_token_maps):
        states_to_token_mask = {}
        for state, next_tokens_to_end_states in states_to_token_maps.items():
            next_tokens = []
            for token_id in next_tokens_to_end_states:
                next_tokens.extend(
                    self.token_equivalence_classes.get(token_id, [token_id])
                )
            states_to_token_mask[state] = torch.tensor(next_tokens)
        return states_to_token_mask

    def get_next_state(self, state: int, token_id: int) -> int:
        canonical_token_id = self.token_equivalence_classes_rev.get(token_id, token_id)
        return super().get_next_state(state, canonical_token_id)

    @classmethod
    def from_interegular_fsm(
        cls,
        interegular_fsm: interegular.fsm.FSM,
        tokenizer: MultimodalTokenizer,
        frozen_tokens: List[str],
    ):
        raise NotImplementedError()
