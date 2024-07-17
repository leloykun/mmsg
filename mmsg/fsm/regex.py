from interegular.patterns import (
    _DOT,
    InvalidSyntax,
    Pattern,
    Unsupported,
    _CharGroup,
    _combine_char_groups,
    _Concatenation,
    _ParsePattern,
)
from interegular.utils.simple_parser import NoMatch, nomatch

from mmsg.integrations.multimodal_tokenizer import MultimodalTokenizer


class _ParsePatternWithModalityMarkers(_ParsePattern):
    def __init__(self, data: str, tokenizer: MultimodalTokenizer):
        super().__init__(data)
        self.flags = None
        self.tokenizer = tokenizer
        self.special_image_tokens_group = _CharGroup(
            frozenset(
                {
                    tokenizer.image_token,
                    tokenizer.boi_token,
                    tokenizer.eoi_token,
                }
            ),
            negated=False,
        )

    def parse(self):
        try:
            return super().parse()
        except NoMatch:
            raise InvalidSyntax

    def atom(self):
        if self.static_b("["):
            return self.repetition(self.chargroup())
        elif self.static_b("\\"):
            return self.repetition(self.escaped())
        elif self.static_b("."):
            return self.repetition(_DOT)
        elif self.static_b("$"):
            raise Unsupported("'$'")
        elif self.static_b("^"):
            raise Unsupported("'^'")
        elif self.static_b(rf"{self.tokenizer.image_token}"):
            # TODO: handle other modalities later
            b = self.repetition(
                _CharGroup(frozenset({self.tokenizer.image_token}), False)
            )
            return _Concatenation(
                tuple(
                    [
                        _CharGroup(frozenset({self.tokenizer.boi_token}), False),
                        b,
                        _CharGroup(frozenset({self.tokenizer.eoi_token}), False),
                    ]
                )
            )
        else:
            c = self.any_but(*self.SPECIAL_CHARS_STANDARD)
            return self.repetition(_CharGroup(frozenset({c}), False))

    def chargroup(self):
        if self.static_b("^"):
            raise Unsupported("'^'")
        else:
            negate = False
        groups = []
        while True:
            try:
                groups.append(self.chargroup_inner())
            except nomatch:
                break
        self.static("]")
        if len(groups) == 1:
            f = tuple(groups)[0]
            return _CharGroup(f.chars, negate ^ f.negated)
        elif len(groups) == 0:
            return _CharGroup(frozenset({}), False)
        else:
            return _combine_char_groups(*groups, negate=negate)


def parse_pattern_with_modality_markers(
    pattern: str, tokenizer: MultimodalTokenizer
) -> Pattern:
    p = _ParsePatternWithModalityMarkers(pattern, tokenizer)
    out = p.parse()
    out = out.simplify()
    return out
