"""Direct token mapping for chess prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Mapping

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


CHESS_START = "<CHESS>"
CHESS_END = "</CHESS>"
PAD_TOKEN = "<PAD>"

DIRECT_TOKEN_CHARS = frozenset(
    "prnbqkPRNBQK./-0123456789abcdefghw<>CHES\n "
)


def load_tokenizer(model_id: str) -> tuple["PreTrainedTokenizerBase", int]:
    """Load a tokenizer and add the chess/pad tokens used by saved adapters."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    special_tokens: dict[str, str | list[str]] = {
        "additional_special_tokens": [CHESS_START, CHESS_END],
    }
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = PAD_TOKEN

    added_tokens = tokenizer.add_special_tokens(special_tokens)
    return tokenizer, added_tokens


@dataclass(frozen=True)
class DirectTokenMapper:
    """Map each supported chess character to exactly one model token."""

    char_to_id: Mapping[str, int]
    fallback_token_id: int

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: "PreTrainedTokenizerBase",
        chars: set[str] | frozenset[str] = DIRECT_TOKEN_CHARS,
    ) -> "DirectTokenMapper":
        char_to_id: dict[str, int] = {}
        bad_chars: list[str] = []

        for char in chars:
            token_ids = tokenizer.encode(char, add_special_tokens=False)
            if len(token_ids) != 1:
                bad_chars.append(repr(char))
                continue
            char_to_id[char] = token_ids[0]

        if bad_chars:
            raise ValueError(
                "Each chess character must map to exactly one token. "
                f"Failed characters: {', '.join(sorted(bad_chars))}"
            )

        fallback = tokenizer.eos_token_id
        if fallback is None:
            raise ValueError("Tokenizer must define an EOS token for fallback IDs.")

        return cls(char_to_id=char_to_id, fallback_token_id=fallback)

    def encode(self, text: str, *, strict: bool = True) -> list[int]:
        """Encode text with direct character-to-token IDs."""

        if strict:
            missing = sorted({char for char in text if char not in self.char_to_id})
            if missing:
                display = ", ".join(repr(char) for char in missing)
                raise ValueError(f"Text contains unmapped chess characters: {display}")

        return [self.char_to_id.get(char, self.fallback_token_id) for char in text]

