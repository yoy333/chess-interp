"""Lightweight tests that avoid downloading models or datasets."""

from __future__ import annotations

import chess

from gpt_chess.config import DataConfig, GGUFConfig
from gpt_chess.data import (
    expanded_board,
    format_prompt,
    iter_game_examples,
    iter_game_texts,
    read_game,
)
from gpt_chess.gguf import build_move_prompt, extract_uci
from gpt_chess.tokenization import DIRECT_TOKEN_CHARS, DirectTokenMapper


def make_dummy_mapper() -> DirectTokenMapper:
    char_to_id = {char: index for index, char in enumerate(sorted(DIRECT_TOKEN_CHARS))}
    return DirectTokenMapper(char_to_id=char_to_id, fallback_token_id=len(char_to_id))


def run() -> None:
    board = chess.Board()
    board_text = expanded_board(board)
    assert len(board_text) == 71

    prompt = format_prompt(board, DataConfig())
    assert prompt.startswith("<CHESS>\n")
    assert prompt.endswith("\n</CHESS>\n")

    game = read_game("1. e4 e5 2. Nf3 Nc6 1-0")
    assert game is not None
    examples = list(iter_game_examples(game, make_dummy_mapper(), DataConfig()))
    assert len(examples) == 4
    assert all(len(example["input_ids"]) == len(example["labels"]) for example in examples)

    line_rows = [
        '[Event "Rated Classical game"]',
        "",
        "1. e4 e5 2. Nf3 Nc6 1-0",
    ]
    assert len(list(iter_game_texts(line_rows))) == 1

    gguf_config = GGUFConfig()
    assert gguf_config.repo_id == "unsloth/Qwen3.5-9B-GGUF"
    assert gguf_config.filename == "Qwen3.5-9B-Q4_0.gguf"
    assert extract_uci("best move: e2e4") == "e2e4"
    assert "Legal UCI moves" in build_move_prompt(board)

    print("gpt_chess smoke tests passed")


if __name__ == "__main__":
    run()

