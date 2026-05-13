"""GGUF move generation using llama.cpp."""

from __future__ import annotations

import re
from dataclasses import replace

import chess

from gpt_chess.config import DataConfig, GGUFConfig
from gpt_chess.data import format_prompt


UCI_RE = re.compile(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b")


def download_gguf(config: GGUFConfig = GGUFConfig()) -> str:
    """Download the configured GGUF file from Hugging Face if needed."""

    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=config.repo_id,
        filename=config.filename,
        local_dir=config.local_dir,
    )


def build_move_prompt(board: chess.Board, data_config: DataConfig = DataConfig()) -> str:
    """Build a natural-language prompt for raw GGUF models."""

    legal_moves = " ".join(move.uci() for move in board.legal_moves)
    return (
        f"Board:\n{format_prompt(board, data_config)}"
        f"Legal UCI moves:\n{legal_moves}\n\n"
        "Return exactly one legal UCI move and nothing else."
    )


def extract_uci(text: str) -> str | None:
    """Extract the first UCI-looking move from generated text."""

    match = UCI_RE.search(text)
    if match is None:
        return None
    return match.group(0)


class GGUFMoveGenerator:
    """Generate UCI moves from a local GGUF model."""

    def __init__(
        self,
        model_path: str,
        config: GGUFConfig = GGUFConfig(),
        data_config: DataConfig = DataConfig(),
    ) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as error:
            raise ImportError(
                "GGUF inference requires `llama-cpp-python`. "
                "Install requirements.txt or `pip install llama-cpp-python`."
            ) from error

        kwargs: dict[str, int | bool] = {
            "n_ctx": config.n_ctx,
            "n_gpu_layers": config.n_gpu_layers,
            "verbose": False,
        }
        if config.n_threads is not None:
            kwargs["n_threads"] = config.n_threads

        self.model = Llama(model_path=model_path, **kwargs)
        self.config = config
        self.data_config = data_config

    def generate_uci(self, board: chess.Board) -> tuple[str | None, str]:
        """Generate and parse one UCI move from the current board."""

        prompt = build_move_prompt(board, self.data_config)
        if self.config.use_chat_template:
            output = self.model.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are choosing a chess move. Reply with exactly "
                            "one legal UCI move and no explanation."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            text = output["choices"][0]["message"]["content"] or ""
            if text.strip():
                return extract_uci(text), text

        output = self.model(
            (
                "You are choosing a chess move. Return exactly one legal UCI move "
                f"and nothing else.\n\n{prompt}\nMove:"
            ),
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            stop=["\n"],
        )
        text = output["choices"][0]["text"]
        return extract_uci(text), text


def load_default_qwen35_q4_0(
    *,
    local_dir: str | None = None,
    n_gpu_layers: int | None = None,
) -> tuple[GGUFMoveGenerator, str]:
    """Download and load Unsloth Qwen3.5 0.8B Q4_0 GGUF."""

    config = GGUFConfig()
    if local_dir is not None:
        config = replace(config, local_dir=local_dir)
    if n_gpu_layers is not None:
        config = replace(config, n_gpu_layers=n_gpu_layers)

    model_path = download_gguf(config)
    return GGUFMoveGenerator(model_path, config), model_path

