"""Generate a chess game with a raw GGUF model."""

from __future__ import annotations

import argparse
import random
from dataclasses import replace

import chess

from gpt_chess.config import DEFAULT_CONFIG, DataConfig, GGUFConfig
from gpt_chess.gguf import GGUFMoveGenerator, download_gguf


def generate_game(
    generator: GGUFMoveGenerator,
    *,
    plies: int,
    seed: int,
    fallback_to_random: bool,
) -> str:
    rng = random.Random(seed)
    board = chess.Board()
    san_moves: list[str] = []

    for _ in range(plies):
        predicted, raw_output = generator.generate_uci(board)
        legal_moves = [move.uci() for move in board.legal_moves]

        if predicted not in legal_moves:
            if not fallback_to_random:
                san_moves.append(f"[invalid output: {raw_output.strip()!r}]")
                break
            predicted = rng.choice(legal_moves)

        move = chess.Move.from_uci(predicted)
        san = board.san(move)
        if board.turn == chess.WHITE:
            san_moves.append(f"{board.fullmove_number}. {san}")
        else:
            san_moves.append(san)
        board.push(move)

        if board.is_game_over():
            break

    return " ".join(san_moves)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf-model-path", default=None)
    parser.add_argument("--gguf-repo-id", default=DEFAULT_CONFIG.gguf.repo_id)
    parser.add_argument("--gguf-filename", default=DEFAULT_CONFIG.gguf.filename)
    parser.add_argument("--gguf-local-dir", default=DEFAULT_CONFIG.gguf.local_dir)
    parser.add_argument("--gguf-n-ctx", type=int, default=DEFAULT_CONFIG.gguf.n_ctx)
    parser.add_argument(
        "--gguf-n-gpu-layers",
        type=int,
        default=DEFAULT_CONFIG.gguf.n_gpu_layers,
    )
    parser.add_argument(
        "--gguf-n-threads",
        type=int,
        default=DEFAULT_CONFIG.gguf.n_threads,
    )
    parser.add_argument(
        "--gguf-max-tokens",
        type=int,
        default=DEFAULT_CONFIG.gguf.max_tokens,
    )
    parser.add_argument("--plies", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fallback-to-random",
        action="store_true",
        help="Keep the game legal if the GGUF model emits an illegal move.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gguf_config = replace(
        GGUFConfig(),
        repo_id=args.gguf_repo_id,
        filename=args.gguf_filename,
        local_dir=args.gguf_local_dir,
        n_ctx=args.gguf_n_ctx,
        n_gpu_layers=args.gguf_n_gpu_layers,
        n_threads=args.gguf_n_threads,
        max_tokens=args.gguf_max_tokens,
    )
    model_path = args.gguf_model_path or download_gguf(gguf_config)
    generator = GGUFMoveGenerator(model_path, gguf_config, DataConfig())
    print(
        generate_game(
            generator,
            plies=args.plies,
            seed=args.seed,
            fallback_to_random=args.fallback_to_random,
        )
    )

