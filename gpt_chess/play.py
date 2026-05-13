"""Generate valid chess games by scoring legal UCI moves."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import chess

from gpt_chess.config import DataConfig
from gpt_chess.data import format_prompt
from gpt_chess.modeling import load_adapter_model
from gpt_chess.tokenization import DirectTokenMapper

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel


def encode_tensor(
    mapper: DirectTokenMapper,
    text: str,
    *,
    device: "torch.device",
) -> "torch.Tensor":
    import torch

    ids = mapper.encode(text)
    return torch.tensor([ids], dtype=torch.long, device=device)


def continuation_logprob(
    model: "PreTrainedModel",
    mapper: DirectTokenMapper,
    prompt: str,
    continuation: str,
) -> float:
    """Return log P(continuation | prompt) using direct chess token IDs."""

    import torch

    device = next(model.parameters()).device
    prompt_ids = encode_tensor(mapper, prompt, device=device)
    continuation_ids = encode_tensor(mapper, continuation, device=device)
    full_ids = torch.cat([prompt_ids, continuation_ids], dim=1)

    with torch.no_grad():
        logits = model(full_ids).logits[:, :-1, :]
        targets = full_ids[:, 1:]
        token_log_probs = logits.log_softmax(dim=-1)
        token_log_probs = token_log_probs.gather(
            -1,
            targets.unsqueeze(-1),
        ).squeeze(-1)

    continuation_len = continuation_ids.shape[1]
    return token_log_probs[:, -continuation_len:].sum().item()


def choose_best_legal_uci(
    model: "PreTrainedModel",
    mapper: DirectTokenMapper,
    board: chess.Board,
    data_config: DataConfig = DataConfig(),
) -> str:
    """Score every legal UCI move and return the model's best choice."""

    return score_legal_uci_moves(model, mapper, board, data_config)[0][1]


def score_legal_uci_moves(
    model: "PreTrainedModel",
    mapper: DirectTokenMapper,
    board: chess.Board,
    data_config: DataConfig = DataConfig(),
) -> list[tuple[float, str]]:
    """Return legal UCI moves sorted by model continuation log-probability."""

    prompt = format_prompt(board, data_config)
    scored_moves = [
        (continuation_logprob(model, mapper, prompt, f"{move.uci()}\n"), move.uci())
        for move in board.legal_moves
    ]
    scored_moves.sort(key=lambda pair: pair[0], reverse=True)
    return scored_moves


def generate_valid_game(
    model: "PreTrainedModel",
    mapper: DirectTokenMapper,
    *,
    plies: int = 12,
    data_config: DataConfig = DataConfig(),
) -> str:
    """Generate SAN movetext by repeatedly applying the best legal UCI move."""

    board = chess.Board()
    san_moves: list[str] = []

    for _ in range(plies):
        best_uci = choose_best_legal_uci(model, mapper, board, data_config)
        move = chess.Move.from_uci(best_uci)
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
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", default="chess_model")
    parser.add_argument("--fallback-model-id", default="gpt2")
    parser.add_argument("--plies", type=int, default=12)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--no-fen-metadata",
        action="store_true",
        help="Use only the 71-token board string inside chess tags.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_config = DataConfig(include_fen_metadata=not args.no_fen_metadata)
    model, _, mapper = load_adapter_model(
        args.adapter_dir,
        fallback_model_id=args.fallback_model_id,
        device=args.device,
    )
    print(generate_valid_game(model, mapper, plies=args.plies, data_config=data_config))

