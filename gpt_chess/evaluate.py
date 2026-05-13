"""Evaluate chess move quality on held-out PGN positions."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import chess

from gpt_chess.config import DEFAULT_CONFIG, DataConfig, GGUFConfig
from gpt_chess.data import iter_game_texts, read_game
from gpt_chess.gguf import GGUFMoveGenerator, download_gguf
from gpt_chess.modeling import load_adapter_model
from gpt_chess.play import score_legal_uci_moves


@dataclass(frozen=True)
class EvalPosition:
    board: chess.Board
    target_uci: str
    ply_index: int


@dataclass(frozen=True)
class MovePrediction:
    target_uci: str
    predicted_uci: str | None
    is_legal: bool
    rank: int | None
    raw_output: str | None = None


def iter_positions(
    dataset,
    *,
    data_config: DataConfig,
    limit_positions: int,
    max_games: int | None,
):
    """Yield target next-move positions from PGN text."""

    yielded = 0
    game_texts = iter_game_texts(row[data_config.text_column] for row in dataset)
    for game_index, game_text in enumerate(game_texts):
        if max_games is not None and game_index >= max_games:
            break

        game = read_game(game_text)
        if game is None:
            continue

        board = game.board()
        moves = list(game.mainline_moves())
        for ply_index, move in enumerate(moves):
            use_position = (
                data_config.position_policy == "all_plies"
                or ply_index == len(moves) - 1
            )
            if use_position:
                yield EvalPosition(
                    board=board.copy(stack=False),
                    target_uci=move.uci(),
                    ply_index=ply_index,
                )
                yielded += 1
                if yielded >= limit_positions:
                    return
            board.push(move)


def evaluate_adapter(
    positions: list[EvalPosition],
    *,
    adapter_dir: str,
    fallback_model_id: str,
    data_config: DataConfig,
    device: str | None,
) -> list[MovePrediction]:
    model, _, mapper = load_adapter_model(
        adapter_dir,
        fallback_model_id=fallback_model_id,
        device=device,
    )

    predictions: list[MovePrediction] = []
    for position in positions:
        scored_moves = score_legal_uci_moves(
            model,
            mapper,
            position.board,
            data_config,
        )
        ranked_moves = [uci for _, uci in scored_moves]
        predicted = ranked_moves[0]
        rank = ranked_moves.index(position.target_uci) + 1
        predictions.append(
            MovePrediction(
                target_uci=position.target_uci,
                predicted_uci=predicted,
                is_legal=True,
                rank=rank,
            )
        )
    return predictions


def evaluate_random(
    positions: list[EvalPosition],
    *,
    seed: int,
) -> list[MovePrediction]:
    rng = random.Random(seed)
    predictions: list[MovePrediction] = []

    for position in positions:
        legal_moves = [move.uci() for move in position.board.legal_moves]
        rng.shuffle(legal_moves)
        predicted = legal_moves[0]
        rank = legal_moves.index(position.target_uci) + 1
        predictions.append(
            MovePrediction(
                target_uci=position.target_uci,
                predicted_uci=predicted,
                is_legal=True,
                rank=rank,
            )
        )
    return predictions


def evaluate_gguf(
    positions: list[EvalPosition],
    *,
    model_path: str,
    gguf_config: GGUFConfig,
    data_config: DataConfig,
) -> list[MovePrediction]:
    generator = GGUFMoveGenerator(model_path, gguf_config, data_config)
    predictions: list[MovePrediction] = []

    for position in positions:
        predicted, raw_output = generator.generate_uci(position.board)
        legal_moves = [move.uci() for move in position.board.legal_moves]
        is_legal = predicted in legal_moves if predicted is not None else False
        predictions.append(
            MovePrediction(
                target_uci=position.target_uci,
                predicted_uci=predicted,
                is_legal=is_legal,
                rank=None,
                raw_output=raw_output,
            )
        )
    return predictions


def summarize(predictions: list[MovePrediction]) -> dict[str, float | int]:
    total = len(predictions)
    legal = sum(prediction.is_legal for prediction in predictions)
    exact = sum(
        prediction.predicted_uci == prediction.target_uci for prediction in predictions
    )
    ranks = [prediction.rank for prediction in predictions if prediction.rank]
    reciprocal_ranks = [1 / rank for rank in ranks]

    return {
        "positions": total,
        "ranked_positions": len(ranks),
        "legal_rate": legal / total if total else 0.0,
        "exact_match_rate": exact / total if total else 0.0,
        "mean_rank": sum(ranks) / len(ranks) if ranks else 0.0,
        "mean_reciprocal_rank": (
            sum(reciprocal_ranks) / len(reciprocal_ranks)
            if reciprocal_ranks
            else 0.0
        ),
    }


def load_eval_dataset(dataset_name: str, dataset_split: str):
    from datasets import load_dataset

    return load_dataset(dataset_name, split=dataset_split)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["adapter", "gguf", "random"],
        default="adapter",
    )
    parser.add_argument("--adapter-dir", default=DEFAULT_CONFIG.model.output_dir)
    parser.add_argument("--fallback-model-id", default=DEFAULT_CONFIG.model.model_id)
    parser.add_argument("--dataset-name", default=DEFAULT_CONFIG.data.dataset_name)
    parser.add_argument(
        "--dataset-split",
        default="train[500:600]",
        help="Use a held-out slice when possible.",
    )
    parser.add_argument("--limit-positions", type=int, default=100)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument(
        "--position-policy",
        choices=["all_plies", "final_ply"],
        default=DEFAULT_CONFIG.data.position_policy,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", default=None)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_config = replace(
        DataConfig(),
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        position_policy=args.position_policy,
    )

    dataset = load_eval_dataset(args.dataset_name, args.dataset_split)
    positions = list(
        iter_positions(
            dataset,
            data_config=data_config,
            limit_positions=args.limit_positions,
            max_games=args.max_games,
        )
    )

    if args.backend == "adapter":
        predictions = evaluate_adapter(
            positions,
            adapter_dir=args.adapter_dir,
            fallback_model_id=args.fallback_model_id,
            data_config=data_config,
            device=args.device,
        )
    elif args.backend == "random":
        predictions = evaluate_random(positions, seed=args.seed)
    else:
        gguf_config = GGUFConfig(
            repo_id=args.gguf_repo_id,
            filename=args.gguf_filename,
            local_dir=args.gguf_local_dir,
            n_ctx=args.gguf_n_ctx,
            n_gpu_layers=args.gguf_n_gpu_layers,
            n_threads=args.gguf_n_threads,
            max_tokens=args.gguf_max_tokens,
        )
        model_path = args.gguf_model_path or download_gguf(gguf_config)
        predictions = evaluate_gguf(
            positions,
            model_path=model_path,
            gguf_config=gguf_config,
            data_config=data_config,
        )

    summary = summarize(predictions)
    print(json.dumps(summary, indent=2))

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "predictions": [asdict(prediction) for prediction in predictions],
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()

