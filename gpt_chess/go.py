"""UCI-style ``go`` command for transformer chess move generators."""

from __future__ import annotations

import argparse
import logging
import os
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import chess

from gpt_chess.config import DEFAULT_CONFIG, DataConfig, GGUFConfig, ModelConfig
from gpt_chess.play import choose_best_legal_uci

if TYPE_CHECKING:
    from gpt_chess.gguf import GGUFMoveGenerator
    from gpt_chess.tokenization import DirectTokenMapper


GoResult = tuple[str | None, str | None]
AsyncCallback = Callable[[GoResult], None]


class EngineStateException(RuntimeError):
    """Raised when the transformer engine is in an invalid command state."""


class MoveGenerator(Protocol):
    """Protocol for raw model generators such as ``GGUFMoveGenerator``."""

    def generate_uci(self, board: chess.Board) -> tuple[str | None, str]:
        """Generate one UCI move plus the raw model text."""


@dataclass
class TransformerUCIEngine:
    """Small UCI-facing wrapper around GPT adapter or Qwen/GGUF move replies.

    UCI search controls are accepted by :meth:`go` for protocol compatibility,
    but transformer inference does not use depth, node, mate, clock, ponder, or
    infinite-search settings. The method immediately asks the configured model
    for a move on the current board and returns ``(bestmove, None)``.
    """

    board: chess.Board = field(default_factory=chess.Board)
    model: Any | None = None
    mapper: "DirectTokenMapper | None" = None
    generator: MoveGenerator | None = None
    data_config: DataConfig = field(default_factory=DataConfig)
    require_legal_reply: bool = True
    _calculating: bool = field(default=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @classmethod
    def from_adapter(
        cls,
        model: Any,
        mapper: "DirectTokenMapper",
        *,
        board: chess.Board | None = None,
        data_config: DataConfig = DataConfig(),
    ) -> "TransformerUCIEngine":
        """Create a UCI wrapper for GPT-2/Qwen Hugging Face adapter scoring."""

        return cls(
            board=board.copy(stack=False) if board is not None else chess.Board(),
            model=model,
            mapper=mapper,
            data_config=data_config,
        )

    @classmethod
    def from_gguf(
        cls,
        generator: "GGUFMoveGenerator",
        *,
        board: chess.Board | None = None,
        require_legal_reply: bool = True,
    ) -> "TransformerUCIEngine":
        """Create a UCI wrapper for Qwen/GGUF raw move generation."""

        return cls(
            board=board.copy(stack=False) if board is not None else chess.Board(),
            generator=generator,
            require_legal_reply=require_legal_reply,
        )

    def set_position(self, board: chess.Board) -> None:
        """Set the current position used by subsequent ``go`` calls."""

        self.board = board.copy(stack=False)

    def push_uci(self, uci: str) -> None:
        """Apply one UCI move to the current position."""

        self.board.push_uci(uci)

    def go(
        self,
        searchmoves: Sequence[str | chess.Move] | None = None,
        ponder: bool = False,
        wtime: int | None = None,
        btime: int | None = None,
        winc: int | None = None,
        binc: int | None = None,
        movestogo: int | None = None,
        depth: int | None = None,
        nodes: int | None = None,
        mate: int | None = None,
        movetime: int | None = None,
        infinite: bool = False,
        async_callback: AsyncCallback | None = None,
    ) -> GoResult:
        """Return the model's best move for the current position.

        The parameters mirror the UCI ``go`` command and python-chess engine
        API. They are intentionally ignored because these transformer backends
        do not search by depth, nodes, mate distance, or clock budget.

        Raises:
            EngineStateException: If another ``go`` call is already running.
        """

        del (
            searchmoves,
            ponder,
            wtime,
            btime,
            winc,
            binc,
            movestogo,
            depth,
            nodes,
            mate,
            movetime,
            infinite,
            async_callback,
        )

        with self._lock:
            if self._calculating:
                raise EngineStateException("engine is already calculating")
            self._calculating = True

        try:
            result = (self._bestmove(), None)
            return result
        finally:
            with self._lock:
                self._calculating = False

    def _bestmove(self) -> str | None:
        if self.board.is_game_over() or not any(self.board.legal_moves):
            return None

        if self.generator is not None:
            move, _raw_output = self.generator.generate_uci(self.board)
            return self._validated_uci(move)

        if self.model is not None and self.mapper is not None:
            move = choose_best_legal_uci(
                self.model,
                self.mapper,
                self.board,
                self.data_config,
            )
            return self._validated_uci(move)

        raise EngineStateException("no GPT adapter or Qwen/GGUF generator configured")

    def _validated_uci(self, move: str | None) -> str | None:
        if move is None:
            return None

        try:
            chess_move = chess.Move.from_uci(move)
        except ValueError:
            return None

        if self.require_legal_reply and chess_move not in self.board.legal_moves:
            return None
        return move


_default_engine = TransformerUCIEngine()


def configure(
    *,
    board: chess.Board | None = None,
    model: Any | None = None,
    mapper: "DirectTokenMapper | None" = None,
    generator: MoveGenerator | None = None,
    data_config: DataConfig = DataConfig(),
    require_legal_reply: bool = True,
) -> TransformerUCIEngine:
    """Configure the module-level engine used by :func:`go`."""

    global _default_engine
    _default_engine = TransformerUCIEngine(
        board=board.copy(stack=False) if board is not None else chess.Board(),
        model=model,
        mapper=mapper,
        generator=generator,
        data_config=data_config,
        require_legal_reply=require_legal_reply,
    )
    return _default_engine


def set_position(board: chess.Board) -> None:
    """Set the current board on the module-level engine."""

    _default_engine.set_position(board)


def push_uci(uci: str) -> None:
    """Apply one UCI move to the module-level engine position."""

    _default_engine.push_uci(uci)


def go(
    searchmoves: Sequence[str | chess.Move] | None = None,
    ponder: bool = False,
    wtime: int | None = None,
    btime: int | None = None,
    winc: int | None = None,
    binc: int | None = None,
    movestogo: int | None = None,
    depth: int | None = None,
    nodes: int | None = None,
    mate: int | None = None,
    movetime: int | None = None,
    infinite: bool = False,
    async_callback: AsyncCallback | None = None,
) -> GoResult:
    """Run UCI ``go`` on the module-level transformer engine."""

    return _default_engine.go(
        searchmoves=searchmoves,
        ponder=ponder,
        wtime=wtime,
        btime=btime,
        winc=winc,
        binc=binc,
        movestogo=movestogo,
        depth=depth,
        nodes=nodes,
        mate=mate,
        movetime=movetime,
        infinite=infinite,
        async_callback=async_callback,
    )


def board_from_args(args: argparse.Namespace) -> chess.Board:
    """Build the current board from CLI position arguments."""

    board = chess.Board() if args.fen is None else chess.Board(args.fen)
    for move in args.moves:
        board.push_uci(move)
    return board


def quiet_model_loading_logs() -> None:
    """Keep CLI output focused on the UCI ``bestmove`` reply."""

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)


def build_engine(args: argparse.Namespace) -> TransformerUCIEngine:
    """Load the requested transformer backend and wrap it as a UCI engine."""

    board = board_from_args(args)

    if args.backend == "base":
        from gpt_chess.modeling import load_base_model_and_tokenizer

        data_config = DataConfig(include_fen_metadata=not args.no_fen_metadata)
        model_config = replace(
            ModelConfig(),
            model_id=args.model_id,
            device_map=None,
            use_lora=False,
        )
        model, _, mapper = load_base_model_and_tokenizer(model_config)
        if args.device is not None:
            model = model.to(args.device)
        model.eval()
        return TransformerUCIEngine.from_adapter(
            model,
            mapper,
            board=board,
            data_config=data_config,
        )

    if args.backend == "adapter":
        from gpt_chess.modeling import load_adapter_model

        adapter_config = Path(args.adapter_dir) / "adapter_config.json"
        if not adapter_config.is_file():
            raise FileNotFoundError(
                "Adapter backend requires a trained PEFT adapter at "
                f"{args.adapter_dir!r}, but adapter_config.json was not found. "
                "Train one first, or use --backend base for plain GPT-2."
            )
        data_config = DataConfig(include_fen_metadata=not args.no_fen_metadata)
        model, _, mapper = load_adapter_model(
            args.adapter_dir,
            fallback_model_id=args.fallback_model_id,
            device=args.device,
        )
        return TransformerUCIEngine.from_adapter(
            model,
            mapper,
            board=board,
            data_config=data_config,
        )

    from gpt_chess.gguf import GGUFMoveGenerator, download_gguf

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
    return TransformerUCIEngine.from_gguf(
        generator,
        board=board,
        require_legal_reply=not args.allow_illegal_reply,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["base", "adapter", "gguf"],
        default="base",
        help="Transformer reply backend to use.",
    )
    parser.add_argument("--model-id", default=DEFAULT_CONFIG.model.model_id)
    parser.add_argument(
        "--fen",
        default=None,
        help="Position FEN. Defaults to the normal starting position.",
    )
    parser.add_argument(
        "--moves",
        nargs="*",
        default=[],
        help="UCI moves to apply after --fen or the starting position.",
    )

    parser.add_argument("--adapter-dir", default=DEFAULT_CONFIG.model.output_dir)
    parser.add_argument("--fallback-model-id", default=DEFAULT_CONFIG.model.model_id)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--no-fen-metadata",
        action="store_true",
        help="Use only the 71-token board string inside chess tags.",
    )

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
    parser.add_argument(
        "--allow-illegal-reply",
        action="store_true",
        help="Print a parsed GGUF move even if it is illegal in the position.",
    )

    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--mate", type=int, default=None)
    parser.add_argument("--movetime", type=int, default=None)
    parser.add_argument("--wtime", type=int, default=None)
    parser.add_argument("--btime", type=int, default=None)
    parser.add_argument("--winc", type=int, default=None)
    parser.add_argument("--binc", type=int, default=None)
    parser.add_argument("--movestogo", type=int, default=None)
    parser.add_argument("--ponder", action="store_true")
    parser.add_argument("--infinite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quiet_model_loading_logs()
    engine = build_engine(args)
    bestmove, ponder = engine.go(
        ponder=args.ponder,
        wtime=args.wtime,
        btime=args.btime,
        winc=args.winc,
        binc=args.binc,
        movestogo=args.movestogo,
        depth=args.depth,
        nodes=args.nodes,
        mate=args.mate,
        movetime=args.movetime,
        infinite=args.infinite,
    )

    if bestmove is None:
        print("bestmove 0000")
    elif ponder is None:
        print(f"bestmove {bestmove}")
    else:
        print(f"bestmove {bestmove} ponder {ponder}")


if __name__ == "__main__":
    main()
