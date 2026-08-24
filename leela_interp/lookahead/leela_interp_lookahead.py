from torch import float32
training_dtype = float32

def printonce(thing):
    from os import getenv
    rank = int(getenv("LOCAL_RANK"))
    if rank == 0:
        print(thing)

def loadDataset(name):
    from dotenv import load_dotenv
    from huggingface_hub import login
    from os import getenv

    load_dotenv()  # Load environment variables from .env file

    login(token=getenv("HUGGINGFACE_TOKEN"))

    from datasets import IterableDataset, iterable_dataset, load_dataset

    printonce("1. Load actual dataset (streaming)")
    try:
        data = load_dataset(name, split="train", streaming=True)
        return data
    except Exception as e:
        printonce(f"Failed to load: {e}")


def toDataset(rows_iter, num_samples=None):
    """Materialize the next num_samples rows of an iterator into a Dataset.

    Takes an *iterator* rather than the dataset itself, deliberately: the
    iterator is created once and kept across batches, so each call consumes
    rows the previous call didn't see. Re-iterating the dataset instead would
    hand every batch the same rows from the head of the stream.

    Only one batch of rows is ever resident, which is also what keeps memory
    flat as num_batches grows. num_samples=None drains the iterator.
    """
    from itertools import islice

    from datasets import Dataset

    printonce(f"1b. Materialize next {num_samples} rows into a Dataset")
    rows = list(rows_iter if num_samples is None else islice(rows_iter, num_samples))
    if not rows:
        return None
    dataset = Dataset.from_list(rows)
    printonce(f"Materialized {len(dataset)} rows.")
    return dataset


def generateActivations(data, num_samples):
    import sys
    import os
    from pathlib import Path

    current_dir = Path.cwd().resolve()
    repo_root = current_dir.parent.parent if current_dir.name == "lookahead" else current_dir
    leela_pytorch_impl_dir = repo_root / "leela_pytorch_impl"

    for path in (repo_root, leela_pytorch_impl_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.append(path_str)

    from model import Lc0Model
    from leela_board import LeelaBoard
    import torch
    import numpy as np

    # %%
    # One GPU per rank, matching how DeepSpeed places the probe. Falls back to
    # CPU when the box has no CUDA device.
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    else:
        device = "cpu"

    ONNX_MODEL_NAME = 'lc0.onnx'
    lc0 = Lc0Model(
        onnx_model_path=os.path.join(leela_pytorch_impl_dir, ONNX_MODEL_NAME),
        device=device,
    )

    printonce("2. Extract Boards and Labels (Balanced)")
    boards = []
    num_boards = 0

    target_num = num_samples

    # These arrays will contain a number 0-63 that represent a square on the chessboard
    first_source_indexes = np.zeros(target_num, dtype=int)
    first_target_indexes = np.zeros(target_num, dtype=int)
    lookahead_source_indexes = np.zeros(target_num, dtype=int)
    lookahead_target_indexes = np.zeros(target_num, dtype=int)

    # must be minimum 1
    LOOKAHEAD_NUM = 2

    known_fens = set()
    for i, item in enumerate(data):
        # fen = item.get('FEN', item.get('fen'))
        fen = item.get("f", "")
        if fen in known_fens:
            continue;
        else:
            known_fens.add(fen)
            
        board = LeelaBoard.from_fen(fen)
        # Ex: ["e2e4", "e7e5"]
        moves = item["m"].split(' ')
        # moves = item['Moves'].split(' ')

        # 1 because the puzzle must start with a move
        # *2 because a "move" is a pair of moves
        if len(moves)<1+LOOKAHEAD_NUM*2:
            continue;

        # Puzzle data starts one move from the start of the puzzle
        # This move is given to the player. i.e. it is played by enemy
        start_uci = moves[0]
        board.push_uci(start_uci)
        # We consider this the first real move.
        # This is the first move that a player would have to find
        first_uci = moves[1]
        lookahead_uci = moves[1+2*LOOKAHEAD_NUM]

        first_source_indexes[num_boards] = board.sq2idx(first_uci[0:2])
        first_target_indexes[num_boards] = board.sq2idx(first_uci[2:4])
        lookahead_source_indexes[num_boards] = board.sq2idx(lookahead_uci[0:2])
        lookahead_target_indexes[num_boards] = board.sq2idx(lookahead_uci[2:4])

        # only appended once the puzzle is known-good, so boards stays aligned
        # with the index arrays
        boards.append(board)
        num_boards += 1

        if num_boards >= target_num:
            break;

    printonce(f"Found {num_boards} positions.")
    # shouldn't really happen but in theory i suppose i should include
    if(num_boards != target_num):
        printonce(f"WARNING: only {num_boards} of {target_num} samples available")

    # Trim to what we actually filled; the arrays were sized optimistically.
    first_source_indexes = first_source_indexes[:num_boards]
    first_target_indexes = first_target_indexes[:num_boards]
    lookahead_source_indexes = lookahead_source_indexes[:num_boards]
    lookahead_target_indexes = lookahead_target_indexes[:num_boards]

    # %%
    labels = np.array(lookahead_target_indexes, dtype=int)

    # %%
    printonce(f"3. Move model and inputs to {device}")
    lc0.to(device)  # keep weights and inputs on the same device
    lc0._device = device
    inputs = lc0.make_inputs(boards).to(device)

    # %%
    from nnsight import NNsight

    printonce("Run forward pass and extract activations with NNsight...")
    # has 15 layers total (0-14)
    LAYER_TO_PROBE = 11

    # Wrap our model in NNsight
    nnsight_model = NNsight(lc0)

    with nnsight_model.trace(inputs) as tracer:
        # Grab the output of the residual stream
        hidden_states = nnsight_model._lc0_model.post_mlp[LAYER_TO_PROBE].output.save()

    printonce("Extracting features")
    X_activations = hidden_states if isinstance(hidden_states, torch.Tensor) else hidden_states.value
    printonce(f"Activations shape: {X_activations.shape}")

    # Index tensors have to sit on the same device as the activations they index.
    batch_idx = torch.arange(num_boards, dtype=int, device=X_activations.device)
    i1 = torch.from_numpy(first_source_indexes).to(X_activations.device)
    i2 = torch.from_numpy(first_target_indexes).to(X_activations.device)

    # Represent the hiddens layers for the targets
    # Should be of the form [batches, h_dim]
    first_source_h = X_activations[batch_idx, i1].to(device)
    first_target_h = X_activations[batch_idx, i2].to(device)

    # TODO: fix to be correct type in first place

    # Hand the batch back on CPU: train_batch moves each micro-batch to the GPU
    # itself, so holding the whole batch of activations in VRAM alongside the
    # probe would only eat memory the lc0 forward pass needs.
    X_activations = X_activations.to(training_dtype)
    first_target_h = first_target_h.to(training_dtype)

    labels_torch = torch.from_numpy(labels)


    return X_activations, first_target_h, labels_torch

def train_batch(model_engine, X_activations, first_target_h, labels_torch):
    # ---- Data ----
    # These are this rank's shard only; see main() for the split.
    # X_activations: (N, num_squares, d_hidden)
    # first_target_h: (N, d_hidden)
    # labels_torch: (N,) — long tensor of correct square index

    # %%
    import torch
    import torch.nn as nn
    import deepspeed.comm as dist
    from torch.utils.data import TensorDataset, DataLoader

    # ---- Config ----
    # DeepSpeed owns the optimizer and lr (see ds_config.json), so this only
    # needs the batching knobs.
    batch_size = model_engine.train_micro_batch_size_per_gpu()
    num_epochs = 7
    val_split = 0.3

    dataset = TensorDataset(X_activations, first_target_h, labels_torch)

    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()  # expects raw logits + integer class labels

    # Every rank must run the same number of optimizer steps or the gradient
    # all-reduce inside model_engine.step() deadlocks. Shards can come out
    # uneven (a shard may run short of usable puzzles), so agree on the
    # smallest step count and have every rank stop there.
    steps_per_epoch = torch.tensor(len(data_loader), dtype=torch.long, device=model_engine.device)
    dist.all_reduce(steps_per_epoch, op=dist.ReduceOp.MIN)
    steps_per_epoch = int(steps_per_epoch.item())
    if steps_per_epoch == 0:
        raise RuntimeError(
            f"a rank has fewer than {batch_size} training samples; "
            "lower train_batch_size or raise samples_per_epoch"
        )

    def gather_metrics(total_loss, correct, total):
        """Sum a rank's running totals across all ranks so the numbers rank 0
        prints describe the whole run, not just its own shard."""
        t = torch.tensor([total_loss, correct, total], dtype=torch.float64, device=model_engine.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss, correct, total = t.tolist()
        return total_loss / total, correct / total

    def evaluate(model_engine, loader):
        model_engine.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for h_y_all, h_t1, labels in loader:
                d = model_engine.device
                h_y_all, h_t1, labels = h_y_all.to(d), h_t1.to(d), labels.to(d)

                logits = model_engine(h_y_all, h_t1)
                loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return gather_metrics(total_loss, correct, total)

    for epoch in range(num_epochs):
        model_engine.train()
        running_loss, correct, total = 0.0, 0, 0

        for step, (h_y_all, h_t1, labels) in enumerate(data_loader):
            if step >= steps_per_epoch:
                break
            d = model_engine.device
            h_y_all = h_y_all.to(d)      # (B, num_squares, d_h)
            h_t1 = h_t1.to(d)            # (B, d_h)
            labels = labels.to(d)        # (B,)

            logits = model_engine(h_y_all, h_t1)     # (B, num_squares)
            loss = criterion(logits, labels)
            model_engine.backward(loss)
            model_engine.step()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss, train_acc = gather_metrics(running_loss, correct, total)
        val_loss, val_acc = evaluate(model_engine, val_loader)

        printonce(f"Epoch {epoch+1:02d} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

import torch
import torch.nn as nn
class BilinearProbe(nn.Module):
    def __init__(self, d_attn, d_h, num_squares = 64):
        """
        d_attn: 64  — attention/bottleneck dimension (also num squares)
        d_h:   768  — hidden state dimension
        """
        super().__init__()
        self.U = nn.Parameter(torch.randn(d_attn, d_h) * 0.01)  # (64, 768)
        self.V = nn.Parameter(torch.randn(d_attn, d_h) * 0.01)  # (64, 768)
        self.c = nn.Parameter(torch.zeros(num_squares))              # (64,)
        self.U.to(training_dtype)
        self.V.to(training_dtype)
        self.c.to(training_dtype)

    def forward(self, h_y_all, h_t1):
        """
        h_y_all: (B, num_sqaures, d_h)  — h^L for each candidate square y
        h_t1:    (B, d_h)     — h^L at position t1
        Returns:  (B, num_squares)     — one logit per square, ready to softmax
        """
        # Project candidate squares: (B, num_squares, d_h) @ (d_h, d_attn) -> (B, num_squares, d_attn)
        Uh = h_y_all @ self.U.T

        # Project conditioning token: (B, d_h) @ (d_h, d_attn) -> (B, d_attn)
        Vh = h_t1 @ self.V.T

        # (B, num_sqaures, d_attn) @ (B, d_attn, 1) -> (B, num_squares, 1) -> (B, num_sqaures)
        logits = (Uh @ Vh.unsqueeze(-1))\
        .squeeze(-1) + self.c
        return logits
    
    def predict(self, h_y_all, h_t1):
        """
        Same input shapes as forward(). Returns (B, d_attn) softmax probabilities
        (or (d_attn,) if the inputs were unbatched).
        """
        logits = self.forward(h_y_all, h_t1)
        probs = torch.softmax(logits, dim=-1)
        return probs
        
def main():
    import deepspeed
    import os

    num_batches = 5
    samples_per_epoch = 600
    # generateActivations drops duplicate FENs and puzzles that are too short,
    # so pull more rows than we actually need.
    ROW_HEADROOM = 4

    d_attn = 32
    d_h = 768
    num_squares = 64
    probe = BilinearProbe(d_attn=d_attn, d_h=d_h, num_squares=num_squares)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "ds_config.json")

    # No training_data here: the probe trains on activations, not on raw puzzle
    # rows. Handing the puzzle dataset to DeepSpeed makes it collate variable
    # -length columns (Themes, etc.), which fails, and yields batches where
    # generateActivations expects single rows. We shard the puzzles ourselves
    # below instead.
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model = probe,
        model_parameters = probe.parameters(),
        config = config_path,
    )

    import torch
    import deepspeed.comm as dist
    from datasets.distributed import split_dataset_by_node

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    samples_per_rank = samples_per_epoch // world_size
    if samples_per_rank == 0:
        raise RuntimeError(
            f"samples_per_epoch={samples_per_epoch} is smaller than world_size={world_size}"
        )
    rows_per_batch = samples_per_rank * ROW_HEADROOM

    # Split the puzzles across ranks *before* generating activations, so the
    # lc0 forward pass is parallelized too rather than every rank redundantly
    # computing the same activations. Shards are disjoint, which is why
    # train_batch can use a plain DataLoader with no DistributedSampler.
    # puzzleStream = loadDataset("Lichess/chess-puzzles")
    puzzleStream = loadDataset("SubhamDB/chess-puzzles")
    puzzleShard = split_dataset_by_node(puzzleStream, rank=rank, world_size=world_size)

    # One iterator for the whole run, advanced batch by batch. Held here rather
    # than inside the loop so each batch materializes rows the previous batches
    # did not consume.
    rows_iter = iter(puzzleShard)

    printonce(
        f"Streaming across {world_size} rank(s): {rows_per_batch} rows and "
        f"{samples_per_rank} target samples per rank per batch, "
        f"{num_batches} batches."
    )

    for batch in range(num_batches):
        printonce(f"=== Batch {batch+1}/{num_batches} ===")
        puzzleData = toDataset(rows_iter, rows_per_batch)

        # A rank whose stream runs dry has to take every other rank down with
        # it, otherwise the survivors hang in train_batch's collectives.
        exhausted = torch.tensor(int(puzzleData is None), dtype=torch.long, device=model_engine.device)
        dist.all_reduce(exhausted, op=dist.ReduceOp.MAX)
        if int(exhausted.item()):
            printonce(f"Puzzle stream exhausted after {batch} batch(es); stopping.")
            break

        activations = generateActivations(puzzleData, samples_per_rank)
        train_batch(model_engine, *activations)
    
if __name__ == "__main__":
    main()
